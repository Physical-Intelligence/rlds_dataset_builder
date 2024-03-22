from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow_datasets as tfds
import tensorflow_hub as hub

# from gello_ur_dataset_builder import dataset_feature_specs


import tensorflow_datasets as tfds
import numpy as np
import rlds

SPECS = tfds.features.FeaturesDict(
    {
        rlds.STEPS: tfds.features.Dataset(
            {
                rlds.OBSERVATION: tfds.features.FeaturesDict(
                    {
                        "base_image": tfds.features.Image(
                            shape=(480, 480, 3),
                            dtype=np.uint8,
                            encoding_format="jpeg",
                            doc="Main camera RGB observation.",
                        ),
                        "wrist_image": tfds.features.Image(
                            shape=(480, 480, 3),
                            dtype=np.uint8,
                            encoding_format="jpeg",
                            doc="Wrist camera RGB observation.",
                        ),
                        "state": tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc="Robot state, consists of 7 robot joint angles, "
                        ),
                    }
                ),
                rlds.ACTION: tfds.features.Tensor(
                    shape=(7,),
                    dtype=np.float32,
                    doc="Robot action, consists of 7 robot joint angles, "
                ),
                rlds.DISCOUNT: tfds.features.Scalar(
                    dtype=np.float32, doc="Discount if provided, default to 1."
                ),
                rlds.REWARD: tfds.features.Scalar(
                    dtype=np.float32,
                    doc="Reward if provided, 1 on final step for demos.",
                ),
                rlds.IS_FIRST: tfds.features.Scalar(
                    dtype=np.bool_, doc="True on first step of the episode."
                ),
                rlds.IS_LAST: tfds.features.Scalar(
                    dtype=np.bool_, doc="True on last step of the episode."
                ),
                rlds.IS_TERMINAL: tfds.features.Scalar(
                    dtype=np.bool_,
                    doc="True on last step of the episode if it is a terminal step, True for demos.",
                ),
                "language_instruction": tfds.features.Text(doc="Language Instruction."),
                "language_embedding": tfds.features.Tensor(
                    shape=(512,),
                    dtype=np.float32,
                    doc="Kona language embedding. "
                    "See https://tfhub.dev/google/universal-sentence-encoder-large/5",
                ),
            }
        ),
        "episode_metadata": tfds.features.FeaturesDict(
            {
                "file_path": tfds.features.Text(doc="Path to the original data file."),
            }
        ),
    }
)


class GelloUrDatasetBuilder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=SPECS)

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='data/episode_*.pkl'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            data = np.load(episode_path, allow_pickle=True)     # this is a list of dicts in our case

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            num_steps = len(data['base_rgb'])
            for i in range(num_steps):
                task = "bus table"
                language_embedding = self._embed([task])[0].numpy()
                episode.append({
                    'observation': {
                        'base_image': data['base_rgb'][i],
                        'wrist_image': data['wrist_rgb'][i],
                        'state': data['joint_positions'][i].astype(np.float32),
                    },
                    'action': data['control'][i].astype(np.float32),
                    'discount': 1.0,
                    'reward': float(i == (num_steps - 1)),
                    'is_first': i == 0,
                    'is_last': i == (num_steps - 1),
                    'is_terminal': i == (num_steps - 1),
                    'language_instruction': task,
                    'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(path)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

