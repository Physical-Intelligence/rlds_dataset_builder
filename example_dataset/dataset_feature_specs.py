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
                            encoding_format="png",
                            doc="Main camera RGB observation.",
                        ),
                        "wrist_image": tfds.features.Image(
                            shape=(480, 480, 3),
                            dtype=np.uint8,
                            encoding_format="png",
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
