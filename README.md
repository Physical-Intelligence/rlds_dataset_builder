# RLDS Dataset Conversion

This repo demonstrates how to convert an existing dataset into RLDS format for X-embodiment experiment integration.
It provides an example for converting a dummy dataset to RLDS. To convert your own dataset, **fork** this repo and 
modify the example code for your dataset following the steps below.

## Installation

First create a conda environment using the provided environment.yml file:
```
conda env create -f environment_ubuntu.yml
```

Then activate the environment using:
```
conda activate rlds_env
```

If you want to manually create an environment, the key packages to install are `tensorflow`, 
`tensorflow_datasets`, `tensorflow_hub`, `apache_beam`, `matplotlib`, `plotly`.


## Run Example RLDS Dataset Creation

Before modifying the code to convert your own dataset, run the provided example dataset creation script to ensure
everything is installed correctly. Run the following lines to create some dummy data and convert it to RLDS.
```
cd example_dataset
python3 create_example_data.py
tfds build
```

This should create a new dataset in `~/tensorflow_datasets/example_dataset`. Please verify that the example
conversion worked before moving on.


## Converting your Own Dataset to RLDS

Now we can modify the provided example to convert your own data. Follow the steps below:

1. **Rename Dataset**: Change the name of the dataset folder from `example_dataset` to the name of your dataset (e.g. robo_net_v2), 
also change the name of `example_dataset_dataset_builder.py` by replacing `example_dataset` with your dataset's name (e.g. robo_net_v2_dataset_builder.py)
and change the class name `ExampleDataset` in the same file to match your dataset's name, using camel case instead of underlines (e.g. RoboNetV2).

2. **Modify Features**: Modify the data fields you plan to store in the dataset. You can find them in `example_dataset/dataset_feature_specs.py`. Please add **all** data fields your raw data contains, i.e. please add additional features for 
additional cameras, audio, tactile features etc. If your type of feature is not demonstrated in the example (e.g. audio),
you can find a list of all supported feature types [here](https://www.tensorflow.org/datasets/api_docs/python/tfds/features?hl=en#classes).
You can store step-wise info like camera images, actions etc in `rlds.STEPS` and episode-wise info like `collector_id` in `episode_metadata`. Please add detailed documentation what each feature consists of (e.g. what are the dimensions of the action space etc.).
Note that we store `language_instruction` in every step even though it is episode-wide information for easier downstream usage (if your dataset
does not define language instructions, please feel free to remove the `language_instruction` and `language_embedding` features).

3. **Modify Dataset Splits**: The function `_split_generator()` determines the splits of the generated dataset (e.g. training, validation etc.).
If your dataset defines a train vs validation split, please provide the corresponding information to `_generate_examples()`, e.g. 
by pointing to the corresponding folders (like in the example) or file IDs etc. If your dataset does not define splits,
remove the `val` split and only include the `train` split. You can then remove all arguments to `_generate_examples()`.

4. **Modify Dataset Conversion Code**: Next, modify the function `_generate_examples()`. Here, your own raw data should be 
loaded, filled into the episode steps and then yielded as a packaged example. Note that the value of the first return argument,
`episode_path` in the example, is only used as a sample ID in the dataset and can be set to any value that is connected to the 
particular stored episode, or any other random value. Just ensure to avoid using the same ID twice.

That's it! You're all set to run dataset conversion. Inside the dataset directory, run:
```
tfds build --overwrite
```
The command line output should finish with a summary of the generated dataset (including size and number of samples). 
Please verify that this output looks as expected and that you can find the generated `tfrecord` files in `~/tensorflow_datasets/<name_of_your_dataset>`.


### Parallelizing Data Processing
By default, dataset conversion is single-threaded. If you are parsing a large dataset, you can use parallel processing.
For this, replace the last two lines of `_generate_examples()` with the commented-out `beam` commands. This will use 
Apache Beam to parallelize data processing. Before starting the processing, you need to install your dataset package 
by filling in the name of your dataset into `setup.py` and running `pip install -e .`

Then, make sure that no GPUs are used during data processing (`export CUDA_VISIBLE_DEVICES=`) and run:
```
tfds build --overwrite --beam_pipeline_options="direct_running_mode=multi_processing,direct_num_workers=10"
```
You can specify the desired number of workers with the `direct_num_workers` argument.

## Visualize Converted Dataset
To verify that the data is converted correctly, please visualize the data using `visualize.ipynb`:

This will display a few random episodes from the dataset with language commands and visualize action and state histograms per dimension.

## Add Transform for Target Spec

For X-embodiment training we are using specific inputs / outputs for the model: input is a single RGB camera, output
is an 8-dimensional action, consisting of end-effector position and orientation, gripper open/close and a episode termination
action.

The final step in adding your dataset to the training mix is to provide a transform function, that transforms a step
from your original dataset above to the required training spec. Please follow the two simple steps below:

1. **Modify Step Transform**: Modify the function `transform_step()` in `example_transform/transform.py`. The function 
takes in a step from your dataset above and is supposed to map it to the desired output spec. The file contains a detailed
description of the desired output spec.

2. **Test Transform**: We provide a script to verify that the resulting __transformed__ dataset outputs match the desired
output spec. Please run the following command: `python3 test_dataset_transform.py <name_of_your_dataset>`
