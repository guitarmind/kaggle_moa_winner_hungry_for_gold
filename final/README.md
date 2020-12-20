# How to Reproduce Winning Solution Locally

This documentation provides the steps of training, inferencing and blending to reproduce our winning solution.

## Local Setup

Install required dependency libraries first from the root project folder.

```
pip install -r requirements.txt
```

It is recommended to run training and inference on a local machine with at least 4 CPUs, 16GB Host Memory and 11GB GPU Memory (Nvidia 2080-Ti or better GPU).


## Training All Single Models

Run the following shell script with the right absolute paths for input dataset and model artifacts. All scripts should be ran from the `final` folder path.

```
sh train.sh <input folder> <model folder>
```

All models will be saved under the `final` folder with fixed folder names.

## Run Inference on All Single Models

Next, run the following shell script with the right absolute path for input dataset and trained model artifacts.

```
sh inference.sh <input folder> <model folder> <output folder>
```

All model predictions will be saved under the target output folder with fixed submission filenames.

## Blend Model Inference Outputs

Finally, run the following python script to create the winning solution prediction file.

```
python blend.py <input folder> <prediction file folder> <output folder>
```

That's it!
