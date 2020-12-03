Below you can find an outline of how to reproduce the solution by team "Hungry for gold🥇🥇" for the kaggle Mechanisms of Action (MoA) Prediction competition.

# Archive Contents
- final: scripts used for the final submissions
    - Best CV: set of model scripts used in our first submission, best CV score blending
    - Best LB: set of model scripts used in our second submission, best public LB score blending
        - Training: jupyter notebooks for each single models to preprocess the input data and save trained model weights. to be run in kaggle GPU notebook environment.
        - Inference: python scripts for each single models to preprocess the input data and make inferences using pre-trained weights.
        - Submission: predicted labels on public test data.
        - a notebook to blend single model predictions

# Hardware
We used kaggle notebook instance with GPU enabled to run all data preprocessing, model training, blending weight search and inference.

https://www.kaggle.com/docs/notebooks

# Software
We used kaggle GPU notebooks to run all our scripts. 

https://github.com/Kaggle/docker-python/blob/master/gpu.Dockerfile

Below are the packages used in addition to the ones included in the default kaggle docker python.

| package name | repository | kaggle dataset |
| --- |--- | --- |
| pytorch-lightning | https://github.com/PyTorchLightning/pytorch-lightning|https://www.kaggle.com/markpeng/pytorch-lightning |
| pytorch-optimizer | https://github.com/jettify/pytorch-optimizer |https://www.kaggle.com/markpeng/pytorch-optimizer |
| pytorch-ranger |https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer|https://www.kaggle.com/markpeng/pytorch-ranger |
| pytorch-tabnet=2.0.0 | https://github.com/dreamquark-ai/tabnet | https://www.kaggle.com/ryati131457/pytorchtabnet |
| ResNeSt| https://github.com/zhanghang1989/ResNeSt | https://www.kaggle.com/markpeng/resnest |
| umap-learn=0.4.6 | https://github.com/lmcinnes/umap | https://www.kaggle.com/kozistr/umaplearn|
| iterative-stratification |https://github.com/trent-b/iterative-stratification |https://www.kaggle.com/yasufuminakama/iterative-stratification |

<!-- python packages are also detailed separately in `requirements.txt` -->

# Data Setup

# Data Processing
Data processing was done separately in each single model scripts.

# Model Build
## Training

## Blend weights search

## Inference

