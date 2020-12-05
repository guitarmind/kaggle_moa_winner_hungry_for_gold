Below you can find an outline of how to reproduce the solution by team "Hungry for gold🥇🥇" for the kaggle Mechanisms of Action (MoA) Prediction competition.

# Archive Contents
- blends: scripts used to determine best model blending weights
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

Below are the packages used in addition to the ones included in the default kaggle docker python. All packages are installed via uploaded kaggle dataset.

| package name | repository | kaggle dataset |
| --- |--- | --- |
| pytorch-lightning=1.0.2 | https://github.com/PyTorchLightning/pytorch-lightning|https://www.kaggle.com/markpeng/pytorch-lightning |
| pytorch-optimizer=0.0.1a17 | https://github.com/jettify/pytorch-optimizer |https://www.kaggle.com/markpeng/pytorch-optimizer |
| pytorch-ranger=0.1.1 |https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer|https://www.kaggle.com/markpeng/pytorch-ranger |
| pytorch-tabnet=2.0.0 | https://github.com/dreamquark-ai/tabnet | https://www.kaggle.com/ryati131457/pytorchtabnet |
| ResNeSt=0.0.6| https://github.com/zhanghang1989/ResNeSt | https://www.kaggle.com/markpeng/resnest |
| umap-learn=0.4.6 | https://github.com/lmcinnes/umap | https://www.kaggle.com/kozistr/umaplearn|
| iterative-stratification=0.1.6 |https://github.com/trent-b/iterative-stratification |https://www.kaggle.com/yasufuminakama/iterative-stratification |

<!-- python packages are also detailed separately in `requirements.txt` -->

# Data Setup
Add https://www.kaggle.com/c/lish-moa/data as input dataset.

# Model Build
## Models Summary
### Best LB blend
| model name| cv | public lb | private lb | training notebook | inference script |
|-|-|-|-|-|-|
|3-stage NN|0.01561|0.01823|0.01618|3-stagenn-train.ipynb|3stage-nn-inference.py|
|2-stage NN + TabNet |0.01615|0.01837|0.01625|2stagenn-tabnet-train.ipynb|2stage-nn-tabnet-inference.ipynb|
|Simple NN (old CV)|0.01585|0.01833|0.01626|simple-nn-using-old-cv-train.ipynb|simple-nn-old-split-inference.py|
|Simple NN (new CV)|0.01564|0.01830|0.01622|simple-nn-new-split-train.ipynb|simple-nn-new-split-inference.py|
|2-heads ResNet |0.01589|0.01836|0.01624|2heads-ResNest-train.ipynb|2heads-ResNest-inference.py|
|EfficientNet B3 Noisy Student |0.01602|0.01850|0.01634|deepinsight-efficientnet-lightning-v7-b3-train.ipynb|deepinsight-efficientnet-lightning-v7-b3-inference.py|
|ResNeSt V2|0.01576|0.01854|0.01636|deepinsight-resnest-lightning-v2-train.ipynb|deepinsight-resnest-lightning-v2-inference.py |

- blend weight search: blends\blend_search_optuna_v7.ipynb
- submission notebook: final\Best LB\fork-of-blending-with-6-models-5old-1new.ipynb

### Best CV blend

| model name| cv | public lb | private lb | training notebook | inference script |
|-|-|-|-|-|-|
|3-stage NN|0.01561|0.01823|0.01618|3stagenn-10folds-train.ipynb|3stagenn-10folds-inference.py|
|2-heads ResNet V2 |0.01566|0.01844|0.01623|2heads-resnest-train.ipynb|2heads-resnest-inference.py|
|EfficientNet B3 Noisy Student |0.01602|0.01850|0.01634|deepinsight-efficientnet-lightning-v7-b3-train.ipynb|deepinsight-efficientnet-lightning-v7-b3-inference.py|
|ResNeSt V1|0.01582|0.01853|0.01636|deepinsight-resnest-lightning-v1-train.ipynb|deepinsight-resnest-lightning-v1-inference.py|
|ResNeSt V2|0.01576|0.01854|0.01636|deepinsight-resnest-lightning-v2-train.ipynb|deepinsight-resnest-lightning-v2-inference.py|

- blend weight search: blends\blending-with-6-models-5old-1new.ipynb
- submission notebook: final\Best CV\blend-search-optuna-v7.ipynb

## Training
All of the trainings can be done by running notebooks above as a kaggle notebook. This generates pickled preprocessing modules, model weights used by inference scripts, and predictions used in blend weights search.

## Blend weights search
Once oof predictions are generated, run the blend weight search notebooks to determine good blending weights for the set of models. The weights in the submission notebooks need to be updated manually.

## Inference
The submission notebooks make inference by running each single-model inference scripts and blending the predictions. All of the training notebooks must be added as dataset to load preprocessing class instances and model weights.

To make predictions on a new dataset, you just need to replace "test_features.csv" in input dataset.
