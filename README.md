# 1st Place Winning Solution - Mechanisms of Action (MoA) Prediction

This documentation outlines how to reproduce the 1st place solution by the ***Hungry for gold🥇🥇*** team for the Mechanisms of Action (MoA) Prediction competition on Kaggle.

[Winning Solution Writeup on Kaggle](https://www.kaggle.com/c/lish-moa/discussion/201510)

[Winning Solution Writeup in PDF](moa_hungry_for_gold_winning_solution_writeup.pdf)

[How to Reproduce Winning Solution Locally](final/README.md)

## Archive Contents

- In the `final` folder: All of the scripts used for the final submissions
    - **Best CV:** A set of model scripts used in our first submission, best CV score blending
    - **Best LB:** A set of model scripts used in our second submission, best public LB score blending
        - **Training:** Includes Jupyter notebooks for each single models to preprocess the input data and save trained model weights. to be run in kaggle GPU notebook environment.
        - **Inference:** Includes Python scripts for each single models to preprocess the input data and make inferences using pre-trained weights. Note that for the `2-StageNN+TabNet` model, we were running it as a notebooks due to unknown Kaggle environment errors to the `UMAP` dependency library "`numba.core`".
        - **Submission:** Includes predicted labels on public test data.
        - A notebook to blend single model predictions

## Hardware

Most of our single models were using Kaggle Notebook instances with GPU enabled to run all data preprocessing, model training, blending weight search and inference.

[https://www.kaggle.com/docs/notebooks](https://www.kaggle.com/docs/notebooks)

For [DeepInsight CNNs](https://www.kaggle.com/c/lish-moa/discussion/195378), such as `EfficientNet B3 NS` and `ResNeSt`, they were trained on a local machine with 64GB RAM and two Nividia 2080-Ti GPUs. Each of them took about 12-25 hours to train for 10-folds.


## Software

We used [Kaggle GPU notebooks](https://github.com/Kaggle/docker-python/blob/master/gpu.Dockerfile) to run all our inference scripts.

Below are the packages used in addition to the ones included in the default Kaggle Docker environment for Python. All packages were installed via uploaded kaggle dataset.

| Package Name | Repository | Kaggle Dataset |
| --- |--- | --- |
| pytorch-lightning=1.0.2 | https://github.com/PyTorchLightning/pytorch-lightning|https://www.kaggle.com/markpeng/pytorch-lightning |
| pytorch-optimizer=0.0.1a17 | https://github.com/jettify/pytorch-optimizer |https://www.kaggle.com/markpeng/pytorch-optimizer |
| pytorch-ranger=0.1.1 |https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer|https://www.kaggle.com/markpeng/pytorch-ranger |
| pytorch-tabnet=2.0.0 | https://github.com/dreamquark-ai/tabnet | https://www.kaggle.com/ryati131457/pytorchtabnet |
| ResNeSt=0.0.6| https://github.com/zhanghang1989/ResNeSt | https://www.kaggle.com/markpeng/resnest |
| umap-learn=0.4.6 | https://github.com/lmcinnes/umap | https://www.kaggle.com/kozistr/umaplearn|
| iterative-stratification=0.1.6 |https://github.com/trent-b/iterative-stratification |https://www.kaggle.com/yasufuminakama/iterative-stratification |
| gen-efficientnet-pytorch |https://github.com/rwightman/gen-efficientnet-pytorch |https://www.kaggle.com/markpeng/gen-efficientnet-pytorch |


## Data Setup

Please add https://www.kaggle.com/c/lish-moa/data as the input dataset.

## Model Build

### Models Summary

#### Best LB Blend

| Model Name| CV | Public LB | Private LB | Training Notebook | Inference Script |
|-|-|-|-|-|-|
|3-stage NN|0.01561|0.01823|0.01618|3-stagenn-train.ipynb|3stage-nn-inference.py|
|2-stage NN + TabNet |0.01615|0.01837|0.01625|2stagenn-tabnet-train.ipynb|2stage-nn-tabnet-inference.ipynb|
|Simple NN (old CV)|0.01585|0.01833|0.01626|simple-nn-using-old-cv-train.ipynb|simple-nn-old-split-inference.py|
|Simple NN (new CV)|0.01564|0.01830|0.01622|simple-nn-new-split-train.ipynb|simple-nn-new-split-inference.py|
|2-heads ResNet |0.01589|0.01836|0.01624|2heads-ResNest-train.ipynb|2heads-ResNest-inference.py|
|EfficientNet B3 Noisy Student |0.01602|0.01850|0.01634|deepinsight-efficientnet-lightning-v7-b3-train.ipynb|deepinsight-efficientnet-lightning-v7-b3-inference.py|
|ResNeSt V2|0.01576|0.01854|0.01636|deepinsight-resnest-lightning-v2-train.ipynb|deepinsight-resnest-lightning-v2-inference.py |

 - Original Submission Notebook: `./final/Best LB/fork-of-blending-with-6-models-5old-1new.ipynb`
 - Cleaned Submission Notebook: `./final/Best LB/final-best-lb-cleaned.ipynb`

#### Best CV Blend

| Model Name| CV | Public LB | Private LB | Training Notebook | Inference Script |
|-|-|-|-|-|-|
|3-stage NN|0.01561|0.01823|0.01618|3stagenn-10folds-train.ipynb|3stagenn-10folds-inference.py|
|2-heads ResNet V2 |0.01566|0.01844|0.01623|2heads-resnest-train.ipynb|2heads-resnest-inference.py|
|EfficientNet B3 Noisy Student |0.01602|0.01850|0.01634|deepinsight-efficientnet-lightning-v7-b3-train.ipynb|deepinsight-efficientnet-lightning-v7-b3-inference.py|
|ResNeSt V1|0.01582|0.01853|0.01636|deepinsight-resnest-lightning-v1-train.ipynb|deepinsight-resnest-lightning-v1-inference.py|
|ResNeSt V2|0.01576|0.01854|0.01636|deepinsight-resnest-lightning-v2-train.ipynb|deepinsight-resnest-lightning-v2-inference.py|

- Original Submission Notebook: `./final/Best CV/blend-search-optuna-v7.ipynb`
- Cleaned Submission Notebook: `./final/Best CV/final-best-CV-cleaned.ipynb`

## Training

All of the training can be done by running notebooks above as a kaggle notebook. This generates pickled preprocessing modules, model weights used by inference scripts, and predictions used in blend weights search.

## Blend Weights Search

Once oof predictions are generated, run the blend weight search notebooks to determine good blending weights for the set of models. The weights in the submission notebooks need to be updated manually.

## Inference

The submission notebooks make inference by running each single-model inference scripts and blending the predictions. All of the training notebooks must be added as dataset to load preprocessing class instances and model weights.

To make predictions on a new dataset, you just need to replace `test_features.csv` in input dataset.

## How to Reproduce Winning Solution Locally

Please refer to [this documentation](final/README.md).

## License

All of our solution code are open-sourced under the [Apache 2.0](LICENSE) license.
