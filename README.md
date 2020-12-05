# 1st Place Winning Solution - Mechanisms of Action (MoA) Prediction

This documentation outlines how to reproduce the 1st place solution by team "Hungry for gold🥇🥇" for the kaggle Mechanisms of Action (MoA) Prediction competition.

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

Below are the packages used in addition to the ones included in the default Kaggle Docker environment for Python.

| Package Name | Repository | Kaggle Dataset |
| --- |--- | --- |
| pytorch-lightning | https://github.com/PyTorchLightning/pytorch-lightning|https://www.kaggle.com/markpeng/pytorch-lightning |
| pytorch-optimizer | https://github.com/jettify/pytorch-optimizer |https://www.kaggle.com/markpeng/pytorch-optimizer |
| pytorch-ranger |https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer|https://www.kaggle.com/markpeng/pytorch-ranger |
| pytorch-tabnet=2.0.0 | https://github.com/dreamquark-ai/tabnet | https://www.kaggle.com/ryati131457/pytorchtabnet |
| ResNeSt| https://github.com/zhanghang1989/ResNeSt | https://www.kaggle.com/markpeng/resnest |
| umap-learn=0.4.6 | https://github.com/lmcinnes/umap | https://www.kaggle.com/kozistr/umaplearn|
| iterative-stratification |https://github.com/trent-b/iterative-stratification |https://www.kaggle.com/yasufuminakama/iterative-stratification |

<!-- python packages are also detailed separately in `requirements.txt` -->

## License

All of our solution code are open-sourced under the [Apache 2.0](LICENSE) license.
