#!/bin/bash

INPUT_FOLDER=$1
MODEL_FOLDER=$2
SCRIPT_PATH="Best LB/Training/Local"

echo "Input Dataset Folder: $INPUT_FOLDER \n"

echo "Training 3-Stage NN Model  ... \n"
python "$SCRIPT_PATH"/3stage-nn-train.py $INPUT_FOLDER "$MODEL_FOLDER"/3-stageNN

echo "Training 2-Stage NN+TabNet Model  ... \n"
python "$SCRIPT_PATH"/2stagenn-tabnet-train.py $INPUT_FOLDER "$MODEL_FOLDER"/2-stageNN-TabNet

echo "Training Simple NN Old CV Model  ... \n"
python "$SCRIPT_PATH"/simple-nn-old-split-train.py $INPUT_FOLDER "$MODEL_FOLDER"/simple-NN-Old-CV

echo "Training Simple NN New CV Model  ... \n"
python "$SCRIPT_PATH"/simple-nn-new-split-train.py $INPUT_FOLDER "$MODEL_FOLDER"/simple-NN-New-CV

echo "Training 2-heads ResNet Model  ... \n"
python "$SCRIPT_PATH"/2heads-ResNest-train.py $INPUT_FOLDER "$MODEL_FOLDER"/2-heads-ResNet

echo "Training DeepInsight EfficientNet B3 NS Model  ... \n"
python "$SCRIPT_PATH"/deepinsight-efficientnet-lightning-v7-b3-train.py $INPUT_FOLDER "$MODEL_FOLDER"/DeepInsight-EfficientNet-B3-NS

echo "Training DeepInsight ResNeSt V2 Model  ... \n"
python "$SCRIPT_PATH"/deepinsight-resnest-lightning-v2-train.py $INPUT_FOLDER "$MODEL_FOLDER"/DeepInsight-ResNeSt-V2

exit 0;