#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Reference:
# https://www.kaggle.com/demetrypascal/fork-of-2heads-looper-super-puper-plate/notebook

kernel_mode = True


# # Preparations

# Let’s load the packages and provide some constants for our script:

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from tensorflow.keras import layers, regularizers, Sequential, Model, backend, callbacks, optimizers, metrics, losses
import tensorflow as tf
import sys
import os
import random
import json
sys.path.append('../input/iterative-stratification')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import pickle
from pickle import dump, load
import glob

import warnings
warnings.filterwarnings('ignore')


# In[3]:


PATH = "../input/lish-moa" if kernel_mode else "/workspace/Kaggle/MoA"
model_output_folder = "../input/improving-mark-s-2-heads-model" if kernel_mode     else f"{PATH}/improving-mark-s-2-heads-model"
os.makedirs(model_output_folder, exist_ok=True)

# SEEDS = [23]
SEEDS = [23, 228, 1488, 1998, 2208, 2077, 404]
KFOLDS = 10

batch_size = 2048
# batch_size = 128
# batch_size = 64

label_smoothing_alpha = 0.0005

P_MIN = label_smoothing_alpha
P_MAX = 1 - P_MIN


# In[4]:


# Import train data, drop sig_id, cp_type
train_features = pd.read_csv(f'{PATH}/train_features.csv')

non_ctl_idx = train_features.loc[
    train_features['cp_type'] != 'ctl_vehicle'].index.to_list()

# Drop training data with ctl vehicle
tr = train_features.iloc[non_ctl_idx, :].reset_index(drop=True)

test_features = pd.read_csv(f'{PATH}/test_features.csv')
te = test_features.copy()


# In[5]:


train_targets_scored = pd.read_csv(f'{PATH}/train_targets_scored.csv')
Y = train_targets_scored.drop('sig_id', axis=1)
Y = Y.iloc[non_ctl_idx, :].copy().reset_index(drop=True).values

train_targets_nonscored = pd.read_csv(f'{PATH}/train_targets_nonscored.csv')
Y0 = train_targets_nonscored.drop('sig_id', axis=1)
Y0 = Y0.iloc[non_ctl_idx, :].copy().reset_index(drop=True).values

sub = pd.read_csv(f'{PATH}/sample_submission.csv')
sub.iloc[:, 1:] = 0


# # Features from t.test

# Here I am getting most important predictors

# In[6]:


# Import predictors from public kernel
json_file_path = '../input/t-test-pca-rfe-logistic-regression/main_predictors.json' if kernel_mode     else "/workspace/Kaggle/MoA/t-test-pca-rfe-logistic-regression/main_predictors.json"

with open(json_file_path, 'r') as j:
    predictors = json.loads(j.read())
    predictors = predictors['start_predictors']


# In[7]:


second_Xtrain = tr[predictors].copy().values

second_Xtest = te[predictors].copy().values
second_Xtrain.shape


# # Keras model

# I got idea of **label smoothing** from this notebook: https://www.kaggle.com/kailex/moa-transfer-recipe-with-smoothing

# In[8]:


def logloss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, P_MIN, P_MAX)
    return -backend.mean(y_true * backend.log(y_pred) +
                         (1 - y_true) * backend.log(1 - y_pred))


# # Inference

# In[9]:


numeric_features = [c for c in train_features.columns if c != "sig_id"]
gene_experssion_features = [c for c in numeric_features if c.startswith("g-")]
cell_viability_features = [c for c in numeric_features if c.startswith("c-")]
len(gene_experssion_features), len(cell_viability_features)


# In[10]:


tr = pd.get_dummies(tr, columns=["cp_time", "cp_dose"])


# In[11]:


tr = tr.drop(['sig_id', 'cp_type'], axis=1)
te = test_features.drop(['sig_id', 'cp_type'], axis=1)


# In[12]:


te = pd.get_dummies(te, columns=["cp_time", "cp_dose"])


# In[13]:


def preprocessor_1(test, seed, scaler=None, pca_gs=None, pca_cs=None):
    # g-mean, c-mean
    test_g_mean = test[gene_experssion_features].mean(axis=1)

    test_c_mean = test[cell_viability_features].mean(axis=1)

    test_columns = test.columns.tolist()

    test = np.concatenate(
        (test, test_g_mean[:, np.newaxis], test_c_mean[:, np.newaxis]), axis=1)

    # Standard Scaler for Numerical Values
    test = pd.DataFrame(data=scaler.transform(test),
                        columns=test_columns + ["g_mean", "c_mean"])
    test_pca_gs = pca_gs.transform(test[gene_experssion_features].values)

    test_pca_cs = pca_cs.transform(test[cell_viability_features].values)

    # Append Features
    test = np.concatenate((test, test_pca_gs, test_pca_cs), axis=1)

    return test


def preprocessor_2(test, scaler=None):
    # Standard Scaler for Numerical Values
    test = scaler.transform(test)

    return test, scaler


def save_pickle(obj, model_output_folder, name):
    dump(obj, open(f"{model_output_folder}/{name}.pkl", 'wb'),
         pickle.HIGHEST_PROTOCOL)


def load_pickle(model_output_folder, name):
    return load(open(f"{model_output_folder}/{name}.pkl", 'rb'))


def mean_logloss(y_pred, y_true):
    logloss = (1 - y_true) * np.log(1 - y_pred +
                                    1e-15) + y_true * np.log(y_pred + 1e-15)
    return np.mean(-logloss)


# In[14]:


tr.shape, te.shape


# In[15]:


oof_predictions = np.zeros((tr.shape[0], Y.shape[1]))

y_pred = np.zeros((te.shape[0], 206))
for s in SEEDS:

    random.seed(s)
    np.random.seed(s)
    tf.random.set_seed(s)

    k = 0
    kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=s)
    for train_index, valid_index in kf.split(tr):
        file_name = f"seed{s}_fold{k}"

        print(f"Inferencing on seed{s} fold{k} ......")

        scaler_1 = load_pickle(model_output_folder, f"{file_name}_scaler_1")
        pca_gs = load_pickle(model_output_folder, f"{file_name}_pca_gs")
        pca_cs = load_pickle(model_output_folder, f"{file_name}_pca_cs")
        X_test_1 = preprocessor_1(te, s, scaler_1, pca_gs, pca_cs)

        scaler_2 = load_pickle(model_output_folder, f"{file_name}_scaler_2")
        X_test_2, scaler_2 = preprocessor_2(second_Xtest, scaler_2)

        n_features = X_test_1.shape[1]
        n_features_2 = X_test_2.shape[1]

        # Model Definition #

        input1_ = layers.Input(shape=(n_features, ))
        input2_ = layers.Input(shape=(n_features_2, ))

        output1 = Sequential([
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(512, activation="elu"),
            layers.BatchNormalization(),
            layers.Dense(256, activation="elu")
        ])(input1_)

        answer1 = Sequential([
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(512, "relu")
        ])(layers.Concatenate()([output1, input2_]))

        answer2 = Sequential([
            layers.BatchNormalization(),
            layers.Dense(512, "elu"),
            layers.BatchNormalization(),
            layers.Dense(256, "relu")
        ])(layers.Concatenate()([output1, input2_, answer1]))

        answer3 = Sequential(
            [layers.BatchNormalization(),
             layers.Dense(256,
                          "elu")])(layers.Concatenate()([answer1, answer2]))

        answer3_ = Sequential([
            layers.BatchNormalization(),
            layers.Dense(256, "relu")
        ])(layers.Concatenate()([answer1, answer2, answer3]))

        answer4 = Sequential([
            layers.BatchNormalization(),
            layers.Dense(
                256,
                kernel_initializer=tf.keras.initializers.lecun_normal(seed=s),
                activation='selu',
                name='last_frozen'),
            layers.BatchNormalization(),
            layers.Dense(
                206,
                kernel_initializer=tf.keras.initializers.lecun_normal(seed=s),
                activation='selu')
        ])(layers.Concatenate()([output1, answer2, answer3, answer3_]))

        # Scored Training #

        answer5 = Sequential(
            [layers.BatchNormalization(),
             layers.Dense(Y.shape[1], "sigmoid")])(answer4)

        m_nn = tf.keras.Model([input1_, input2_], answer5)

        m_nn.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                     loss=losses.BinaryCrossentropy(
                         label_smoothing=label_smoothing_alpha),
                     metrics=logloss)

        # Load final model
        m_nn.load_weights(f'{model_output_folder}/{file_name}_final.h5')

        # Generate Submission Prediction #
        fold_submit_preds = m_nn.predict([X_test_1, X_test_2],
                                         batch_size=batch_size)
        y_pred += fold_submit_preds / (KFOLDS * len(SEEDS))
        print(fold_submit_preds[:5, :])

        k += 1

        print('\n')


# In[16]:


oof_predictions = glob.glob(f'{model_output_folder}/oof_*.npy')[0]
oof_predictions = np.load(oof_predictions)

oof_loss = mean_logloss(oof_predictions, Y)
print(f"OOF Validation Loss: {oof_loss:.6f}")


# # Submission

# In[17]:


sub.iloc[:, 1:] = y_pred
# sub.iloc[:, 1:] = np.clip(y_pred, P_MIN, P_MAX)


# In[18]:


sub


# In[19]:


# Set ctl_vehicle to 0
sub.iloc[test_features['cp_type'] == 'ctl_vehicle', 1:] = 0

# Save Submission
sub.to_csv('submission_improving-mark-s-2-heads-model.csv', index=False)
# sub.to_csv('submission.csv', index=False)


# In[20]:


sub


# In[ ]:




