#!/usr/bin/env python
# coding: utf-8

# In[1]:

kernel_mode = False

import sys
if kernel_mode:
    sys.path.append('../input/iterative-stratification/')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import argparse
model_artifact_name = "simple-nn-old-cv"
parser = argparse.ArgumentParser(description='Inferencing SimpleNN Old CV')
parser.add_argument('input', metavar='INPUT',
                    help='Input folder', default=".")
parser.add_argument('model', metavar='MODEL',
                    help='Model folder', default=".")
parser.add_argument('output', metavar='OUTPUT',
                    help='Output folder', default=".")
parser.add_argument('--batch-size', type=int, default=2048,
                    help='Batch size')
args = parser.parse_args()
input_folder = args.input
model_folder = args.model
output_folder = args.output

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
import seaborn as sns

from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')


# In[3]:


from sklearn.preprocessing import QuantileTransformer


# In[4]:


dataset_folder = "../input/lish-moa" if kernel_mode else input_folder
model_output_folder = "../input/simple-nn-using-old-cv" if kernel_mode \
    else model_folder
BATCH_SIZE = args.batch_size

if kernel_mode:
    os.listdir(dataset_folder)


# In[5]:


train_features = pd.read_csv(f'{dataset_folder}/train_features.csv')
train_targets_scored = pd.read_csv(
    f'{dataset_folder}/train_targets_scored.csv')
train_targets_nonscored = pd.read_csv(
    f'{dataset_folder}/train_targets_nonscored.csv')

test_features = pd.read_csv(f'{dataset_folder}/test_features.csv')
sample_submission = pd.read_csv(f'{dataset_folder}/sample_submission.csv')


# In[6]:


GENES = [col for col in train_features.columns if col.startswith('g-')]
CELLS = [col for col in train_features.columns if col.startswith('c-')]


# In[7]:


IS_TRAIN = False


# In[8]:


for col in (GENES + CELLS):

    vec_len = len(train_features[col].values)
    vec_len_test = len(test_features[col].values)
    raw_vec = train_features[col].values.reshape(vec_len, 1)
    if IS_TRAIN:
        transformer = QuantileTransformer(
            n_quantiles=100, random_state=0, output_distribution="normal")
        transformer.fit(raw_vec)
        pd.to_pickle(transformer, f'{model_output_folder}/{col}_quantile_transformer.pkl')
    else:
        transformer = pd.read_pickle(f'{model_output_folder}/{col}_quantile_transformer.pkl')

    train_features[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
    test_features[col] = transformer.transform(
        test_features[col].values.reshape(vec_len_test, 1)).reshape(1, vec_len_test)[0]


# In[9]:


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed=42)


# In[10]:


# GENES
n_comp = 90  # <--Update

data = pd.concat([pd.DataFrame(train_features[GENES]),
                  pd.DataFrame(test_features[GENES])])
if IS_TRAIN:
    fa = FactorAnalysis(n_components=n_comp,
                        random_state=1903).fit(data[GENES])
    pd.to_pickle(fa, f'{model_output_folder}/factor_analysis_g.pkl')
else:
    fa = pd.read_pickle(f'{model_output_folder}/factor_analysis_g.pkl')
data2 = fa.transform(data[GENES])
train2 = data2[:train_features.shape[0]]
test2 = data2[-test_features.shape[0]:]

train2 = pd.DataFrame(train2, columns=[f'pca_G-{i}' for i in range(n_comp)])
test2 = pd.DataFrame(test2, columns=[f'pca_G-{i}' for i in range(n_comp)])

train_features = pd.concat((train_features, train2), axis=1)
test_features = pd.concat((test_features, test2), axis=1)


# In[11]:


# CELLS
n_comp = 50  # <--Update

data = pd.concat([pd.DataFrame(train_features[CELLS]),
                  pd.DataFrame(test_features[CELLS])])
if IS_TRAIN:
    fa = FactorAnalysis(n_components=n_comp,
                        random_state=1903).fit(data[CELLS])
    pd.to_pickle(fa, f'{model_output_folder}/factor_analysis_c.pkl')
else:
    fa = pd.read_pickle(f'{model_output_folder}/factor_analysis_c.pkl')
data2 = fa.transform(data[CELLS])
train2 = data2[:train_features.shape[0]]
test2 = data2[-test_features.shape[0]:]

train2 = pd.DataFrame(train2, columns=[f'pca_C-{i}' for i in range(n_comp)])
test2 = pd.DataFrame(test2, columns=[f'pca_C-{i}' for i in range(n_comp)])

train_features = pd.concat((train_features, train2), axis=1)
test_features = pd.concat((test_features, test2), axis=1)


# In[12]:


train_features.shape


# In[13]:


from sklearn.feature_selection import VarianceThreshold


var_thresh = QuantileTransformer(
    n_quantiles=100, random_state=0, output_distribution="normal")

data = train_features.append(test_features)
if IS_TRAIN:
    transformer = QuantileTransformer(
        n_quantiles=100, random_state=123, output_distribution="normal")
    transformer.fit(data.iloc[:, 5:])
    pd.to_pickle(transformer, f'{model_output_folder}/{col}_quantile_transformer2.pkl')
else:
    transformer = pd.read_pickle(f'{model_output_folder}/{col}_quantile_transformer2.pkl')
data_transformed = transformer.transform(data.iloc[:, 5:])

train_features_transformed = data_transformed[: train_features.shape[0]]
test_features_transformed = data_transformed[-test_features.shape[0]:]


train_features = pd.DataFrame(train_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']].values.reshape(
    -1, 4),                              columns=['sig_id', 'cp_type', 'cp_time', 'cp_dose'])
train_features = pd.concat(
    [train_features, pd.DataFrame(train_features_transformed)], axis=1)


test_features = pd.DataFrame(test_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']].values.reshape(
    -1, 4),                             columns=['sig_id', 'cp_type', 'cp_time', 'cp_dose'])

test_features = pd.concat(
    [test_features, pd.DataFrame(test_features_transformed)], axis=1)

train_features.shape


# In[14]:


train_features


# In[15]:


from pickle import load, dump


# In[16]:


from sklearn.cluster import KMeans


def fe_cluster_genes(train, test, n_clusters_g=45, SEED=123):

    features_g = list(train.columns[4:776])

    def create_cluster(train, test, features, kind='g', n_clusters=n_clusters_g):
        train_ = train[features].copy()
        test_ = test[features].copy()
        kmeans_genes = load(
            open(f'{model_output_folder}/kmeans_genes.pkl', 'rb'))
        train[f'clusters_{kind}'] = kmeans_genes.predict(train_.values)
        test[f'clusters_{kind}'] = kmeans_genes.predict(test_.values)
        train = pd.get_dummies(train, columns=[f'clusters_{kind}'])
        test = pd.get_dummies(test, columns=[f'clusters_{kind}'])
        return train, test

    train, test = create_cluster(
        train, test, features_g, kind='g', n_clusters=n_clusters_g)
    return train, test

train_features, test_features = fe_cluster_genes(train_features, test_features)


# In[17]:


def fe_cluster_cells(train, test, n_clusters_c=15, SEED=123):

    features_c = list(train.columns[776:876])

    def create_cluster(train, test, features, kind='c', n_clusters=n_clusters_c):
        train_ = train[features].copy()
        test_ = test[features].copy()
        kmeans_cells = load(
            open(f'{model_output_folder}/kmeans_cells.pkl', 'rb'))
        train[f'clusters_{kind}'] = kmeans_cells.predict(train_.values)
        test[f'clusters_{kind}'] = kmeans_cells.predict(test_.values)
        train = pd.get_dummies(train, columns=[f'clusters_{kind}'])
        test = pd.get_dummies(test, columns=[f'clusters_{kind}'])
        return train, test

    train, test = create_cluster(
        train, test, features_c, kind='c', n_clusters=n_clusters_c)
    return train, test

train_features, test_features = fe_cluster_cells(train_features, test_features)


# In[18]:


def fe_stats(train, test):

    features_g = list(train.columns[4:776])
    features_c = list(train.columns[776:876])

    for df in train, test:
        df['g_sum'] = df[features_g].sum(axis=1)
        df['g_mean'] = df[features_g].mean(axis=1)
        df['g_std'] = df[features_g].std(axis=1)
        df['g_kurt'] = df[features_g].kurtosis(axis=1)
        df['g_skew'] = df[features_g].skew(axis=1)
        df['c_sum'] = df[features_c].sum(axis=1)
        df['c_mean'] = df[features_c].mean(axis=1)
        df['c_std'] = df[features_c].std(axis=1)
        df['c_kurt'] = df[features_c].kurtosis(axis=1)
        df['c_skew'] = df[features_c].skew(axis=1)
        df['gc_sum'] = df[features_g + features_c].sum(axis=1)
        df['gc_mean'] = df[features_g + features_c].mean(axis=1)
        df['gc_std'] = df[features_g + features_c].std(axis=1)
        df['gc_kurt'] = df[features_g + features_c].kurtosis(axis=1)
        df['gc_skew'] = df[features_g + features_c].skew(axis=1)

    return train, test

train_features, test_features = fe_stats(train_features, test_features)


# In[19]:


train = train_features.merge(train_targets_scored, on='sig_id')
train = train[train['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)
test = test_features[test_features['cp_type']
                     != 'ctl_vehicle'].reset_index(drop=True)

target = train[train_targets_scored.columns]


# In[20]:


train = train.drop('cp_type', axis=1)
test = test.drop('cp_type', axis=1)


# In[21]:


train


# In[22]:


target_cols = target.drop('sig_id', axis=1).columns.values.tolist()


# In[23]:


folds = train.copy()

mskf = MultilabelStratifiedKFold(n_splits=5)

for f, (t_idx, v_idx) in enumerate(mskf.split(X=train, y=target)):
    folds.loc[v_idx, 'kfold'] = int(f)

folds['kfold'] = folds['kfold'].astype(int)
folds


# In[24]:


print(train.shape)
print(folds.shape)
print(test.shape)
print(target.shape)
print(sample_submission.shape)


# # Dataset Classes

# In[25]:


class MoADataset:

    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return (self.features.shape[0])

    def __getitem__(self, idx):
        dct = {
            'x': torch.tensor(self.features[idx, :], dtype=torch.float),
            'y': torch.tensor(self.targets[idx, :], dtype=torch.float)
        }
        return dct


class TestDataset:

    def __init__(self, features):
        self.features = features

    def __len__(self):
        return (self.features.shape[0])

    def __getitem__(self, idx):
        dct = {
            'x': torch.tensor(self.features[idx, :], dtype=torch.float)
        }
        return dct


# In[26]:


def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    final_loss = 0

    for data in dataloader:
        optimizer.zero_grad()
        inputs, targets = data['x'].to(device), data['y'].to(device)
#         print(inputs.shape)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        final_loss += loss.item()

    final_loss /= len(dataloader)

    return final_loss


def valid_fn(model, loss_fn, dataloader, device):
    model.eval()
    final_loss = 0
    valid_preds = []

    for data in dataloader:
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        final_loss += loss.item()
        valid_preds.append(outputs.sigmoid().detach().cpu().numpy())

    final_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds)

    return final_loss, valid_preds


def inference_fn(model, dataloader, device):
    model.eval()
    preds = []

    for data in dataloader:
        inputs = data['x'].to(device)

        with torch.no_grad():
            outputs = model(inputs)

        preds.append(outputs.sigmoid().detach().cpu().numpy())

    preds = np.concatenate(preds)

    return preds


# In[27]:


import torch
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F


class SmoothBCEwLogits(_WeightedLoss):

    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets: torch.Tensor, n_labels: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
                                           self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets, self.weight)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss


# In[28]:


class Model(nn.Module):      # <-- Update

    def __init__(self, num_features, num_targets, hidden_size):
        super(Model, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dense1 = nn.utils.weight_norm(
            nn.Linear(num_features, hidden_size))

        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.25)
        self.dense2 = nn.Linear(hidden_size, hidden_size)

        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.25)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))

    def forward(self, x):
        x = self.batch_norm1(x)
        x = F.leaky_relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)

        return x


# In[29]:


def process_data(data):
    data = pd.get_dummies(data, columns=['cp_time', 'cp_dose'])
    return data


# In[30]:


feature_cols = [c for c in process_data(folds).columns if c not in target_cols]
feature_cols = [c for c in feature_cols if c not in ['kfold', 'sig_id']]
len(feature_cols)


# In[31]:


# HyperParameters

DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 25
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 1e-5
NFOLDS = 5  # <-- Update
EARLY_STOPPING_STEPS = 10
EARLY_STOP = False

num_features = len(feature_cols)
num_targets = len(target_cols)
hidden_size = 2048


# In[32]:


def run_training(fold, seed):

    seed_everything(seed)

    train = process_data(folds)
    test_ = process_data(test)

    trn_idx = train[train['kfold'] != fold].index
    val_idx = train[train['kfold'] == fold].index

    train_df = train[train['kfold'] != fold].reset_index(drop=True)
    valid_df = train[train['kfold'] == fold].reset_index(drop=True)

    x_train, y_train = train_df[
        feature_cols].values, train_df[target_cols].values
    x_valid, y_valid = valid_df[
        feature_cols].values, valid_df[target_cols].values

    train_dataset = MoADataset(x_train, y_train)
    valid_dataset = MoADataset(x_valid, y_valid)
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = Model(
        num_features=num_features,
        num_targets=num_targets,
        hidden_size=hidden_size,
    )

    model.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=5e-3, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,
                                              max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader))

    loss_fn = nn.BCEWithLogitsLoss()

    loss_tr = SmoothBCEwLogits(smoothing=0.001)

    oof = np.zeros((len(train), target.iloc[:, 1:].shape[1]))

    #--------------------- PREDICTION---------------------
    x_test = test_[feature_cols].values
    testdataset = TestDataset(x_test)
    testloader = torch.utils.data.DataLoader(
        testdataset, batch_size=BATCH_SIZE, shuffle=False)

    model = Model(
        num_features=num_features,
        num_targets=num_targets,
        hidden_size=hidden_size,

    )

    model.load_state_dict(torch.load(f"{model_output_folder}/SEED{seed}_FOLD{fold}_.pth"))
    model.to(DEVICE)

 #   if not IS_TRAIN:
    valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, DEVICE)
    oof[val_idx] = valid_preds

    predictions = np.zeros((len(test_), target.iloc[:, 1:].shape[1]))
    predictions = inference_fn(model, testloader, DEVICE)

    return oof, predictions


# In[33]:


def run_k_fold(NFOLDS, seed):
    oof = np.zeros((len(train), len(target_cols)))
    predictions = np.zeros((len(test), len(target_cols)))

    for fold in range(NFOLDS):
        oof_, pred_ = run_training(fold, seed)

        predictions += pred_ / NFOLDS
        oof += oof_

    return oof, predictions


# In[34]:


# Averaging on multiple SEEDS

SEED = [940, 1513, 1269, 1392, 1119, 1303]  # <-- Update
oof = np.zeros((len(train), len(target_cols)))
predictions = np.zeros((len(test), len(target_cols)))

for seed in SEED:

    oof_, predictions_ = run_k_fold(NFOLDS, seed)
    oof += oof_ / len(SEED)
    predictions += predictions_ / len(SEED)

train[target_cols] = oof
test[target_cols] = predictions


# In[35]:


train_targets_scored


# In[36]:


len(target_cols)


# In[37]:


valid_results = train_targets_scored.drop(columns=target_cols).merge(
    train[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)


y_true = train_targets_scored[target_cols].values
y_pred = valid_results[target_cols].values

score = 0
for i in range(len(target_cols)):
    score_ = log_loss(y_true[:, i], y_pred[:, i])
    score += score_ / target.shape[1]

print("CV log_loss: ", score)


# In[38]:


sub = sample_submission.drop(columns=target_cols).merge(
    test[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)
# sub.to_csv('submission.csv', index=False)
sub.to_csv(f'{output_folder}/submission_simpleNN_oldcv_0.01836.csv', index=False)


# In[39]:


sub.shape


# In[ ]:
