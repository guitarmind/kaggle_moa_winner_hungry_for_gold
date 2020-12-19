#!/usr/bin/env python
# coding: utf-8

# ## 101-preprocess.ipynb

# In[1]:


import sys

import argparse
model_artifact_name = "2-stage-nn-tabnet"
parser = argparse.ArgumentParser(description='Inferencing 2-Stage NN+TabNet')
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
os.makedirs(f'{model_folder}/model', exist_ok=True)
os.makedirs(f'{model_folder}/interim', exist_ok=True)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from scipy.sparse.csgraph import connected_components
from umap import UMAP
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, RepeatedMultilabelStratifiedKFold

import numpy as np
import scipy as sp
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
import seaborn as sns
import time
# import joblib

from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

print(f"is cuda available: {torch.cuda.is_available()}")

import warnings
# warnings.filterwarnings('ignore')

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

DEFAULT_SEED = 512
seed_everything(seed_value=DEFAULT_SEED)


# In[2]:


# file name prefix
NB = '101'

IS_TRAIN = False ################################################################

MODEL_DIR = f"{model_folder}/model" # "../model"
INT_DIR = f"{model_folder}/interim" # "../interim"

DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')

# label smoothing
PMIN = 0.0
PMAX = 1.0

# submission smoothing
SMIN = 0.0
SMAX = 1.0


# In[3]:


train_features = pd.read_csv(f'{input_folder}/train_features.csv')
train_targets_scored = pd.read_csv(f'{input_folder}/train_targets_scored.csv')
train_targets_nonscored = pd.read_csv(f'{input_folder}/train_targets_nonscored.csv')

test_features = pd.read_csv(f'{input_folder}/test_features.csv')
sample_submission = pd.read_csv(f'{input_folder}/sample_submission.csv')


# In[4]:


# test_features_dummy = pd.read_csv('../input/dummytestfeatures/test_features_dummy.csv')
# test_features = pd.concat([test_features, test_features_dummy]).reset_index(drop=True)


# In[5]:


from sklearn.preprocessing import QuantileTransformer

GENES = [col for col in train_features.columns if col.startswith('g-')]
CELLS = [col for col in train_features.columns if col.startswith('c-')]

for col in (GENES + CELLS):
    vec_len = len(train_features[col].values)
    vec_len_test = len(test_features[col].values)
    raw_vec = pd.concat([train_features, test_features])[col].values.reshape(vec_len+vec_len_test, 1)
    if IS_TRAIN:
        transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution="normal")
        transformer.fit(raw_vec)
        pd.to_pickle(transformer, f'{MODEL_DIR}/{NB}_{col}_quantile_transformer.pkl')
    else:
        transformer = pd.read_pickle(f'{MODEL_DIR}/{NB}_{col}_quantile_transformer.pkl')        

    train_features[col] = transformer.transform(train_features[col].values.reshape(vec_len, 1)).reshape(1, vec_len)[0]
    test_features[col] = transformer.transform(test_features[col].values.reshape(vec_len_test, 1)).reshape(1, vec_len_test)[0]


# In[6]:


# GENES
n_comp = 50
n_dim = 15

data = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES])])

if IS_TRAIN:
    pca = PCA(n_components=n_comp, random_state=DEFAULT_SEED).fit(train_features[GENES])
    umap = UMAP(n_components=n_dim, random_state=DEFAULT_SEED).fit(train_features[GENES])
    pd.to_pickle(pca, f"{MODEL_DIR}/{NB}_pca_g.pkl")
    pd.to_pickle(umap, f"{MODEL_DIR}/{NB}_umap_g.pkl")
else:
    pca = pd.read_pickle(f"{MODEL_DIR}/{NB}_pca_g.pkl")
    umap = pd.read_pickle(f"{MODEL_DIR}/{NB}_umap_g.pkl")
    
data2 = pca.transform(data[GENES])
data3 = umap.transform(data[GENES])

train2 = data2[:train_features.shape[0]]
test2 = data2[-test_features.shape[0]:]
train3 = data3[:train_features.shape[0]]
test3 = data3[-test_features.shape[0]:]

train2 = pd.DataFrame(train2, columns=[f'pca_G-{i}' for i in range(n_comp)])
train3 = pd.DataFrame(train3, columns=[f'umap_G-{i}' for i in range(n_dim)])
test2 = pd.DataFrame(test2, columns=[f'pca_G-{i}' for i in range(n_comp)])
test3 = pd.DataFrame(test3, columns=[f'umap_G-{i}' for i in range(n_dim)])

train_features = pd.concat((train_features, train2, train3), axis=1)
test_features = pd.concat((test_features, test2, test3), axis=1)

#CELLS
n_comp = 15
n_dim = 5

data = pd.concat([pd.DataFrame(train_features[CELLS]), pd.DataFrame(test_features[CELLS])])


if IS_TRAIN:
    pca = PCA(n_components=n_comp, random_state=DEFAULT_SEED).fit(train_features[CELLS])
    umap = UMAP(n_components=n_dim, random_state=DEFAULT_SEED).fit(train_features[CELLS])
    pd.to_pickle(pca, f"{MODEL_DIR}/{NB}_pca_c.pkl")
    pd.to_pickle(umap, f"{MODEL_DIR}/{NB}_umap_c.pkl")
else:
    pca = pd.read_pickle(f"{MODEL_DIR}/{NB}_pca_c.pkl")
    umap = pd.read_pickle(f"{MODEL_DIR}/{NB}_umap_c.pkl")   

data2 = pca.transform(data[CELLS])
data3 = umap.transform(data[CELLS])

train2 = data2[:train_features.shape[0]]
test2 = data2[-test_features.shape[0]:]
train3 = data3[:train_features.shape[0]]
test3 = data3[-test_features.shape[0]:]

train2 = pd.DataFrame(train2, columns=[f'pca_C-{i}' for i in range(n_comp)])
train3 = pd.DataFrame(train3, columns=[f'umap_C-{i}' for i in range(n_dim)])
test2 = pd.DataFrame(test2, columns=[f'pca_C-{i}' for i in range(n_comp)])
test3 = pd.DataFrame(test3, columns=[f'umap_C-{i}' for i in range(n_dim)])

train_features = pd.concat((train_features, train2, train3), axis=1)
test_features = pd.concat((test_features, test2, test3), axis=1)

# drop_cols = [f'c-{i}' for i in range(n_comp,len(CELLS))]


# In[7]:


from sklearn.feature_selection import VarianceThreshold

if IS_TRAIN:
    var_thresh = VarianceThreshold(threshold=0.5).fit(train_features.iloc[:, 4:])
    pd.to_pickle(var_thresh, f"{MODEL_DIR}/{NB}_variance_thresh0_5.pkl")
else:
    var_thresh = pd.read_pickle(f"{MODEL_DIR}/{NB}_variance_thresh0_5.pkl")
                                
data = train_features.append(test_features)
data_transformed = var_thresh.transform(data.iloc[:, 4:])

train_features_transformed = data_transformed[ : train_features.shape[0]]
test_features_transformed = data_transformed[-test_features.shape[0] : ]


train_features = pd.DataFrame(train_features[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),                              columns=['sig_id','cp_type','cp_time','cp_dose'])

train_features = pd.concat([train_features, pd.DataFrame(train_features_transformed)], axis=1)


test_features = pd.DataFrame(test_features[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),                             columns=['sig_id','cp_type','cp_time','cp_dose'])

test_features = pd.concat([test_features, pd.DataFrame(test_features_transformed)], axis=1)

print(train_features.shape)
print(test_features.shape)


# In[8]:


train = train_features[train_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)
test = test_features[test_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)

train = train.drop('cp_type', axis=1)
test = test.drop('cp_type', axis=1)


# In[9]:


train.to_pickle(f"{INT_DIR}/{NB}_train_preprocessed.pkl")
test.to_pickle(f"{INT_DIR}/{NB}_test_preprocessed.pkl")


# ## 203-101-nonscored-pred-2layers.ipynb

# In[10]:


# file name prefix
NB = '203'

# IS_TRAIN = True

# MODEL_DIR = "model" # "../model"
# INT_DIR = "interim" # "../interim"

DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')

# label smoothing
PMIN = 0.0
PMAX = 1.0

# submission smoothing
SMIN = 0.0
SMAX = 1.0

# model hyper params
HIDDEN_SIZE = 2048

# training hyper params
EPOCHS = 15
BATCH_SIZE = args.batch_size
NFOLDS = 10 # 10
NREPEATS = 1
NSEEDS = 5 # 5

# Adam hyper params
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5

# scheduler hyper params
PCT_START = 0.2
DIV_FACS = 1e3
MAX_LR = 1e-2


# In[11]:


def process_data(data):    
    data = pd.get_dummies(data, columns=['cp_time','cp_dose'])
    return data

class MoADataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float),
            'y' : torch.tensor(self.targets[idx, :], dtype=torch.float)            
        }
        return dct
    
class TestDataset:
    def __init__(self, features):
        self.features = features
        
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float)
        }
        return dct

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

def calc_valid_log_loss(train, target, target_cols):
    y_pred = train[target_cols].values
    y_true = target[target_cols].values
    
    y_true_t = torch.from_numpy(y_true.astype(np.float64)).clone()
    y_pred_t = torch.from_numpy(y_pred.astype(np.float64)).clone()
    
    return torch.nn.BCELoss()(y_pred_t, y_true_t).to('cpu').detach().numpy().copy()


# In[12]:


class Model(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size=HIDDEN_SIZE):
        super(Model, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(0.2)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))
               
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.25)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))
    
    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.relu(self.dense1(x))
                
        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)
        
        return x


# In[13]:


def run_training(train, test, trn_idx, val_idx, feature_cols, target_cols, fold, seed):
    
    seed_everything(seed)
    
    train_ = process_data(train)
    test_ = process_data(test)
    
    train_df = train_.loc[trn_idx,:].reset_index(drop=True)
    valid_df = train_.loc[val_idx,:].reset_index(drop=True)
    
    x_train, y_train  = train_df[feature_cols].values, train_df[target_cols].values
    x_valid, y_valid =  valid_df[feature_cols].values, valid_df[target_cols].values
    
    train_dataset = MoADataset(x_train, y_train)
    valid_dataset = MoADataset(x_valid, y_valid)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = Model(
        num_features=len(feature_cols),
        num_targets=len(target_cols),
    )
    
    model.to(DEVICE)
       
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=PCT_START, div_factor=DIV_FACS, 
                                              max_lr=MAX_LR, epochs=EPOCHS, steps_per_epoch=len(trainloader))
    loss_fn = nn.BCEWithLogitsLoss()

    oof = np.zeros((len(train), target.iloc[:, 1:].shape[1]))
    best_loss = np.inf
    best_loss_epoch = -1
    
    if IS_TRAIN:
        for epoch in range(EPOCHS):

            train_loss = train_fn(model, optimizer, scheduler, loss_fn, trainloader, DEVICE)
            valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, DEVICE)

            if valid_loss < best_loss:            
                best_loss = valid_loss
                best_loss_epoch = epoch
                oof[val_idx] = valid_preds
                model.to('cpu')
                torch.save(model.state_dict(), f"{MODEL_DIR}/{NB}_nonscored_SEED{seed}_FOLD{fold}_.pth")
                model.to(DEVICE)

            if epoch % 10 == 0 or epoch == EPOCHS-1:
                print(f"seed: {seed}, FOLD: {fold}, EPOCH: {epoch}, train_loss: {train_loss:.6f}, valid_loss: {valid_loss:.6f}, best_loss: {best_loss:.6f}, best_loss_epoch: {best_loss_epoch}")                           
    
    #--------------------- PREDICTION---------------------
    x_test = test_[feature_cols].values
    testdataset = TestDataset(x_test)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = Model(
        num_features=len(feature_cols),
        num_targets=len(target_cols),
    )
    
    model.load_state_dict(torch.load(f"{MODEL_DIR}/{NB}_nonscored_SEED{seed}_FOLD{fold}_.pth"))
    model.to(DEVICE)
    
    if not IS_TRAIN:
        valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, DEVICE)
        oof[val_idx] = valid_preds

    predictions = np.zeros((len(test_), target.iloc[:, 1:].shape[1]))
    predictions = inference_fn(model, testloader, DEVICE)
    
    return oof, predictions


# In[14]:


def run_k_fold(train, test, feature_cols, target_cols, NFOLDS, seed):
    oof = np.zeros((len(train), len(target_cols)))
    predictions = np.zeros((len(test), len(target_cols)))
    
    mskf = RepeatedMultilabelStratifiedKFold(n_splits=NFOLDS, n_repeats=NREPEATS, random_state=None)
    
    for f, (t_idx, v_idx) in enumerate(mskf.split(X=train, y=target)):
        oof_, pred_ = run_training(train, test, t_idx, v_idx, feature_cols, target_cols, f, seed)
        
        predictions += pred_ / NFOLDS / NREPEATS
        oof += oof_ / NREPEATS
        
    return oof, predictions


# In[15]:


def run_seeds(train, test, feature_cols, target_cols, nfolds=NFOLDS, nseed=NSEEDS):
    seed_list = range(nseed)
    oof = np.zeros((len(train), len(target_cols)))
    predictions = np.zeros((len(test), len(target_cols)))

    time_start = time.time()

    for seed in seed_list:

        oof_, predictions_ = run_k_fold(train, test, feature_cols, target_cols, nfolds, seed)
        oof += oof_ / nseed
        predictions += predictions_ / nseed
        print(f"seed {seed}, elapsed time: {time.time() - time_start}")

    train[target_cols] = oof
    test[target_cols] = predictions


# In[16]:


train_features = pd.read_csv(f'{input_folder}/train_features.csv')
train_targets_scored = pd.read_csv(f'{input_folder}/train_targets_scored.csv')
train_targets_nonscored = pd.read_csv(f'{input_folder}/train_targets_nonscored.csv')

test_features = pd.read_csv(f'{input_folder}/test_features.csv')
sample_submission = pd.read_csv(f'{input_folder}/sample_submission.csv')


# In[17]:


train = pd.read_pickle(f"{INT_DIR}/101_train_preprocessed.pkl")
test = pd.read_pickle(f"{INT_DIR}/101_test_preprocessed.pkl")



# ### non-scored labels prediction

# In[23]:


# remove nonscored labels if all values == 0
train_targets_nonscored = train_targets_nonscored.loc[:, train_targets_nonscored.sum() != 0]
print(train_targets_nonscored.shape)

train = train.merge(train_targets_nonscored, on='sig_id')


# In[24]:


target = train[train_targets_nonscored.columns]
target_cols = target.drop('sig_id', axis=1).columns.values.tolist()
feature_cols = [c for c in process_data(train).columns if c not in target_cols and c not in ['kfold','sig_id']]


# In[25]:


run_seeds(train, test, feature_cols, target_cols)


# In[26]:


print(f"train shape: {train.shape}")
print(f"test  shape: {test.shape}")
print(f"features : {len(feature_cols)}")
print(f"targets  : {len(target_cols)}")


# In[27]:


valid_loss_total = calc_valid_log_loss(train, target, target_cols)
print(f"CV loss: {valid_loss_total}")


# In[28]:


train.to_pickle(f"{INT_DIR}/{NB}_train_nonscored_pred.pkl")
test.to_pickle(f"{INT_DIR}/{NB}_test_nonscored_pred.pkl")


# In[29]:


valid_results = train_targets_nonscored.drop(columns=target_cols).merge(train[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)

y_true = train_targets_nonscored[target_cols].values
y_true = y_true > 0.5
y_pred = valid_results[target_cols].values

score = 0
for i in range(len(target_cols)):
    score_ = log_loss(y_true[:, i], y_pred[:, i])
    score += score_ / target.shape[1]
    
print("CV log_loss: ", score)


# ## 503-203-tabnet-with-nonscored-features-10fold3seed

# In[30]:
# In[31]:


from pytorch_tabnet.tab_model import TabNetRegressor


# In[32]:


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
seed_everything(42)


# In[33]:


# file name prefix
NB = '503'
NB_PREV = '203'

# IS_TRAIN = False

# MODEL_DIR = "../input/moa503/503-tabnet" # "../model"
# INT_DIR = "../input/moa503/203-nonscored-pred" # "../interim"

DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')

# label smoothing
PMIN = 0.0
PMAX = 1.0

# submission smoothing
SMIN = 0.0
SMAX = 1.0

# model hyper params

# training hyper params
# EPOCHS = 25
# BATCH_SIZE = 256
NFOLDS = 10 # 10
NREPEATS = 1
NSEEDS = 3 # 5

# Adam hyper params
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5

# scheduler hyper params
PCT_START = 0.2
DIV_FACS = 1e3
MAX_LR = 1e-2


# In[34]:


train_features = pd.read_csv(f'{input_folder}/train_features.csv')
train_targets_scored = pd.read_csv(f'{input_folder}/train_targets_scored.csv')
train_targets_nonscored = pd.read_csv(f'{input_folder}/train_targets_nonscored.csv')

test_features = pd.read_csv(f'{input_folder}/test_features.csv')
sample_submission = pd.read_csv(f'{input_folder}/sample_submission.csv')


# In[35]:


# test_features_dummy = pd.read_csv('../input/dummytestfeatures/test_features_dummy.csv')
# test_features = pd.concat([test_features, test_features_dummy]).reset_index(drop=True)


# In[ ]:





# In[36]:


print("(nsamples, nfeatures)")
print(train_features.shape)
print(train_targets_scored.shape)
print(train_targets_nonscored.shape)
print(test_features.shape)
print(sample_submission.shape)


# In[37]:


GENES = [col for col in train_features.columns if col.startswith('g-')]
CELLS = [col for col in train_features.columns if col.startswith('c-')]


# In[38]:


from sklearn.preprocessing import QuantileTransformer

use_test_for_preprocessing = False

for col in (GENES + CELLS):

    if IS_TRAIN:
        transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution="normal")
        if use_test_for_preprocessing:
            raw_vec = pd.concat([train_features, test_features])[col].values.reshape(vec_len+vec_len_test, 1)
            transformer.fit(raw_vec)
        else:
            raw_vec = train_features[col].values.reshape(vec_len, 1)
            transformer.fit(raw_vec)
        pd.to_pickle(transformer, f'{MODEL_DIR}/{NB}_{col}_quantile_transformer.pkl')
    else:
        transformer = pd.read_pickle(f'{MODEL_DIR}/{NB}_{col}_quantile_transformer.pkl') 

    vec_len = len(train_features[col].values)
    vec_len_test = len(test_features[col].values)


    train_features[col] = transformer.transform(train_features[col].values.reshape(vec_len, 1)).reshape(1, vec_len)[0]
    test_features[col] = transformer.transform(test_features[col].values.reshape(vec_len_test, 1)).reshape(1, vec_len_test)[0]


# In[39]:


# GENES

n_comp = 90

data = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES])])
if IS_TRAIN:
    fa = FactorAnalysis(n_components=n_comp, random_state=42).fit(data[GENES])
    pd.to_pickle(fa, f'{MODEL_DIR}/{NB}_factor_analysis_g.pkl')
else:
    fa = pd.read_pickle(f'{MODEL_DIR}/{NB}_factor_analysis_g.pkl')
    
data2 = (fa.transform(data[GENES]))
train2 = data2[:train_features.shape[0]]; test2 = data2[-test_features.shape[0]:]

train2 = pd.DataFrame(train2, columns=[f'pca_G-{i}' for i in range(n_comp)])
test2 = pd.DataFrame(test2, columns=[f'pca_G-{i}' for i in range(n_comp)])

# drop_cols = [f'c-{i}' for i in range(n_comp,len(GENES))]
train_features = pd.concat((train_features, train2), axis=1)
test_features = pd.concat((test_features, test2), axis=1)

#CELLS

n_comp = 50

data = pd.concat([pd.DataFrame(train_features[CELLS]), pd.DataFrame(test_features[CELLS])])

if IS_TRAIN:
    fa = FactorAnalysis(n_components=n_comp, random_state=42).fit(data[CELLS])
    pd.to_pickle(fa, f'{MODEL_DIR}/{NB}_factor_analysis_c.pkl')
else:
    fa = pd.read_pickle(f'{MODEL_DIR}/{NB}_factor_analysis_c.pkl')

data2 = (fa.transform(data[CELLS]))
train2 = data2[:train_features.shape[0]]; test2 = data2[-test_features.shape[0]:]

train2 = pd.DataFrame(train2, columns=[f'pca_C-{i}' for i in range(n_comp)])
test2 = pd.DataFrame(test2, columns=[f'pca_C-{i}' for i in range(n_comp)])

# drop_cols = [f'c-{i}' for i in range(n_comp,len(CELLS))]
train_features = pd.concat((train_features, train2), axis=1)
test_features = pd.concat((test_features, test2), axis=1)


# In[40]:


# features_g = list(train_features.columns[4:776])
# train_ = train_features[features_g].copy()
# test_ = test_features[features_g].copy()
# data = pd.concat([train_, test_], axis = 0)
# km = KMeans(n_clusters=35, random_state=123).fit(data)


# In[41]:


# km.predict(data)


# In[42]:


# km.labels_


# In[43]:


from sklearn.cluster import KMeans
def fe_cluster(train, test, n_clusters_g = 35, n_clusters_c = 5, SEED = 123):
    
    features_g = list(train.columns[4:776])
    features_c = list(train.columns[776:876])
    
    def create_cluster(train, test, features, kind = 'g', n_clusters = n_clusters_g):
        train_ = train[features].copy()
        test_ = test[features].copy()
        data = pd.concat([train_, test_], axis = 0)
        
        if IS_TRAIN:
            kmeans = KMeans(n_clusters = n_clusters, random_state = SEED).fit(data)
            pd.to_pickle(kmeans, f"{MODEL_DIR}/{NB}_kmeans_{kind}.pkl")
        else:
            kmeans = pd.read_pickle(f"{MODEL_DIR}/{NB}_kmeans_{kind}.pkl")
            
        train[f'clusters_{kind}'] = kmeans.predict(train_)
        test[f'clusters_{kind}'] = kmeans.predict(test_)
        train = pd.get_dummies(train, columns = [f'clusters_{kind}'])
        test = pd.get_dummies(test, columns = [f'clusters_{kind}'])
        return train, test
    
    train, test = create_cluster(train, test, features_g, kind = 'g', n_clusters = n_clusters_g)
    train, test = create_cluster(train, test, features_c, kind = 'c', n_clusters = n_clusters_c)
    return train, test

train_features ,test_features=fe_cluster(train_features,test_features)


# In[44]:


print(train_features.shape)
print(test_features.shape)


# In[45]:


def fe_stats(train, test):
    
    features_g = list(train.columns[4:776])
    features_c = list(train.columns[776:876])
    
    for df in train, test:
#         df['g_sum'] = df[features_g].sum(axis = 1)
        df['g_mean'] = df[features_g].mean(axis = 1)
        df['g_std'] = df[features_g].std(axis = 1)
        df['g_kurt'] = df[features_g].kurtosis(axis = 1)
        df['g_skew'] = df[features_g].skew(axis = 1)
#         df['c_sum'] = df[features_c].sum(axis = 1)
        df['c_mean'] = df[features_c].mean(axis = 1)
        df['c_std'] = df[features_c].std(axis = 1)
        df['c_kurt'] = df[features_c].kurtosis(axis = 1)
        df['c_skew'] = df[features_c].skew(axis = 1)
#         df['gc_sum'] = df[features_g + features_c].sum(axis = 1)
        df['gc_mean'] = df[features_g + features_c].mean(axis = 1)
        df['gc_std'] = df[features_g + features_c].std(axis = 1)
        df['gc_kurt'] = df[features_g + features_c].kurtosis(axis = 1)
        df['gc_skew'] = df[features_g + features_c].skew(axis = 1)
        
    return train, test

train_features,test_features=fe_stats(train_features,test_features)


# In[46]:


print(train_features.shape)
print(test_features.shape)


# In[47]:


remove_vehicle = True

if remove_vehicle:
    trt_idx = train_features['cp_type']=='trt_cp'
    train_features = train_features.loc[trt_idx].reset_index(drop=True)
    train_targets_scored = train_targets_scored.loc[trt_idx].reset_index(drop=True)
    train_targets_nonscored = train_targets_nonscored.loc[trt_idx].reset_index(drop=True)
else:
    pass


# In[48]:


# train = train_features.merge(train_targets_scored, on='sig_id')
train = train_features.merge(train_targets_scored, on='sig_id')
train = train[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
test = test_features[test_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)

# target = train[train_targets_scored.columns]
target = train[train_targets_scored.columns]
target_cols = target.drop('sig_id', axis=1).columns.values.tolist()

train = train.drop('cp_type', axis=1)
test = test.drop('cp_type', axis=1)


# In[49]:


print(target.shape)
print(train_features.shape)
print(test_features.shape)
print(train.shape)
print(test.shape)


# In[50]:


train_nonscored_pred = pd.read_pickle(f'{INT_DIR}/{NB_PREV}_train_nonscored_pred.pkl')
test_nonscored_pred = pd.read_pickle(f'{INT_DIR}/{NB_PREV}_test_nonscored_pred.pkl')


# In[51]:


# remove nonscored labels if all values == 0
train_targets_nonscored = train_targets_nonscored.loc[:, train_targets_nonscored.sum() != 0]

# nonscored_targets = [c for c in train_targets_nonscored.columns if c != "sig_id"]


# In[52]:


train = train.merge(train_nonscored_pred[train_targets_nonscored.columns], on='sig_id')
test = test.merge(test_nonscored_pred[train_targets_nonscored.columns], on='sig_id')


# In[53]:


from sklearn.preprocessing import QuantileTransformer

nonscored_target = [c for c in train_targets_nonscored.columns if c != "sig_id"]

for col in (nonscored_target):

    vec_len = len(train[col].values)
    vec_len_test = len(test[col].values)
#     raw_vec = pd.concat([train, test])[col].values.reshape(vec_len+vec_len_test, 1)
    raw_vec = train[col].values.reshape(vec_len, 1)
    if IS_TRAIN:
        transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution="normal")
        transformer.fit(raw_vec)
        pd.to_pickle(transformer, f'{MODEL_DIR}/{NB}_{col}_quantile_transformer.pkl')
    else:
        transformer = pd.read_pickle(f'{MODEL_DIR}/{NB}_{col}_quantile_transformer.pkl')        

    train[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
    test[col] = transformer.transform(test[col].values.reshape(vec_len_test, 1)).reshape(1, vec_len_test)[0]


# In[54]:


feature_cols = [c for c in train.columns if c not in target_cols]
feature_cols = [c for c in feature_cols if c not in ['sig_id']]
len(feature_cols)


# In[55]:


num_features=len(feature_cols)
num_targets=len(target_cols)


# In[56]:


import torch
import torch.nn as nn
from pytorch_tabnet.metrics import Metric

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0, n_cls=2):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing + smoothing / n_cls
        self.smoothing = smoothing / n_cls

    def forward(self, x, target):
        probs = torch.nn.functional.sigmoid(x,)
        # ylogy + (1-y)log(1-y)
        #with torch.no_grad():
        target1 = self.confidence * target + (1-target) * self.smoothing
        #print(target1.cpu())
        loss = -(torch.log(probs+1e-15) * target1 + (1-target1) * torch.log(1-probs+1e-15))
        #print(loss.cpu())
        #nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        #nll_loss = nll_loss.squeeze(1)
        #smooth_loss = -logprobs.mean(dim=-1)
        #loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
    
class SmoothedLogLossMetric(Metric):
    """
    BCE with logit loss
    """
    def __init__(self, smoothing=0.001):
        self._name = f"{smoothing:.3f}" # write an understandable name here
        self._maximize = False
        self._lossfn = LabelSmoothing(smoothing)

    def __call__(self, y_true, y_score):
        """
        """
        y_true = torch.from_numpy(y_true.astype(np.float32)).clone()
        y_score = torch.from_numpy(y_score.astype(np.float32)).clone()
#         print("smoothed log loss metric: ", self._lossfn(y_score, y_true).to('cpu').detach().numpy().copy())
        return self._lossfn(y_score, y_true).to('cpu').detach().numpy().copy().take(0)
    
class LogLossMetric(Metric):
    """
    BCE with logit loss
    """
    def __init__(self, smoothing=0.0):
        self._name = f"{smoothing:.3f}" # write an understandable name here
        self._maximize = False
        self._lossfn = LabelSmoothing(smoothing)

    def __call__(self, y_true, y_score):
        """
        """
        y_true = torch.from_numpy(y_true.astype(np.float32)).clone()
        y_score = torch.from_numpy(y_score.astype(np.float32)).clone()
#         print("log loss metric: ", self._lossfn(y_score, y_true).to('cpu').detach().numpy().copy())
        return self._lossfn(y_score, y_true).to('cpu').detach().numpy().copy().take(0)


# In[57]:


def process_data(data):
#     data = pd.get_dummies(data, columns=['cp_time','cp_dose'])
    data.loc[:, 'cp_time'] = data.loc[:, 'cp_time'].map({24: 0, 48: 1, 72: 2, 0: 0, 1: 1, 2: 2})
    data.loc[:, 'cp_dose'] = data.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1, 0: 0, 1: 1})   
    return data

def run_training_tabnet(train, test, trn_idx, val_idx, feature_cols, target_cols, fold, seed, filename="tabnet"):
    
    seed_everything(seed)
    
    train_ = process_data(train)
    test_ = process_data(test)
    
    train_df = train_.loc[trn_idx,:].reset_index(drop=True)
    valid_df = train_.loc[val_idx,:].reset_index(drop=True)
    
    x_train, y_train  = train_df[feature_cols].values, train_df[target_cols].values
    x_valid, y_valid =  valid_df[feature_cols].values, valid_df[target_cols].values
        
    model = TabNetRegressor(n_d=32, n_a=32, n_steps=1, lambda_sparse=0,
                            cat_dims=[3, 2], cat_emb_dim=[1, 1], cat_idxs=[0, 1],
                            optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                            mask_type='entmax',  # device_name=DEVICE,
                            scheduler_params=dict(milestones=[100, 150], gamma=0.9),#)
                            scheduler_fn=torch.optim.lr_scheduler.MultiStepLR,
                            verbose=10,
                            seed = seed)
    
    loss_fn = LabelSmoothing(0.001)
#     eval_metric = SmoothedLogLossMetric(0.001)
#     eval_metric_nosmoothing = SmoothedLogLossMetric(0.)
       
    oof = np.zeros((len(train), target.iloc[:, 1:].shape[1]))
    
    if IS_TRAIN:
#         print("isnan", np.any(np.isnan(x_train)))
        model.fit(X_train=x_train, y_train=y_train,
                  eval_set=[(x_valid, y_valid)], eval_metric=[LogLossMetric, SmoothedLogLossMetric],
                  max_epochs=200, patience=50, batch_size=1024, virtual_batch_size=128,
                    num_workers=0, drop_last=False, loss_fn=loss_fn
                  )
        model.save_model(f"{MODEL_DIR}/{NB}_{filename}_SEED{seed}_FOLD{fold}")
            
    #--------------------- PREDICTION---------------------
    x_test = test_[feature_cols].values
    
    model = TabNetRegressor(n_d=32, n_a=32, n_steps=1, lambda_sparse=0,
                            cat_dims=[3, 2], cat_emb_dim=[1, 1], cat_idxs=[0, 1],
                            optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                            mask_type='entmax',  # device_name=DEVICE,
                            scheduler_params=dict(milestones=[100, 150], gamma=0.9),#)
                            scheduler_fn=torch.optim.lr_scheduler.MultiStepLR,
                            verbose=10,
                            seed = seed)
    
    model.load_model(f"{MODEL_DIR}/{NB}_{filename}_SEED{seed}_FOLD{fold}.model")

    valid_preds = model.predict(x_valid)

    valid_preds = torch.sigmoid(torch.as_tensor(valid_preds)).detach().cpu().numpy()
    oof[val_idx] = valid_preds
        
    predictions = model.predict(x_test)
    predictions = torch.sigmoid(torch.as_tensor(predictions)).detach().cpu().numpy()
    
    return oof, predictions


# In[58]:


def run_k_fold(train, test, feature_cols, target_cols, NFOLDS, seed):
    oof = np.zeros((len(train), len(target_cols)))
    predictions = np.zeros((len(test), len(target_cols)))
    
    mskf = MultilabelStratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state = seed)
    
    for f, (t_idx, v_idx) in enumerate(mskf.split(X=train, y=target)):
        oof_, pred_ = run_training_tabnet(train, test, t_idx, v_idx, feature_cols, target_cols, f, seed)
        
        predictions += pred_ / NFOLDS / NREPEATS
        oof += oof_ / NREPEATS
        
    return oof, predictions

def run_seeds(train, test, feature_cols, target_cols, nfolds=NFOLDS, nseed=NSEEDS):
    seed_list = range(nseed)
    oof = np.zeros((len(train), len(target_cols)))
    predictions = np.zeros((len(test), len(target_cols)))

    time_start = time.time()

    for seed in seed_list:

        oof_, predictions_ = run_k_fold(train, test, feature_cols, target_cols, nfolds, seed)
        oof += oof_ / nseed
        predictions += predictions_ / nseed
        print(f"seed {seed}, elapsed time: {time.time() - time_start}")

    train[target_cols] = oof
    test[target_cols] = predictions


# In[59]:


train.to_pickle(f"{INT_DIR}/{NB}_pre_train.pkl")
test.to_pickle(f"{INT_DIR}/{NB}_pre_test.pkl")


# In[60]:


run_seeds(train, test, feature_cols, target_cols, NFOLDS, NSEEDS)


# In[61]:


train.to_pickle(f"{INT_DIR}/{NB}_train.pkl")
test.to_pickle(f"{INT_DIR}/{NB}_test.pkl")


# In[62]:


# train[target_cols] = np.maximum(PMIN, np.minimum(PMAX, train[target_cols]))
valid_results = train_targets_scored.drop(columns=target_cols).merge(train[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)

y_true = train_targets_scored[target_cols].values
y_true = y_true > 0.5
y_pred = valid_results[target_cols].values

score = 0
for i in range(len(target_cols)):
    score_ = log_loss(y_true[:, i], y_pred[:, i])
    score += score_ / target.shape[1]
    
print("CV log_loss: ", score)


# In[63]:


sub = sample_submission.drop(columns=target_cols).merge(test[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)
sub.to_csv(f'{output_folder}/submission_2stage_nn_tabnet_0.01837.csv', index=False)


# In[64]:


sub


# In[ ]:




