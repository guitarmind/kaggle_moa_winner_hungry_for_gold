#!/usr/bin/env python
# coding: utf-8



# In[1]:


import sys

import argparse
model_artifact_name = "3-stage-nn"
parser = argparse.ArgumentParser(description='Inferencing 3-Stage NN')
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

BATCH_SIZE = args.batch_size

from scipy.sparse.csgraph import connected_components
from umap import UMAP
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
import seaborn as sns
import time

from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA,FactorAnalysis
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
print(torch.cuda.is_available())
import warnings


# In[2]:


torch.__version__


# In[3]:


NB = '25'

IS_TRAIN = False
MODEL_DIR = f"{model_folder}/model" # "../model"
INT_DIR = f"{model_folder}/interim" # "../interim"

NSEEDS = 5  # 5
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 15
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_STEPS = 10
EARLY_STOP = False

NFOLDS = 5  # 5

PMIN = 0.0005
PMAX = 0.9995
SMIN = 0.0
SMAX = 1.0


# In[4]:


train_features = pd.read_csv(f'{input_folder}/train_features.csv')
train_targets_scored = pd.read_csv(f'{input_folder}/train_targets_scored.csv')
train_targets_nonscored = pd.read_csv(f'{input_folder}/train_targets_nonscored.csv')

test_features = pd.read_csv(f'{input_folder}/test_features.csv')
sample_submission = pd.read_csv(f'{input_folder}/sample_submission.csv')


# In[5]:



# In[6]:


train_targets_nonscored = train_targets_nonscored.loc[:, train_targets_nonscored.sum() != 0]
print(train_targets_nonscored.shape)


# In[7]:


for c in train_targets_nonscored.columns:
    if c != "sig_id":
        train_targets_nonscored[c] = np.maximum(PMIN, np.minimum(PMAX, train_targets_nonscored[c]))


# In[8]:


print("(nsamples, nfeatures)")
print(train_features.shape)
print(train_targets_scored.shape)
print(train_targets_nonscored.shape)
print(test_features.shape)
print(sample_submission.shape)


# In[9]:


GENES = [col for col in train_features.columns if col.startswith('g-')]
CELLS = [col for col in train_features.columns if col.startswith('c-')]


# In[10]:


def seed_everything(seed=1903):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=1903)


# In[ ]:





# In[11]:


# GENES
n_comp = 90
n_dim = 45

data = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES])])

if IS_TRAIN:
    fa = FactorAnalysis(n_components=n_comp, random_state=1903).fit(data[GENES])
    pd.to_pickle(fa, f'{MODEL_DIR}/{NB}_factor_analysis_g.pkl')
    umap = UMAP(n_components=n_dim, random_state=1903).fit(data[GENES])
    pd.to_pickle(umap, f'{MODEL_DIR}/{NB}_umap_g.pkl')
else:
    fa = pd.read_pickle(f'{MODEL_DIR}/{NB}_factor_analysis_g.pkl')
    umap = pd.read_pickle(f'{MODEL_DIR}/{NB}_umap_g.pkl')

data2 = (fa.transform(data[GENES]))
data3 = (umap.transform(data[GENES]))

train2 = data2[:train_features.shape[0]]
test2 = data2[-test_features.shape[0]:]
train3 = data3[:train_features.shape[0]]
test3 = data3[-test_features.shape[0]:]

train2 = pd.DataFrame(train2, columns=[f'fa_G-{i}' for i in range(n_comp)])
train3 = pd.DataFrame(train3, columns=[f'umap_G-{i}' for i in range(n_dim)])
test2 = pd.DataFrame(test2, columns=[f'fa_G-{i}' for i in range(n_comp)])
test3 = pd.DataFrame(test3, columns=[f'umap_G-{i}' for i in range(n_dim)])

train_features = pd.concat((train_features, train2, train3), axis=1)
test_features = pd.concat((test_features, test2, test3), axis=1)

#CELLS
n_comp = 50
n_dim = 25

data = pd.concat([pd.DataFrame(train_features[CELLS]), pd.DataFrame(test_features[CELLS])])

if IS_TRAIN:
    fa = FactorAnalysis(n_components=n_comp, random_state=1903).fit(data[CELLS])
    pd.to_pickle(fa, f'{MODEL_DIR}/{NB}_factor_analysis_c.pkl')
    umap = UMAP(n_components=n_dim, random_state=1903).fit(data[CELLS])
    pd.to_pickle(umap, f'{MODEL_DIR}/{NB}_umap_c.pkl')
else:
    fa = pd.read_pickle(f'{MODEL_DIR}/{NB}_factor_analysis_c.pkl')
    umap = pd.read_pickle(f'{MODEL_DIR}/{NB}_umap_c.pkl')
    
data2 = (fa.transform(data[CELLS]))
data3 = (umap.transform(data[CELLS]))

train2 = data2[:train_features.shape[0]]
test2 = data2[-test_features.shape[0]:]
train3 = data3[:train_features.shape[0]]
test3 = data3[-test_features.shape[0]:]

train2 = pd.DataFrame(train2, columns=[f'fa_C-{i}' for i in range(n_comp)])
train3 = pd.DataFrame(train3, columns=[f'umap_C-{i}' for i in range(n_dim)])
test2 = pd.DataFrame(test2, columns=[f'fa_C-{i}' for i in range(n_comp)])
test3 = pd.DataFrame(test3, columns=[f'umap_C-{i}' for i in range(n_dim)])

train_features = pd.concat((train_features, train2, train3), axis=1)
test_features = pd.concat((test_features, test2, test3), axis=1)



# In[12]:


data3


# In[13]:


from sklearn.preprocessing import QuantileTransformer

for col in (GENES + CELLS):
    vec_len = len(train_features[col].values)
    vec_len_test = len(test_features[col].values)
    raw_vec = pd.concat([train_features, test_features])[col].values.reshape(vec_len+vec_len_test, 1)
    if IS_TRAIN:
        transformer = QuantileTransformer(n_quantiles=100, random_state=123, output_distribution="normal")
        transformer.fit(raw_vec)
        pd.to_pickle(transformer, f'{MODEL_DIR}/{NB}_{col}_quantile_transformer.pkl')
    else:
        transformer = pd.read_pickle(f'{MODEL_DIR}/{NB}_{col}_quantile_transformer.pkl')        

    train_features[col] = transformer.transform(train_features[col].values.reshape(vec_len, 1)).reshape(1, vec_len)[0]
    test_features[col] = transformer.transform(test_features[col].values.reshape(vec_len_test, 1)).reshape(1, vec_len_test)[0]


# In[14]:




# In[16]:


print(train_features.shape)
print(test_features.shape)


# In[ ]:





# In[17]:


# train = train_features.merge(train_targets_scored, on='sig_id')
train = train_features.merge(train_targets_nonscored, on='sig_id')
train = train[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
test = test_features[test_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)

# target = train[train_targets_scored.columns]
target = train[train_targets_nonscored.columns]


# In[18]:


train = train.drop('cp_type', axis=1)
test = test.drop('cp_type', axis=1)


# In[19]:


print(target.shape)
print(train_features.shape)
print(test_features.shape)
print(train.shape)
print(test.shape)


# In[20]:


target_cols = target.drop('sig_id', axis=1).columns.values.tolist()


# In[21]:


folds = train.copy()

mskf = MultilabelStratifiedKFold(n_splits=NFOLDS)

for f, (t_idx, v_idx) in enumerate(mskf.split(X=train, y=target)):
    folds.loc[v_idx, 'kfold'] = int(f)

folds['kfold'] = folds['kfold'].astype(int)
folds


# In[22]:


print(train.shape)
print(folds.shape)
print(test.shape)
print(target.shape)
print(sample_submission.shape)


# In[23]:


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


# In[24]:


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


# In[25]:


class Model(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size):
        super(Model, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(0.15)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))
        
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.3)
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.25)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))
    
    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.leaky_relu(self.dense1(x))
        
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))
        
        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)
        
        return x


# In[26]:


def process_data(data):
    
    data = pd.get_dummies(data, columns=['cp_time','cp_dose'])

    return data


# In[27]:


feature_cols = [c for c in process_data(folds).columns if c not in target_cols]
feature_cols = [c for c in feature_cols if c not in ['kfold','sig_id']]
len(feature_cols)


# In[28]:


num_features=len(feature_cols)
num_targets=len(target_cols)
hidden_size=2048



# In[29]:


def run_training(fold, seed):
    
    seed_everything(seed)
    
    train = process_data(folds)
    test_ = process_data(test)
    
    trn_idx = train[train['kfold'] != fold].index
    val_idx = train[train['kfold'] == fold].index
    
    train_df = train[train['kfold'] != fold].reset_index(drop=True)
    valid_df = train[train['kfold'] == fold].reset_index(drop=True)
    
    x_train, y_train  = train_df[feature_cols].values, train_df[target_cols].values
    x_valid, y_valid =  valid_df[feature_cols].values, valid_df[target_cols].values
    
    train_dataset = MoADataset(x_train, y_train)
    valid_dataset = MoADataset(x_valid, y_valid)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = Model(
        num_features=num_features,
        num_targets=num_targets,
        hidden_size=hidden_size,
    )
    
    model.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.2, div_factor=1e3, 
                                              max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader))
    loss_fn = nn.BCEWithLogitsLoss()
    
    early_stopping_steps = EARLY_STOPPING_STEPS
    early_step = 0
    
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
                torch.save(model.state_dict(), f"{MODEL_DIR}/{NB}-nonscored-SEED{seed}-FOLD{fold}_.pth")

            elif(EARLY_STOP == True):
                early_step += 1
                if (early_step >= early_stopping_steps):
                    break

            if epoch % 10 == 0 or epoch == EPOCHS-1:
                print(f"seed: {seed}, FOLD: {fold}, EPOCH: {epoch}, train_loss: {train_loss:.6f}, valid_loss: {valid_loss:.6f}, best_loss: {best_loss:.6f}, best_loss_epoch: {best_loss_epoch}")            
    
    #--------------------- PREDICTION---------------------
    x_test = test_[feature_cols].values
    testdataset = TestDataset(x_test)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = Model(
        num_features=num_features,
        num_targets=num_targets,
        hidden_size=hidden_size,
    )
    
    model.load_state_dict(torch.load(f"{MODEL_DIR}/{NB}-nonscored-SEED{seed}-FOLD{fold}_.pth"))
    model.to(DEVICE)
    
    if not IS_TRAIN:
        valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, DEVICE)
        oof[val_idx] = valid_preds    
    
    predictions = np.zeros((len(test_), target.iloc[:, 1:].shape[1]))
    predictions = inference_fn(model, testloader, DEVICE)
    
    return oof, predictions


# In[30]:


def run_k_fold(NFOLDS, seed):
    oof = np.zeros((len(train), len(target_cols)))
    predictions = np.zeros((len(test), len(target_cols)))
    
    for fold in range(NFOLDS):
        oof_, pred_ = run_training(fold, seed)
        
        predictions += pred_ / NFOLDS
        oof += oof_
        
    return oof, predictions


# In[31]:


SEED = range(NSEEDS) 
oof = np.zeros((len(train), len(target_cols)))
predictions = np.zeros((len(test), len(target_cols)))

time_start = time.time()

for seed in SEED:
    
    oof_, predictions_ = run_k_fold(NFOLDS, seed)
    oof += oof_ / len(SEED)
    predictions += predictions_ / len(SEED)
    print(f"elapsed time: {time.time() - time_start}")

train[target_cols] = oof
test[target_cols] = predictions

print(oof.shape)
print(predictions.shape)


# In[32]:


train.to_pickle(f"{INT_DIR}/{NB}-train_nonscore_pred.pkl")
test.to_pickle(f"{INT_DIR}/{NB}-test_nonscore_pred.pkl")


# In[33]:


len(target_cols)


# In[34]:


train[target_cols] = np.maximum(PMIN, np.minimum(PMAX, train[target_cols]))
valid_results = train_targets_nonscored.drop(columns=target_cols).merge(train[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)

y_true = train_targets_nonscored[target_cols].values
y_true = y_true > 0.5
y_pred = valid_results[target_cols].values

score = 0
for i in range(len(target_cols)):
    score_ = log_loss(y_true[:, i], y_pred[:, i])
    score += score_ / target.shape[1]
    
print("CV log_loss: ", score)


# In[35]:



EPOCHS = 25
# NFOLDS = 5


# In[36]:





# In[37]:


nonscored_target = [c for c in train[train_targets_nonscored.columns] if c != "sig_id"]


# In[38]:


nonscored_target


# In[39]:


train = pd.read_pickle(f"{INT_DIR}/{NB}-train_nonscore_pred.pkl")
test = pd.read_pickle(f"{INT_DIR}/{NB}-test_nonscore_pred.pkl")


# In[40]:



train = train.merge(train_targets_scored, on='sig_id')

target = train[train_targets_scored.columns]


# In[41]:



for col in (nonscored_target):

    vec_len = len(train[col].values)
    vec_len_test = len(test[col].values)
    raw_vec = train[col].values.reshape(vec_len, 1)
    if IS_TRAIN:
        transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution="normal")
        transformer.fit(raw_vec)
        pd.to_pickle(transformer, f"{MODEL_DIR}/{NB}_{col}_quantile_nonscored.pkl")
    else:
        transformer = pd.read_pickle(f"{MODEL_DIR}/{NB}_{col}_quantile_nonscored.pkl")

    train[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
    test[col] = transformer.transform(test[col].values.reshape(vec_len_test, 1)).reshape(1, vec_len_test)[0]


# In[42]:


target_cols = target.drop('sig_id', axis=1).columns.values.tolist()


# In[43]:


train


# In[44]:


folds = train.copy()

mskf = MultilabelStratifiedKFold(n_splits=NFOLDS)

for f, (t_idx, v_idx) in enumerate(mskf.split(X=train, y=target)):
    folds.loc[v_idx, 'kfold'] = int(f)

folds['kfold'] = folds['kfold'].astype(int)
folds


# In[45]:


print(train.shape)
print(folds.shape)
print(test.shape)
print(target.shape)
print(sample_submission.shape)


# In[46]:


def process_data(data):
    
    data = pd.get_dummies(data, columns=['cp_time','cp_dose'])

    
    return data


# In[47]:


feature_cols = [c for c in process_data(folds).columns if c not in target_cols]
feature_cols = [c for c in feature_cols if c not in ['kfold','sig_id']]
len(feature_cols)


# In[48]:


num_features=len(feature_cols)
num_targets=len(target_cols)
hidden_size=2048
# hidden_size=4096
# hidden_size=9192


# In[49]:


def run_training(fold, seed):
    
    seed_everything(seed)
    
    train = process_data(folds)
    test_ = process_data(test)
    
    trn_idx = train[train['kfold'] != fold].index
    val_idx = train[train['kfold'] == fold].index
    
    train_df = train[train['kfold'] != fold].reset_index(drop=True)
    valid_df = train[train['kfold'] == fold].reset_index(drop=True)
    
    x_train, y_train  = train_df[feature_cols].values, train_df[target_cols].values
    x_valid, y_valid =  valid_df[feature_cols].values, valid_df[target_cols].values
    
    train_dataset = MoADataset(x_train, y_train)
    valid_dataset = MoADataset(x_valid, y_valid)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = Model(
        num_features=num_features,
        num_targets=num_targets,
        hidden_size=hidden_size,
    )
    
    model.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
#     scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.3, div_factor=1000, 
#                                               max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader))
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.2, div_factor=1e3, 
                                              max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader))
    loss_fn = nn.BCEWithLogitsLoss()
    
    early_stopping_steps = EARLY_STOPPING_STEPS
    early_step = 0
    
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
                torch.save(model.state_dict(), f"{MODEL_DIR}/{NB}-scored-SEED{seed}-FOLD{fold}_.pth")

            elif(EARLY_STOP == True):
                early_step += 1
                if (early_step >= early_stopping_steps):
                    break

            if epoch % 10 == 0 or epoch == EPOCHS-1:
                print(f"seed: {seed}, FOLD: {fold}, EPOCH: {epoch}, train_loss: {train_loss:.6f}, valid_loss: {valid_loss:.6f}, best_loss: {best_loss:.6f}, best_loss_epoch: {best_loss_epoch}")            
   
    #--------------------- PREDICTION---------------------
    x_test = test_[feature_cols].values
    testdataset = TestDataset(x_test)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = Model(
        num_features=num_features,
        num_targets=num_targets,
        hidden_size=hidden_size,
    )
    
    model.load_state_dict(torch.load(f"{MODEL_DIR}/{NB}-scored-SEED{seed}-FOLD{fold}_.pth"))
    model.to(DEVICE)
    
    if not IS_TRAIN:
        valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, DEVICE)
        oof[val_idx] = valid_preds    
    
    predictions = np.zeros((len(test_), target.iloc[:, 1:].shape[1]))
    predictions = inference_fn(model, testloader, DEVICE)
    
    return oof, predictions


# In[50]:


def run_k_fold(NFOLDS, seed):
    oof = np.zeros((len(train), len(target_cols)))
    predictions = np.zeros((len(test), len(target_cols)))
    
    for fold in range(NFOLDS):
        oof_, pred_ = run_training(fold, seed)
        
        predictions += pred_ / NFOLDS
        oof += oof_
        
    return oof, predictions


# In[51]:


SEED = range(NSEEDS)  #[0, 1, 2, 3 ,4]#, 5, 6, 7, 8, 9, 10]
oof = np.zeros((len(train), len(target_cols)))
predictions = np.zeros((len(test), len(target_cols)))

time_start = time.time()

for seed in SEED:
    
    oof_, predictions_ = run_k_fold(NFOLDS, seed)
    oof += oof_ / len(SEED)
    predictions += predictions_ / len(SEED)
    print(f"elapsed time: {time.time() - time_start}")

train[target_cols] = oof
test[target_cols] = predictions


# In[52]:


train.to_pickle(f"{INT_DIR}/{NB}-train-score-pred.pkl")
test.to_pickle(f"{INT_DIR}/{NB}-test-score-pred.pkl")


# In[53]:


len(target_cols)


# In[54]:


train[target_cols] = np.maximum(PMIN, np.minimum(PMAX, train[target_cols]))

valid_results = train_targets_scored.drop(columns=target_cols).merge(train[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)

y_true = train_targets_scored[target_cols].values
y_true = y_true > 0.5
y_pred = valid_results[target_cols].values

score = 0
for i in range(len(target_cols)):
    score_ = log_loss(y_true[:, i], y_pred[:, i])
    score += score_ / target.shape[1]
    
print("CV log_loss: ", score)




# In[55]:




# In[56]:


train = pd.read_pickle(f"{INT_DIR}/{NB}-train-score-pred.pkl")
test = pd.read_pickle(f"{INT_DIR}/{NB}-test-score-pred.pkl")


# In[57]:


EPOCHS = 25
# NFOLDS = 5


# In[58]:


PMIN = 0.0005
PMAX = 0.9995
for c in train_targets_scored.columns:
    if c != "sig_id":
        train_targets_scored[c] = np.maximum(PMIN, np.minimum(PMAX, train_targets_scored[c]))


# In[59]:


train_targets_scored.columns


# In[60]:


train = train[train_targets_scored.columns]
train.columns = [c + "_pred" if (c != 'sig_id' and c in train_targets_scored.columns) else c for c in train.columns]


# In[61]:


test = test[train_targets_scored.columns]
test.columns = [c + "_pred" if (c != 'sig_id' and c in train_targets_scored.columns) else c for c in test.columns]


# In[62]:


train


# In[63]:



train = train.merge(train_targets_scored, on='sig_id')

target = train[train_targets_scored.columns]


# In[64]:



# In[65]:


from sklearn.preprocessing import QuantileTransformer

scored_target_pred = [c + "_pred" for c in train_targets_scored.columns if c != 'sig_id']

for col in (scored_target_pred):

    vec_len = len(train[col].values)
    vec_len_test = len(test[col].values)
    raw_vec = train[col].values.reshape(vec_len, 1)
#     transformer.fit(raw_vec)
    if IS_TRAIN:
        transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution="normal")
        transformer.fit(raw_vec)
        pd.to_pickle(transformer, f"{MODEL_DIR}/{NB}_{col}_quantile_scored.pkl")
    else:
        transformer = pd.read_pickle(f"{MODEL_DIR}/{NB}_{col}_quantile_scored.pkl")

    train[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
    test[col] = transformer.transform(test[col].values.reshape(vec_len_test, 1)).reshape(1, vec_len_test)[0]


# In[66]:




# In[67]:


target_cols = target.drop('sig_id', axis=1).columns.values.tolist()


# In[68]:


train


# In[69]:


folds = train.copy()

mskf = MultilabelStratifiedKFold(n_splits=NFOLDS)

for f, (t_idx, v_idx) in enumerate(mskf.split(X=train, y=target)):
    folds.loc[v_idx, 'kfold'] = int(f)

folds['kfold'] = folds['kfold'].astype(int)
folds


# In[70]:


print(train.shape)
print(folds.shape)
print(test.shape)
print(target.shape)
print(sample_submission.shape)


# In[71]:


folds


# In[72]:


def process_data(data):
    
    return data


# In[73]:


feature_cols = [c for c in folds.columns if c not in target_cols]
feature_cols = [c for c in feature_cols if c not in ['kfold','sig_id']]
len(feature_cols)


# In[74]:


feature_cols


# In[75]:


folds


# In[76]:


EPOCHS = 25
num_features=len(feature_cols)
num_targets=len(target_cols)
hidden_size=1024
# hidden_size=4096
# hidden_size=9192


# In[77]:


def run_training(fold, seed):
    
    seed_everything(seed)
    
    train = process_data(folds)
    test_ = process_data(test)
    
    trn_idx = train[train['kfold'] != fold].index
    val_idx = train[train['kfold'] == fold].index
    
    train_df = train[train['kfold'] != fold].reset_index(drop=True)
    valid_df = train[train['kfold'] == fold].reset_index(drop=True)
    
    x_train, y_train  = train_df[feature_cols].values, train_df[target_cols].values
    x_valid, y_valid =  valid_df[feature_cols].values, valid_df[target_cols].values
    
    train_dataset = MoADataset(x_train, y_train)
    valid_dataset = MoADataset(x_valid, y_valid)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = Model(
        num_features=num_features,
        num_targets=num_targets,
        hidden_size=hidden_size,
    )
    
    model.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.2, div_factor=1e3, 
                                              max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader))
    loss_fn = nn.BCEWithLogitsLoss()
    
    early_stopping_steps = EARLY_STOPPING_STEPS
    early_step = 0
    
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
                torch.save(model.state_dict(), f"{MODEL_DIR}/{NB}-scored2-SEED{seed}-FOLD{fold}_.pth")
            elif(EARLY_STOP == True):
                early_step += 1
                if (early_step >= early_stopping_steps):
                    break

            if epoch % 10 == 0 or epoch == EPOCHS-1:
                print(f"seed: {seed}, FOLD: {fold}, EPOCH: {epoch}, train_loss: {train_loss:.6f}, valid_loss: {valid_loss:.6f}, best_loss: {best_loss:.6f}, best_loss_epoch: {best_loss_epoch}")                           
    
    #--------------------- PREDICTION---------------------
    x_test = test_[feature_cols].values
    testdataset = TestDataset(x_test)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = Model(
        num_features=num_features,
        num_targets=num_targets,
        hidden_size=hidden_size,
    )
    
    model.load_state_dict(torch.load(f"{MODEL_DIR}/{NB}-scored2-SEED{seed}-FOLD{fold}_.pth"))
    model.to(DEVICE)
    
    if not IS_TRAIN:
        valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, DEVICE)
        oof[val_idx] = valid_preds     
    
    predictions = np.zeros((len(test_), target.iloc[:, 1:].shape[1]))
    predictions = inference_fn(model, testloader, DEVICE)
    
    return oof, predictions


# In[78]:


def run_k_fold(NFOLDS, seed):
    oof = np.zeros((len(train), len(target_cols)))
    predictions = np.zeros((len(test), len(target_cols)))
    
    for fold in range(NFOLDS):
        oof_, pred_ = run_training(fold, seed)
        
        predictions += pred_ / NFOLDS
        oof += oof_
        
    return oof, predictions


# In[79]:


SEED = range(NSEEDS)  # [0, 1, 2, 3 ,4]#, 5, 6, 7, 8, 9, 10]
oof = np.zeros((len(train), len(target_cols)))
predictions = np.zeros((len(test), len(target_cols)))

time_start = time.time()

for seed in SEED:
    
    oof_, predictions_ = run_k_fold(NFOLDS, seed)
    oof += oof_ / len(SEED)
    predictions += predictions_ / len(SEED)
    print(f"elapsed time: {time.time() - time_start}")

train[target_cols] = oof
test[target_cols] = predictions


# In[80]:


train.to_pickle(f"{INT_DIR}/{NB}-train-score-stack-pred.pkl")
test.to_pickle(f"{INT_DIR}/{NB}-test-score-stack-pred.pkl")


# In[81]:


train[target_cols] = np.maximum(PMIN, np.minimum(PMAX, train[target_cols]))
valid_results = train_targets_scored.drop(columns=target_cols).merge(train[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)

y_true = train_targets_scored[target_cols].values
y_true = y_true > 0.5
y_pred = valid_results[target_cols].values

y_pred = np.minimum(SMAX, np.maximum(SMIN, y_pred))

score = 0
for i in range(len(target_cols)):
    score_ = log_loss(y_true[:, i], y_pred[:, i])
    score += score_ / target.shape[1]
    
print("CV log_loss: ", score)


# In[82]:



sub = sample_submission.drop(columns=target_cols).merge(test[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)
sub.to_csv(f'{output_folder}/submission_3stage_nn_0.01822.csv', index=False)


# In[83]:


sub


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




