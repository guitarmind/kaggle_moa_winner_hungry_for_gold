#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Credits:
 - 2heads: https://www.kaggle.com/demetrypascal/2heads-deep-resnets-pipeline-smoothing-transfer

"""

kernel_mode = True

import sys
if kernel_mode:
    sys.path.insert(0, "../input/iterative-stratification")
    sys.path.insert(0, "../input/pytorch-lightning")
    sys.path.insert(0, "../input/gen-efficientnet-pytorch")
    sys.path.insert(0, "../input/resnest")

import os
import numpy as np
import pandas as pd
import time
import random
import math
import glob
import pickle
from pickle import dump, load

import matplotlib.pyplot as plt
pd.options.display.max_columns = None

from sklearn.preprocessing import StandardScaler, QuantileTransformer, LabelEncoder, RobustScaler
from sklearn.decomposition import KernelPCA, PCA

from scipy.special import erfinv as sp_erfinv

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torch.autograd import Function
import torch.optim as optim

from torch.nn import Linear, BatchNorm1d, ReLU
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics.functional import classification

import geffnet

import cv2

from scipy.stats import kurtosis
from scipy.stats import skew

import optuna

import warnings
warnings.filterwarnings('ignore')

import gc
gc.enable()

rand_seed = 1120


# In[2]:


if kernel_mode:
    get_ipython().system('mkdir -p /root/.cache/torch/hub/checkpoints/')
    get_ipython().system('cp ../input/gen-efficientnet-pretrained/tf_efficientnet_*.pth /root/.cache/torch/hub/checkpoints/')
    get_ipython().system('ls -la /root/.cache/torch/hub/checkpoints/')


# In[3]:


dataset_folder = "../input/lish-moa" if kernel_mode else "/workspace/Kaggle/MoA/"
model_list = {
    "tabnet_baseline": {
        "model_path":
        "../input/supervised-tabnet-v2-seeds-output" if kernel_mode else
        "/workspace/Kaggle/MoA/completed/supervised_tabnet_v2_seeds",
        "params": {
            "input_dim": 875,
            "output_dim": 206,
            "features_dim": 32,
            "n_steps": 1,
            "gamma": 1.3,
            "n_independent": 2,
            "n_shared": 2,
            "mask_type": "entmax",
            "lambda_sparse": 0,
            "momentum": 0.02,
            "epsilon": 1e-15
        }
    },
    "2heads_deep_resnets_v1": {
        "model_path":
        "../input/2heads-deep-resnets-v1-seeds-output" if kernel_mode else
        "/workspace/Kaggle/MoA/completed/2heads_deep_resnets_v1_seeds",
        "params": {
            "input_dim": 997,
            "output_dim": 206
        }
    },
    "deepinsight_efficientnet_v7_b3": {
        "model_path":
        f"../input/deepinsight-efficientnet-v7-b3" if kernel_mode else
        f"/workspace/Kaggle/MoA/completed/deepinsight_efficientnet_v7_b3",
    },
    "deepinsight_efficientnet_v7_b3_seed2": {
        "model_path":
        f"../input/deepinsight-efficientnet-v7-b3-seed2" if kernel_mode else
        f"/workspace/Kaggle/MoA/completed/deepinsight_efficientnet_v7_b3_seed2",
    },
    "deepinsight_ResNeSt_v1_resnest50": {
        "model_path":
        f"../input/deepinsight-resnest-v1-resnest50" if kernel_mode else
        f"/workspace/Kaggle/MoA/completed/deepinsight_ResNeSt_v1_resnest50",
    }
}

infer_batch_size = 2048

if kernel_mode:
    gpus = [0]
    num_workers = 2
else:
    # gpus = [0, 1]
    gpus = [0]
    # gpus = [1]
    num_workers = 4


# In[4]:


train_features = pd.read_csv(f"{dataset_folder}/train_features.csv",
                             engine='c')
train_labels = pd.read_csv(f"{dataset_folder}/train_targets_scored.csv",
                           engine='c')

train_extra_labels = pd.read_csv(
    f"{dataset_folder}/train_targets_nonscored.csv", engine='c')

test_features = pd.read_csv(f"{dataset_folder}/test_features.csv", engine='c')

sample_submission = pd.read_csv(f"{dataset_folder}/sample_submission.csv",
                                engine='c')


# In[5]:


# Sort by sig_id to ensure that all row orders match
train_features = train_features.sort_values(
    by=["sig_id"], axis=0, inplace=False).reset_index(drop=True)
train_labels = train_labels.sort_values(by=["sig_id"], axis=0,
                                        inplace=False).reset_index(drop=True)
train_extra_labels = train_extra_labels.sort_values(
    by=["sig_id"], axis=0, inplace=False).reset_index(drop=True)

sample_submission = sample_submission.sort_values(
    by=["sig_id"], axis=0, inplace=False).reset_index(drop=True)


# In[6]:


train_features.shape, train_labels.shape, train_extra_labels.shape


# In[7]:


test_features.shape


# In[8]:


train_features["cp_type"].value_counts()


# In[9]:


train_features["cp_dose"].value_counts()


# In[10]:


category_features = ["cp_type", "cp_dose"]
numeric_features = [c for c in train_features.columns if c != "sig_id" and c not in category_features]
all_features = category_features + numeric_features
gene_experssion_features = [c for c in numeric_features if c.startswith("g-")]
cell_viability_features = [c for c in numeric_features if c.startswith("c-")]
len(numeric_features), len(gene_experssion_features), len(cell_viability_features)


# In[11]:


train_classes = [c for c in train_labels.columns if c != "sig_id"]
train_extra_classes = [c for c in train_extra_labels.columns if c != "sig_id"]
len(train_classes), len(train_extra_classes)


# ## Utility Functions

# In[12]:


def save_pickle(obj, model_output_folder, seed, fold_i, name):
    dump(
        obj,
        open(f"{model_output_folder}/seed{seed}/fold{fold_i}_{name}.pkl",
             'wb'), pickle.HIGHEST_PROTOCOL)


def load_pickle(model_output_folder, seed, fold_i, name):
    return load(
        open(f"{model_output_folder}/seed{seed}/fold{fold_i}_{name}.pkl",
             'rb'))


# ## Label Encoding

# In[13]:


# Usually we should do it in the K-fold loop, but here the cardinality is static so we process them at once
category_dims = []
for c in category_features:
    le = LabelEncoder()
    train_features[c] = le.fit_transform(train_features[c])
    test_features[c] = le.fit_transform(test_features[c])
    category_dims.append(len(le.classes_))


# In[14]:


category_dims


# ## Feature Engineering
# Done per K-fold to avoid overfitting training set

# In[15]:


def extract_stats_features(train, valid, test):
    for df in [train, valid, test]:
        df['g_mean'] = df[gene_experssion_features].mean(axis=1)
        df['c_mean'] = df[cell_viability_features].mean(axis=1)

        df['g_sum'] = df[gene_experssion_features].sum(axis=1)
        df['c_sum'] = df[cell_viability_features].sum(axis=1)

        df['g_med'] = np.median(df[gene_experssion_features], axis=1)
        df['c_med'] = np.median(df[cell_viability_features], axis=1)

        df['g_std'] = df[gene_experssion_features].sum(axis=1)
        df['c_std'] = df[cell_viability_features].sum(axis=1)

        df['g_kurt'] = df[gene_experssion_features].kurtosis(axis=1)
        df['c_kurt'] = df[cell_viability_features].kurtosis(axis=1)

        df['g_skew'] = df[gene_experssion_features].skew(axis=1)
        df['c_skew'] = df[cell_viability_features].skew(axis=1)


def extract_pca_components(train,
                           valid,
                           test,
                           n_components,
                           prefix,
                           pca=None,
                           seed=None):
    if pca is not None:
        train_components = pca.transform(train)
        valid_components = pca.transform(valid)
        test_components = pca.transform(test)
    else:
        pca = PCA(n_components=n_components, random_state=seed)
        train_components = pca.fit_transform(train)
        valid_components = pca.transform(valid)
        test_components = pca.transform(test)

    columns = [f'pca_{prefix}{i + 1}' for i in range(n_components)]
    train_components = pd.DataFrame(train_components, columns=columns)
    valid_components = pd.DataFrame(valid_components, columns=columns)
    test_components = pd.DataFrame(test_components, columns=columns)

    return train_components, valid_components, test_components, pca


# In[16]:


def remove_low_variance(train, valid, test, features, var_thresh=None):
    if var_thresh is not None:
        train_var = var_thresh.transform(train[features].copy())
        valid_var = var_thresh.transform(valid[features].copy())
        test_var = var_thresh.transform(test[features].copy())
    else:
        var_thresh = VarianceThreshold(0.8)
        train_var = var_thresh.fit_transform(train[features].copy())
        valid_var = var_thresh.transform(valid[features].copy())
        test_var = var_thresh.transform(test[features].copy())

    selected_features = [
        features[i] for i in list(var_thresh.get_support(indices=True))
    ]

    train = pd.DataFrame(train_var, columns=selected_features)
    valid = pd.DataFrame(valid_var, columns=selected_features)
    test = pd.DataFrame(test_var, columns=selected_features)

    return train, valid, test, var_thresh


# In[17]:


def extract_clusters(train,
                     valid,
                     test,
                     features,
                     prefix,
                     n_clusters,
                     kmeans=None,
                     seed=None):
    if kmeans is not None:
        train[f'clusters_{prefix}'] = kmeans.predict(train[features])
        valid[f'clusters_{prefix}'] = kmeans.predict(valid[features])
        test[f'clusters_{prefix}'] = kmeans.predict(test[features])
    else:
        kmeans = KMeans(init='k-means++',
                        max_iter=300,
                        n_clusters=n_clusters,
                        random_state=seed)
        train[f'clusters_{prefix}'] = kmeans.fit_predict(train[features])
        valid[f'clusters_{prefix}'] = kmeans.predict(valid[features])
        test[f'clusters_{prefix}'] = kmeans.predict(test[features])

    train = pd.get_dummies(train, columns=[f'clusters_{prefix}'])
    valid = pd.get_dummies(valid, columns=[f'clusters_{prefix}'])
    test = pd.get_dummies(test, columns=[f'clusters_{prefix}'])

    # Post processing for clusters that don't exist in validation or test set
    cluster_columns = [c for c in train.columns if c.startswith("clusters_")]
    for col in cluster_columns:
        if col not in valid.columns:
            valid[col] = 0
        if col not in test.columns:
            test[col] = 0

    return train, valid, test, kmeans


# ## Baseline TabNet Inference

# In[18]:


torch.cuda.empty_cache()
gc.collect()


# In[19]:


class BaselineTabNetMoADataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, scaler, is_train=True):
        self.features = features
        self.labels = labels
        self.scaler = scaler

        if is_train:
            self.features = scaler.fit_transform(self.features)
        else:
            self.features = scaler.transform(self.features)

    def __getitem__(self, index):
        if self.labels is not None:
            return self.features[index, :], self.labels[index, :]
        else:
            # Return dummy label
            return self.features[index, :], -1

    def __len__(self):
        return self.features.shape[0]


# ### TabNet Model Definition
# Based on https://github.com/dreamquark-ai/tabnet

# In[20]:


"""
Other possible implementations:
https://github.com/KrisKorrel/sparsemax-pytorch/blob/master/sparsemax.py
https://github.com/msobroza/SparsemaxPytorch/blob/master/mnist/sparsemax.py
https://github.com/vene/sparse-structured-attention/blob/master/pytorch/torchsparseattn/sparsemax.py
"""


# credits to Yandex https://github.com/Qwicen/node/blob/master/lib/nn_utils.py
def _make_ix_like(input, dim=0):
    d = input.size(dim)
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


class SparsemaxFunction(Function):
    """
    An implementation of sparsemax (Martins & Astudillo, 2016). See
    :cite:`DBLP:journals/corr/MartinsA16` for detailed description.
    By Ben Peters and Vlad Niculae
    """

    @staticmethod
    def forward(ctx, input, dim=-1):
        """sparsemax: normalizing sparse transform (a la softmax)
        Parameters:
            input (Tensor): any shape
            dim: dimension along which to apply sparsemax
        Returns:
            output (Tensor): same shape as input
        """
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input -= max_val  # same numerical stability trick as for softmax
        tau, supp_size = SparsemaxFunction._threshold_and_support(
            input, dim=dim)
        output = torch.clamp(input - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None

    @staticmethod
    def _threshold_and_support(input, dim=-1):
        """Sparsemax building block: compute the threshold
        Args:
            input: any dimension
            dim: dimension along which to apply the sparsemax
        Returns:
            the threshold value
        """

        input_srt, _ = torch.sort(input, descending=True, dim=dim)
        input_cumsum = input_srt.cumsum(dim) - 1
        rhos = _make_ix_like(input, dim)
        support = rhos * input_srt > input_cumsum

        support_size = support.sum(dim=dim).unsqueeze(dim)
        tau = input_cumsum.gather(dim, support_size - 1)
        tau /= support_size.to(input.dtype)
        return tau, support_size


class Sparsemax(nn.Module):
    def __init__(self, dim=-1):
        self.dim = dim
        super(Sparsemax, self).__init__()

    def forward(self, input):
        return sparsemax(input, self.dim)


class Entmax15Function(Function):
    """
    An implementation of exact Entmax with alpha=1.5 (B. Peters, V. Niculae, A. Martins). See
    :cite:`https://arxiv.org/abs/1905.05702 for detailed description.
    Source: https://github.com/deep-spin/entmax
    """

    @staticmethod
    def forward(ctx, input, dim=-1):
        ctx.dim = dim

        max_val, _ = input.max(dim=dim, keepdim=True)
        input = input - max_val  # same numerical stability trick as for softmax
        input = input / 2  # divide by 2 to solve actual Entmax

        tau_star, _ = Entmax15Function._threshold_and_support(input, dim)
        output = torch.clamp(input - tau_star, min=0)**2
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        Y, = ctx.saved_tensors
        gppr = Y.sqrt()  # = 1 / g'' (Y)
        dX = grad_output * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr
        return dX, None

    @staticmethod
    def _threshold_and_support(input, dim=-1):
        Xsrt, _ = torch.sort(input, descending=True, dim=dim)

        rho = _make_ix_like(input, dim)
        mean = Xsrt.cumsum(dim) / rho
        mean_sq = (Xsrt**2).cumsum(dim) / rho
        ss = rho * (mean_sq - mean**2)
        delta = (1 - ss) / rho

        # NOTE this is not exactly the same as in reference algo
        # Fortunately it seems the clamped values never wrongly
        # get selected by tau <= sorted_z. Prove this!
        delta_nz = torch.clamp(delta, 0)
        tau = mean - torch.sqrt(delta_nz)

        support_size = (tau <= Xsrt).sum(dim).unsqueeze(dim)
        tau_star = tau.gather(dim, support_size - 1)
        return tau_star, support_size


class Entmoid15(Function):
    """ A highly optimized equivalent of labda x: Entmax15([x, 0]) """

    @staticmethod
    def forward(ctx, input):
        output = Entmoid15._forward(input)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def _forward(input):
        input, is_pos = abs(input), input >= 0
        tau = (input + torch.sqrt(F.relu(8 - input**2))) / 2
        tau.masked_fill_(tau <= input, 2.0)
        y_neg = 0.25 * F.relu(tau - input, inplace=True)**2
        return torch.where(is_pos, 1 - y_neg, y_neg)

    @staticmethod
    def backward(ctx, grad_output):
        return Entmoid15._backward(ctx.saved_tensors[0], grad_output)

    @staticmethod
    def _backward(output, grad_output):
        gppr0, gppr1 = output.sqrt(), (1 - output).sqrt()
        grad_input = grad_output * gppr0
        q = grad_input / (gppr0 + gppr1)
        grad_input -= q * gppr0
        return grad_input


class Entmax15(nn.Module):
    def __init__(self, dim=-1):
        self.dim = dim
        super(Entmax15, self).__init__()

    def forward(self, input):
        return entmax15(input, self.dim)
    
sparsemax = SparsemaxFunction.apply
entmax15 = Entmax15Function.apply
entmoid15 = Entmoid15.apply


# In[21]:


def initialize_non_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(4 * input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    return


def initialize_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    return


class GBN(torch.nn.Module):
    """
        Ghost Batch Normalization
        https://arxiv.org/abs/1705.08741
    """

    def __init__(self, input_dim, virtual_batch_size=128, momentum=0.01):
        super(GBN, self).__init__()

        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = BatchNorm1d(self.input_dim, momentum=momentum)

    def forward(self, x):
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(x_) for x_ in chunks]

        return torch.cat(res, dim=0)


class GLU_Layer(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 fc=None,
                 virtual_batch_size=128,
                 momentum=0.02):
        super(GLU_Layer, self).__init__()

        self.output_dim = output_dim
        if fc:
            self.fc = fc
        else:
            self.fc = Linear(input_dim, 2 * output_dim, bias=False)
        initialize_glu(self.fc, input_dim, 2 * output_dim)

        self.bn = GBN(
            2 * output_dim,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        out = torch.mul(x[:, :self.output_dim],
                        torch.sigmoid(x[:, self.output_dim:]))
        return out


class GLU_Block(torch.nn.Module):
    """
        Independant GLU block, specific to each step
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 n_glu=2,
                 first=False,
                 shared_layers=None,
                 virtual_batch_size=128,
                 momentum=0.02):
        super(GLU_Block, self).__init__()
        self.first = first
        self.shared_layers = shared_layers
        self.n_glu = n_glu
        self.glu_layers = torch.nn.ModuleList()

        params = {
            'virtual_batch_size': virtual_batch_size,
            'momentum': momentum
        }

        fc = shared_layers[0] if shared_layers else None
        self.glu_layers.append(
            GLU_Layer(input_dim, output_dim, fc=fc, **params))
        for glu_id in range(1, self.n_glu):
            fc = shared_layers[glu_id] if shared_layers else None
            self.glu_layers.append(
                GLU_Layer(output_dim, output_dim, fc=fc, **params))

    def forward(self, x):
        scale = torch.sqrt(torch.FloatTensor([0.5]).to(x.device))
        if self.first:  # the first layer of the block has no scale multiplication
            x = self.glu_layers[0](x)
            layers_left = range(1, self.n_glu)
        else:
            layers_left = range(self.n_glu)

        for glu_id in layers_left:
            x = torch.add(x, self.glu_layers[glu_id](x))
            x = x * scale
        return x


# In[22]:


class EmbeddingGenerator(torch.nn.Module):
    """
        Classical embeddings generator
    """

    def __init__(self, input_dim, cat_dims, cat_idxs, cat_emb_dim):
        """ This is an embedding module for an entier set of features

        Parameters
        ----------
        input_dim : int
            Number of features coming as input (number of columns)
        cat_dims : list of int
            Number of modalities for each categorial features
            If the list is empty, no embeddings will be done
        cat_idxs : list of int
            Positional index for each categorical features in inputs
        cat_emb_dim : int or list of int
            Embedding dimension for each categorical features
            If int, the same embdeding dimension will be used for all categorical features
        """
        super(EmbeddingGenerator, self).__init__()
        if cat_dims == [] or cat_idxs == []:
            self.skip_embedding = True
            self.post_embed_dim = input_dim
            return

        self.skip_embedding = False
        if isinstance(cat_emb_dim, int):
            self.cat_emb_dims = [cat_emb_dim] * len(cat_idxs)
        else:
            self.cat_emb_dims = cat_emb_dim

        # check that all embeddings are provided
        if len(self.cat_emb_dims) != len(cat_dims):
            msg = """ cat_emb_dim and cat_dims must be lists of same length, got {len(self.cat_emb_dims)}
                      and {len(cat_dims)}"""
            raise ValueError(msg)
        self.post_embed_dim = int(input_dim + np.sum(self.cat_emb_dims) -
                                  len(self.cat_emb_dims))

        self.embeddings = torch.nn.ModuleList()

        # Sort dims by cat_idx
        sorted_idxs = np.argsort(cat_idxs)
        cat_dims = [cat_dims[i] for i in sorted_idxs]
        self.cat_emb_dims = [self.cat_emb_dims[i] for i in sorted_idxs]

        for cat_dim, emb_dim in zip(cat_dims, self.cat_emb_dims):
            self.embeddings.append(torch.nn.Embedding(cat_dim, emb_dim))

        # record continuous indices
        self.continuous_idx = torch.ones(input_dim, dtype=torch.bool)
        self.continuous_idx[cat_idxs] = 0

    def forward(self, x):
        """
        Apply embdeddings to inputs
        Inputs should be (batch_size, input_dim)
        Outputs will be of size (batch_size, self.post_embed_dim)
        """
        if self.skip_embedding:
            # no embeddings required
            return x

        cols = []
        cat_feat_counter = 0
        for feat_init_idx, is_continuous in enumerate(self.continuous_idx):
            # Enumerate through continuous idx boolean mask to apply embeddings
            if is_continuous:
                cols.append(x[:, feat_init_idx].float().view(-1, 1))
            else:
                cols.append(self.embeddings[cat_feat_counter](
                    x[:, feat_init_idx].long()))
                cat_feat_counter += 1
        # concat
        post_embeddings = torch.cat(cols, dim=1)
        return post_embeddings


# In[23]:


class FeatTransformer(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 shared_layers,
                 n_glu_independent,
                 virtual_batch_size=128,
                 momentum=0.02):
        super(FeatTransformer, self).__init__()
        """
        Initialize a feature transformer.

        Parameters
        ----------
        - input_dim : int
            Input size
        - output_dim : int
            Outpu_size
        - n_glu_independant
        - shared_blocks : torch.nn.ModuleList
            The shared block that should be common to every step
        - momentum : float
            Float value between 0 and 1 which will be used for momentum in batch norm
        """

        params = {
            'n_glu': n_glu_independent,
            'virtual_batch_size': virtual_batch_size,
            'momentum': momentum
        }

        if shared_layers is None:
            # no shared layers
            self.shared = torch.nn.Identity()
            is_first = True
        else:
            self.shared = GLU_Block(
                input_dim,
                output_dim,
                first=True,
                shared_layers=shared_layers,
                n_glu=len(shared_layers),
                virtual_batch_size=virtual_batch_size,
                momentum=momentum)
            is_first = False

        if n_glu_independent == 0:
            # no independent layers
            self.specifics = torch.nn.Identity()
        else:
            spec_input_dim = input_dim if is_first else output_dim
            self.specifics = GLU_Block(
                spec_input_dim, output_dim, first=is_first, **params)

    def forward(self, x):
        x = self.shared(x)
        x = self.specifics(x)
        return x


# In[24]:


class AttentiveTransformer(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 virtual_batch_size=128,
                 momentum=0.02,
                 mask_type="sparsemax"):
        """
        Initialize an attention transformer.

        Parameters
        ----------
        - input_dim : int
            Input size
        - output_dim : int
            Outpu_size
        - momentum : float
            Float value between 0 and 1 which will be used for momentum in batch norm
        - mask_type: str
            Either "sparsemax" or "entmax" : this is the masking function to use
        """
        super(AttentiveTransformer, self).__init__()
        self.fc = Linear(input_dim, output_dim, bias=False)
        initialize_non_glu(self.fc, input_dim, output_dim)
        self.bn = GBN(
            output_dim,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum)

        if mask_type == "sparsemax":
            # Sparsemax
            self.selector = Sparsemax(dim=-1)
        elif mask_type == "entmax":
            # Entmax
            self.selector = Entmax15(dim=-1)
        else:
            raise NotImplementedError("Please choose either sparsemax" +
                                      "or entmax as masktype")

    def forward(self, priors, processed_feat):
        x = self.fc(processed_feat)
        x = self.bn(x)
        x = torch.mul(x, priors)
        x = self.selector(x)
        return x


# In[25]:


class TabNetNoEmbeddings(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 n_d=8,
                 n_a=8,
                 n_steps=3,
                 gamma=1.3,
                 n_independent=2,
                 n_shared=2,
                 epsilon=1e-15,
                 virtual_batch_size=128,
                 momentum=0.02,
                 mask_type="sparsemax"):
        """
        Defines main part of the TabNet network without the embedding layers.

        Parameters
        ----------
        - input_dim : int
            Number of features
        - output_dim : int or list of int for multi task classification
            Dimension of network output
            examples : one for regression, 2 for binary classification etc...
        - n_d : int
            Dimension of the prediction  layer (usually between 4 and 64)
        - n_a : int
            Dimension of the attention  layer (usually between 4 and 64)
        - n_steps: int
            Number of sucessive steps in the newtork (usually betwenn 3 and 10)
        - gamma : float
            Float above 1, scaling factor for attention updates (usually betwenn 1.0 to 2.0)
        - momentum : float
            Float value between 0 and 1 which will be used for momentum in all batch norm
        - n_independent : int
            Number of independent GLU layer in each GLU block (default 2)
        - n_shared : int
            Number of independent GLU layer in each GLU block (default 2)
        - epsilon: float
            Avoid log(0), this should be kept very low
        - mask_type: str
            Either "sparsemax" or "entmax" : this is the masking function to use
        """
        super(TabNetNoEmbeddings, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_multi_task = isinstance(output_dim, list)
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size
        self.mask_type = mask_type
        self.initial_bn = BatchNorm1d(self.input_dim, momentum=0.01)

        if self.n_shared > 0:
            shared_feat_transform = torch.nn.ModuleList()
            for i in range(self.n_shared):
                if i == 0:
                    shared_feat_transform.append(
                        Linear(self.input_dim, 2 * (n_d + n_a), bias=False))
                else:
                    shared_feat_transform.append(
                        Linear(n_d + n_a, 2 * (n_d + n_a), bias=False))

        else:
            shared_feat_transform = None

        self.initial_splitter = FeatTransformer(
            self.input_dim,
            n_d + n_a,
            shared_feat_transform,
            n_glu_independent=self.n_independent,
            virtual_batch_size=self.virtual_batch_size,
            momentum=momentum)

        self.feat_transformers = torch.nn.ModuleList()
        self.att_transformers = torch.nn.ModuleList()

        for step in range(n_steps):
            transformer = FeatTransformer(
                self.input_dim,
                n_d + n_a,
                shared_feat_transform,
                n_glu_independent=self.n_independent,
                virtual_batch_size=self.virtual_batch_size,
                momentum=momentum)
            attention = AttentiveTransformer(
                n_a,
                self.input_dim,
                virtual_batch_size=self.virtual_batch_size,
                momentum=momentum,
                mask_type=self.mask_type)
            self.feat_transformers.append(transformer)
            self.att_transformers.append(attention)

        if self.is_multi_task:
            self.multi_task_mappings = torch.nn.ModuleList()
            for task_dim in output_dim:
                task_mapping = Linear(n_d, task_dim, bias=False)
                initialize_non_glu(task_mapping, n_d, task_dim)
                self.multi_task_mappings.append(task_mapping)
        else:
            self.final_mapping = Linear(n_d, output_dim, bias=False)
            initialize_non_glu(self.final_mapping, n_d, output_dim)

    def forward(self, x):
        res = 0
        x = self.initial_bn(x)

        prior = torch.ones(x.shape).to(x.device)
        M_loss = 0
        att = self.initial_splitter(x)[:, self.n_d:]

        for step in range(self.n_steps):
            M = self.att_transformers[step](prior, att)
            M_loss += torch.mean(
                torch.sum(torch.mul(M, torch.log(M + self.epsilon)), dim=1))
            # update prior
            prior = torch.mul(self.gamma - M, prior)
            # output
            masked_x = torch.mul(M, x)
            out = self.feat_transformers[step](masked_x)
            d = ReLU()(out[:, :self.n_d])
            res = torch.add(res, d)
            # update attention
            att = out[:, self.n_d:]

        M_loss /= self.n_steps

        if self.is_multi_task:
            # Result will be in list format
            out = []
            for task_mapping in self.multi_task_mappings:
                out.append(task_mapping(res))
        else:
            out = self.final_mapping(res)
        return out, M_loss

    def forward_masks(self, x):
        x = self.initial_bn(x)

        prior = torch.ones(x.shape).to(x.device)
        M_explain = torch.zeros(x.shape).to(x.device)
        att = self.initial_splitter(x)[:, self.n_d:]
        masks = {}

        for step in range(self.n_steps):
            M = self.att_transformers[step](prior, att)
            masks[step] = M
            # update prior
            prior = torch.mul(self.gamma - M, prior)
            # output
            masked_x = torch.mul(M, x)
            out = self.feat_transformers[step](masked_x)
            d = ReLU()(out[:, :self.n_d])
            # explain
            step_importance = torch.sum(d, dim=1)
            M_explain += torch.mul(M, step_importance.unsqueeze(dim=1))
            # update attention
            att = out[:, self.n_d:]

        return M_explain, masks


# ### Custom Model

# In[26]:


class TabNet(pl.LightningModule):
    def __init__(self,
                 input_dim,
                 output_dim,
                 n_d=8,
                 n_a=8,
                 n_steps=3,
                 gamma=1.3,
                 cat_idxs=[],
                 cat_dims=[],
                 cat_emb_dim=1,
                 n_independent=2,
                 n_shared=2,
                 epsilon=1e-15,
                 virtual_batch_size=128,
                 momentum=0.02,
                 learning_rate=1e-3,
                 mask_type="sparsemax"):
        super(TabNet, self).__init__()
        self.cat_idxs = cat_idxs or []
        self.cat_dims = cat_dims or []
        self.cat_emb_dim = cat_emb_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.mask_type = mask_type

        if self.n_steps <= 0:
            raise ValueError("n_steps should be a positive integer.")
        if self.n_independent == 0 and self.n_shared == 0:
            raise ValueError("n_shared and n_independant can't be both zero.")

        self.virtual_batch_size = virtual_batch_size
        self.embedder = EmbeddingGenerator(input_dim, cat_dims, cat_idxs,
                                           cat_emb_dim)
        self.post_embed_dim = self.embedder.post_embed_dim
        self.tabnet = TabNetNoEmbeddings(self.post_embed_dim, output_dim, n_d,
                                         n_a, n_steps, gamma, n_independent,
                                         n_shared, epsilon, virtual_batch_size,
                                         momentum, mask_type)

        self.scaler = StandardScaler(with_mean=True, with_std=True)

        # Save passed hyperparameters
        self.save_hyperparameters("n_d", "n_a", "n_steps", "gamma",
                                  "cat_emb_dim", "n_independent", "n_shared",
                                  "epsilon", "virtual_batch_size", "momentum",
                                  "learning_rate", "mask_type")

    def forward(self, x):
        x = self.embedder(x)
        return self.tabnet(x)

    def forward_masks(self, x):
        x = self.embedder(x)
        return self.tabnet.forward_masks(x)

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.type_as(x)
        logits, _ = self(x)
        return {"pred_logits": logits}

    def test_epoch_end(self, output_results):
        all_outputs = torch.cat([out["pred_logits"] for out in output_results],
                                dim=0)
        pred_probs = F.sigmoid(all_outputs).detach().cpu().numpy()
        return {"pred_probs": pred_probs}

    def setup(self, stage=None):

        self.train_dataset = BaselineTabNetMoADataset(
            train_features.loc[train_index, all_features].copy().values,
            train_labels.loc[train_index,
                             train_classes].copy().values, self.scaler)

        self.test_dataset = BaselineTabNetMoADataset(
            test_features.loc[:, all_features].copy(),
            None,
            self.scaler,
            is_train=False)

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        test_dataloader = DataLoader(self.test_dataset,
                                     batch_size=infer_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=False)
        print(f"Test iterations: {len(test_dataloader)}")
        return test_dataloader

    def configure_optimizers(self):
        pass


# In[27]:


# params = model_list["tabnet_baseline"]["params"]
# baseline_model = TabNet(params["input_dim"],
#                         params["output_dim"],
#                         n_d=params["features_dim"],
#                         n_a=params["features_dim"],
#                         n_steps=params["n_steps"],
#                         gamma=params["gamma"],
#                         cat_idxs=[],
#                         cat_dims=[],
#                         cat_emb_dim=1,
#                         n_independent=params["n_independent"],
#                         n_shared=params["n_shared"],
#                         epsilon=params["epsilon"],
#                         virtual_batch_size=virtual_batch_size,
#                         momentum=params["momentum"],
#                         mask_type=params["mask_type"])
# print(baseline_model)


# In[28]:


def get_model(model_name, model_path, input_dim, output_dim):
    if model_name.startswith("tabnet"):
        model = TabNet.load_from_checkpoint(model_path,
                                            input_dim=input_dim,
                                            output_dim=output_dim)
        model.freeze()
        model.eval()
        return model


# ### Inference

# In[29]:


rounds = 3
kfolds = 10
seed_everything(rand_seed)
trial_round_seeds = np.random.randint(42, 10000, size=rounds)


# In[30]:


model_name = "tabnet_baseline"

baseline_tabnet_kfold_submit_preds = np.zeros(
    (test_features.shape[0], len(train_classes)))

for r, trial_round_seed in enumerate(trial_round_seeds):
    print(
        f"Inferencing model in Trial Round {r} with Seed={trial_round_seed} ......"
    )
    skf = MultilabelStratifiedKFold(n_splits=kfolds,
                                    shuffle=True,
                                    random_state=trial_round_seed)

    label_counts = np.sum(train_labels.drop("sig_id", axis=1), axis=0)
    y_labels = label_counts.index.tolist()

    # Ensure Reproducibility
    seed_everything(trial_round_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    for i, (train_index, val_index) in enumerate(
            skf.split(train_features, train_labels[y_labels])):
        print(f"Inferencing on Fold {i} ......")
        print(train_index.shape, val_index.shape)

        input_dim = len(category_features) + len(numeric_features)
        output_dim = len(y_labels)  # Multi-label

        model_path = glob.glob(
            f'{model_list[model_name]["model_path"]}/{model_name}/seed{trial_round_seed}/fold{i}/epoch*.ckpt'
        )[0]
        print(f"Loading model from {model_path}")
        model = get_model(model_name, model_path, input_dim, output_dim)

        trainer = Trainer(
            logger=False,
            gpus=gpus,
            distributed_backend="dp",  # multiple-gpus, 1 machine
            benchmark=False,
            deterministic=True)
        output = trainer.test(model, verbose=False)[0]
        submit_preds = output["pred_probs"]
        baseline_tabnet_kfold_submit_preds += submit_preds / (rounds * kfolds)

        del model, trainer
        torch.cuda.empty_cache()
        gc.collect()


# In[31]:


print(baseline_tabnet_kfold_submit_preds.shape)
baseline_tabnet_kfold_submit_preds


# ## 2-Heads Deep ResNet Inference

# ### PCA RFE Selected Features

# In[32]:


pca_rfe_features = [
    "g-0", "g-7", "g-8", "g-10", "g-13", "g-17", "g-20", "g-22", "g-24",
    "g-26", "g-28", "g-29", "g-30", "g-31", "g-32", "g-34", "g-35", "g-36",
    "g-37", "g-38", "g-39", "g-41", "g-46", "g-48", "g-50", "g-51", "g-52",
    "g-55", "g-58", "g-59", "g-61", "g-62", "g-63", "g-65", "g-66", "g-67",
    "g-68", "g-70", "g-72", "g-74", "g-75", "g-79", "g-83", "g-84", "g-85",
    "g-86", "g-90", "g-91", "g-94", "g-95", "g-96", "g-97", "g-98", "g-100",
    "g-102", "g-105", "g-106", "g-112", "g-113", "g-114", "g-116", "g-121",
    "g-123", "g-126", "g-128", "g-131", "g-132", "g-134", "g-135", "g-138",
    "g-139", "g-140", "g-142", "g-144", "g-145", "g-146", "g-147", "g-148",
    "g-152", "g-155", "g-157", "g-158", "g-160", "g-163", "g-164", "g-165",
    "g-170", "g-173", "g-174", "g-175", "g-177", "g-178", "g-181", "g-183",
    "g-185", "g-186", "g-189", "g-192", "g-194", "g-195", "g-196", "g-197",
    "g-199", "g-201", "g-202", "g-206", "g-208", "g-210", "g-213", "g-214",
    "g-215", "g-220", "g-226", "g-228", "g-229", "g-235", "g-238", "g-241",
    "g-242", "g-243", "g-244", "g-245", "g-248", "g-250", "g-251", "g-254",
    "g-257", "g-259", "g-261", "g-266", "g-270", "g-271", "g-272", "g-275",
    "g-278", "g-282", "g-287", "g-288", "g-289", "g-291", "g-293", "g-294",
    "g-297", "g-298", "g-301", "g-303", "g-304", "g-306", "g-308", "g-309",
    "g-310", "g-311", "g-314", "g-315", "g-316", "g-317", "g-320", "g-321",
    "g-322", "g-327", "g-328", "g-329", "g-332", "g-334", "g-335", "g-336",
    "g-337", "g-339", "g-342", "g-344", "g-349", "g-350", "g-351", "g-353",
    "g-354", "g-355", "g-357", "g-359", "g-360", "g-364", "g-365", "g-366",
    "g-367", "g-368", "g-369", "g-374", "g-375", "g-377", "g-379", "g-385",
    "g-386", "g-390", "g-392", "g-393", "g-400", "g-402", "g-406", "g-407",
    "g-409", "g-410", "g-411", "g-414", "g-417", "g-418", "g-421", "g-423",
    "g-424", "g-427", "g-429", "g-431", "g-432", "g-433", "g-434", "g-437",
    "g-439", "g-440", "g-443", "g-449", "g-458", "g-459", "g-460", "g-461",
    "g-464", "g-467", "g-468", "g-470", "g-473", "g-477", "g-478", "g-479",
    "g-484", "g-485", "g-486", "g-488", "g-489", "g-491", "g-494", "g-496",
    "g-498", "g-500", "g-503", "g-504", "g-506", "g-508", "g-509", "g-512",
    "g-522", "g-529", "g-531", "g-534", "g-539", "g-541", "g-546", "g-551",
    "g-553", "g-554", "g-559", "g-561", "g-562", "g-565", "g-568", "g-569",
    "g-574", "g-577", "g-578", "g-586", "g-588", "g-590", "g-594", "g-595",
    "g-596", "g-597", "g-599", "g-600", "g-603", "g-607", "g-615", "g-618",
    "g-619", "g-620", "g-625", "g-628", "g-629", "g-632", "g-634", "g-635",
    "g-636", "g-638", "g-639", "g-641", "g-643", "g-644", "g-645", "g-646",
    "g-647", "g-648", "g-663", "g-664", "g-665", "g-668", "g-669", "g-670",
    "g-671", "g-672", "g-673", "g-674", "g-677", "g-678", "g-680", "g-683",
    "g-689", "g-691", "g-693", "g-695", "g-701", "g-702", "g-703", "g-704",
    "g-705", "g-706", "g-708", "g-711", "g-712", "g-720", "g-721", "g-723",
    "g-724", "g-726", "g-728", "g-731", "g-733", "g-738", "g-739", "g-742",
    "g-743", "g-744", "g-745", "g-749", "g-750", "g-752", "g-760", "g-761",
    "g-764", "g-766", "g-768", "g-770", "g-771", "c-0", "c-1", "c-2", "c-3",
    "c-4", "c-5", "c-6", "c-7", "c-8", "c-9", "c-10", "c-11", "c-12", "c-13",
    "c-14", "c-15", "c-16", "c-17", "c-18", "c-19", "c-20", "c-21", "c-22",
    "c-23", "c-24", "c-25", "c-26", "c-27", "c-28", "c-29", "c-30", "c-31",
    "c-32", "c-33", "c-34", "c-35", "c-36", "c-37", "c-38", "c-39", "c-40",
    "c-41", "c-42", "c-43", "c-44", "c-45", "c-46", "c-47", "c-48", "c-49",
    "c-50", "c-51", "c-52", "c-53", "c-54", "c-55", "c-56", "c-57", "c-58",
    "c-59", "c-60", "c-61", "c-62", "c-63", "c-64", "c-65", "c-66", "c-67",
    "c-68", "c-69", "c-70", "c-71", "c-72", "c-73", "c-74", "c-75", "c-76",
    "c-77", "c-78", "c-79", "c-80", "c-81", "c-82", "c-83", "c-84", "c-85",
    "c-86", "c-87", "c-88", "c-89", "c-90", "c-91", "c-92", "c-93", "c-94",
    "c-95", "c-96", "c-97", "c-98", "c-99"
]
len(pca_rfe_features)


# In[33]:


rfe_feature_indices = [
    train_features[all_features].columns.get_loc(c)
    for c in pca_rfe_features + category_features + ["cp_time"]
    if c in train_features
]
len(rfe_feature_indices)


# ### Model Definition
# 

# In[34]:


torch.cuda.empty_cache()
gc.collect()


# In[35]:


class TwoHeadsMoADataset(torch.utils.data.Dataset):
    def __init__(self,
                 features,
                 rfe_feature_indices,
                 labels,
                 scaler,
                 is_train=True):
        self.features = features
        self.rfe_feature_indices = rfe_feature_indices
        self.labels = labels
        self.scaler = scaler

        if is_train:
            self.features = scaler.fit_transform(self.features)
            self.rfe_features = np.copy(self.features[:, rfe_feature_indices])
        else:
            self.features = scaler.transform(self.features)
            self.rfe_features = np.copy(self.features[:, rfe_feature_indices])

    def __getitem__(self, index):
        if self.labels is not None:
            return {
                "x": self.features[index, :],
                "rfe_x": self.rfe_features[index, :],
                "y": self.labels[index, :]
            }
        else:
            # Return dummy label
            return {
                "x": self.features[index, :],
                "rfe_x": self.rfe_features[index, :],
                "y": -1
            }

    def __len__(self):
        return self.features.shape[0]


# In[36]:


# https://github.com/rwightman/gen-efficientnet-pytorch/blob/master/geffnet/efficientnet_builder.py#L672
def initialize_weight_goog(m, n='', fix_group_fanout=True):
    # weight init as per Tensorflow Official impl
    if isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        fan_out = m.weight.size(0)  # fan-out
        fan_in = 0
        if 'routing_fn' in n:
            fan_in = m.weight.size(1)
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        #         nn.init.xavier_normal_(m.weight.data)
        m.bias.data.zero_()


class DeepResNet(pl.LightningModule):
    def __init__(
            self,
            input_dim,
            output_dim,
            training_set=(None, None),  # tuple
            valid_set=(None, None),  # tuple
            test_set=None,
            momentum=0.1,
            learning_rate=1e-3):
        super(DeepResNet, self).__init__()

        self.train_data, self.train_labels = training_set
        self.valid_data, self.valid_labels = valid_set
        self.test_data = test_set

        self.head_1 = nn.Sequential(
            nn.BatchNorm1d(input_dim, momentum=momentum), nn.Dropout(p=0.2),
            Linear(input_dim, 512, bias=True), nn.ELU(),
            nn.BatchNorm1d(512, momentum=momentum), Linear(512, 256,
                                                           bias=True))

        self.head_2 = nn.Sequential(
            nn.BatchNorm1d(256 + len(rfe_feature_indices), momentum=momentum),
            nn.Dropout(p=0.3),
            Linear(256 + len(rfe_feature_indices), 512, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(512, momentum=momentum),
            Linear(512, 512, bias=True),
            nn.ELU(),
            nn.BatchNorm1d(512, momentum=momentum),
            Linear(512, 256, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(256, momentum=momentum),
            Linear(256, 256, bias=True),
            nn.ELU(),
        )

        # Avg+Max Concat
        #         self.final_layers = nn.Sequential(
        #             nn.BatchNorm1d(512, momentum=momentum), Linear(512,
        #                                                            256, bias=True),
        #             nn.SELU(), nn.BatchNorm1d(256, momentum=momentum),
        #             Linear(256, output_dim, bias=True))

        self.final_layers = nn.Sequential(
            nn.BatchNorm1d(256, momentum=momentum), Linear(256,
                                                           256, bias=True),
            nn.SELU(), nn.BatchNorm1d(256, momentum=momentum),
            Linear(256, output_dim, bias=True))

        if self.training:
            for m in self.head_1.modules():
                initialize_weight_goog(m)
            for m in self.head_2.modules():
                initialize_weight_goog(m)
            for m in self.final_layers.modules():
                initialize_weight_goog(m)


#         self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.scaler = RobustScaler(with_centering=True,
                                   with_scaling=True,
                                   quantile_range=(25.0, 75.0))

        # Save passed hyperparameters
        self.save_hyperparameters("momentum", "learning_rate")

    def forward(self, x, rfe_x):
        out_1 = self.head_1(x)

        concat_out = torch.cat([out_1, rfe_x], dim=1)

        out_2 = self.head_2(concat_out)

        avg_out = (out_1 + out_2) / 2
        #         max_out = torch.max(out_1, out_2)
        #         concat_out = torch.cat([avg_out, max_out], dim=1)

        #         out_3 = self.final_layers(concat_out)

        out_3 = self.final_layers(avg_out)

        return out_3

    def training_step(self, batch, batch_idx):
        x = batch["x"]
        rfe_x = batch["rfe_x"]
        y = batch["y"]
        x = x.float()
        rfe_x = rfe_x.type_as(x)
        y = y.type_as(x)
        logits = self(x, rfe_x)

        # Label smoothing
        #         logits = F.sigmoid(logits)
        #         logits = torch.clamp(logits, min=P_MIN, max=P_MAX)
        #         loss = F.binary_cross_entropy(logits, y, reduction="mean")

        loss = F.binary_cross_entropy_with_logits(logits, y, reduction="mean")

        self.log('train_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["x"]
        rfe_x = batch["rfe_x"]
        y = batch["y"]
        x = x.float()
        rfe_x = rfe_x.type_as(x)
        y = y.type_as(x)
        logits = self(x, rfe_x)

        val_loss = F.binary_cross_entropy_with_logits(logits,
                                                      y,
                                                      reduction="mean")

        self.log('val_loss',
                 val_loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

        return val_loss

    def test_step(self, batch, batch_idx):
        x = batch["x"]
        rfe_x = batch["rfe_x"]
        y = batch["y"]
        x = x.float()
        rfe_x = rfe_x.type_as(x)
        y = y.type_as(x)
        logits = self(x, rfe_x)
        return {"pred_logits": logits}

    def test_epoch_end(self, output_results):
        all_outputs = torch.cat([out["pred_logits"] for out in output_results],
                                dim=0)
        pred_probs = F.sigmoid(all_outputs).detach().cpu().numpy()
        print(pred_probs)
        return {"pred_probs": pred_probs}

    def setup(self, stage=None):

        self.train_dataset = TwoHeadsMoADataset(self.train_data,
                                                rfe_feature_indices,
                                                self.train_labels, self.scaler)

        self.val_dataset = TwoHeadsMoADataset(self.valid_data,
                                              rfe_feature_indices,
                                              self.valid_labels,
                                              self.scaler,
                                              is_train=False)

        self.test_dataset = TwoHeadsMoADataset(self.test_data,
                                               rfe_feature_indices,
                                               None,
                                               self.scaler,
                                               is_train=False)

    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      pin_memory=True,
                                      drop_last=False)
        print(f"Train iterations: {len(train_dataloader)}")
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(self.val_dataset,
                                    batch_size=infer_batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=True,
                                    drop_last=False)
        print(f"Validate iterations: {len(val_dataloader)}")
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(self.test_dataset,
                                     batch_size=infer_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=False)
        print(f"Test iterations: {len(test_dataloader)}")
        return test_dataloader

    def configure_optimizers(self):
        print(f"Initial Learning Rate: {self.hparams.learning_rate:.6f}")
        optimizer = optim.Adam(self.parameters(),
                               lr=self.hparams.learning_rate,
                               weight_decay=weight_decay)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=T_max,
                                                         eta_min=0,
                                                         last_epoch=-1)
        return [optimizer], [scheduler]


# In[37]:


def get_model(model_path, training_set, valid_set, test_set, input_dim,
              output_dim):
    model = DeepResNet.load_from_checkpoint(model_path,
                                            input_dim=input_dim,
                                            output_dim=output_dim,
                                            training_set=training_set,
                                            valid_set=valid_set,
                                            test_set=test_set)
    model.freeze()
    model.eval()
    return model


# ### Inference

# In[38]:


rounds = 3
kfolds = 10
seed_everything(rand_seed)
trial_round_seeds = np.random.randint(42, 10000, size=rounds)


# In[39]:


def extract_2head_stats_features(train, test):
    for df in [train, test]:
        df['g_mean'] = df[gene_experssion_features].mean(axis=1)
        df['c_mean'] = df[cell_viability_features].mean(axis=1)

        df['g_sum'] = df[gene_experssion_features].sum(axis=1)
        df['c_sum'] = df[cell_viability_features].sum(axis=1)

        df['g_med'] = np.median(df[gene_experssion_features], axis=1)
        df['c_med'] = np.median(df[cell_viability_features], axis=1)

        df['g_std'] = df[gene_experssion_features].sum(axis=1)
        df['c_std'] = df[cell_viability_features].sum(axis=1)

        df['g_kurt'] = df[gene_experssion_features].kurtosis(axis=1)
        df['c_kurt'] = df[cell_viability_features].kurtosis(axis=1)

        df['g_skew'] = df[gene_experssion_features].skew(axis=1)
        df['c_skew'] = df[cell_viability_features].skew(axis=1)


def extract_2heads_pca_components(train, test, n_components, prefix, seed):
    pca = PCA(n_components=n_components, random_state=seed)
    train_components = pca.fit_transform(train)
    test_components = pca.transform(test)

    columns = [f'pca_{prefix}{i + 1}' for i in range(n_components)]
    train_components = pd.DataFrame(train_components, columns=columns)
    test_components = pd.DataFrame(test_components, columns=columns)

    return train_components, test_components


# In[40]:


model_name = "2heads_deep_resnets_v1"

twoheads_kfold_submit_preds = np.zeros(
    (test_features.shape[0], len(train_classes)))

twoheads_train_features = train_features.copy()
twoheads_test_features = test_features.copy()
extract_2head_stats_features(twoheads_train_features, twoheads_test_features)

for r, trial_round_seed in enumerate(trial_round_seeds):
    print(
        f"Inferencing model in Trial Round {r} with Seed={trial_round_seed} ......"
    )
    skf = MultilabelStratifiedKFold(n_splits=kfolds,
                                    shuffle=True,
                                    random_state=trial_round_seed)
    label_counts = np.sum(train_labels.drop("sig_id", axis=1), axis=0)
    y_labels = label_counts.index.tolist()

    train_gene_pca, test_gene_pca = extract_2heads_pca_components(
        twoheads_train_features[gene_experssion_features],
        twoheads_test_features[gene_experssion_features],
        n_components=80,
        prefix="g",
        seed=trial_round_seed)
    train_cell_pca, test_cell_pca = extract_2heads_pca_components(
        twoheads_train_features[cell_viability_features],
        twoheads_test_features[cell_viability_features],
        n_components=30,
        prefix="c",
        seed=trial_round_seed)

    seed_train_features = pd.concat([twoheads_train_features, train_gene_pca],
                                    axis=1)
    seed_test_features = pd.concat([twoheads_test_features, test_gene_pca],
                                   axis=1)

    seed_train_features = pd.concat([seed_train_features, train_cell_pca],
                                    axis=1)
    seed_test_features = pd.concat([seed_test_features, test_cell_pca], axis=1)

    # Update feature list
    numeric_features = [
        c for c in seed_train_features.columns
        if c != "sig_id" and c not in category_features
    ]
    fold_all_features = category_features + numeric_features

    # Ensure Reproducibility
    seed_everything(trial_round_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    for i, (train_index, val_index) in enumerate(
            skf.split(train_features, train_labels[y_labels])):
        print(f"Inferencing on Fold {i} ......")
        print(train_index.shape, val_index.shape)

        model_path = glob.glob(
            f'{model_list[model_name]["model_path"]}/{model_name}/seed{trial_round_seed}/fold{i}/epoch*.ckpt'
        )[0]

        seed_train = seed_train_features.loc[train_index,
                                             fold_all_features].copy().values
        seed_train_labels = train_labels.loc[train_index,
                                             train_classes].copy().values
        seed_valid = seed_train_features.loc[val_index,
                                             fold_all_features].copy().values
        seed_valid_labels = train_labels.loc[val_index,
                                             train_classes].copy().values
        seed_test = seed_test_features[fold_all_features].copy().values

        input_dim = len(fold_all_features)
        output_dim = len(y_labels)  # Multi-label

        print(f"Loading model from {model_path}")
        model = get_model(model_path, (seed_train, seed_train_labels),
                          (seed_valid, seed_valid_labels), seed_test,
                          input_dim, output_dim)

        trainer = Trainer(
            logger=False,
            gpus=gpus,
            distributed_backend="dp",  # multiple-gpus, 1 machine
            benchmark=False,
            deterministic=True)
        output = trainer.test(model, verbose=False)[0]
        submit_preds = output["pred_probs"]
        twoheads_kfold_submit_preds += submit_preds / (rounds * kfolds)

        del model, trainer
        torch.cuda.empty_cache()
        gc.collect()


# In[41]:


del twoheads_train_features, twoheads_test_features
gc.collect()


# In[42]:


print(twoheads_kfold_submit_preds.shape)
twoheads_kfold_submit_preds


# ## DeepInsight EfficientNet B3 NoisyStudent

# In[43]:


model_type = "b3"
pretrained_model = f"tf_efficientnet_{model_type}_ns"
experiment_name = f"deepinsight_efficientnet_v7_{model_type}"

batch_size = 48
infer_batch_size = 384
# infer_batch_size = 512
image_size = 300  # B3
drop_rate = 0.3  # B3
resolution = 300

# DeepInsight Transform
perplexity = 5

drop_connect_rate = 0.2
fc_size = 512


# ### Load MoA Data

# In[44]:


train_features = pd.read_csv(f"{dataset_folder}/train_features.csv",
                             engine='c')
train_labels = pd.read_csv(f"{dataset_folder}/train_targets_scored.csv",
                           engine='c')

train_extra_labels = pd.read_csv(
    f"{dataset_folder}/train_targets_nonscored.csv", engine='c')

test_features = pd.read_csv(f"{dataset_folder}/test_features.csv", engine='c')

sample_submission = pd.read_csv(f"{dataset_folder}/sample_submission.csv",
                                engine='c')


# In[45]:


# Sort by sig_id to ensure that all row orders match
train_features = train_features.sort_values(
    by=["sig_id"], axis=0, inplace=False).reset_index(drop=True)
train_labels = train_labels.sort_values(by=["sig_id"], axis=0,
                                        inplace=False).reset_index(drop=True)
train_extra_labels = train_extra_labels.sort_values(
    by=["sig_id"], axis=0, inplace=False).reset_index(drop=True)

sample_submission = sample_submission.sort_values(
    by=["sig_id"], axis=0, inplace=False).reset_index(drop=True)


# In[46]:


category_features = ["cp_type", "cp_dose"]
numeric_features = [
    c for c in train_features.columns
    if c != "sig_id" and c not in category_features
]
all_features = category_features + numeric_features
gene_experssion_features = [c for c in numeric_features if c.startswith("g-")]
cell_viability_features = [c for c in numeric_features if c.startswith("c-")]
len(numeric_features), len(gene_experssion_features), len(
    cell_viability_features)


# In[47]:


train_classes = [c for c in train_labels.columns if c != "sig_id"]
train_extra_classes = [c for c in train_extra_labels.columns if c != "sig_id"]
len(train_classes), len(train_extra_classes)


# ### Feature Encoding

# In[48]:


for df in [train_features, test_features]:
    df['cp_type'] = df['cp_type'].map({'ctl_vehicle': 0, 'trt_cp': 1})
    df['cp_dose'] = df['cp_dose'].map({'D1': 0, 'D2': 1})
    df['cp_time'] = df['cp_time'].map({24: 0, 48: 0.5, 72: 1})


# ### DeepInsight Transform - t-SNE 2D Embeddings

# In[49]:


# Modified from DeepInsight Transform
# https://github.com/alok-ai-lab/DeepInsight/blob/master/pyDeepInsight/image_transformer.py

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from scipy.spatial import ConvexHull
from matplotlib import pyplot as plt
import inspect


class DeepInsightTransformer:
    """Transform features to an image matrix using dimensionality reduction

    This class takes in data normalized between 0 and 1 and converts it to a
    CNN compatible 'image' matrix

    """
    def __init__(self,
                 feature_extractor='tsne',
                 perplexity=30,
                 pixels=100,
                 random_state=None,
                 n_jobs=None):
        """Generate an ImageTransformer instance

        Args:
            feature_extractor: string of value ('tsne', 'pca', 'kpca') or a
                class instance with method `fit_transform` that returns a
                2-dimensional array of extracted features.
            pixels: int (square matrix) or tuple of ints (height, width) that
                defines the size of the image matrix.
            random_state: int or RandomState. Determines the random number
                generator, if present, of a string defined feature_extractor.
            n_jobs: The number of parallel jobs to run for a string defined
                feature_extractor.
        """
        self.random_state = random_state
        self.n_jobs = n_jobs

        if isinstance(feature_extractor, str):
            fe = feature_extractor.casefold()
            if fe == 'tsne_exact'.casefold():
                fe = TSNE(n_components=2,
                          metric='cosine',
                          perplexity=perplexity,
                          n_iter=1000,
                          method='exact',
                          random_state=self.random_state,
                          n_jobs=self.n_jobs)
            elif fe == 'tsne'.casefold():
                fe = TSNE(n_components=2,
                          metric='cosine',
                          perplexity=perplexity,
                          n_iter=1000,
                          method='barnes_hut',
                          random_state=self.random_state,
                          n_jobs=self.n_jobs)
            elif fe == 'pca'.casefold():
                fe = PCA(n_components=2, random_state=self.random_state)
            elif fe == 'kpca'.casefold():
                fe = KernelPCA(n_components=2,
                               kernel='rbf',
                               random_state=self.random_state,
                               n_jobs=self.n_jobs)
            else:
                raise ValueError(("Feature extraction method '{}' not accepted"
                                  ).format(feature_extractor))
            self._fe = fe
        elif hasattr(feature_extractor, 'fit_transform') and                 inspect.ismethod(feature_extractor.fit_transform):
            self._fe = feature_extractor
        else:
            raise TypeError('Parameter feature_extractor is not a '
                            'string nor has method "fit_transform"')

        if isinstance(pixels, int):
            pixels = (pixels, pixels)

        # The resolution of transformed image
        self._pixels = pixels
        self._xrot = None

    def fit(self, X, y=None, plot=False):
        """Train the image transformer from the training set (X)

        Args:
            X: {array-like, sparse matrix} of shape (n_samples, n_features)
            y: Ignored. Present for continuity with scikit-learn
            plot: boolean of whether to produce a scatter plot showing the
                feature reduction, hull points, and minimum bounding rectangle

        Returns:
            self: object
        """
        # Transpose to get (n_features, n_samples)
        X = X.T

        # Perform dimensionality reduction
        x_new = self._fe.fit_transform(X)

        # Get the convex hull for the points
        chvertices = ConvexHull(x_new).vertices
        hull_points = x_new[chvertices]

        # Determine the minimum bounding rectangle
        mbr, mbr_rot = self._minimum_bounding_rectangle(hull_points)

        # Rotate the matrix
        # Save the rotated matrix in case user wants to change the pixel size
        self._xrot = np.dot(mbr_rot, x_new.T).T

        # Determine feature coordinates based on pixel dimension
        self._calculate_coords()

        # plot rotation diagram if requested
        if plot is True:
            # Create subplots
            fig, ax = plt.subplots(1, 1, figsize=(10, 7), squeeze=False)
            ax[0, 0].scatter(x_new[:, 0],
                             x_new[:, 1],
                             cmap=plt.cm.get_cmap("jet", 10),
                             marker="x",
                             alpha=1.0)
            ax[0, 0].fill(x_new[chvertices, 0],
                          x_new[chvertices, 1],
                          edgecolor='r',
                          fill=False)
            ax[0, 0].fill(mbr[:, 0], mbr[:, 1], edgecolor='g', fill=False)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.show()
        return self

    @property
    def pixels(self):
        """The image matrix dimensions

        Returns:
            tuple: the image matrix dimensions (height, width)

        """
        return self._pixels

    @pixels.setter
    def pixels(self, pixels):
        """Set the image matrix dimension

        Args:
            pixels: int or tuple with the dimensions (height, width)
            of the image matrix

        """
        if isinstance(pixels, int):
            pixels = (pixels, pixels)
        self._pixels = pixels
        # recalculate coordinates if already fit
        if hasattr(self, '_coords'):
            self._calculate_coords()

    def _calculate_coords(self):
        """Calculate the matrix coordinates of each feature based on the
        pixel dimensions.
        """
        ax0_coord = np.digitize(self._xrot[:, 0],
                                bins=np.linspace(min(self._xrot[:, 0]),
                                                 max(self._xrot[:, 0]),
                                                 self._pixels[0])) - 1
        ax1_coord = np.digitize(self._xrot[:, 1],
                                bins=np.linspace(min(self._xrot[:, 1]),
                                                 max(self._xrot[:, 1]),
                                                 self._pixels[1])) - 1
        self._coords = np.stack((ax0_coord, ax1_coord))

    def transform(self, X, empty_value=0):
        """Transform the input matrix into image matrices

        Args:
            X: {array-like, sparse matrix} of shape (n_samples, n_features)
                where n_features matches the training set.
            empty_value: numeric value to fill elements where no features are
                mapped. Default = 0 (although it was 1 in the paper).

        Returns:
            A list of n_samples numpy matrices of dimensions set by
            the pixel parameter
        """

        # Group by location (x1, y1) of each feature
        # Tranpose to get (n_features, n_samples)
        img_coords = pd.DataFrame(np.vstack(
            (self._coords, X.clip(0, 1))).T).groupby(
                [0, 1],  # (x1, y1)
                as_index=False).mean()

        img_matrices = []
        blank_mat = np.zeros(self._pixels)
        if empty_value != 0:
            blank_mat[:] = empty_value
        for z in range(2, img_coords.shape[1]):
            img_matrix = blank_mat.copy()
            img_matrix[img_coords[0].astype(int),
                       img_coords[1].astype(int)] = img_coords[z]
            img_matrices.append(img_matrix)

        return img_matrices

    def transform_3d(self, X, empty_value=0):
        """Transform the input matrix into image matrices

        Args:
            X: {array-like, sparse matrix} of shape (n_samples, n_features)
                where n_features matches the training set.
            empty_value: numeric value to fill elements where no features are
                mapped. Default = 0 (although it was 1 in the paper).

        Returns:
            A list of n_samples numpy matrices of dimensions set by
            the pixel parameter
        """

        # Group by location (x1, y1) of each feature
        # Tranpose to get (n_features, n_samples)
        img_coords = pd.DataFrame(np.vstack(
            (self._coords, X.clip(0, 1))).T).groupby(
                [0, 1],  # (x1, y1)
                as_index=False)
        avg_img_coords = img_coords.mean()
        min_img_coords = img_coords.min()
        max_img_coords = img_coords.max()

        img_matrices = []
        blank_mat = np.zeros((3, self._pixels[0], self._pixels[1]))
        if empty_value != 0:
            blank_mat[:, :, :] = empty_value
        for z in range(2, avg_img_coords.shape[1]):
            img_matrix = blank_mat.copy()
            img_matrix[0, avg_img_coords[0].astype(int),
                       avg_img_coords[1].astype(int)] = avg_img_coords[z]
            img_matrix[1, min_img_coords[0].astype(int),
                       min_img_coords[1].astype(int)] = min_img_coords[z]
            img_matrix[2, max_img_coords[0].astype(int),
                       max_img_coords[1].astype(int)] = max_img_coords[z]
            img_matrices.append(img_matrix)

        return img_matrices

    def fit_transform(self, X, empty_value=0):
        """Train the image transformer from the training set (X) and return
        the transformed data.

        Args:
            X: {array-like, sparse matrix} of shape (n_samples, n_features)
            empty_value: numeric value to fill elements where no features are
                mapped. Default = 0 (although it was 1 in the paper).

        Returns:
            A list of n_samples numpy matrices of dimensions set by
            the pixel parameter
        """
        self.fit(X)
        return self.transform(X, empty_value=empty_value)

    def fit_transform_3d(self, X, empty_value=0):
        """Train the image transformer from the training set (X) and return
        the transformed data.

        Args:
            X: {array-like, sparse matrix} of shape (n_samples, n_features)
            empty_value: numeric value to fill elements where no features are
                mapped. Default = 0 (although it was 1 in the paper).

        Returns:
            A list of n_samples numpy matrices of dimensions set by
            the pixel parameter
        """
        self.fit(X)
        return self.transform_3d(X, empty_value=empty_value)

    def feature_density_matrix(self):
        """Generate image matrix with feature counts per pixel

        Returns:
            img_matrix (ndarray): matrix with feature counts per pixel
        """
        fdmat = np.zeros(self._pixels)
        # Group by location (x1, y1) of each feature
        # Tranpose to get (n_features, n_samples)
        coord_cnt = (
            pd.DataFrame(self._coords.T).assign(count=1).groupby(
                [0, 1],  # (x1, y1)
                as_index=False).count())
        fdmat[coord_cnt[0].astype(int),
              coord_cnt[1].astype(int)] = coord_cnt['count']
        return fdmat

    @staticmethod
    def _minimum_bounding_rectangle(hull_points):
        """Find the smallest bounding rectangle for a set of points.

        Modified from JesseBuesking at https://stackoverflow.com/a/33619018
        Returns a set of points representing the corners of the bounding box.

        Args:
            hull_points : an nx2 matrix of hull coordinates

        Returns:
            (tuple): tuple containing
                coords (ndarray): coordinates of the corners of the rectangle
                rotmat (ndarray): rotation matrix to align edges of rectangle
                    to x and y
        """

        pi2 = np.pi / 2.

        # Calculate edge angles
        edges = hull_points[1:] - hull_points[:-1]
        angles = np.arctan2(edges[:, 1], edges[:, 0])
        angles = np.abs(np.mod(angles, pi2))
        angles = np.unique(angles)

        # Find rotation matrices
        rotations = np.vstack([
            np.cos(angles),
            np.cos(angles - pi2),
            np.cos(angles + pi2),
            np.cos(angles)
        ]).T
        rotations = rotations.reshape((-1, 2, 2))

        # Apply rotations to the hull
        rot_points = np.dot(rotations, hull_points.T)

        # Find the bounding points
        min_x = np.nanmin(rot_points[:, 0], axis=1)
        max_x = np.nanmax(rot_points[:, 0], axis=1)
        min_y = np.nanmin(rot_points[:, 1], axis=1)
        max_y = np.nanmax(rot_points[:, 1], axis=1)

        # Find the box with the best area
        areas = (max_x - min_x) * (max_y - min_y)
        best_idx = np.argmin(areas)

        # Return the best box
        x1 = max_x[best_idx]
        x2 = min_x[best_idx]
        y1 = max_y[best_idx]
        y2 = min_y[best_idx]
        rotmat = rotations[best_idx]

        # Generate coordinates
        coords = np.zeros((4, 2))
        coords[0] = np.dot([x1, y2], rotmat)
        coords[1] = np.dot([x2, y2], rotmat)
        coords[2] = np.dot([x2, y1], rotmat)
        coords[3] = np.dot([x1, y1], rotmat)

        return coords, rotmat


# In[50]:


class LogScaler:
    """Log normalize and scale data

    Log normalization and scaling procedure as described as norm-2 in the
    DeepInsight paper supplementary information.
    
    Note: The dimensions of input matrix is (N samples, d features)
    """
    def __init__(self):
        self._min0 = None
        self._max = None

    """
    Use this as a preprocessing step in inference mode.
    """

    def fit(self, X, y=None):
        # Min. of training set per feature
        self._min0 = X.min(axis=0)

        # Log normalized X by log(X + _min0 + 1)
        X_norm = np.log(
            X +
            np.repeat(np.abs(self._min0)[np.newaxis, :], X.shape[0], axis=0) +
            1).clip(min=0, max=None)

        # Global max. of training set from X_norm
        self._max = X_norm.max()

    """
    For training set only.
    """

    def fit_transform(self, X, y=None):
        # Min. of training set per feature
        self._min0 = X.min(axis=0)

        # Log normalized X by log(X + _min0 + 1)
        X_norm = np.log(
            X +
            np.repeat(np.abs(self._min0)[np.newaxis, :], X.shape[0], axis=0) +
            1).clip(min=0, max=None)

        # Global max. of training set from X_norm
        self._max = X_norm.max()

        # Normalized again by global max. of training set
        return (X_norm / self._max).clip(0, 1)

    """
    For validation and test set only.
    """

    def transform(self, X, y=None):
        # Adjust min. of each feature of X by _min0
        for i in range(X.shape[1]):
            X[:, i] = X[:, i].clip(min=self._min0[i], max=None)

        # Log normalized X by log(X + _min0 + 1)
        X_norm = np.log(
            X +
            np.repeat(np.abs(self._min0)[np.newaxis, :], X.shape[0], axis=0) +
            1).clip(min=0, max=None)

        # Normalized again by global max. of training set
        return (X_norm / self._max).clip(0, 1)


# ### Dataset

# In[51]:


class MoAImageDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, transformer):
        self.features = features
        self.labels = labels
        self.transformer = transformer

    def __getitem__(self, index):
        normalized = self.features[index, :]
        normalized = np.expand_dims(normalized, axis=0)

        # Note: we are setting empty_value=0
        image = self.transformer.transform_3d(normalized, empty_value=0)[0]

        return {"x": image, "y": self.labels[index, :]}

    def __len__(self):
        return self.features.shape[0]


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, transformer):
        self.features = features
        self.labels = labels
        self.transformer = transformer

    def __getitem__(self, index):
        normalized = self.features[index, :]
        normalized = np.expand_dims(normalized, axis=0)

        # Note: we are setting empty_value=0
        image = self.transformer.transform_3d(normalized, empty_value=0)[0]

        return {"x": image, "y": -1}

    def __len__(self):
        return self.features.shape[0]


# ### Model Definition

# In[52]:


# Reference: https://github.com/rwightman/gen-efficientnet-pytorch/blob/master/geffnet/efficientnet_builder.py#L672
def initialize_weight_goog(m, n='', fix_group_fanout=True):
    # weight init as per Tensorflow Official impl
    # https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_model.py
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        if fix_group_fanout:
            fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        fan_out = m.weight.size(0)  # fan-out
        fan_in = 0
        if 'routing_fn' in n:
            fan_in = m.weight.size(1)
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        m.weight.data.uniform_(-init_range, init_range)
        m.bias.data.zero_()


def initialize_weight_default(m, n=''):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight,
                                 mode='fan_in',
                                 nonlinearity='linear')


# In[53]:


class MoAEfficientNet(pl.LightningModule):
    def __init__(
            self,
            pretrained_model_name,
            training_set=(None, None),  # tuple
            valid_set=(None, None),  # tuple
            test_set=None,
            transformer=None,
            num_classes=206,
            in_chans=3,
            drop_rate=0.,
            drop_connect_rate=0.,
            fc_size=512,
            learning_rate=1e-3,
            weight_init='goog'):
        super(MoAEfficientNet, self).__init__()

        self.train_data, self.train_labels = training_set
        self.valid_data, self.valid_labels = valid_set
        self.test_data = test_set
        self.transformer = transformer

        self.backbone = getattr(geffnet, pretrained_model)(
            pretrained=True,
            in_chans=in_chans,
            drop_rate=drop_rate,
            drop_connect_rate=drop_connect_rate,
            weight_init=weight_init)

        self.backbone.classifier = nn.Sequential(
            nn.Linear(self.backbone.classifier.in_features, fc_size,
                      bias=True), nn.ELU(),
            nn.Linear(fc_size, num_classes, bias=True))

        if self.training:
            for m in self.backbone.classifier.modules():
                initialize_weight_goog(m)

        # Save passed hyperparameters
        self.save_hyperparameters("pretrained_model_name", "num_classes",
                                  "in_chans", "drop_rate", "drop_connect_rate",
                                  "weight_init", "fc_size", "learning_rate")

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x = batch["x"]
        y = batch["y"]
        x = x.float()
        y = y.type_as(x)
        logits = self(x)

        loss = F.binary_cross_entropy_with_logits(logits, y, reduction="mean")

        self.log('train_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["x"]
        y = batch["y"]
        x = x.float()
        y = y.type_as(x)
        logits = self(x)

        val_loss = F.binary_cross_entropy_with_logits(logits,
                                                      y,
                                                      reduction="mean")

        self.log('val_loss',
                 val_loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

        return val_loss

    def test_step(self, batch, batch_idx):
        x = batch["x"]
        y = batch["y"]
        x = x.float()
        y = y.type_as(x)
        logits = self(x)
        return {"pred_logits": logits}

    def test_epoch_end(self, output_results):
        all_outputs = torch.cat([out["pred_logits"] for out in output_results],
                                dim=0)
        print("Logits:", all_outputs)
        pred_probs = F.sigmoid(all_outputs).detach().cpu().numpy()
        print("Predictions: ", pred_probs)
        return {"pred_probs": pred_probs}

    def setup(self, stage=None):
        self.train_dataset = MoAImageDataset(self.train_data,
                                             self.train_labels,
                                             self.transformer)

        self.val_dataset = MoAImageDataset(self.valid_data, self.valid_labels,
                                           self.transformer)

        self.test_dataset = TestDataset(self.test_data, None, self.transformer)

    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      pin_memory=True,
                                      drop_last=False)
        print(f"Train iterations: {len(train_dataloader)}")
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(self.val_dataset,
                                    batch_size=infer_batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=True,
                                    drop_last=False)
        print(f"Validate iterations: {len(val_dataloader)}")
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(self.test_dataset,
                                     batch_size=infer_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=False)
        print(f"Test iterations: {len(test_dataloader)}")
        return test_dataloader

    def configure_optimizers(self):
        print(f"Initial Learning Rate: {self.hparams.learning_rate:.6f}")
        optimizer = optim.Adam(self.parameters(),
                               lr=self.hparams.learning_rate,
                               weight_decay=weight_decay)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=T_max,
                                                         eta_min=0,
                                                         last_epoch=-1)

        return [optimizer], [scheduler]


# ### Inference

# In[54]:


kfolds = 10
skf = MultilabelStratifiedKFold(n_splits=kfolds,
                                shuffle=True,
                                random_state=rand_seed)

label_counts = np.sum(train_labels.drop("sig_id", axis=1), axis=0)
y_labels = label_counts.index.tolist()


# In[55]:


def get_model(model_path, test_set, transformer):
    model = MoAEfficientNet.load_from_checkpoint(
        model_path,
        pretrained_model_name=pretrained_model,
        training_set=(None, None),  # tuple
        valid_set=(None, None),  # tuple
        test_set=test_set,
        transformer=transformer,
        drop_rate=drop_rate,
        drop_connect_rate=drop_connect_rate,
        fc_size=fc_size,
        weight_init='goog')

    model.freeze()
    model.eval()
    return model


def save_pickle(obj, model_output_folder, fold_i, name):
    dump(obj, open(f"{model_output_folder}/fold{fold_i}_{name}.pkl", 'wb'),
         pickle.HIGHEST_PROTOCOL)


def load_pickle(model_output_folder, fold_i, name):
    return load(open(f"{model_output_folder}/fold{fold_i}_{name}.pkl", 'rb'))


# In[56]:


# Ensure Reproducibility
seed_everything(rand_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

effnetb3_kfold_submit_preds = np.zeros(
    (test_features.shape[0], len(train_classes)))
for i, (train_index, val_index) in enumerate(
        skf.split(train_features, train_labels[y_labels])):
    print(f"Inferencing on Fold {i} ......")
    print(train_index.shape, val_index.shape)

    model_path = glob.glob(
        f'{model_list[experiment_name]["model_path"]}/{experiment_name}/fold{i}/epoch*.ckpt'
    )[0]

    test = test_features[all_features].copy().values

    # Load LogScaler (Norm-2 Normalization)
    scaler = load_pickle(
        f'{model_list[experiment_name]["model_path"]}/{experiment_name}', i,
        "log-scaler")
    test = scaler.transform(test)

    # Load DeepInsight Feature Map
    transformer = load_pickle(
        f'{model_list[experiment_name]["model_path"]}/{experiment_name}', i,
        "deepinsight-transform")

    print(f"Loading model from {model_path}")
    model = get_model(model_path, test_set=test, transformer=transformer)

    trainer = Trainer(
        logger=False,
        gpus=gpus,
        distributed_backend="dp",  # multiple-gpus, 1 machine
        precision=16,
        benchmark=False,
        deterministic=True)
    output = trainer.test(model, verbose=False)[0]
    submit_preds = output["pred_probs"]
    effnetb3_kfold_submit_preds += submit_preds / kfolds

    del model, trainer, scaler, transformer, test
    torch.cuda.empty_cache()
    gc.collect()


# ## DeepInsight EfficientNet B3 NoisyStudent (Seed=419)

# In[57]:


rand_seed = 419

model_type = "b3"
pretrained_model = f"tf_efficientnet_{model_type}_ns"
experiment_name = f"deepinsight_efficientnet_v7_{model_type}_seed2"

batch_size = 48
infer_batch_size = 384
# infer_batch_size = 512
image_size = 300  # B3
drop_rate = 0.3  # B3
resolution = 300

# DeepInsight Transform
perplexity = 5

drop_connect_rate = 0.2
fc_size = 512


# In[58]:


kfolds = 10
skf = MultilabelStratifiedKFold(n_splits=kfolds,
                                shuffle=True,
                                random_state=rand_seed)

label_counts = np.sum(train_labels.drop("sig_id", axis=1), axis=0)
y_labels = label_counts.index.tolist()


# In[59]:


# Ensure Reproducibility
seed_everything(rand_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

effnetb3_seed2_kfold_submit_preds = np.zeros(
    (test_features.shape[0], len(train_classes)))
for i, (train_index, val_index) in enumerate(
        skf.split(train_features, train_labels[y_labels])):
    print(f"Inferencing on Fold {i} ......")
    print(train_index.shape, val_index.shape)

    model_path = glob.glob(
        f'{model_list[experiment_name]["model_path"]}/{experiment_name}/fold{i}/epoch*.ckpt'
    )[0]

    test = test_features[all_features].copy().values

    # Load LogScaler (Norm-2 Normalization)
    scaler = load_pickle(
        f'{model_list[experiment_name]["model_path"]}/{experiment_name}', i,
        "log-scaler")
    test = scaler.transform(test)

    # Load DeepInsight Feature Map
    transformer = load_pickle(
        f'{model_list[experiment_name]["model_path"]}/{experiment_name}', i,
        "deepinsight-transform")

    print(f"Loading model from {model_path}")
    model = get_model(model_path, test_set=test, transformer=transformer)

    trainer = Trainer(
        logger=False,
        gpus=gpus,
        distributed_backend="dp",  # multiple-gpus, 1 machine
        precision=16,
        benchmark=False,
        deterministic=True)
    output = trainer.test(model, verbose=False)[0]
    submit_preds = output["pred_probs"]
    effnetb3_seed2_kfold_submit_preds += submit_preds / kfolds

    del model, trainer, scaler, transformer, test
    torch.cuda.empty_cache()
    gc.collect()


# ## DeepInsight ResNeSt V1

# In[60]:


training_mode = False


# In[61]:


import resnest
from resnest.torch import resnest50, resnest101, resnest200, resnest269,     resnest50_fast_2s2x40d, resnest50_fast_1s2x40d, resnest50_fast_1s1x64d


# In[62]:


if kernel_mode:
    get_ipython().system('mkdir -p /root/.cache/torch/hub/checkpoints/')
    get_ipython().system('cp ../input/deepinsight-resnest-v1-resnest50/*.pth /root/.cache/torch/hub/checkpoints/')
    get_ipython().system('ls -la /root/.cache/torch/hub/checkpoints/')


# In[63]:


kfolds = 10
skf = MultilabelStratifiedKFold(n_splits=kfolds,
                                shuffle=True,
                                random_state=rand_seed)

label_counts = np.sum(train_labels.drop("sig_id", axis=1), axis=0)
y_labels = label_counts.index.tolist()


# In[64]:


model_type = "resnest50"
pretrained_model = f"resnest50_fast_2s2x40d"
experiment_name = f"deepinsight_ResNeSt_v1_{model_type}"

if kernel_mode:
    dataset_folder = "../input/lish-moa"
    model_output_folder = f"./{experiment_name}" if training_mode         else f"../input/deepinsight-resnest-v1-resnest50/{experiment_name}"
else:
    dataset_folder = "/workspace/Kaggle/MoA"
    model_output_folder = f"{dataset_folder}/{experiment_name}" if training_mode         else f"/workspace/Kaggle/MoA/completed/deepinsight_ResNeSt_v1_resnest50/{experiment_name}"

if training_mode:
    os.makedirs(model_output_folder, exist_ok=True)

    # Dedicated logger for experiment
    exp_logger = TensorBoardLogger(model_output_folder,
                                   name=f"overall_logs",
                                   default_hp_metric=False)

# debug_mode = True
debug_mode = False

num_workers = 6
# gpus = [0, 1]
gpus = [0]
# gpus = [1]

epochs = 200
patience = 16

# learning_rate = 1e-3
learning_rate = 0.000352  # Suggested Learning Rate from LR finder (V7)
learning_rate *= len(gpus)
weight_decay = 1e-6
# weight_decay = 0

# T_max = 10  # epochs
T_max = 5  # epochs
T_0 = 5  # epochs

accumulate_grad_batches = 1
gradient_clip_val = 10.0

if "resnest50" in model_type:
    batch_size = 128
    infer_batch_size = 256 if not kernel_mode else 384
    image_size = 224
    resolution = 224
elif model_type == "resnest101":
    batch_size = 48
    infer_batch_size = 96
    image_size = 256
    resolution = 256
elif model_type == "resnest200":
    batch_size = 12
    infer_batch_size = 24
    image_size = 320
    resolution = 320
elif model_type == "resnest269":
    batch_size = 4
    infer_batch_size = 8
    image_size = 416
    resolution = 416

# Prediction Clipping Thresholds
prob_min = 0.001
prob_max = 0.999

# Swap Noise
swap_prob = 0.1
swap_portion = 0.15

label_smoothing = 0.001

# DeepInsight Transform
perplexity = 5

fc_size = 512

final_drop = 0.0
dropblock_prob = 0.0


# ### Dataset

# In[65]:


class MoAImageSwapDataset(torch.utils.data.Dataset):
    def __init__(self,
                 features,
                 labels,
                 transformer,
                 swap_prob=0.15,
                 swap_portion=0.1):
        self.features = features
        self.labels = labels
        self.transformer = transformer
        self.swap_prob = swap_prob
        self.swap_portion = swap_portion

        # self.crop = CropToFixedSize(width=image_size, height=image_size)

    def __getitem__(self, index):
        normalized = self.features[index, :]

        # Swap row featurs randomly
        normalized = self.add_swap_noise(index, normalized)
        normalized = np.expand_dims(normalized, axis=0)

        # Note: we are setting empty_value=0
        image = self.transformer.transform_3d(normalized, empty_value=0)[0]

        # Resize to target size
        image = cv2.resize(image.transpose((1, 2, 0)),
                           (image_size, image_size),
                           interpolation=cv2.INTER_CUBIC)
        image = image.transpose((2, 0, 1))

        return {"x": image, "y": self.labels[index, :]}

    def add_swap_noise(self, index, X):
        if np.random.rand() < self.swap_prob:
            swap_index = np.random.randint(self.features.shape[0], size=1)[0]
            # Select only gene expression and cell viability features
            swap_features = np.random.choice(
                np.array(range(3, self.features.shape[1])),
                size=int(self.features.shape[1] * self.swap_portion),
                replace=False)
            X[swap_features] = self.features[swap_index, swap_features]

        return X

    def __len__(self):
        return self.features.shape[0]


# In[66]:


class MoAImageDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, transformer):
        self.features = features
        self.labels = labels
        self.transformer = transformer

    def __getitem__(self, index):
        normalized = self.features[index, :]
        normalized = np.expand_dims(normalized, axis=0)

        # Note: we are setting empty_value=0
        image = self.transformer.transform_3d(normalized, empty_value=0)[0]

        # Resize to target size
        image = cv2.resize(image.transpose((1, 2, 0)),
                           (image_size, image_size),
                           interpolation=cv2.INTER_CUBIC)
        image = image.transpose((2, 0, 1))

        return {"x": image, "y": self.labels[index, :]}

    def __len__(self):
        return self.features.shape[0]


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, transformer):
        self.features = features
        self.labels = labels
        self.transformer = transformer

    def __getitem__(self, index):
        normalized = self.features[index, :]
        normalized = np.expand_dims(normalized, axis=0)

        # Note: we are setting empty_value=0
        image = self.transformer.transform_3d(normalized, empty_value=0)[0]

        # Resize to target size
        image = cv2.resize(image.transpose((1, 2, 0)),
                           (image_size, image_size),
                           interpolation=cv2.INTER_CUBIC)
        image = image.transpose((2, 0, 1))

        return {"x": image, "y": -1}

    def __len__(self):
        return self.features.shape[0]


# ### Model Definition

# In[67]:


def initialize_weights(layer):
    for m in layer.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            fan_out = m.weight.size(0)  # fan-out
            fan_in = 0
            init_range = 1.0 / math.sqrt(fan_in + fan_out)
            m.weight.data.uniform_(-init_range, init_range)
            m.bias.data.zero_()


# In[68]:


class MoAResNeSt(pl.LightningModule):
    def __init__(
            self,
            pretrained_model_name,
            training_set=(None, None),  # tuple
            valid_set=(None, None),  # tuple
            test_set=None,
            transformer=None,
            num_classes=206,
            final_drop=0.0,
            dropblock_prob=0,
            fc_size=512,
            learning_rate=1e-3):
        super(MoAResNeSt, self).__init__()

        self.train_data, self.train_labels = training_set
        self.valid_data, self.valid_labels = valid_set
        self.test_data = test_set
        self.transformer = transformer

        self.backbone = getattr(resnest.torch, pretrained_model)(
            pretrained=True,
            final_drop=final_drop,
            dropblock_prob=dropblock_prob)

        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, fc_size, bias=True),
            nn.ELU(), nn.Linear(fc_size, num_classes, bias=True))

        if self.training:
            initialize_weights(self.backbone.fc)

        # Save passed hyperparameters
        self.save_hyperparameters("pretrained_model_name", "num_classes",
                                  "final_drop", "dropblock_prob", "fc_size",
                                  "learning_rate")

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x = batch["x"]
        y = batch["y"]
        x = x.float()
        y = y.type_as(x)
        logits = self(x)

        # loss = F.binary_cross_entropy_with_logits(logits, y, reduction="mean")

        # Label smoothing
        loss = SmoothBCEwLogits(smoothing=label_smoothing)(logits, y)

        self.log('train_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["x"]
        y = batch["y"]
        x = x.float()
        y = y.type_as(x)
        logits = self(x)

        val_loss = F.binary_cross_entropy_with_logits(logits,
                                                      y,
                                                      reduction="mean")

        self.log('val_loss',
                 val_loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

        return val_loss

    def test_step(self, batch, batch_idx):
        x = batch["x"]
        y = batch["y"]
        x = x.float()
        y = y.type_as(x)
        logits = self(x)
        return {"pred_logits": logits}

    def test_epoch_end(self, output_results):
        all_outputs = torch.cat([out["pred_logits"] for out in output_results],
                                dim=0)
        print("Logits:", all_outputs)
        pred_probs = F.sigmoid(all_outputs).detach().cpu().numpy()
        print("Predictions: ", pred_probs)
        return {"pred_probs": pred_probs}

    def setup(self, stage=None):
        #         self.train_dataset = MoAImageDataset(self.train_data,
        #                                              self.train_labels,
        #                                              self.transformer)
        self.train_dataset = MoAImageSwapDataset(self.train_data,
                                                 self.train_labels,
                                                 self.transformer,
                                                 swap_prob=swap_prob,
                                                 swap_portion=swap_portion)

        self.val_dataset = MoAImageDataset(self.valid_data, self.valid_labels,
                                           self.transformer)

        self.test_dataset = TestDataset(self.test_data, None, self.transformer)

    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      pin_memory=True,
                                      drop_last=False)
        print(f"Train iterations: {len(train_dataloader)}")
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(self.val_dataset,
                                    batch_size=infer_batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=True,
                                    drop_last=False)
        print(f"Validate iterations: {len(val_dataloader)}")
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(self.test_dataset,
                                     batch_size=infer_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=False)
        print(f"Test iterations: {len(test_dataloader)}")
        return test_dataloader

    def configure_optimizers(self):
        print(f"Initial Learning Rate: {self.hparams.learning_rate:.6f}")
        #         optimizer = optim.Adam(self.parameters(),
        #                                lr=self.hparams.learning_rate,
        #                                weight_decay=weight_decay)
        #         optimizer = torch.optim.SGD(self.parameters(),
        #                                     lr=self.hparams.learning_rate,
        #                                     momentum=0.9,
        #                                     dampening=0,
        #                                     weight_decay=weight_decay,
        #                                     nesterov=False)

        optimizer = torch_optimizer.RAdam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay,
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=T_max,
                                                         eta_min=0,
                                                         last_epoch=-1)

        #         scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #             optimizer,
        #             T_0=T_0,
        #             T_mult=1,
        #             eta_min=0,
        #             last_epoch=-1)

        #         scheduler = optim.lr_scheduler.OneCycleLR(
        #             optimizer=optimizer,
        #             pct_start=0.1,
        #             div_factor=1e3,
        #             max_lr=1e-1,
        #             # max_lr=1e-2,
        #             epochs=epochs,
        #             steps_per_epoch=len(self.train_images) // batch_size)

        return [optimizer], [scheduler]


# In[69]:


def get_model(training_set, valid_set, test_set, transformer, model_path=None):
    if training_mode:
        model = MoAResNeSt(
            pretrained_model_name=pretrained_model,
            training_set=training_set,  # tuple
            valid_set=valid_set,  # tuple
            test_set=test_set,
            transformer=transformer,
            num_classes=len(train_classes),
            final_drop=final_drop,
            dropblock_prob=dropblock_prob,
            fc_size=fc_size,
            learning_rate=learning_rate)
    else:
        model = MoAResNeSt.load_from_checkpoint(
            model_path,
            pretrained_model_name=pretrained_model,
            training_set=training_set,  # tuple
            valid_set=valid_set,  # tuple
            test_set=test_set,
            transformer=transformer,
            num_classes=len(train_classes),
            fc_size=fc_size)
        model.freeze()
        model.eval()
    return model


# In[70]:


def norm2_normalization(train, valid, test):
    scaler = LogScaler()
    train = scaler.fit_transform(train)
    valid = scaler.transform(valid)
    test = scaler.transform(test)
    return train, valid, test, scaler


def quantile_transform(train, valid, test):
    q_scaler = QuantileTransformer(n_quantiles=1000,
                                   output_distribution='normal',
                                   ignore_implicit_zeros=False,
                                   subsample=100000,
                                   random_state=rand_seed)
    train = q_scaler.fit_transform(train)
    valid = q_scaler.transform(valid)
    test = q_scaler.transform(test)

    # Transform to [0, 1]
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    train = min_max_scaler.fit_transform(train)
    valid = min_max_scaler.transform(valid)
    test = min_max_scaler.transform(test)

    return train, valid, test, q_scaler, min_max_scaler


def extract_feature_map(train,
                        feature_extractor='tsne_exact',
                        resolution=100,
                        perplexity=30):
    transformer = DeepInsightTransformer(feature_extractor=feature_extractor,
                                         pixels=resolution,
                                         perplexity=perplexity,
                                         random_state=rand_seed,
                                         n_jobs=-1)
    transformer.fit(train)
    return transformer


# ### Inference

# In[71]:


# Ensure Reproducibility
seed_everything(rand_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

best_model = None
oof_predictions = np.zeros((train_features.shape[0], len(train_classes)))
resnest_kfold_submit_preds = np.zeros((test_features.shape[0], len(train_classes)))
for i, (train_index, val_index) in enumerate(
        skf.split(train_features, train_labels[y_labels])):
    if training_mode:
        print(f"Training on Fold {i} ......")
        print(train_index.shape, val_index.shape)

        logger = TensorBoardLogger(model_output_folder,
                                   name=f"fold{i}/logs",
                                   default_hp_metric=False)

        train = train_features.loc[train_index, all_features].copy().values
        fold_train_labels = train_labels.loc[train_index,
                                             train_classes].copy().values
        valid = train_features.loc[val_index, all_features].copy().values
        fold_valid_labels = train_labels.loc[val_index,
                                             train_classes].copy().values
        test = test_features[all_features].copy().values

        # LogScaler (Norm-2 Normalization)
        print("Running norm-2 normalization ......")
        train, valid, test, scaler = norm2_normalization(train, valid, test)
        save_pickle(scaler, model_output_folder, i, "log-scaler")

        # Extract DeepInsight Feature Map
        print("Extracting feature map ......")
        transformer = extract_feature_map(train,
                                          feature_extractor='tsne_exact',
                                          resolution=resolution,
                                          perplexity=perplexity)
        save_pickle(transformer, model_output_folder, i,
                    "deepinsight-transform")

        model = get_model(training_set=(train, fold_train_labels),
                          valid_set=(valid, fold_valid_labels),
                          test_set=test,
                          transformer=transformer)

        callbacks = [
            EarlyStopping(monitor='val_loss_epoch',
                          min_delta=1e-6,
                          patience=patience,
                          verbose=True,
                          mode='min',
                          strict=True),
            LearningRateMonitor(logging_interval='step')
        ]
        # https://pytorch-lightning.readthedocs.io/en/latest/generated/pytorch_lightning.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint
        checkpoint_callback = ModelCheckpoint(
            filepath=f"{model_output_folder}/fold{i}" +
            "/{epoch}-{train_loss_epoch:.6f}-{val_loss_epoch:.6f}" +
            f"-image_size={image_size}-resolution={resolution}-perplexity={perplexity}-fc={fc_size}",
            save_top_k=1,
            save_weights_only=False,
            save_last=False,
            verbose=True,
            monitor='val_loss_epoch',
            mode='min',
            prefix='')

        if debug_mode:
            # Find best LR
            # https://pytorch-lightning.readthedocs.io/en/latest/lr_finder.html
            trainer = Trainer(
                gpus=[gpus[0]],
                distributed_backend="dp",  # multiple-gpus, 1 machine
                auto_lr_find=True,
                benchmark=False,
                deterministic=True,
                logger=logger,
                accumulate_grad_batches=accumulate_grad_batches,
                gradient_clip_val=gradient_clip_val,
                precision=16,
                max_epochs=1)

            # Run learning rate finder
            lr_finder = trainer.tuner.lr_find(
                model,
                min_lr=1e-7,
                max_lr=1e2,
                num_training=100,
                mode='exponential',
                early_stop_threshold=100.0,
            )
            fig = lr_finder.plot(suggest=True)
            fig.show()

            # Pick point based on plot, or get suggestion
            suggested_lr = lr_finder.suggestion()

            # Update hparams of the model
            model.hparams.learning_rate = suggested_lr
            print(
                f"Suggested Learning Rate: {model.hparams.learning_rate:.6f}")

        else:
            trainer = Trainer(
                gpus=gpus,
                distributed_backend="dp",  # multiple-gpus, 1 machine
                max_epochs=epochs,
                benchmark=False,
                deterministic=True,
                # fast_dev_run=True,
                checkpoint_callback=checkpoint_callback,
                callbacks=callbacks,
                accumulate_grad_batches=accumulate_grad_batches,
                gradient_clip_val=gradient_clip_val,
                precision=16,
                logger=logger)
            trainer.fit(model)

            # Load best model
            seed_everything(rand_seed)
            best_model = MoAResNeSt.load_from_checkpoint(
                checkpoint_callback.best_model_path,
                pretrained_model_name=pretrained_model,
                training_set=(train, fold_train_labels),  # tuple
                valid_Set=(valid, fold_valid_labels),  # tuple
                test_set=test,
                transformer=transformer,
                fc_size=fc_size)
            best_model.freeze()

            print("Predicting on validation set ......")
            output = trainer.test(ckpt_path="best",
                                  test_dataloaders=model.val_dataloader(),
                                  verbose=False)[0]
            fold_preds = output["pred_probs"]
            oof_predictions[val_index, :] = fold_preds

            print(fold_preds[:5, :])
            fold_valid_loss = mean_logloss(fold_preds, fold_valid_labels)
            print(f"Fold {i} Validation Loss: {fold_valid_loss:.6f}")

            # Generate submission predictions
            print("Predicting on test set ......")
            best_model.setup()
            output = trainer.test(best_model, verbose=False)[0]
            submit_preds = output["pred_probs"]
            print(test_features.shape, submit_preds.shape)

            resnest_kfold_submit_preds += submit_preds / kfolds

        del model, trainer, train, valid, test, scaler, transformer
    else:
        print(f"Inferencing on Fold {i} ......")
        print(train_index.shape, val_index.shape)

        model_path = glob.glob(
            f'{model_list[experiment_name]["model_path"]}/{experiment_name}/fold{i}/epoch*.ckpt'
        )[0]

        test = test_features[all_features].copy().values

        # Load LogScaler (Norm-2 Normalization)
        scaler = load_pickle(
            f'{model_list[experiment_name]["model_path"]}/{experiment_name}',
            i, "log-scaler")
        test = scaler.transform(test)

        # Load DeepInsight Feature Map
        transformer = load_pickle(
            f'{model_list[experiment_name]["model_path"]}/{experiment_name}',
            i, "deepinsight-transform")

        print(f"Loading model from {model_path}")
        model = get_model(training_set=(None, None),
                          valid_set=(None, None),
                          test_set=test,
                          transformer=transformer,
                          model_path=model_path)

        trainer = Trainer(
            logger=False,
            gpus=gpus,
            distributed_backend="dp",  # multiple-gpus, 1 machine
            precision=16,
            benchmark=False,
            deterministic=True)
        output = trainer.test(model, verbose=False)[0]
        submit_preds = output["pred_probs"]
        resnest_kfold_submit_preds += submit_preds / kfolds

        del model, trainer, scaler, transformer, test

    torch.cuda.empty_cache()
    gc.collect()

    if debug_mode:
        break


# In[72]:


print(resnest_kfold_submit_preds.shape)
resnest_kfold_submit_preds


# ## Blend Models

# In[73]:


torch.cuda.empty_cache()
gc.collect()


# In[74]:


def mean_logloss(y_pred, y_true):
    logloss = (1 - y_true) * np.log(1 - y_pred +
                                    1e-15) + y_true * np.log(y_pred + 1e-15)
    return np.mean(-logloss)


# In[ ]:





# In[75]:


baseline_tabnet_oof_preds = glob.glob(
    f'{model_list["tabnet_baseline"]["model_path"]}/oof_*.npy')[0]
baseline_tabnet_oof_preds = np.load(baseline_tabnet_oof_preds)

twheads_oof_preds = glob.glob(
    f'{model_list["2heads_deep_resnets_v1"]["model_path"]}/oof_*.npy')[0]
twheads_oof_preds = np.load(twheads_oof_preds)

# guassrank_v1_oof_preds = glob.glob(
#     f'{model_list["rankgauss_pca_nn_v1"]["model_path"]}/oof_*.npy')[0]
# guassrank_v1_oof_preds = np.load(guassrank_v1_oof_preds)

effnet_v7_b3_oof_preds = glob.glob(
    f'{model_list["deepinsight_efficientnet_v7_b3"]["model_path"]}/oof_*.npy'
)[0]
effnet_v7_b3_oof_preds = np.load(effnet_v7_b3_oof_preds)

effnet_seed2_v7_b3_oof_preds = glob.glob(
    f'{model_list["deepinsight_efficientnet_v7_b3_seed2"]["model_path"]}/oof_*.npy'
)[0]
effnet_seed2_v7_b3_oof_preds = np.load(effnet_seed2_v7_b3_oof_preds)
    
resnest_v1_oof_preds = glob.glob(
    f'{model_list["deepinsight_ResNeSt_v1_resnest50"]["model_path"]}/oof_*.npy'
)[0]
resnest_v1_oof_preds = np.load(resnest_v1_oof_preds)


# In[76]:


oof_loss = mean_logloss(baseline_tabnet_oof_preds, train_labels[train_classes].values)
print(f"OOF Validation Loss of baseline_tabnet_oof_preds: {oof_loss:.6f}")


# In[77]:


oof_loss = mean_logloss(twheads_oof_preds, train_labels[train_classes].values)
print(f"OOF Validation Loss of twheads_oof_preds: {oof_loss:.6f}")


# In[78]:


# oof_loss = mean_logloss(guassrank_v1_oof_preds, train_labels[train_classes].values)
# print(f"OOF Validation Loss of rankgauss_pca_nn_v1: {oof_loss:.6f}")
# OOF Validation Loss of rankgauss_pca_nn_v1: 0.015130


# In[79]:


oof_loss = mean_logloss(effnet_v7_b3_oof_preds, train_labels[train_classes].values)
print(f"OOF Validation Loss of deepinsight_efficientnet_v7_b3: {oof_loss:.6f}")


# In[80]:


oof_loss = mean_logloss(effnet_seed2_v7_b3_oof_preds, train_labels[train_classes].values)
print(f"OOF Validation Loss of deepinsight_efficientnet_v7_b3_seed2: {oof_loss:.6f}")


# In[81]:


oof_loss = mean_logloss(effnet_seed2_v7_b3_oof_preds, train_labels[train_classes].values)
print(f"OOF Validation Loss of deepinsight_efficientnet_v7_b3_seed2: {oof_loss:.6f}")


# In[82]:


oof_loss = mean_logloss(resnest_v1_oof_preds, train_labels[train_classes].values)
print(f"OOF Validation Loss of resnest_v1_oof_preds: {oof_loss:.6f}")


# ### Search Best Blend Weights by Optuna

# In[83]:


blend_search = False
# blend_search = True
n_trials = 3000


# In[84]:


torch.cuda.empty_cache()
gc.collect()


# In[85]:


def objective(trial):
    w1 = trial.suggest_float("w1", 0, 1.0)
    w2 = trial.suggest_float("w2", 0, 1.0)
    w3 = trial.suggest_float("w3", 0, 1.0)
    w4 = trial.suggest_float("w4", 0, 1.0)
    w5 = trial.suggest_float("w5", 0, 1.0)
    blend = w1 * baseline_tabnet_oof_preds +         w2 * twheads_oof_preds +         w3 * effnet_v7_b3_oof_preds +         w4 * effnet_seed2_v7_b3_oof_preds +         w5 * resnest_v1_oof_preds
    blend = np.clip(blend, 0, 1.0)

    loss = mean_logloss(blend, train_labels[train_classes].values)
    return loss


# In[86]:


if blend_search:
    study_name = "moa_blend_effnetb3_resnestv1"
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=0,
        interval_steps=1,
    )
    sampler = optuna.samplers.TPESampler(seed=rand_seed)
    study = optuna.create_study(direction="minimize",
                                pruner=pruner,
                                sampler=sampler,
                                study_name=study_name,
                                storage=f'sqlite:///{study_name}.db',
                                load_if_exists=True)

    study.optimize(objective,
                   n_trials=n_trials,
                   timeout=None,
                   gc_after_trial=True,
                   n_jobs=-1)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


# In[87]:


# With ResNeSt V1
# Number of finished trials: 3000
# Best trial:
#   Value: 0.014093936959273327
#   Params: 
#     w1: 0.05961977033877462
#     w2: 0.20936627033493893
#     w3: 0.21618640167696873
#     w4: 0.19696602747953756
#     w5: 0.31871835363380363


# In[88]:


# With EfficientNet-B3 NS seed=419
# Number of finished trials: 3000
# Best trial:
#   Value: 0.014199481939912025
#   Params: 
#     w1: 0.14970493278685254
#     w2: 0.27499801392456136
#     w3: 0.00022822436409621776
#     w4: 0.31379453587676504
#     w5: 0.26238691389652163


# In[89]:


# With EfficientNet-B3 NS seed=1120
# Number of finished trials: 3000
# Best trial:
#   Value: 0.014484323717445898
#   Params: 
#     w1: 0.18954821976796626
#     w2: 0.5136980542036813
#     w3: 0.00023326930556589673
#     w4: 0.29925906518060896


# In[90]:


w1 = 0.05961977033877462
w2 = 0.20936627033493893
w3 = 0.21618640167696873
w4 = 0.19696602747953756
w5 = 0.31871835363380363


# In[91]:


best_oof_blend = w1 * baseline_tabnet_oof_preds +     w2 * twheads_oof_preds +     w3 * effnet_v7_b3_oof_preds +     w4 * effnet_seed2_v7_b3_oof_preds +     w5 * resnest_v1_oof_preds
best_oof_blend = np.clip(best_oof_blend, 0, 1.0)
mean_logloss(best_oof_blend, train_labels[train_classes].values)


# In[92]:


oof_even_blend = (baseline_tabnet_oof_preds +                   twheads_oof_preds +                   effnet_v7_b3_oof_preds +                   effnet_seed2_v7_b3_oof_preds +                   resnest_v1_oof_preds) / 5
oof_even_blend = np.clip(oof_even_blend, 0, 1.0)
# oof_even_blend = np.clip(oof_even_blend, 0.0001, 0.9999)
mean_logloss(oof_even_blend, train_labels[train_classes].values)


# In[ ]:





# In[93]:


baseline_tabnet_kfold_submit_preds.shape,     twoheads_kfold_submit_preds.shape,     effnetb3_kfold_submit_preds.shape,     effnetb3_seed2_kfold_submit_preds.shape,     resnest_kfold_submit_preds.shape


# In[94]:


# final_kfold_submit_preds = w1 * baseline_tabnet_kfold_submit_preds + \
#     w2 * twoheads_kfold_submit_preds + \
#     w3 * effnetb3_kfold_submit_preds + \
#     w4 * effnetb3_seed2_kfold_submit_preds + \
#     w5 * resnest_kfold_submit_preds
final_kfold_submit_preds = (baseline_tabnet_kfold_submit_preds +                             twoheads_kfold_submit_preds +                             effnetb3_kfold_submit_preds +                             effnetb3_seed2_kfold_submit_preds +                             resnest_kfold_submit_preds) / 5

final_kfold_submit_preds = np.clip(final_kfold_submit_preds, 0, 1.0)


# In[95]:


final_kfold_submit_preds


# ## Final Submission

# In[96]:


submission = pd.DataFrame(data=test_features["sig_id"].values,
                          columns=["sig_id"])
submission = submission.reindex(columns=["sig_id"] + train_classes)
submission[train_classes] = final_kfold_submit_preds
# Set control type to 0 as control perturbations have no MoAs
submission.loc[test_features['cp_type'] == 0, submission.columns[1:]] = 0
submission.to_csv('submission_supervised_tabnet_v2_seeds_0.01857.csv', index=False)


# In[97]:


submission


# ## EOF

# In[ ]:




