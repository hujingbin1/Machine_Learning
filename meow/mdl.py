import os
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from log import log
from step3_model_train import lstm
import numpy as np
# 确定设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
class MeowModel(object):
    def __init__(self, cacheDir):
        self.estimator = Ridge(
            alpha=0.1,
            random_state=None,
            fit_intercept=False,
            tol=1e-8
        )

    def fit(self, xdf, ydf):
        self.estimator.fit(
            X=xdf.to_numpy(),
            y=ydf.to_numpy(),
        )
        # xdf_tensor = torch.tensor(xdf.head().to_numpy())
        # ydf_tensor = torch.tensor(ydf.head().to_numpy())
        # print(xdf_tensor)
        # print(ydf_tensor)
        log.inf("Done fitting")

    def predict(self, xdf):
        return self.estimator.predict(xdf.to_numpy())
