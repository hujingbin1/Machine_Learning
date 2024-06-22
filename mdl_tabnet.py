import os
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from log import log
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def rmspe(y_true, y_pred):
    # Function to calculate the root mean squared percentage error
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

class RMSPE(Metric):
    def __init__(self):
        self._name = "rmspe"
        self._maximize = False

    def __call__(self, y_true, y_score):
        return np.sqrt(np.mean(np.square((y_true - y_score) / y_true)))

def RMSPELoss(y_pred, y_true):
    return torch.sqrt(torch.mean(((y_true - y_pred) / y_true) ** 2)).clone()

#tabnet网络
class MeowModel(object):
    def __init__(self, cacheDir,input_dim=72, hidden_dim=128, n_layers=3, out_dim=1, learning_rate=0.001, batchsize = 2048, n_epochs=100, dropout=0):
        self.model = None
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.out_dim = out_dim
        self.batch_size = batchsize
        self.learning_rate = learning_rate
        self.max_epochs = n_epochs
        self.best_loss = 100000

    def fit(self, X, y):
        nunique = X.nunique()
        types = X.dtypes

        categorical_columns = []
        categorical_dims =  {}

        for col in X.columns:
            if col == 'symbol':
                l_enc = LabelEncoder()
                X[col] = l_enc.fit_transform(X[col].values)
                categorical_columns.append(col)
                categorical_dims[col] = len(l_enc.classes_)
            else:
                scaler = StandardScaler()
                X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1))

        cat_idxs = [ i for i, f in enumerate(X.columns.tolist()) if f in categorical_columns]
        cat_dims = [ categorical_dims[f] for i, f in enumerate(X.columns.tolist()) if f in categorical_columns]


        tabnet_params = dict(
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=1,
            n_d=16,
            n_a=16,
            n_steps=2,
            gamma=2,
            n_independent=2,
            n_shared=2,
            lambda_sparse=0,
            optimizer_fn=Adam,
            optimizer_params=dict(lr=(2e-2)),
            mask_type="entmax",
            scheduler_params=dict(T_0=200, T_mult=1, eta_min=1e-4, last_epoch=-1, verbose=False),
            scheduler_fn=CosineAnnealingWarmRestarts,
            seed=42,
            verbose=10
        )
        self.model = TabNetRegressor(**tabnet_params)

        kfold = KFold(n_splits=5, random_state=42, shuffle=True)

        predictions = np.zeros((X.shape[0], 1))

        for fold, (trn_ind, val_ind) in enumerate(kfold.split(X)):
            print(f'Training fold {fold + 1}')
            X_train, X_val = X.iloc[trn_ind].values, X.iloc[val_ind].values
            y_train, y_val = y.iloc[trn_ind].values.reshape(-1, 1), y.iloc[val_ind].values.reshape(-1, 1)

            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                max_epochs=200,
                patience=50,
                batch_size=1024 * 20,
                virtual_batch_size=128 * 20,
                num_workers=4,
                drop_last=False,
                eval_metric=[RMSPE],
                loss_fn=RMSPELoss
            )

            saving_path_name = f"./fold{fold}"
            saved_filepath = self.model.save_model(saving_path_name)

            predictions[val_ind] = self.model.predict(X_val)

        print(f'OOF score across folds: {rmspe(y, predictions.flatten())}')


    def predict(self, X):
        preds = self.model.predict(X)

