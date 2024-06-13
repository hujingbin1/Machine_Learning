import os
import torch
import torch.nn as nn
from log import log
import torch.nn.functional as F  # 导入激活函数模块

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MLPNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_layers, out_dim, dropout):
        super(MLPNet, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(p=dropout))
        
        # Hidden layers
        for i in range(1, n_layers):
            self.layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=dropout))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dims[-1], out_dim))
        
    def forward(self, _x):
        x = _x.view(_x.size(0), -1)  # Flatten the input
        for layer in self.layers:
            x = layer(x)
        return x