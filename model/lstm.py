import os
import torch
import torch.nn as nn
from log import log
import torch.nn.functional as F 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#定义LSTM网络
class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, out_dim, dropout):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_dim, out_dim)
    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (batch, seq_len, input_size)
        b, s, h = x.shape  # x is output, size (batch, seq_len, hidden_size)
        x = x.view(s*b, h)
        x = self.dropout(x)
        x = self.fc(x)
        x = x.view(s, b, -1)
        return x
