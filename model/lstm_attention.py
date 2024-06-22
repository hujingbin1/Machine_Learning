import os
import torch
import torch.nn as nn
from log import log
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#定义LSTM+Atention网络
class LSTMAttentionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, out_dim, dropout):
        super(LSTMAttentionNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, out_dim)
    def forward(self, _x):
        x, _ = self.lstm(_x)
        b, s, h = x.shape
        x = x.view(s*b, h)
        attention_weights = torch.softmax(self.attention(x).squeeze(-1), dim=-1)
        context_vector = torch.sum(x* attention_weights.unsqueeze(-1), dim=1)
        x = self.fc(context_vector)
        x = self.dropout(x)
        x = x.view(s, b, -1)
        return x

