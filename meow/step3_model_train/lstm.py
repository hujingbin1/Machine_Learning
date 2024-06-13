import os
import torch
import torch.nn as nn
from log import log
import torch.nn.functional as F  # 导入激活函数模块

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, out_dim, dropout):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        # 确保LSTM接受batch_first=True，对输入数据格式要求是(batch_size, seq_len, input_size)
        self.lstm = nn.LSTM(input_dim, hidden_d im, n_layers, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)  # 可选：在LSTM之后再加一层Dropout
        self.fc = nn.Linear(hidden_dim, out_dim)
    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (batch, seq_len, input_size) 1024*1*6
        # print(x.shape)
        b, s, h = x.shape  # x is output, size (batch, seq_len, hidden_size)
        x = x.view(s*b, h) # 1024*20
        x = self.dropout(x)  # 应用Dropout
        x = self.fc(x)  #1024*1
        # x = F.sigmoid(x)  # 添加ReLU激活函数  # 修改这里
        x = x.view(s, b, -1)
        return x

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class LSTMNet(nn.Module):
#     def __init__(self, input_dim, hidden_dim1, hidden_dim2, fc_dim, out_dim, dropout):
#         super(LSTMNet, self).__init__()
#         self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
#         self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
#         self.fc1 = nn.Linear(hidden_dim2, fc_dim)
#         self.fc2 = nn.Linear(fc_dim, out_dim)
#         self.dropout = nn.Dropout(p=dropout)

#     def forward(self, x):
#         x, _ = self.lstm1(x)
#         x, _ = self.lstm2(x)
#         x = self.dropout(x)
#         x = self.fc1(x)
#         x = torch.relu(x)
#         x = self.fc2(x)
#         return x