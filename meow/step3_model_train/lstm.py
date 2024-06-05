import os
import torch
import torch.nn as nn
from log import log

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, out_dim):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        # 确保LSTM接受batch_first=True，对输入数据格式要求是(batch_size, seq_len, input_size)
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        
        # # 初始化隐藏状态和细胞状态
        # h0 = torch.zeros(self.n_layers, self.hidden_dim).to(device)
        # c0 = torch.zeros(self.n_layers, self.hidden_dim).to(device)
        
        # # 通过LSTM并取最后一个时间步的输出
        # out, _ = self.lstm(x, (h0, c0))# 使用初始化的h0和c0
        # out = out[:, -1, :]  # 取序列的最后一个时间步输出用于预测
        # out = self.fc(out)
        # return out
        out, _ = self.lstm(x) # LSTM层
        out = self.fc(out) # 全连接层
        return out