# import os
# import torch
# import torch.nn as nn
# from log import log
# import torch.nn.functional as F  # 导入激活函数模块

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# class MLPNet(nn.Module):
#     def __init__(self, input_dim, hidden_dim, n_layers, out_dim, dropout):
#         super(MLPNet, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.n_layers = n_layers
#         # 确保LSTM接受batch_first=True，对输入数据格式要求是(batch_size, seq_len, input_size)
#         self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
#         self.dropout = nn.Dropout(p=dropout)  # 可选：在LSTM之后再加一层Dropout
#         self.fc = nn.Linear(hidden_dim, out_dim)
#     def forward(self, _x):
#         x, _ = self.lstm(_x)  # _x is input, size (batch, seq_len, input_size) 1024*1*6
#         # print(x.shape)
#         b, s, h = x.shape  # x is output, size (batch, seq_len, hidden_size)
#         x = x.view(s*b, h) # 1024*20
#         x = self.dropout(x)  # 应用Dropout
#         x = self.fc(x)  #1024*1
#         x = x.view(s, b, -1)  
#         return x