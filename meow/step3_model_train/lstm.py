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
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)  # 可选：在LSTM之后再加一层Dropout
        self.fc = nn.Linear(hidden_dim, out_dim)
    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (batch, seq_len, input_size) 1024*1*6【seq_len应该是窗口的大小】
        # print(x.shape)
        b, s, h = x.shape  # x is output, size (batch, seq_len, hidden_size)
        x = x.view(s*b, h) # 1024*20
        x = self.dropout(x)  # 应用Dropout
        x = self.fc(x)  #1024*1
        # x = F.sigmoid(x)  # 添加ReLU激活函数  # 修改这里
        x = x.view(s, b, -1)
        return x

# import os
# import torch
# import torch.nn as nn
# from log import log
# import torch.nn.functional as F  # 导入激活函数模块

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class AttentionLayer(nn.Module):
#     def __init__(self, hidden_dim):
#         super(AttentionLayer, self).__init__()
#         self.weight = nn.Parameter(torch.Tensor(hidden_dim, 1))
#         nn.init.xavier_uniform_(self.weight.data)
    
#     def forward(self, lstm_output):
#         # lstm_output shape: (batch_size, seq_len, hidden_dim)
#         weights = torch.bmm(lstm_output, self.weight.unsqueeze(2)).squeeze(2)  # (batch_size, seq_len)
#         alpha = F.softmax(weights, dim=1).unsqueeze(2)  # (batch_size, seq_len, 1)
#         context = torch.bmm(alpha, lstm_output).squeeze(2)  # (batch_size, hidden_dim)
#         return context

# class LSTMNet(nn.Module):
#     def __init__(self, input_dim, hidden_dim, n_layers, out_dim, dropout):
#         super(LSTMNet, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.n_layers = n_layers
#         self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
#         self.dropout = nn.Dropout(p=dropout)
#         self.attention = AttentionLayer(hidden_dim)  # 新增注意力层
#         self.fc = nn.Linear(hidden_dim, out_dim)
    
#     def forward(self, _x):
#         # LSTM前向传播
#         lstm_out, _ = self.lstm(_x)  # lstm_out shape: (batch_size, seq_len, hidden_dim)
        
#         # 应用注意力机制
#         context = self.attention(lstm_out)  # context shape: (batch_size, hidden_dim)
        
#         # Dropout
#         x = self.dropout(context)
        
#         # 全连接层
#         x = self.fc(x)  # x shape: (batch_size, out_dim)
        
#         return x

# # 注意：这里未对x进行view操作，因为我们直接使用注意力层输出作为全连接层的输入