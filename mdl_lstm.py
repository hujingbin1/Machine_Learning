import os
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from log import log
from model import lstm
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

#LSTM网络
class MeowModel(object):
    def __init__(self, cacheDir, input_dim=72, hidden_dim=128, n_layers=3, out_dim=1, learning_rate=0.001,
                 batchsize=2048, n_epochs=100, dropout=0):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.out_dim = out_dim
        self.batch_size = batchsize
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.model = lstm.LSTMNet(input_dim, hidden_dim, n_layers, out_dim, dropout).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.best_loss = 100000

    def fit(self, xdf, ydf):

        dataset = TensorDataset(torch.tensor(xdf.to_numpy(), dtype=torch.float32).to(device),
                                torch.tensor(ydf.to_numpy(), dtype=torch.float32).to(device))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=True)
        torch.autograd.set_detect_anomaly(True)
        
        loss_list = []
        for epoch in range(self.n_epochs):
            for inputs, targets in dataloader:
                inputs = inputs.unsqueeze(1).to(device)  # 增加序列长度维度，shape 变为 (batch_size, 1, 6)
                targets = targets.to(device)
                outputs = self.model(inputs).squeeze(dim=1)
                loss = self.criterion(outputs, targets)
                print(loss)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                loss_list.append(loss)
                
            # 每个epoch结束时检查是否需要保存模型
            if (sum(loss_list) / len(loss_list)) < self.best_loss:
                self.best_loss = sum(loss_list) / len(loss_list)
                self.best_model_weights = self.model.state_dict()  # 保存当前最佳模型的权重
                torch.save(self.model.state_dict(), 'best_model.pth')
                log.inf(f'Epoch {epoch}: Loss improved. Saving best model so far.')
            log.inf(f'Epoch {epoch}: Loss = {loss.item()}')

        # 训练循环结束后，输出完成信息，并加载最佳模型参数
        log.inf(f'Done fitting after {self.n_epochs} epochs. Loading the best model.')
        self.model.load_state_dict(self.best_model_weights)

    def predict(self, X):

        batch_size = self.batch_size
        self.model.load_state_dict(torch.load('best_model.pth'))
        self.model.eval()

        X_tensor = X.clone().detach().to(torch.float32).to(device)        
        all_predictions = []
        num_batches = (len(X_tensor) + batch_size - 1) // batch_size
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X_tensor))
                batch_data = X_tensor[start_idx:end_idx]
                batch_predictions = self.model(batch_data)
                batch_predictions = batch_predictions.cpu().numpy()
                all_predictions.extend(batch_predictions)

        all_predictions = np.concatenate(all_predictions, axis=0)
        return all_predictions
