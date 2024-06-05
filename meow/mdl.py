import os
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
# class MeowModel(object):
#     def __init__(self, cacheDir):
#         self.estimator = Ridge(
#             alpha=0.5,
#             random_state=None,
#             fit_intercept=False,
#             tol=1e-8
#         )

#     def fit(self, xdf, ydf):
#         self.estimator.fit(
#             X=xdf.to_numpy(),
#             y=ydf.to_numpy(),
#         )
#         xdf_tensor = torch.tensor(xdf.head().to_numpy())
#         ydf_tensor = torch.tensor(ydf.head().to_numpy())
#         print(xdf_tensor)
#         print(ydf_tensor)
#         log.inf("Done fitting")

#     def predict(self, xdf):
#         return self.estimator.predict(xdf.to_numpy())



class MeowModel(object):
    def __init__(self, cacheDir, input_dim=6, hidden_dim=20, n_layers=2, out_dim=1, learning_rate=0.01, batchsize = 10240, n_epochs=100):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.out_dim = out_dim
        self.batch_size = batchsize
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        # 显式地将模型置于目标设备上
        self.model = lstm.LSTMNet(input_dim, hidden_dim, n_layers, out_dim).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        self.best_loss = 100000
        
    def fit(self, xdf, ydf):

        dataset = TensorDataset(torch.tensor(xdf.to_numpy(), dtype=torch.float32).to(device),
                            torch.tensor(ydf.to_numpy(), dtype=torch.float32).to(device))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=True)
        
        cnt = 0
        for epoch in range(self.n_epochs):
            for inputs, targets in dataloader:
                cnt = cnt + 1
                outputs = self.model(inputs.view(self.batch_size, xdf.shape[1])).to(device)
                loss = self.criterion(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print(cnt)
             # 每个epoch结束时检查是否需要保存模型
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_model_weights = self.model.state_dict()  # 保存当前最佳模型的权重
                torch.save(self.model.state_dict(), 'best_model.pth')
                log.inf(f'Epoch {epoch}: Loss improved. Saving best model so far.')
            cnt = 0
            log.inf(f'Epoch {epoch}: Loss = {loss.item()}')

        # 训练循环结束后，输出完成信息，并加载最佳模型参数
        log.inf(f'Done fitting after {self.n_epochs} epochs. Loading the best model.')
        self.model.load_state_dict(self.best_model_weights)
        
    def predict(self, X):
	# 加载保存的最优模型权重
        self.model.load_state_dict(torch.load('best_model.pth'))
		
        with torch.no_grad():
			# 确保预测数据也在目标设备上
            X_tensor = X.clone().detach().requires_grad_(True).to(torch.float32)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy()