import os
from sklearn.linear_model import Ridge
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
class MeowModel(object):
    def __init__(self, cacheDir,input_dim=78, hidden_dim=256, n_layers=3, out_dim=1, learning_rate=0.001, batchsize = 64, n_epochs=50, dropout=0):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.out_dim = out_dim
        self.batch_size = batchsize
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        # 显式地将模型置于目标设备上
        self.model = lstm.LSTMNet(input_dim, hidden_dim, n_layers, out_dim, dropout).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.best_loss = 100000
        
    def fit(self, xdf, ydf):

        dataset = TensorDataset(torch.tensor(xdf.to_numpy(), dtype=torch.float32).to(device),
                            torch.tensor(ydf.to_numpy(), dtype=torch.float32).to(device))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=True)
        
        torch.autograd.set_detect_anomaly(True)
        cnt = 0
        loss_list = []
        for epoch in range(self.n_epochs):
            for inputs, targets in dataloader:
                cnt += 1
                # with open("./inputs.txt", 'w') as f:
                #     f.write("input\n")
                #     f.write(str(inputs))
                #     f.write('\n')
                #     f.write("targets\n")
                #     f.write(str(targets))
                #     f.write('\n')
                # 将inputs和targets移动到device上
                inputs = inputs.unsqueeze(1).to(device)# 增加序列长度维度，shape 变为 (batch_size, 1, 6)
                targets = targets.to(device)
                outputs = self.model(inputs).squeeze(dim=1)
                loss = self.criterion(outputs, targets)
                print(loss)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                print(cnt)
                loss_list.append(loss)
			# 每个epoch结束时检查是否需要保存模型
            if (sum(loss_list)/len(loss_list)) < self.best_loss:
                self.best_loss = sum(loss_list)/len(loss_list)
                self.best_model_weights = self.model.state_dict()  # 保存当前最佳模型的权重
                torch.save(self.model.state_dict(), 'best_model.pth')
                log.inf(f'Epoch {epoch}: Loss improved. Saving best model so far.')
            cnt = 0
            log.inf(f'Epoch {epoch}: Loss = {sum(loss_list)/len(loss_list)}')

		# 训练循环结束后，输出完成信息，并加载最佳模型参数
        log.inf(f'Done fitting after {self.n_epochs} epochs. Loading the best model.')
        self.model.load_state_dict(self.best_model_weights)

    def predict(self, X):
		# 加载保存的最优模型权重
        batch_size=self.batch_size
        self.model.load_state_dict(torch.load('best_model.pth'))
        self.model.eval()  # 确保模型处于评估模式，这会影响某些层如Dropout或BatchNorm的行为

		# 确保预测数据也在目标设备上，同时转换为float32
        X_tensor = X.clone().detach().to(torch.float32).to(device)
        print(X_tensor.shape)
		# 初始化预测结果列表
        all_predictions = []

		# 计算需要多少个批次
        num_batches = (len(X_tensor) + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in range(num_batches):
		        # 获取当前批次的数据
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X_tensor))
                batch_data = X_tensor[start_idx:end_idx]

		        # 通过模型进行预测
                batch_predictions = self.model(batch_data)

		        # 将预测结果从当前设备转移到CPU，并转换为numpy数组
                batch_predictions = batch_predictions.cpu().numpy()
        
                # 将当前批次的预测结果添加到总的预测结果列表中
                all_predictions.extend(batch_predictions)

		# 将所有批次的预测结果合并为一个numpy数组
        all_predictions = np.concatenate(all_predictions, axis=0)

        # print(all_predictions.shape)
        return all_predictions