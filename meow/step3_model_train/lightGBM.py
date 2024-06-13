import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# 定义 RMSPE 评估指标
def rmspe(y_true, y_pred):
    """计算均方根百分比误差（RMSPE）"""
    return np.sqrt(np.mean(np.square((y_true - y_pred)/y_true)))

def feval_rmspe(y_pred, lgb_train):
    """自定义 RMSPE 评估函数，返回给 LightGBM 使用"""
    y_true = lgb_train.get_label()
    return 'RMSPE', rmspe(y_true, y_pred), False

# 读取数据
result = pd.read_csv('data/train.csv')

# 过滤出特定 stock_id 的数据
stock_id = 0  # 示例 stock_id，可以根据需要调整
selected_data = result[result.stock_id == stock_id].reset_index(drop=True)

# 提取目标值（y）和特征（X）
selected_y = selected_data['target']
train_features = selected_data.drop(['time_id', 'target', 'stock_id'], axis=1)

# 使用 5 折交叉验证
kf = KFold(n_splits=5, random_state=2021, shuffle=True)

for train_index, test_index in kf.split(train_features):
    # 分割训练集和测试集
    X_train, X_test = train_features.iloc[train_index], train_features.iloc[test_index]
    y_train, y_test = selected_y.iloc[train_index], selected_y.iloc[test_index]
    
    # 创建 LightGBM 数据集
    train_dataset = lgb.Dataset(X_train, y_train, weight=1/np.square(y_train))
    validation_dataset = lgb.Dataset(X_test, y_test, weight=1/np.square(y_test))
    
    # LightGBM 参数设置
    params = {
        'objective': 'regression',  # 回归任务
        'metric': 'rmse',  # 评估指标
        'learning_rate': 0.05,  # 学习率
        'num_leaves': 31,  # 叶子节点数
        'feature_fraction': 0.9,  # 每次迭代中随机选择的特征比例
        'bagging_fraction': 0.8,  # 每次迭代中随机选择的数据比例
        'bagging_freq': 5,  # 执行bagging的频率
        'verbose': -1  # 训练过程不输出
    }
    
    # 训练 LightGBM 模型
    model = lgb.train(params, 
                      train_set=train_dataset, 
                      valid_sets=[train_dataset, validation_dataset],  # 验证集用于早停
                      early_stopping_rounds=50,  # 早停设置
                      feval=feval_rmspe,  # 自定义评估函数
                      verbose_eval=20,  # 每 20 轮输出一次日志
                      num_boost_round=1000)  # 最大迭代次数
    
    # 预测
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    # 打印 RMSPE 结果
    print("RMSPE = ", rmspe(y_test, y_pred))
    
    # 绘制特征重要性图
    lgb.plot_importance(model, max_num_features=20)
    plt.show()
