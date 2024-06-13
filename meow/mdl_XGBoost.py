'''
'n_estimators': 树的数量,即模型使用的弱学习器(decision tree)的数量。一般来说,这个值越大,模型的性能会更好,但同时也会增加训练时间。
'booster': 使用的boosting算法, 在这里设置为 'gbtree'，表示使用基于树的boosting方法。
'objective': 优化目标函数。在这里设置为 'reg:squarederror'，表示使用平方误差回归。
'max_depth': 每个decision tree的最大深度。这个值控制了模型的复杂度,较大的值可能会导致过拟合。
'lambda': L2正则化系数。用于防止模型过拟合。
'subsample': 每棵树使用的数据占全部训练集的比例。小于1.0可以起到随机采样的作用,增加泛化能力。
'colsample_bytree': 每棵树随机选择的特征占全部特征的比例。这个可以增加模型的多样性。
'min_child_weight': 每个叶子节点最小的样本权重和。防止过拟合。
'eta': 学习率。控制每棵树更新量的大小。值越小训练越慢,但可能会更稳定。
'seed': 随机种子,确保结果的可复现性。
'nthread': 使用的CPU线程数。
'''
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

class MeowModel(object):
    def __init__(self, cacheDir):
        # self.cacheDir = './'
        self.model_path = os.path.join('./', "xgboost_model.json")
        self.model = None

    def fit(self, xdf, ydf):
        X = xdf.to_numpy()
        y = ydf.to_numpy()
        params = {
            'n_estimators': 100,
            'booster': 'gbtree',
            'objective': 'reg:squarederror',
            'max_depth': 7,
            'lambda': 3,
            'subsample': 0.9,
            'colsample_bytree': 1,
            'min_child_weight': 3,
            'eta': 0.3,
            'seed': 1000,
            'nthread': 4,
        }

        self.model = xgb.XGBRegressor(**params)
        self.model.fit(X, y)
        self.model.save_model(self.model_path)

    def predict(self, xdf):
        if self.model is None:
            self.model = xgb.XGBRegressor()
            self.model.load_model(self.model_path)
        X_new = xdf.to_numpy()
        return self.model.predict(X_new)

