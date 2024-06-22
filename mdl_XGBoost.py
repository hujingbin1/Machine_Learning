import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os


# XGBoost模型
class MeowModel(object):
    def __init__(self, cacheDir):
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
