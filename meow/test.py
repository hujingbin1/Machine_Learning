import os
from log import log
from dl import MeowDataLoader
from feat_all_feature import MeowFeatureGenerator
from mdl import MeowModel
from eval import MeowEvaluator
from tradingcalendar import Calendar
import torch
import numpy as np

# 确定设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class MeowEngine(object):
    def __init__(self, h5dir, cacheDir):
        self.calendar = Calendar()
        self.h5dir = h5dir
        if not os.path.exists(h5dir):
            raise ValueError("Data directory not exists: {}".format(self.h5dir))
        if not os.path.isdir(h5dir):
            raise ValueError("Invalid data directory: {}".format(self.h5dir))
        self.cacheDir = cacheDir # this is not used in sample code
        self.dloader = MeowDataLoader(h5dir=h5dir)
        self.featGenerator = MeowFeatureGenerator(cacheDir=cacheDir)
        self.model = MeowModel(cacheDir=cacheDir)
        self.evaluator = MeowEvaluator(cacheDir=cacheDir)

    def fit(self, startDate, endDate):
        dates = self.calendar.range(startDate, endDate)
        rawData = self.dloader.loadDates(dates)
        log.inf("Running model fitting...")
        xdf, ydf = self.featGenerator.genFeatures(rawData)
        self.model.fit(xdf, ydf)

    def predict(self, xdf):
        # return self.model.predict(torch.tensor(xdf.to_numpy(), dtype=torch.float32).unsqueeze(1).to(device))
        return self.model.predict(xdf)
    
    def eval(self, startDate, endDate):
        log.inf("Running model evaluation...")
        dates = self.calendar.range(startDate, endDate)
        rawData = self.dloader.loadDates(dates)
        xdf, ydf = self.featGenerator.genFeatures(rawData)
        ydf.loc[:, "forecast"] = self.predict(xdf)
        self.evaluator.eval(ydf)

if __name__ == "__main__":
    engine = MeowEngine(h5dir="./dataset/h5", cacheDir=None)
    engine.fit(20230601, 20231129)
    engine.eval(20231201, 20231229)

