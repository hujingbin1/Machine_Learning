import os
import pandas as pd
import numpy as np
from tradingcalendar import Calendar
from log import log

def dealDataFrame(df):
    """
    处理一个dataframe，修正其中的np.nan np.inf，使其变为平均值。
    不会处理0，需要额外对于0的处理。
    """
    column_name = df.columns[3:]
    for name in column_name:
        df = df.replace(np.inf,np.nan)
        # 计算每组的均值
        mean_values = df.groupby('symbol')[name].transform('mean')
        mean_values = mean_values.fillna(mean_values.mean())
        df[name] = df[name].fillna(mean_values)
    return df

#这部分做数据集的加载
class MeowDataLoader(object):
    def __init__(self, h5dir):
        self.h5dir = h5dir
        self.calendar = Calendar()
    
    def loadDates(self, dates):
        if len(dates) == 0:
            raise ValueError("Dates empty")
        log.inf("Loading data of {} dates from {} to {}...".format(len(dates), min(dates), max(dates)))
        return pd.concat(self.loadDate(x) for x in dates)

    def loadDate(self, date):
        if not self.calendar.isTradingDay(date):
            raise ValueError("Not a trading day: {}".format(date))
        h5File = os.path.join(self.h5dir, "{}.h5".format(date))
        df = pd.read_hdf(h5File)
        ###
        df = dealDataFrame(df)
        df.loc[:, "date"] = date
        precols = ["symbol", "interval", "date"]
        df = df[precols + [x for x in df.columns if x not in precols]] # re-arrange columns
        return df
