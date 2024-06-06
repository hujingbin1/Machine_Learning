import os
import numpy as np
import pandas as pd
from log import log

'''
这里的feature选取相关系数最高的二十个特征 均为Log(A/B)或者A/B类型的 不区分正/负相关 只看绝对值
'''
def div_fun(col):
	col = col.fillna(1)
	col = col.replace(np.inf, 1)
	col[col<1] = 1
	return col

def log_fun(col):
	col = col.fillna(1)
	col = col.replace(np.inf, 1)
	col[col<1] = 1
	col = np.log(col)
	return col

class MeowFeatureGenerator(object):
    @classmethod
    def featureNames(cls):
        return [
            "log-nAddBuy-nCxlBuy",
            "low-open",
            "log-open-low",
            "log-tradeBuyQty-sellVwad",
            "log-tradeBuyTurnover-sellVwad",
            "log-atr10_19-addBuyQty",
            "log-nAddSell-nAddBuy",
            "log-atr10_19-nAddBuy",
            "log-atr10_19-tradeBuyQty",
            "nAddBuy-nCxlBuy",
            "log-addBuyQty-atr5_9",
            "log-atr0_4-addBuyQty",
            "log-nAddBuy-addSellHigh",
            "log-nCxlBuy-addBuyQty",
            "log-addBuyQty-nAddSell",
            "log-atr10_19-nTradeBuy",
            "log-nAddSell-nTradeBuy",
            "log-nTradeBuy-nCxlBuy",
            "log-sellVwad-addBuyQty",
            "log-tradeBuyQty-atr5_9"
            # "nCxlBuy-nAddBuy",
            # "log-nAddBuy-atr5_9",
            # "log-tradeSellLow-nAddBuy",
            # "log-nAddBuy-tradeSellHigh",
            # "log-addSellLow-nAddBuy",
            # "log-tradeBuyQty-atr0_4",
            # "log-nAddSell-tradeBuyQty",
            # "log-cxlBuyHigh-nAddBuy",
			# "log-nAddBuy-atr0_4",
   			# "tradeBuyTurnover-nCxlBuy"
        ]

    def __init__(self, cacheDir):
        self.cacheDir = cacheDir
        self.ycol = "fret12"
        self.mcols = ["symbol", "date", "interval"]

    

    def genFeatures(self, df):
        log.inf("Generating {} features from raw data...".format(len(self.featureNames())))
        feature = {}
        col = {}
        #feature1
        col[1] = (df["nAddBuy"]/df["nCxlBuy"])
        df.loc[:, "log-nAddBuy-nCxlBuy"] = log_fun(col[1])
        feature["log-nAddBuy-nCxlBuy"] = df.loc[:, "log-nAddBuy-nCxlBuy"]
        #feature2
        col[2] = (df["low"]/df["open"])
        df.loc[:, "low-open"] = div_fun(col[2])
        feature["low-open"] = df.loc[:, "low-open"]
        #feature3
        col[3] = (df["open"]/df["low"])
        df.loc[:, "log-open-low"] = log_fun(col[3])
        feature["log-open-low"] = df.loc[:, "log-open-low"]
        #feature4
        col[4] = (df["tradeBuyQty"]/df["sellVwad"])
        df.loc[:, "log-tradeBuyQty-sellVwad"] = log_fun(col[4])
        feature["log-tradeBuyQty-sellVwad"] = df.loc[:, "log-tradeBuyQty-sellVwad"]
        #feature5
        col[5] = (df["tradeBuyTurnover"]/df["sellVwad"])
        df.loc[:, "log-tradeBuyTurnover-sellVwad"] = log_fun(col[5])
        feature["log-tradeBuyTurnover-sellVwad"] = df.loc[:, "log-tradeBuyTurnover-sellVwad"]
        #feature6
        col[6] = (df["atr10_19"]/df["addBuyQty"])
        df.loc[:, "log-atr10_19-addBuyQty"] = log_fun(col[6])
        feature["log-atr10_19-addBuyQty"] = df.loc[:, "log-atr10_19-addBuyQty"]
        #feature7
        col[7] = (df["nAddSell"]/df["nAddBuy"])
        df.loc[:, "log-nAddSell-nAddBuy"] = log_fun(col[7])
        feature["log-nAddSell-nAddBuy"] = df.loc[:, "log-nAddSell-nAddBuy"]
        #feature8
        col[8] = (df["atr10_19"]/df["nAddBuy"])
        df.loc[:, "log-atr10_19-nAddBuy"] = log_fun(col[8])
        feature["log-atr10_19-nAddBuy"] = df.loc[:, "log-atr10_19-nAddBuy"]
        #feature9
        col[9] = (df["atr10_19"]/df["tradeBuyQty"])
        df.loc[:, "log-atr10_19-tradeBuyQty"] = log_fun(col[9])
        feature["log-atr10_19-tradeBuyQty"] = df.loc[:, "log-atr10_19-tradeBuyQty"]
        #feature10
        col[10] = (df["nAddBuy"]/df["nCxlBuy"])
        df.loc[:, "nAddBuy-nCxlBuy"] = div_fun(col[10])
        feature["nAddBuy-nCxlBuy"] = df.loc[:, "nAddBuy-nCxlBuy"]
        #feature11
        col[11] = (df["addBuyQty"]/df["atr5_9"])
        df.loc[:, "log-addBuyQty-atr5_9"] = log_fun(col[11])
        feature["log-addBuyQty-atr5_9"] = df.loc[:, "log-addBuyQty-atr5_9"]
        #feature12
        col[12] = (df["atr0_4"]/df["addBuyQty"])
        df.loc[:, "log-atr0_4-addBuyQty"] = log_fun(col[12])
        feature["log-atr0_4-addBuyQty"] = df.loc[:, "log-atr0_4-addBuyQty"]
        #feature13
        col[13] = (df["nAddBuy"]/df["addSellHigh"])
        df.loc[:, "log-nAddBuy-addSellHigh"] = log_fun(col[13])
        feature["log-nAddBuy-addSellHigh"] = df.loc[:, "log-nAddBuy-addSellHigh"]
        #feature14
        col[14] = (df["nCxlBuy"]/df["addBuyQty"])
        df.loc[:, "log-nCxlBuy-addBuyQty"] = log_fun(col[14])
        feature["log-nCxlBuy-addBuyQty"] = df.loc[:, "log-nCxlBuy-addBuyQty"]
        #feature15
        col[15] = (df["addBuyQty"]/df["nAddSell"])
        df.loc[:, "log-addBuyQty-nAddSell"] = log_fun(col[15])
        feature["log-addBuyQty-nAddSell"] = df.loc[:, "log-addBuyQty-nAddSell"]
        #feature16
        col[16] = (df["atr10_19"]/df["nTradeBuy"])
        df.loc[:, "log-atr10_19-nTradeBuy"] = log_fun(col[16])
        feature["log-atr10_19-nTradeBuy"] = df.loc[:, "log-atr10_19-nTradeBuy"]
        #feature17
        col[17] = (df["nAddSell"]/df["nTradeBuy"])
        df.loc[:, "log-nAddSell-nTradeBuy"] = log_fun(col[17])
        feature["log-nAddSell-nTradeBuy"] = df.loc[:, "log-nAddSell-nTradeBuy"]
        #feature18
        col[18] = (df["nTradeBuy"]/df["nCxlBuy"])
        df.loc[:, "log-nTradeBuy-nCxlBuy"] = log_fun(col[18])
        feature["log-nTradeBuy-nCxlBuy"] = df.loc[:, "log-nTradeBuy-nCxlBuy"]
        #feature19
        col[19] = (df["sellVwad"]/df["addBuyQty"])
        df.loc[:, "log-sellVwad-addBuyQty"] = log_fun(col[19])
        feature["log-sellVwad-addBuyQty"] = df.loc[:, "log-sellVwad-addBuyQty"]
        #feature20
        col[20] = (df["tradeBuyQty"]/df["atr5_9"])
        df.loc[:, "log-tradeBuyQty-atr5_9"] = log_fun(col[20])
        feature["log-tradeBuyQty-atr5_9"] = df.loc[:, "log-tradeBuyQty-atr5_9"]

		#ydf
        df.loc[:, "bret12"] = (df["midpx"] - df["midpx"].shift(12)) / df["midpx"].shift(12) # backward return
        cxbret = df.groupby("interval")[["bret12"]].mean().reset_index().rename(columns={"bret12": "cx_bret12"})
        df = df.merge(cxbret, on="interval", how="left")
        df.loc[:, "lagret12"] = df["bret12"] - df["cx_bret12"]
        
        xdf = df[self.mcols + self.featureNames()].set_index(self.mcols)
        ydf = df[self.mcols + [self.ycol]].set_index(self.mcols)
        return xdf.fillna(0), ydf.fillna(0)
