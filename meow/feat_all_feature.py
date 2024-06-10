import os
import numpy as np
import pandas as pd
from log import log

'''
这里的feature选取相关系数最高的三十个特征 均为Log(A/B)或者A/B类型的 不区分正/负相关 只看绝对值
'''
def div_fun(col):
	col = col.fillna(1)
	col = col.replace(np.inf, 1)
    ### 是否考虑不进行置1
	col[col<1] = 1
	return col

### 处理na与inf,把na替换为na_value,inf替换为inf_value
def deal_Na_Inf(col,na_value = 1,inf_value = 1):
    col = col.fillna(na_value)
    col = col.replace(np.inf, inf_value)
    
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
            ###下面的特征为通过log(A/B),A/B两种方式枚举出来的特征。
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
            "nCxlBuy-nAddBuy",
            "log-nAddBuy-atr5_9",
            "log-tradeSellLow-nAddBuy",
            "log-nAddBuy-tradeSellHigh",
            "log-addSellLow-nAddBuy",
            "log-tradeBuyQty-atr0_4",
            "log-nAddSell-tradeBuyQty",
            "log-cxlBuyHigh-nAddBuy",
			"log-nAddBuy-atr0_4",
   			"tradeBuyTurnover-nCxlBuy",
            ### 延用部分初始数据

            ### 下面的特征为baseline中直接提供的特征。
            "ob_imb0",
            "ob_imb4",
            "ob_imb9",
            "trade_imb",
            "trade_imbema5",
            # "lagret12"
            ### 其他特征
            # "lastpx-sub-open",
            
        ]

    def __init__(self, cacheDir):
        self.cacheDir = cacheDir
        self.ycol = "fret12"
        self.mcols = ["symbol", "date", "interval"]

    

    def genFeatures(self, df):
        log.inf("Generating {} features from raw data...".format(len(self.featureNames())))
        feature = {}
        col = {}
        "代码生成的feature"
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
        #feature21
        col[21] = (df["nCxlBuy"]/df["nAddBuy"])
        df.loc[:,"nCxlBuy-nAddBuy"] = div_fun(col[21])
        feature["nCxlBuy-nAddBuy"] = df.log[:,"nCxlBuy-nAddBuy"]
        #feature22
        col[22] = (df["nAddbuy"]/df["atr5_9"])
        df.loc[:,"log-nAddBuy-atr5_9"] = log_fun(col[22])
        feature["log-nAddBuy-atr5_9"] = df.loc[:,"log-nAddBuy-atr5_9"]
        #feature23
        col[23] = (df["tradeSellLow"]/df["nAddBuy"])
        df.loc[:,"log-tradeSellLow-nAddBuy"] = log_fun(col[23])
        feature["log-tradeSellLow-nAddBuy"] = df.loc[:,"log-tradeSellLow-nAddBuy"]
        #feature24
        col[24] = (df["nAddBuy"]/df["tradeSellHigh"])
        df.loc[:,"log-nAddBuy-tradeSellHigh"] = log_fun(col[24])
        feature["log-nAddBuy-tradeSellHigh"] = df.loc[:,"log-nAddBuy-tradeSellHigh"]
        #feature25
        col[25] = (df["addSellLow"]/df["nAddBuy"])
        df.loc[:,"log-addSellLow-nAddBuy"] = log_fun(col[25])
        feature["log-addSellLow-nAddBuy"] = df.loc[:,"log-addSellLow-nAddBuy"]
        #feature26
        col[26] = (df["tradeBuyQty"]/df["atr0_4"])
        df.loc[:,"log-tradeBuyQty-atr0_4"] = log_fun(col[26])
        feature["log-tradeBuyQty-atr0_4"] = df.loc[:,"log-tradeBuyQty-atr0_4"]
        #feature27
        col[27] = (df["nAddSell"]/df["tradeBuyQty"])
        df.loc[:,"log-nAddSell-tradeBuyQty"] = log_fun(col[27])
        feature["log-nAddSell-tradeBuyQty"] = df.loc[:,"log-nAddSell-tradeBuyQty"]
        #feature28
        col[28] = (df["cxlBuyHigh"]/df["nAddBuy"]) 
        df.loc[:,"log-cxlBuyHigh-nAddBuy"] = log_fun(col[28])
        feature["log-cxlBuyHigh-nAddBuy"] = df.loc[:,"log-cxlBuyHigh-nAddBuy"]
        #feature29
        col[29] = (df["nAddBuy"]/df["atr0_4"])
        df.loc[:,"log-nAddBuy-atr0_4"] = log_fun(col[29])
        feature["log-nAddBuy-atr0_4"] = df.loc[:,"log-nAddBuy-atr0_4"]
        #feature30
        col[30] = (df["tradeBuyTurnover"]/df["nCxlBuy"])
        df.loc[:,"tradeBuyTurnover-nCxlBuy"] = div_fun(col[30])
        feature["tradeBuyTurnover-nCxlBuy"] = df.loc[:,"tradeBuyTurnover-nCxlBuy"]
        
        "延用老师提供的feature"
        ### feature31
        col[31] = (df["asize0"] - df["bsize0"]) / (df["asize0"] + df["bsize0"])
        df.loc[:, "ob_imb0"] = deal_Na_Inf(col[31,1,1])
        feature["ob_imb0"] = df.loc[:,"ob_imb0"]
        ### feature32
        col[32] = (df["asize0_4"] - df["bsize0_4"]) / (df["asize0_4"] + df["bsize0_4"])
        df.loc[:, "ob_imb4"] = deal_Na_Inf(col[32],1,1)
        feature["ob_imb4"] = df.loc[:,"ob_imb4"]
        ### feature33
        col[33] = (df["asize5_9"] - df["bsize5_9"]) / (df["asize5_9"] + df["bsize5_9"])
        df.loc[:, "ob_imb9"] = deal_Na_Inf(col[33],1,1)
        feature["ob_imb9"] = df.loc[:,"ob_imb9"]
        ### feature34
        col[34] = (df["tradeBuyQty"] - df["tradeSellQty"]) / (df["tradeBuyQty"] + df["tradeSellQty"])
        df.loc[:, "trade_imb"] = deal_Na_Inf(col[34],1,1)
        feature["trade_imb"] = df.loc[:,"trade_imb"]
        ### feature35
        col[35] = df["trade_imb"].ewm(halflife=5).mean()
        df.loc[:, "trade_imbema5"] = deal_Na_Inf(col[35],1,1)
        feature["trade_imbema5"] = df.loc[:,"trade_imbema5"]
        
        
		#ydf
        df.loc[:, "bret12"] = (df["midpx"] - df["midpx"].shift(12)) / df["midpx"].shift(12) # backward return
        cxbret = df.groupby("interval")[["bret12"]].mean().reset_index().rename(columns={"bret12": "cx_bret12"})
        df = df.merge(cxbret, on="interval", how="left")
        df.loc[:, "lagret12"] = df["bret12"] - df["cx_bret12"]
        
        xdf = df[self.mcols + self.featureNames()].set_index(self.mcols)
        ydf = df[self.mcols + [self.ycol]].set_index(self.mcols)
        return xdf.fillna(0), ydf.fillna(0)
