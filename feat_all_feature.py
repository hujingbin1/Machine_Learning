import os
import numpy as np
import pandas as pd
from log import log


def div_fun(col):
    col = col.fillna(1)
    col = col.replace(np.inf, 1)
    # TODO: 是否考虑不进行置1
    col[col < 1] = 1
    return col


# 处理na与inf,把na替换为na_value,inf替换为inf_value
def deal_Na_Inf(col, na_value=1, inf_value=1):
    col = col.fillna(na_value)
    col = col.replace(np.inf, inf_value)
    return col


def log_fun(col):
    col = col.fillna(1)
    col = col.replace(np.inf, 1)
    col[col < 1] = 1
    col = np.log(col)
    return col


def normalize(group):
    # 计算最大值和最小值，用于归一化
    min_val = group.min()
    max_val = group.max()
    # 归一化公式：(x - min) / (max - min)，并保留两位小数（可选）
    return (group - min_val) / (max_val - min_val)


class MeowFeatureGenerator(object):
    @classmethod
    def featureNames(cls):
        return [
            ###下面的特征为通过log(A/B),A/B,B/A两种方式枚举出来的特征。
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
            "log-tradeBuyQty-atr5_9",
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
            ### 下面的特征为baseline中直接提供的特征。
            "ob_imb0",
            "ob_imb4",
            "ob_imb9",
            "trade_imb",
            "trade_imbema5",
            "lagret12",
            ### 通过增长趋势显现出来的特征。
            "trend-nCxlSell",
            "trend-atr10_19",
            "trend-nTradeBuy",
            "trend-nTradeSell",
            "trend-tradeBuyQty",
            "trend-cxlSellQty",
            "trend-tradeBuyTurnover",
            "trend-cxlSellTurnover",
            "trend-nCxlBuy",
            "trend-nAddBuy",
            "trend-tradeSellTurnover",
            "trend-tradeSellQty",
            #### 这些虽然也是trend，但是在corr方面没有足够的吸引力
            #### 但是其理论意义我觉得可以作为feature。
            "trend-open",
            "trend-high",
            "trend-low",
            "trend-bid0",
            "trend-ask0",
            "trend-bid19",
            "trend-ask19",
            "trend-btr10_19",  # 因为atr10_19在上面的属性中，所以也采用这个属性。
            # diff类特征,(A-B)/(A+B),注意到有部分数据高度相关，人为剔除了几个。
            "diff(midpx-tradeBuyHigh)",
            "diff(midpx-tradeSellHigh)",
            "diff(midpx-lastpx)",
            "diff(lastpx-ask0)",
            "diff(open-bid0)",
            "diff(low-ask0)",
            "diff(open-low)",
            ####下面是几个认为感觉有意义的
            "diff(nAddBuy-nAddSell)",
            "diff(addBuyQty-addSellQty)",
            "diff(addBuyTurnover-addSellTurnover)",
            "diff(nCxlBuy-nCxlSell)",
            "diff(cxlBuyQty-cxlSellQty)",
            "diff(cxlBuyTurnover-cxlSellTurnover)",
            "diff(nTradeBuy-nTradeSell)",
            "diff(tradeBuyQty-tradeSellQty)",
            "diff(tradeBuyTurnover-tradeSellTurnover)",

            ###手工制作特征
            "(ask0/bid0)-1",
            "wap0",
            "atr0_4/asize0_4",
            "btr0_4/bsize0_4",
            "atr10_19/asize10_19",
            "btr10_19/bsize10_19",
            ### 基础特征
            # "norm-tradeBuyQty",
            # "norm-tradeSellQty",
            ### 时间特征
            # "week-day", #星期几
            # "continue-time" #这只股票开盘的时间/10
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
        # feature1
        col[1] = (df["nAddBuy"] / df["nCxlBuy"])
        df.loc[:, "log-nAddBuy-nCxlBuy"] = log_fun(col[1])
        feature["log-nAddBuy-nCxlBuy"] = df.loc[:, "log-nAddBuy-nCxlBuy"]
        # feature2
        col[2] = (df["low"] / df["open"])
        df.loc[:, "low-open"] = div_fun(col[2])
        feature["low-open"] = df.loc[:, "low-open"]
        # feature3
        col[3] = (df["open"] / df["low"])
        df.loc[:, "log-open-low"] = log_fun(col[3])
        feature["log-open-low"] = df.loc[:, "log-open-low"]
        # feature4
        col[4] = (df["tradeBuyQty"] / df["sellVwad"])
        df.loc[:, "log-tradeBuyQty-sellVwad"] = log_fun(col[4])
        feature["log-tradeBuyQty-sellVwad"] = df.loc[:, "log-tradeBuyQty-sellVwad"]
        # feature5
        col[5] = (df["tradeBuyTurnover"] / df["sellVwad"])
        df.loc[:, "log-tradeBuyTurnover-sellVwad"] = log_fun(col[5])
        feature["log-tradeBuyTurnover-sellVwad"] = df.loc[:, "log-tradeBuyTurnover-sellVwad"]
        # feature6
        col[6] = (df["atr10_19"] / df["addBuyQty"])
        df.loc[:, "log-atr10_19-addBuyQty"] = log_fun(col[6])
        feature["log-atr10_19-addBuyQty"] = df.loc[:, "log-atr10_19-addBuyQty"]
        # feature7
        col[7] = (df["nAddSell"] / df["nAddBuy"])
        df.loc[:, "log-nAddSell-nAddBuy"] = log_fun(col[7])
        feature["log-nAddSell-nAddBuy"] = df.loc[:, "log-nAddSell-nAddBuy"]
        # feature8
        col[8] = (df["atr10_19"] / df["nAddBuy"])
        df.loc[:, "log-atr10_19-nAddBuy"] = log_fun(col[8])
        feature["log-atr10_19-nAddBuy"] = df.loc[:, "log-atr10_19-nAddBuy"]
        # feature9
        col[9] = (df["atr10_19"] / df["tradeBuyQty"])
        df.loc[:, "log-atr10_19-tradeBuyQty"] = log_fun(col[9])
        feature["log-atr10_19-tradeBuyQty"] = df.loc[:, "log-atr10_19-tradeBuyQty"]
        # feature10
        col[10] = (df["nAddBuy"] / df["nCxlBuy"])
        df.loc[:, "nAddBuy-nCxlBuy"] = div_fun(col[10])
        feature["nAddBuy-nCxlBuy"] = df.loc[:, "nAddBuy-nCxlBuy"]
        # feature11
        col[11] = (df["addBuyQty"] / df["atr5_9"])
        df.loc[:, "log-addBuyQty-atr5_9"] = log_fun(col[11])
        feature["log-addBuyQty-atr5_9"] = df.loc[:, "log-addBuyQty-atr5_9"]
        # feature12
        col[12] = (df["atr0_4"] / df["addBuyQty"])
        df.loc[:, "log-atr0_4-addBuyQty"] = log_fun(col[12])
        feature["log-atr0_4-addBuyQty"] = df.loc[:, "log-atr0_4-addBuyQty"]
        # feature13
        col[13] = (df["nAddBuy"] / df["addSellHigh"])
        df.loc[:, "log-nAddBuy-addSellHigh"] = log_fun(col[13])
        feature["log-nAddBuy-addSellHigh"] = df.loc[:, "log-nAddBuy-addSellHigh"]
        # feature14
        col[14] = (df["nCxlBuy"] / df["addBuyQty"])
        df.loc[:, "log-nCxlBuy-addBuyQty"] = log_fun(col[14])
        feature["log-nCxlBuy-addBuyQty"] = df.loc[:, "log-nCxlBuy-addBuyQty"]
        # feature15
        col[15] = (df["addBuyQty"] / df["nAddSell"])
        df.loc[:, "log-addBuyQty-nAddSell"] = log_fun(col[15])
        feature["log-addBuyQty-nAddSell"] = df.loc[:, "log-addBuyQty-nAddSell"]
        # feature16
        col[16] = (df["atr10_19"] / df["nTradeBuy"])
        df.loc[:, "log-atr10_19-nTradeBuy"] = log_fun(col[16])
        feature["log-atr10_19-nTradeBuy"] = df.loc[:, "log-atr10_19-nTradeBuy"]
        # feature17
        col[17] = (df["nAddSell"] / df["nTradeBuy"])
        df.loc[:, "log-nAddSell-nTradeBuy"] = log_fun(col[17])
        feature["log-nAddSell-nTradeBuy"] = df.loc[:, "log-nAddSell-nTradeBuy"]
        # feature18
        col[18] = (df["nTradeBuy"] / df["nCxlBuy"])
        df.loc[:, "log-nTradeBuy-nCxlBuy"] = log_fun(col[18])
        feature["log-nTradeBuy-nCxlBuy"] = df.loc[:, "log-nTradeBuy-nCxlBuy"]
        # feature19
        col[19] = (df["sellVwad"] / df["addBuyQty"])
        df.loc[:, "log-sellVwad-addBuyQty"] = log_fun(col[19])
        feature["log-sellVwad-addBuyQty"] = df.loc[:, "log-sellVwad-addBuyQty"]
        # feature20
        col[20] = (df["tradeBuyQty"] / df["atr5_9"])
        df.loc[:, "log-tradeBuyQty-atr5_9"] = log_fun(col[20])
        feature["log-tradeBuyQty-atr5_9"] = df.loc[:, "log-tradeBuyQty-atr5_9"]
        # feature21
        col[21] = (df["nCxlBuy"] / df["nAddBuy"])
        df.loc[:, "nCxlBuy-nAddBuy"] = div_fun(col[21])
        feature["nCxlBuy-nAddBuy"] = df.loc[:, "nCxlBuy-nAddBuy"]
        # feature22
        col[22] = (df["nAddBuy"] / df["atr5_9"])
        df.loc[:, "log-nAddBuy-atr5_9"] = log_fun(col[22])
        feature["log-nAddBuy-atr5_9"] = df.loc[:, "log-nAddBuy-atr5_9"]
        # feature23
        col[23] = (df["tradeSellLow"] / df["nAddBuy"])
        df.loc[:, "log-tradeSellLow-nAddBuy"] = log_fun(col[23])
        feature["log-tradeSellLow-nAddBuy"] = df.loc[:, "log-tradeSellLow-nAddBuy"]
        # feature24
        col[24] = (df["nAddBuy"] / df["tradeSellHigh"])
        df.loc[:, "log-nAddBuy-tradeSellHigh"] = log_fun(col[24])
        feature["log-nAddBuy-tradeSellHigh"] = df.loc[:, "log-nAddBuy-tradeSellHigh"]
        # feature25
        col[25] = (df["addSellLow"] / df["nAddBuy"])
        df.loc[:, "log-addSellLow-nAddBuy"] = log_fun(col[25])
        feature["log-addSellLow-nAddBuy"] = df.loc[:, "log-addSellLow-nAddBuy"]
        # feature26
        col[26] = (df["tradeBuyQty"] / df["atr0_4"])
        df.loc[:, "log-tradeBuyQty-atr0_4"] = log_fun(col[26])
        feature["log-tradeBuyQty-atr0_4"] = df.loc[:, "log-tradeBuyQty-atr0_4"]
        # feature27
        col[27] = (df["nAddSell"] / df["tradeBuyQty"])
        df.loc[:, "log-nAddSell-tradeBuyQty"] = log_fun(col[27])
        feature["log-nAddSell-tradeBuyQty"] = df.loc[:, "log-nAddSell-tradeBuyQty"]
        # feature28
        col[28] = (df["cxlBuyHigh"] / df["nAddBuy"])
        df.loc[:, "log-cxlBuyHigh-nAddBuy"] = log_fun(col[28])
        feature["log-cxlBuyHigh-nAddBuy"] = df.loc[:, "log-cxlBuyHigh-nAddBuy"]
        # feature29
        col[29] = (df["nAddBuy"] / df["atr0_4"])
        df.loc[:, "log-nAddBuy-atr0_4"] = log_fun(col[29])
        feature["log-nAddBuy-atr0_4"] = df.loc[:, "log-nAddBuy-atr0_4"]
        # feature30
        col[30] = (df["tradeBuyTurnover"] / df["nCxlBuy"])
        df.loc[:, "tradeBuyTurnover-nCxlBuy"] = div_fun(col[30])
        feature["tradeBuyTurnover-nCxlBuy"] = df.loc[:, "tradeBuyTurnover-nCxlBuy"]

        "延用老师提供的feature"
        ### feature31
        col[31] = (df["asize0"] - df["bsize0"]) / (df["asize0"] + df["bsize0"])
        df.loc[:, "ob_imb0"] = deal_Na_Inf(col[31], 1, 1)
        feature["ob_imb0"] = df.loc[:, "ob_imb0"]
        ### feature32
        col[32] = (df["asize0_4"] - df["bsize0_4"]) / (df["asize0_4"] + df["bsize0_4"])
        df.loc[:, "ob_imb4"] = deal_Na_Inf(col[32], 1, 1)
        feature["ob_imb4"] = df.loc[:, "ob_imb4"]
        ### feature33
        col[33] = (df["asize5_9"] - df["bsize5_9"]) / (df["asize5_9"] + df["bsize5_9"])
        df.loc[:, "ob_imb9"] = deal_Na_Inf(col[33], 1, 1)
        feature["ob_imb9"] = df.loc[:, "ob_imb9"]
        ### feature34
        col[34] = (df["tradeBuyQty"] - df["tradeSellQty"]) / (df["tradeBuyQty"] + df["tradeSellQty"])
        df.loc[:, "trade_imb"] = deal_Na_Inf(col[34], 1, 1)
        feature["trade_imb"] = df.loc[:, "trade_imb"]
        ### feature35
        col[35] = df["trade_imb"].ewm(halflife=5).mean()
        df.loc[:, "trade_imbema5"] = deal_Na_Inf(col[35], 1, 1)
        feature["trade_imbema5"] = df.loc[:, "trade_imbema5"]
        ###trend类型feature
        ### feature36
        shift = df["nCxlSell"].shift(periods=12, fill_value=df["nCxlSell"].mean())
        shift = shift.replace(0, shift.mean())
        col[36] = (df["nCxlSell"] - shift) / shift
        df.loc[:, "trend-nCxlSell"] = deal_Na_Inf(col[36])
        feature["trend-nCxlSell"] = df.loc[:, "trend-nCxlSell"]
        ### feature37
        shift = df["atr10_19"].shift(periods=12, fill_value=df["atr10_19"].mean())
        shift = shift.replace(0, shift.mean())
        col[37] = (df["atr10_19"] - shift) / shift
        df.loc[:, "trend-atr10_19"] = deal_Na_Inf(col[37])
        feature["trend-atr10_19"] = df.loc[:, "trend-atr10_19"]
        ### feature38
        shift = df["nTradeBuy"].shift(periods=12, fill_value=df["nTradeBuy"].mean())
        shift = shift.replace(0, shift.mean())
        col[38] = (df["nTradeBuy"] - shift) / shift
        df.loc[:, "trend-nTradeBuy"] = deal_Na_Inf(col[38])
        feature["trend-nTradeBuy"] = df.loc[:, "trend-nTradeBuy"]
        ### feature39
        shift = df["nTradeSell"].shift(periods=12, fill_value=df["nTradeSell"].mean())
        shift = shift.replace(0, shift.mean())
        col[39] = (df["nTradeSell"] - shift) / shift
        df.loc[:, "trend-nTradeSell"] = deal_Na_Inf(col[39])
        feature["trend-nTradeSell"] = df.loc[:, "trend-nTradeSell"]
        ### feature40
        shift = df["tradeBuyQty"].shift(periods=12, fill_value=df["tradeBuyQty"].mean())
        shift = shift.replace(0, shift.mean())
        col[40] = (df["tradeBuyQty"] - shift) / shift
        df.loc[:, "trend-tradeBuyQty"] = deal_Na_Inf(col[40])
        feature["trend-tradeBuyQty"] = df.loc[:, "trend-tradeBuyQty"]
        ### feature41
        shift = df["cxlSellQty"].shift(periods=12, fill_value=df["cxlSellQty"].mean())
        shift = shift.replace(0, shift.mean())
        col[41] = (df["cxlSellQty"] - shift) / shift
        df.loc[:, "trend-cxlSellQty"] = deal_Na_Inf(col[41])
        feature["trend-cxlSellQty"] = df.loc[:, "trend-cxlSellQty"]
        ### feature42
        shift = df["tradeBuyTurnover"].shift(periods=12, fill_value=df["tradeBuyTurnover"].mean())
        shift = shift.replace(0, shift.mean())
        col[42] = (df["tradeBuyTurnover"] - shift) / shift
        df.loc[:, "trend-tradeBuyTurnover"] = deal_Na_Inf(col[42])
        feature["trend-tradeBuyTurnover"] = df.loc[:, "trend-tradeBuyTurnover"]
        ### feature43
        shift = df["cxlSellTurnover"].shift(periods=12, fill_value=df["cxlSellTurnover"].mean())
        shift = shift.replace(0, shift.mean())
        col[43] = (df["cxlSellTurnover"] - shift) / shift
        df.loc[:, "trend-cxlSellTurnover"] = deal_Na_Inf(col[43])
        feature["trend-cxlSellTurnover"] = df.loc[:, "trend-cxlSellTurnover"]
        ### feature44
        shift = df["nCxlBuy"].shift(periods=12, fill_value=df["nCxlBuy"].mean())
        shift = shift.replace(0, shift.mean())
        col[44] = (df["nCxlBuy"] - shift) / shift
        df.loc[:, "trend-nCxlBuy"] = deal_Na_Inf(col[44])
        feature["trend-nCxlBuy"] = df.loc[:, "trend-nCxlBuy"]
        ### feature45
        shift = df["nAddBuy"].shift(periods=12, fill_value=df["nAddBuy"].mean())
        shift = shift.replace(0, shift.mean())
        col[45] = (df["nAddBuy"] - shift) / shift
        df.loc[:, "trend-nAddBuy"] = deal_Na_Inf(col[45])
        feature["trend-nAddBuy"] = df.loc[:, "trend-nAddBuy"]
        ### feature46
        shift = df["tradeSellTurnover"].shift(periods=12, fill_value=df["tradeSellTurnover"].mean())
        shift = shift.replace(0, shift.mean())
        col[46] = (df["tradeSellTurnover"] - shift) / shift
        df.loc[:, "trend-tradeSellTurnover"] = deal_Na_Inf(col[46])
        feature["trend-tradeSellTurnover"] = df.loc[:, "trend-tradeSellTurnover"]
        ### feature47
        shift = df["tradeSellQty"].shift(periods=12, fill_value=df["tradeSellQty"].mean())
        shift = shift.replace(0, shift.mean())
        col[47] = (df["tradeSellQty"] - shift) / shift
        df.loc[:, "trend-tradeSellQty"] = deal_Na_Inf(col[47])
        feature["trend-tradeSellQty"] = df.loc[:, "trend-tradeSellQty"]
        ############corr不明显，但是比较有意义的trend
        ### feature48
        shift = df["open"].shift(periods=12, fill_value=df["open"].mean())
        shift = shift.replace(0, shift.mean())
        col[48] = (df["open"] - shift) / shift
        df.loc[:, "trend-open"] = deal_Na_Inf(col[48])
        feature["trend-open"] = df.loc[:, "trend-open"]
        ### feature49
        shift = df["high"].shift(periods=12, fill_value=df["high"].mean())
        shift = shift.replace(0, shift.mean())
        col[49] = (df["high"] - shift) / shift
        df.loc[:, "trend-high"] = deal_Na_Inf(col[49])
        feature["trend-high"] = df.loc[:, "trend-high"]
        ### feature50
        shift = df["low"].shift(periods=12, fill_value=df["low"].mean())
        shift = shift.replace(0, shift.mean())
        col[50] = (df["low"] - shift) / shift
        df.loc[:, "trend-low"] = deal_Na_Inf(col[50])
        feature["trend-low"] = df.loc[:, "trend-low"]
        ### feature51
        shift = df["bid0"].shift(periods=12, fill_value=df["bid0"].mean())
        shift = shift.replace(0, shift.mean())
        col[51] = (df["bid0"] - shift) / shift
        df.loc[:, "trend-bid0"] = deal_Na_Inf(col[51])
        feature["trend-bid0"] = df.loc[:, "trend-bid0"]
        ### feature52
        shift = df["ask0"].shift(periods=12, fill_value=df["ask0"].mean())
        shift = shift.replace(0, shift.mean())
        col[52] = (df["ask0"] - shift) / shift
        df.loc[:, "trend-ask0"] = deal_Na_Inf(col[52])
        feature["trend-ask0"] = df.loc[:, "trend-ask0"]
        ### feature53
        shift = df["bid19"].shift(periods=12, fill_value=df["bid19"].mean())
        shift = shift.replace(0, shift.mean())
        col[53] = (df["bid19"] - shift) / shift
        df.loc[:, "trend-bid19"] = deal_Na_Inf(col[53])
        feature["trend-bid19"] = df.loc[:, "trend-bid19"]
        ### feature54
        shift = df["ask19"].shift(periods=12, fill_value=df["ask19"].mean())
        shift = shift.replace(0, shift.mean())
        col[54] = (df["ask19"] - shift) / shift
        df.loc[:, "trend-ask19"] = deal_Na_Inf(col[54])
        feature["trend-ask19"] = df.loc[:, "trend-ask19"]
        ### feature55
        shift = df["btr10_19"].shift(periods=12, fill_value=df["btr10_19"].mean())
        shift = shift.replace(0, shift.mean())
        col[55] = (df["btr10_19"] - shift) / shift
        df.loc[:, "trend-btr10_19"] = deal_Na_Inf(col[55])
        feature["trend-btr10_19"] = df.loc[:, "trend-btr10_19"]

        ###diff类特征，计算公式(A-B)/(A+B)
        ###feature56
        col[56] = (df["midpx"] - df["tradeBuyHigh"]) / (df["midpx"] + df["tradeBuyHigh"])
        df.loc[:, "diff(midpx-tradeBuyHigh)"] = deal_Na_Inf(col[56])
        feature["diff(midpx-tradeBuyHigh)"] = df.loc[:, "diff(midpx-tradeBuyHigh)"]
        ###feature57
        col[57] = (df["midpx"] - df["tradeSellHigh"]) / (df["midpx"] + df["tradeSellHigh"])
        df.loc[:, "diff(midpx-tradeSellHigh)"] = deal_Na_Inf(col[57])
        feature["diff(midpx-tradeSellHigh)"] = df.loc[:, "diff(midpx-tradeSellHigh)"]
        ###feature58
        col[58] = (df["midpx"] - df["lastpx"]) / (df["midpx"] + df["lastpx"])
        df.loc[:, "diff(midpx-lastpx)"] = deal_Na_Inf(col[58])
        feature["diff(midpx-lastpx)"] = df.loc[:, "diff(midpx-lastpx)"]
        ###feature59
        col[59] = (df["lastpx"] - df["ask0"]) / (df["lastpx"] + df["ask0"])
        df.loc[:, "diff(lastpx-ask0)"] = deal_Na_Inf(col[59])
        feature["diff(lastpx-ask0)"] = df.loc[:, "diff(lastpx-ask0)"]
        ###feature60
        col[60] = (df["open"] - df["bid0"]) / (df["open"] + df["bid0"])
        df.loc[:, "diff(open-bid0)"] = deal_Na_Inf(col[60])
        feature["diff(open-bid0)"] = df.loc[:, "diff(open-bid0)"]
        ###feature61
        col[61] = (df["low"] - df["ask0"]) / (df["low"] + df["ask0"])
        df.loc[:, "diff(low-ask0)"] = deal_Na_Inf(col[61])
        feature["diff(low-ask0)"] = df.loc[:, "diff(low-ask0)"]
        ###feature62
        col[62] = (df["open"] - df["low"]) / (df["open"] + df["low"])
        df.loc[:, "diff(open-low)"] = deal_Na_Inf(col[62])
        feature["diff(open-low)"] = df.loc[:, "diff(open-low)"]
        ###feature63
        col[63] = (df["nAddBuy"] - df["nAddSell"]) / (df["nAddBuy"] + df["nAddSell"])
        df.loc[:, "diff(nAddBuy-nAddSell)"] = deal_Na_Inf(col[63])
        feature["diff(nAddBuy-nAddSell)"] = df.loc[:, "diff(nAddBuy-nAddSell)"]
        ###feature64
        col[64] = (df["addBuyQty"] - df["addSellQty"]) / (df["addBuyQty"] + df["addSellQty"])
        df.loc[:, "diff(addBuyQty-addSellQty)"] = deal_Na_Inf(col[64])
        feature["diff(addBuyQty-addSellQty)"] = df.loc[:, "diff(addBuyQty-addSellQty)"]
        ###feature65
        col[65] = (df["addBuyTurnover"] - df["addSellTurnover"]) / (df["addBuyTurnover"] + df["addSellTurnover"])
        df.loc[:, "diff(addBuyTurnover-addSellTurnover)"] = deal_Na_Inf(col[65])
        feature["diff(addBuyTurnover-addSellTurnover)"] = df.loc[:, "diff(addBuyTurnover-addSellTurnover)"]
        ###feature66
        col[66] = (df["nCxlBuy"] - df["nCxlSell"]) / (df["nCxlBuy"] + df["nCxlSell"])
        df.loc[:, "diff(nCxlBuy-nCxlSell)"] = deal_Na_Inf(col[66])
        feature["diff(nCxlBuy-nCxlSell)"] = df.loc[:, "diff(nCxlBuy-nCxlSell)"]
        ###feature67
        col[67] = (df["cxlBuyQty"] - df["cxlSellQty"]) / (df["cxlBuyQty"] + df["cxlSellQty"])
        df.loc[:, "diff(cxlBuyQty-cxlSellQty)"] = deal_Na_Inf(col[67])
        feature["diff(cxlBuyQty-cxlSellQty)"] = df.loc[:, "diff(cxlBuyQty-cxlSellQty)"]
        ###feature68
        col[68] = (df["cxlBuyTurnover"] - df["cxlSellTurnover"]) / (df["cxlBuyTurnover"] + df["cxlSellTurnover"])
        df.loc[:, "diff(cxlBuyTurnover-cxlSellTurnover)"] = deal_Na_Inf(col[68])
        feature["diff(cxlBuyTurnover-cxlSellTurnover)"] = df.loc[:, "diff(cxlBuyTurnover-cxlSellTurnover)"]
        ###feature69
        col[69] = (df["nTradeBuy"] - df["nTradeSell"]) / (df["nTradeBuy"] + df["nTradeSell"])
        df.loc[:, "diff(nTradeBuy-nTradeSell)"] = deal_Na_Inf(col[69])
        feature["diff(nTradeBuy-nTradeSell)"] = df.loc[:, "diff(nTradeBuy-nTradeSell)"]
        ###feature70
        col[70] = (df["tradeBuyQty"] - df["tradeSellQty"]) / (df["tradeBuyQty"] + df["tradeSellQty"])
        df.loc[:, "diff(tradeBuyQty-tradeSellQty)"] = deal_Na_Inf(col[70])
        feature["diff(tradeBuyQty-tradeSellQty)"] = df.loc[:, "diff(tradeBuyQty-tradeSellQty)"]
        ###feature71
        col[71] = (df["tradeBuyTurnover"] - df["tradeSellTurnover"]) / (
                    df["tradeBuyTurnover"] + df["tradeSellTurnover"])
        df.loc[:, "diff(tradeBuyTurnover-tradeSellTurnover)"] = deal_Na_Inf(col[71])
        feature["diff(tradeBuyTurnover-tradeSellTurnover)"] = df.loc[:, "diff(tradeBuyTurnover-tradeSellTurnover)"]
        #######加入6个手工特征
        ###feature72
        col[72] = (df["ask0"] / df["bid0"] - 1)
        df.loc[:, "(ask0/bid0)-1"] = deal_Na_Inf(col[72])
        feature["(ask0/bid0)-1"] = df.loc[:, "(ask0/bid0)-1"]
        ###feature73
        col[73] = (df["bid0"] * df["asize0"] + df["ask0"] * df["bsize0"]) / (df["bsize0"] + df["asize0"])
        df.loc[:, "wap0"] = deal_Na_Inf(col[73])
        feature["wap0"] = df.loc[:, "wap0"]
        ###feature74
        col[74] = (df["atr0_4"] / df["asize0_4"])
        df.loc[:, "atr0_4/asize0_4"] = deal_Na_Inf(col[74])
        feature["atr0_4/asize0_4"] = df.loc[:, "atr0_4/asize0_4"]
        ###feature75-"btr0_4/bsize0_4",
        col[75] = (df["btr0_4"] / df["bsize0_4"])
        df.loc[:, "btr0_4/bsize0_4"] = deal_Na_Inf(col[75])
        feature["btr0_4/bsize0_4"] = df.loc[:, "btr0_4/bsize0_4"]
        ###feature76 -"atr10_19/asize10_19"
        col[76] = (df["atr10_19"] / df["asize10_19"])
        df.loc[:, "atr10_19/asize10_19"] = deal_Na_Inf(col[76])
        feature["atr10_19/asize10_19"] = df.loc[:, "atr10_19/asize10_19"]
        ###feature77-"btr10_19/bsize10_19"
        col[77] = (df["btr10_19"] / df["bsize10_19"])
        df.loc[:, "btr10_19/bsize10_19"] = deal_Na_Inf(col[77])
        feature["btr10_19/bsize10_19"] = df.loc[:, "btr10_19/bsize10_19"]
        ###feature78 - "norm-tradeBuyQty" 
        # col[78] = df.groupby("symbol")["tradeBuyQty"].apply(normalize).reset_index()["tradeSellQty"]*10
        col[78] = df["tradeBuyQty"]
        df.loc[:, "norm-tradeBuyQty"] = deal_Na_Inf(10 * normalize(col[78]))
        feature["norm-tradeBuyQty"] = df.loc[:, "norm-tradeBuyQty"]
        ###feature79 - "norm-tradeBuyQty" 
        # col[79] = df.groupby("symbol")["tradeSellQty"].apply(normalize).reset_index()["tradeSellQty"]*10
        col[79] = df["tradeSellQty"]
        df.loc[:, "norm-tradeSellQty"] = deal_Na_Inf(10 * normalize(col[79]))
        feature["norm-tradeSellQty"] = df.loc[:, "norm-tradeSellQty"]
        ###feature80
        col[80] = (df["date"] + 4) % 7 + 1
        df.loc[:, "week-day"] = deal_Na_Inf(col[80])
        feature["week-day"] = df.loc[:, "week-day"]
        ###feature81
        col[81] = df.groupby('symbol').cumcount() + 1
        df.loc[:, "continue-time"] = col[81] / 10
        feature["continue-time"] = df.loc[:, "continue-time"]
        ### feature82
        df.loc[:, "bret12"] = (df["midpx"] - df["midpx"].shift(12)) / df["midpx"].shift(12)  # backward return
        cxbret = df.groupby("interval")[["bret12"]].mean().reset_index().rename(columns={"bret12": "cx_bret12"})
        df = df.merge(cxbret, on="interval", how="left")
        df.loc[:, "lagret12"] = df["bret12"] - df["cx_bret12"]

        xdf = df[self.mcols + self.featureNames()].set_index(self.mcols)
        ydf = df[self.mcols + [self.ycol]].set_index(self.mcols)

        print(df.columns)
        print(xdf.columns)
        print(ydf.columns)
        return xdf.fillna(0), ydf.fillna(0)
