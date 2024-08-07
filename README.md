# 股票时序数据预测——机器学习大作业

## 环境

Python: 3.10

参考meow/requirements.txt 进行环境配置

"pip install -r requirements.txt"

## 数据集

将数据集压缩文件放入meow/dataset/data_zip, 解压文件放于meow/dataset/h5路径下， 运行meow/dataset/h5_to_csv路径下的id.py， 获取h5文件名并写入txt ，运行meow/dataset/h5_to_csv路径下的data_process.py, 将所有h5文件转为csv文件， 放置于meow/dataset/csv路径下

## 文件路径说明

meow/dataset 数据集
meow/step* 每一步处理的代码（见名知意）
meow/test 测试文件或者数据，放临时文件

reference/pytorch 之前学习pytorch写的一点东西 可以参考这个pipeline
reference/Time-Series-Forecast 参考别人的代码仓库

.gitignore 忽略上传到git的文件格式

## 数据分析

### 主要代码文件：`step1_data_analysis/da.ipynb`

#### 1

author:赵子豪
对数据的简单处理
corr-20230601.txt 对20230601该日的数据的相关性分析

corr-20230601.txt 对20230601该日的数据的相关性分析的排序版,排序规则，按照corr绝对值进行降序

correlation-all.txt 对训练集日期的数据的相关性分析

correlation-all-sort.txt 对训练集日期的数据的相关性分析的排序版,排序规则，按照corr绝对值进行降序

feature-create.txt 对所有训练集日期，制造feature

目前的计算规则比较简单，主要是除法与取对数。
计算规则，对于任意两个属性A,B,分别计算(A/B),(B/A),log(A/B),log(B/A)和预测值fret12的corr属性（即相关系数），按照绝对值的降序排列

关于feature-create.txt的左边一列数据

log-nAddBuy-nCxlBuy   表示 log(nAddBuy/nCxlBuy)

low-open              表示 low/open

#### 2

更新了trend类型数据的计算。计算公式如下$\frac{ask0[i]-ask0[i-12]}{ask0[i-12]}$

其产生的文件下如下：

20230601-trend-sort.txt 对20230601单日的数据进行增减趋势分析。

corr-all-trend-sort.txt 对所有训练集的数据进行增减趋势分析。

顺便写了一个处理读入的dataframe的异常值的函数，对于异常值的填充更加合理化。后续可以添加在读入数据之后的处理之中。（注意，如果使用除法，要处理除数为0或者除之后为inf的情况）

```python
def dealDataFrame(df):
    """
    处理一个dataframe，修正其中的np.nan np.inf，使其变为与缺失值所在行股票编号相同的平均值。
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
```

#### 3

更新了 diff 类型数据的计算。计算公式 A,B为两列数据,$\frac{A-B}{A+B}$。

产生的文件如下
20230601-diff-sort.csv 对20230601单日的数据进行diff计算，并计算其与fret12的相关性。按照绝对值降序排列。

corr-all-diff-sort.csv 对所有数据进行diff计算，并计算其与fret12的相关性。按照绝对值降序排列。

我根据corr-all-diff-sort.csv的结果选出了7个feature，根据我个人的理解选择了9个feature，可以在feat_all_feature.py中看到我加入的feature。

#### 4

加入6个手工特征，思路来源于 <https://xueqiu.com/8188497048/198528860>

加入2个基础feature，"norm-tradeBuyQty","norm-tradeSellQty"。

加入2个时间feature

week-day ：表示星期几

continue-time : 表示开盘的时间（单位：min）/10

## 模型建立

author:胡景斌

简单整理了一下模型文件
meow/step3_model_train/lstm.py 为LSTM网络建立文件，可以在这里修改网络结构

meow/mdl.py为Ridge线性模型
meow/mdl_lstm.py为LSTM模型
meow/mdl_XGBoost.py为XGBoost模型

`python meow.py`训练Ridge线性回归模型 并进行推理
`python meow_lstm.py`训练LSTM模型 并进行推理
`python meow_XGBoost.py`训练XGBoost决策树模型 并进行推理

xgboost_model.json为存储的XGBoost模型的权重信息，可以直接读取进行推理

## 结果记录

hjb进行了模型测试和结果记录

LSTM（72 features）: Meow evaluation summary: Pearson correlation=0.107296385683008, R2=-0.000476972814861, MSE=0.000023766325315

Ridge（72 features）: Meow evaluation summary: Pearson correlation=0.131434137144724, R2=0.017256343648482, MSE=0.000023345070475

Ridge（78 features）:Meow evaluation summary: Pearson correlation=0.131807431710562, R2=0.017361341592724, MSE=0.000023342576249

XGBoost(72 features, 'n_estimators': 100,
            'booster': 'gbtree',
            'objective': 'reg:squarederror',
            'max_depth': 7,
            'lambda': 3,
            'subsample': 0.7,
            'colsample_bytree': 1,
            'min_child_weight': 3,
            'eta': 0.3,
            'seed': 1000,
            'nthread': 4):Meow evaluation summary: Pearson correlation=0.132621148659139, R2=0.015362901662225, MSE=0.000023390049179

XGBoost(82 features, 'n_estimators': 100,
            'booster': 'gbtree',
            'objective': 'reg:squarederror',
            'max_depth': 7,
            'lambda': 3,
            'subsample': 0.9,
            'colsample_bytree': 1,
            'min_child_weight': 3,
            'eta': 0.3,
            'seed': 1000,
            'nthread': 4):Meow evaluation summary: Pearson correlation=0.042762092957075, R2=-0.409263039603044, MSE=0.000033477036218

XGBoost(78 features(去掉时间和数量基础特征), 'n_estimators': 100,
            'booster': 'gbtree',
            'objective': 'reg:squarederror',
            'max_depth': 7,
            'lambda': 3,
            'subsample': 0.9,
            'colsample_bytree': 1,
            'min_child_weight': 3,
            'eta': 0.3,
            'seed': 1000,
            'nthread': 4):Meow evaluation summary: Pearson correlation=0.137069601064915, R2=0.017185462925854, MSE=0.000023346754246

### 修改记录

#### feat_all_feature

zzh在feat_all_feature中增加了部分特征，通过注释将他们进行了简单地分类。

留一个疑问，对于NA和INF的情况，填0，还是填1会比较合适？

对于log计算，为了避免log(x)，在x趋近于0的时候，log(x)接近负无穷，所以在x小于1的时候，置x=1,但是对于普通的除法，有必要在x<1的时候置1吗？

hjb在feat_all_features中修改了拼写错误

hjb在feat_all_features中再次修改了拼写错误，去掉了时间特征和基础特征，发现效果提升；加上这两种特征，效果下降很明显

### meow_lstm

hjb修改了输入通道数，目前为72，调整了部分参数

## 其他修改

1.zzh 在dl.py中，读取h5文件后面加了一行代码,`df = dealDataFrame(df)`，用于去除df中的na,inf。
