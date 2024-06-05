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
