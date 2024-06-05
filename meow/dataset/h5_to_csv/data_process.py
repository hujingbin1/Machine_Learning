import pandas as pd
import os

with open('./id.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    
# 去除每行末尾的换行符
lines = [line.strip() for line in lines]

for filename in lines:
	# 加载H5文件
	store = pd.HDFStore('../h5/'+filename+'.h5')

	print(store.keys())

	df = store['h5']

	# 关闭HDF5存储
	store.close()

	# 将DataFrame保存为CSV文件
	df.to_csv('../csv/'+filename+'.csv', index=False)
	break
 	
