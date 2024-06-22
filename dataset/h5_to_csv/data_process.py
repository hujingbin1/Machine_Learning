import pandas as pd
import os

with open('./id.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    
lines = [line.strip() for line in lines]

for filename in lines:
	store = pd.HDFStore('../h5/'+filename+'.h5')
	print(store.keys())
	df = store['h5']
	store.close()
	df.to_csv('../csv/'+filename+'.csv', index=False)
	break

