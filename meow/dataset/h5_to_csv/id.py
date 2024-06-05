import pandas as pd
import os

# 获取当前目录
current_dir = os.getcwd()

# 获取上一级目录
parent_dir = os.path.dirname(current_dir)

h5_dir = os.path.join(parent_dir,'h5')

# 使用os.listdir()获取上级目录下的所有文件和目录名
all_entries = os.listdir(h5_dir)

# 获取上级目录下的所有.h5文件名，去掉后缀并排序
files_h5 = [os.path.splitext(entry)[0] for entry in os.listdir(h5_dir) if entry.endswith('.h5')]

# 按文件名升序排序
files_sorted = sorted(files_h5)

# 写入文件
with open("./id.txt", "w", encoding='utf-8') as fp:
    for file_name in files_sorted:
        fp.write(file_name)
        fp.write("\n") # 对每个文件名单独写入并添加换行符