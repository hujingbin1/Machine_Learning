import pandas as pd
import os

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
h5_dir = os.path.join(parent_dir,'h5')
all_entries = os.listdir(h5_dir)
files_h5 = [os.path.splitext(entry)[0] for entry in os.listdir(h5_dir) if entry.endswith('.h5')]
files_sorted = sorted(files_h5)

with open("./id.txt", "w", encoding='utf-8') as fp:
    for file_name in files_sorted:
        fp.write(file_name)
        fp.write("\n") 