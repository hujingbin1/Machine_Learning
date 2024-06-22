import os
import pandas as pd

def h5_to_csv(file_path):
    df = pd.read_hdf(file_path)
    csv_file_name = os.path.splitext(os.path.basename(file_path))[0] + '.csv'
    csv_file_name = "../csv/" + csv_file_name
    df.to_csv(csv_file_name, index=False)
    print(f"{file_path} 已成功转换为 {csv_file_name}")

def convert_folder_h5_to_csv(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".h5"):
            file_path = os.path.join(folder_path, filename)
            h5_to_csv(file_path)

# 指定包含H5文件的文件夹路径
folder_path = '.'  # 将此路径替换为你的H5文件所在文件夹的实际路径
convert_folder_h5_to_csv(folder_path)