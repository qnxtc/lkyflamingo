import pandas as pd
import numpy as np


def csv_to_npz(csv_file_path, npz_file_path):
    try:
        # 尝试以 ZIP 格式读取文件
        df = pd.read_csv(csv_file_path, compression='zip')

        # 将 DataFrame 转换为 NumPy 数组
        data_array = df.values

        # 保存为 NPZ 文件
        # 使用 savez_compressed 进行压缩保存，可减小文件大小
        np.savez_compressed(npz_file_path, data=data_array)

        print(f"成功将 {csv_file_path} 转换为 {npz_file_path}")
    except FileNotFoundError:
        print(f"未找到文件: {csv_file_path}")
    except Exception as e:
        print(f"转换过程中出现错误: {e}")


# 指定 CSV 文件路径和要保存的 NPZ 文件路径
csv_file = 'dataset/PEMS04/PEMS04.csv.gz'
npz_file = 'dataset/PEMS04/PEMS04.npz'

# 调用函数进行转换
csv_to_npz(csv_file, npz_file)