import numpy as np

# 加载 npz 文件
npz_file = np.load('dataset/PEMS04/PEMS04.npz')

# 获取特定数组（假设数组名为 'data'）
array_data = npz_file['data']

# 查看数组的形状
shape = array_data.shape
print("数组形状:", shape)

# 查看数组的数据类型
dtype = array_data.dtype
print("数组数据类型:", dtype)

# 查看数组的前几行内容（如果数组较大）
if array_data.size > 10:
    print("数组前几行内容:", array_data[:10])
else:
    print("数组内容:", array_data)

# 关闭 npz 文件（虽然在大多数情况下 Python 会自动处理，但显式关闭是个好习惯）
npz_file.close()