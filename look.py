import pickle

import numpy as np

vector_type = 'uint32'
DATA = np.zeros((80000,), dtype=vector_type)

client_id = {1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32,
             34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
             61, 62, 63, 64}

check_type = "encode"  # encode or decode
rounds = 1  # 第几轮
for i in client_id:
    with open(f"log/client-{check_type}-{i}-{rounds}.pkl", "rb") as f:
        DATA += pickle.load(f)

print(DATA)

with open(f"log/server-{check_type}-0-{rounds}.pkl", "rb") as f:
    data = pickle.load(f)
    print(data)

cha = data - DATA
print(cha)
