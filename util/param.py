#这里可以改掉线率（readme文档里有说）
import math

import numpy as np
import pandas as pd
from Cryptodome.Cipher import ChaCha20
from Cryptodome.Random import get_random_bytes

# System parameters
vector_len = 16000
# 向量的长度，可能用于表示数据的维度。
vector_type = 'uint32'
# 向量的数据类型，这里是无符号 32 位整数。
committee_size = 60
# 委员会的大小，即委员会成员的数量。
fraction = 1 / 3
# 一个分数，可能用于某种比例计算。

# Waiting time
# Set according to a target dropout rate (e.g., 1%) 
# and message lantecy (see model/LatencyModel.py)
# 这个掉线率怎么设置的？
# 这些是不同阶段的等待时间，使用 pandas 的 Timedelta 对象表示。
# 注释中提到这些时间是根据目标掉线率和消息延迟设置的，不同的名称可能对应不同的协议或流程阶段。
#原版 wt_flamingo_report = pd.Timedelta('10s')
# 在第一个文件中修改（config.py）
wt_flamingo_report = pd.Timedelta('10s')  # 增大等待时间降低掉线率
wt_flamingo_crosscheck = pd.Timedelta('3s')
wt_flamingo_reconstruction = pd.Timedelta('3s')

wt_google_adkey = pd.Timedelta('10s')
wt_google_graph = pd.Timedelta('10s')
wt_google_share = pd.Timedelta('30s')  # ensure all user_choice received messages
wt_google_collection = pd.Timedelta('10s')
wt_google_crosscheck = pd.Timedelta('3s')
wt_google_recontruction = pd.Timedelta('2s')

# WARNING: 
# this should be a random seed from beacon service;
# we use a fixed one for simplicity
root_seed = get_random_bytes(32)
nonce = b'\x00\x00\x00\x00\x00\x00\x00\x00'
# 信标服务这里提到了
# root_seed：根种子，用于生成随机数。注释提醒这个种子应该来自信标服务，但为了简单起见，这里使用 get_random_bytes(32) 生成一个 32 字节的随机种子。
# nonce：一个固定的 8 字节的常量，在 ChaCha20 加密算法中作为随机数使用。

def assert_power_of_two(x):
    return (math.ceil(math.log2(x)) == math.floor(math.log2(x)));
# 这个函数用于判断一个数 x 是否为 2 的幂次方。它通过计算以 2 为底的对数的向上取整和向下取整，如果两者相等，则说明 x 是 2 的幂次方。

# choose committee members
def choose_committee(root_seed, committee_size, num_clients):
    prg_committee_holder = ChaCha20.new(key=root_seed, nonce=nonce)

    data = b"secr" * committee_size * 128
    prg_committee_bytes = prg_committee_holder.encrypt(data)
    committee_numbers = np.frombuffer(prg_committee_bytes, dtype=vector_type)

    user_committee = set()
    cnt = 0
    while (len(user_committee) < committee_size):
        sampled_id = committee_numbers[cnt] % num_clients
        (user_committee).add(sampled_id + 1)
        cnt += 1

    return user_committee
# 6. 选择委员会成员函数
# 功能：从 num_clients 个客户端中选择 committee_size 个客户端作为委员会成员。
# 步骤：
# 使用 ChaCha20 加密算法，以 root_seed 为密钥，nonce 为随机数，创建一个加密器 prg_committee_holder。
# 生成一个由字符串 "secr" 重复 committee_size * 128 次组成的字节序列 data，并使用加密器对其进行加密，得到 prg_committee_bytes。
# 将加密后的字节序列转换为 vector_type 类型的数组 committee_numbers。
# 创建一个空的集合 user_committee 用于存储委员会成员的 ID。
# 循环从 committee_numbers 中取出元素，对 num_clients 取模得到一个客户端 ID，将其加 1 后添加到 user_committee 中，直到集合的大小达到 committee_size。
# 返回委员会成员的 ID 集合。

# choose neighbors
def findNeighbors(root_seed, current_iteration, num_clients, id, neighborhood_size):
    neighbors_list = set()  # a set, instead of a list

    # compute PRF(root, iter_num), output a seed. can use AES
    prf = ChaCha20.new(key=root_seed, nonce=nonce)
    current_seed = prf.encrypt(current_iteration.to_bytes(32, 'big'))

    # compute PRG(seed), a binary string
    prg = ChaCha20.new(key=current_seed, nonce=nonce)

    # compute number of bytes we need for a graph
    # 知道这个字节数要干什么
    num_choose = math.ceil(math.log2(num_clients))  # number of neighbors I choose
    num_choose = num_choose * neighborhood_size

    bytes_per_client = math.ceil(math.log2(num_clients) / 8)
    segment_len = num_choose * bytes_per_client
    num_rand_bytes = segment_len * num_clients
    data = b"a" * num_rand_bytes
    graph_string = prg.encrypt(data)

    # find the segment for myself
    my_segment = graph_string[(id - 1) *
                              segment_len: (id - 1) * (segment_len) + segment_len]

    # define the number of bits within bytes_per_client that can be convert to int (neighbor's ID)
    bits_per_client = math.ceil(math.log2(num_clients))
    # default number of clients is power of two
    for i in range(num_choose):
        tmp = my_segment[i * bytes_per_client: i *
                                               bytes_per_client + bytes_per_client]
        tmp_neighbor = int.from_bytes(
            tmp, 'big') & ((1 << bits_per_client) - 1)

        if tmp_neighbor == id - 1:
            # print("client", self.id, " random neighbor choice happened to be itself, skip")
            continue
        if tmp_neighbor in neighbors_list:
            # print("client", self.id, "already chose", tmp_neighbor, "skip")
            continue
        neighbors_list.add(tmp_neighbor)

    # now we have a list for who I chose
    # find my ID in the rest, see which segment I am in. add to neighbors_list
    for i in range(num_clients):
        if i == id - 1:
            continue
        seg = graph_string[i * segment_len: i *
                                            (segment_len) + segment_len]
        ls = parse_segment_to_list(
            seg, num_choose, bits_per_client, bytes_per_client)
        if id - 1 in ls:
            # add current segment owner into neighbors_list
            neighbors_list.add(i)

    return neighbors_list
# 7. 查找邻居节点函数
# 功能：为指定 ID 的客户端查找其邻居节点。
# 步骤：
# 创建一个空的集合 neighbors_list 用于存储邻居节点的 ID。
# 使用 ChaCha20 加密算法，以 root_seed 为密钥，nonce 为随机数，对当前迭代次数 current_iteration 进行加密，得到当前种子 current_seed。
# 使用 current_seed 作为密钥，nonce 作为随机数，创建一个新的加密器 prg。
# 计算需要的字节数，生成一个由字符 "a" 重复组成的字节序列 data，并使用 prg 对其进行加密，得到 graph_string。
# 从 graph_string 中提取出当前客户端的分段 my_segment。
# 循环从 my_segment 中提取字节序列，将其转换为整数作为邻居节点的 ID，排除自身和已经选择过的邻居节点，将符合条件的邻居节点 ID 添加到 neighbors_list 中。
# 遍历所有客户端，对于每个客户端的分段，调用 parse_segment_to_list 函数将其解析为邻居节点 ID 列表，如果当前客户端的 ID 在该列表中，则将该客户端的 ID 添加到 neighbors_list 中。
# 返回邻居节点的 ID 集合。

def parse_segment_to_list(segment, num_choose, bits_per_client, bytes_per_client):
    cur_ls = set()
    # take a segment (byte string), parse it to a list
    for i in range(num_choose):
        cur_bytes = segment[i * bytes_per_client: i *
                                                  bytes_per_client + bytes_per_client]

        cur_no = int.from_bytes(cur_bytes, 'big') & (
                (1 << bits_per_client) - 1)

        cur_ls.add(cur_no)

    return cur_ls
# 8. 解析分段为列表函数
# 功能：将一个字节序列分段解析为邻居节点 ID 列表。
# 步骤：
# 创建一个空的集合 cur_ls 用于存储邻居节点的 ID。
# 循环从分段中提取字节序列，将其转换为整数，并进行位运算，得到邻居节点的 ID，将其添加到 cur_ls 中。
# 返回邻居节点的 ID 集合。