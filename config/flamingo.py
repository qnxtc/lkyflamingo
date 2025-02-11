# Our custom modules.
# Some config files require additional command line parameters to easily
# control agent or simulation hyperparameters during coarse parallelization.
import argparse
# Standard modules.
from datetime import timedelta
from math import floor
from sys import exit
from time import time

import numpy as np
import pandas as pd
# ML data and training
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Kernel import Kernel
from agent.flamingo.SA_ClientAgent import SA_ClientAgent as ClientAgent
###############################################
from agent.flamingo.SA_Manage import SA_Manage as Manage
from agent.flamingo.SA_ServiceAgent import SA_ServiceAgent as ServiceAgent
from model.LatencyModel import LatencyModel
from util import param
from util import util
####新增的####
import torch
from agent.flamingo.SA_ClientAgent import TrafficLSTM
####新增的###

# 1. 导入模块
# 标准库模块：
# argparse：用于解析命令行参数，方便用户在运行程序时指定不同的配置。
# datetime.timedelta：用于处理时间间隔。
# math.floor：用于向下取整。
# sys.exit：用于终止程序运行。
# time：用于记录时间，计算程序运行时长。
# numpy：用于进行高效的数值计算。
# pandas：用于数据处理和时间处理。
# 机器学习相关模块：
# pmlb.fetch_data：从 Penn Machine Learning Benchmark (PMLB) 数据集存储库中获取数据。
# sklearn.model_selection.train_test_split：用于将数据集划分为训练集和测试集。
# sklearn.preprocessing.StandardScaler：用于对数据进行标准化处理。
# 自定义模块：
# Kernel：可能是自定义的核心模拟内核。
# SA_ClientAgent、SA_Manage、SA_ServiceAgent：分别代表客户端代理、管理对象和服务端代理。
# LatencyModel：用于模拟代理之间的延迟。
# param 和 util：可能包含一些参数和工具函数。

parser = argparse.ArgumentParser(description='Detailed options for PPFL config.')
parser.add_argument('-a', '--clear_learning', action='store_true',
                    help='Learning in the clear (vs SMP protocol)')
parser.add_argument('-c', '--config', required=True,
                    help='Name of config file to execute')
parser.add_argument('-i', '--num_iterations', type=int, default=5,
                    help='Number of iterations for the secure multiparty protocol)')
parser.add_argument('-k', '--skip_log', action='store_true',
                    help='Skip writing agent logs to disk')
parser.add_argument('-l', '--log_dir', default=None,
                    help='Log directory name (default: unix timestamp at program start)')
parser.add_argument('-n', '--num_clients', type=int, default=5,
                    help='Number of clients for the secure multiparty protocol)')
parser.add_argument('-o', '--neighborhood_size', type=int, default=1,
                    help='Number of neighbors a client has (should only enter the multiplication factor of log(n))')
parser.add_argument('--round_time', type=int, default=10,
                    help='Fixed time the server waits for one round')
parser.add_argument('-s', '--seed', type=int, default=None,
                    help='numpy.random.seed() for simulation')
#############原来的################
# parser.add_argument('-t', '--dataset', default='car_evaluation',
#                     help='Set ML dataset')
#############原来的################
#############新增的################
parser.add_argument('-t', '--dataset', default='PEMS04',
                    help='Set ML dataset')
#############新增的################
#############原来的################
#parser.add_argument('-e', '--vector_length', type=int, default=80000,
#                   help='set input vector length')
#############原来的################
#############新增的################
parser.add_argument('-e', '--vector_length', type=int, default=35011,
                   help='set input vector length')
#############新增的################
parser.add_argument('-x', '--constant', type=int, default=100,
                    help='Constant +x for encoding')
parser.add_argument('-y', '--multiplier', type=int, default=16,
                    help='Multiplier 2^y for encoding')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Maximum verbosity!')
parser.add_argument('-p', '--parallel_mode', type=bool, default=True,
                    help='turn on parallel mode at server side')
parser.add_argument('-d', '--debug_mode', type=bool, default=False,
                    help='print debug info')
parser.add_argument('--config_help', action='store_true',
                    help='Print argument options for this config file')
###############################################
parser.add_argument('-m', '--manage_number', type=int, default=2,
                    help='manages numbers')
parser.add_argument('-fd', '--d_final_sum', action="store_true",
                    help='save final sum decode data')
parser.add_argument('-fe', '--e_final_sum', action="store_true",
                    help='save final sum encode data')

args, remaining_args = parser.parse_known_args()

if args.config_help:
    parser.print_help()
    exit()
# 2. 命令行参数解析
# 使用 argparse.ArgumentParser 创建一个参数解析器，通过 add_argument 方法添加各种命令行参数，如是否明文学习、配置文件名称、迭代次数等。
# parse_known_args 方法解析命令行参数，将解析后的参数存储在 args 中，未解析的参数存储在 remaining_args 中。
# 如果用户指定了 --config_help 参数，则打印参数帮助信息并退出程序。

# Historical date to simulate.  Required even if not relevant.
historical_date = pd.to_datetime('2023-01-01')

# Requested log directory.
log_dir = args.log_dir
skip_log = args.skip_log

# Random seed specification on the command line.  Default: None (by clock).
# If none, we select one via a specific random method and pass it to seed()
# so we can record it for future use.  (You cannot reasonably obtain the
# automatically generated seed when seed() is called without a parameter.)

# Note that this seed is used to (1) make any random decisions within this
# config file itself and (2) to generate random number seeds for the
# (separate) Random objects given to each agent.  This ensure that when
# the agent population is appended, prior agents will continue to behave
# in the same manner save for influences by the new agents.  (i.e. all prior
# agents still have their own separate PRNG sequence, and it is the same as
# before)

seed = args.seed
if not seed: seed = int(pd.Timestamp.now().timestamp() * 1000000) % (2 ** 32 - 1)
np.random.seed(seed)

dataset = args.dataset

# Config parameter that causes util.util.print to suppress most output.
util.silent_mode = not args.verbose
num_clients = args.num_clients
neighborhood_size = args.neighborhood_size
round_time = args.round_time
num_iterations = args.num_iterations
parallel_mode = args.parallel_mode
debug_mode = args.debug_mode

if not param.assert_power_of_two(num_clients):
    raise ValueError("Number of clients must be power of 2")
# 3. 初始化参数
# historical_date：模拟的历史日期。
# log_dir 和 skip_log：日志目录和是否跳过日志记录的标志。
# seed：随机数种子，如果用户未指定，则根据当前时间生成一个种子，并设置 numpy 的随机数种子。
# dataset：要使用的机器学习数据集。
# util.silent_mode：根据用户是否指定 --verbose 参数来设置是否静默模式。
# num_clients：客户端的数量，要求必须是 2 的幂次方，否则抛出异常。

# split_size = args.split_size
# max_logreg_iterations = args.max_logreg_iterations
# epsilon = args.epsilon
# learning_rate = args.learning_rate
# clear_learning = args.clear_learning
# collusion = args.collusion

### How many client agents will there be?   1000 in 125 subgraphs of 8 fits ln(n), for example
# num_subgraphs = args.num_subgraphs

print("Silent mode: {}".format(util.silent_mode))
print("Configuration seed: {}\n".format(seed))

# Since the simulator often pulls historical data, we use a real-world
# nanosecond timestamp (pandas.Timestamp) for our discrete time "steps",
# which are considered to be nanoseconds.  For other (or abstract) time
# units, one can either configure the Timestamp interval, or simply
# interpret the nanoseconds as something else.

# What is the earliest available time for an agent to act during the
# simulation?
midnight = historical_date
kernelStartTime = midnight

# When should the Kernel shut down?
kernelStopTime = midnight + pd.to_timedelta('2000:00:00')

# This will configure the kernel with a default computation delay
# (time penalty) for each agent's wakeup and recvMsg.  An agent
# can change this at any time for itself.  (nanoseconds)
defaultComputationDelay = 1000000000 * 0.1  # five seconds
# 4. 时间设置
# kernelStartTime 和 kernelStopTime：分别是模拟内核的开始时间和结束时间。
# defaultComputationDelay：每个代理唤醒和接收消息的默认计算延迟。

# IMPORTANT NOTE CONCERNING AGENT IDS: the id passed to each agent must:
#    1. be unique
#    2. equal its index in the agents list
# This is to avoid having to call an extra getAgentListIndexByID()
# in the kernel every single time an agent must be referenced.


### Configure the Kernel.
kernel = Kernel("Base Kernel",
                random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))

### Obtain random state for whatever latency model will be used.
latency_rstate = np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64'))
# 5. 内核和随机状态配置
# kernel：初始化模拟内核，设置名称和随机状态。
# latency_rstate：为延迟模型设置随机状态。


### Configure the agents.  When conducting "agent of change" experiments, the
### new agents should be added at the END only.
agent_count = 0
agents = []
agent_types = []

### What accuracy multiplier will be used?
accy_multiplier = 100000

### What will be the scale of the shared secret?
secret_scale = 1000000

### FOR MACHINE LEARNING APPLICATIONS: LOAD DATA HERE
#
#   The data should be loaded only once (for speed).  Data should usually be
#   shuffled, split into training and test data, and passed to the client
#   parties.
#
#   X_data should be a numpy array with column-wise features and row-wise
#   examples.  y_data should contain the same number of rows (examples)
#   and a single column representing the label.
#
#   Usually this will be passed through a function to shuffle and split
#   the data into the structures expected by the PPFL clients.  For example:
#   X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.25, random_state = shuffle_seed)
#   12345
###原来的（基于PMLB的分类数据）########
# X_input, y_input = fetch_data(dataset, local_cache_dir="dataset", return_X_y=True)
# scaler = StandardScaler()
# scaler.fit(X_input)
# X_input = scaler.transform(X_input)
###原来的（基于PMLB的分类数据）########

######新增的（时间序列数据加载）#######
def load_pems_data():
    data = np.load('F:\DATA\pycharm\lkyflamingo\dataset\PEMS04\PEMS04.npz')['data']  # shape: (16992, 307, 3)

    # 数据标准化
    scaler = StandardScaler()
    data_2d = data.reshape(-1, 3)
    scaled_data = scaler.fit_transform(data_2d).reshape(data.shape)

    # 构建时间序列样本
    seq_length = 12
    pred_length = 3
    X, y = [], []
    for i in range(len(data) - seq_length - pred_length):
        X.append(scaled_data[i:i + seq_length])
        y.append(scaled_data[i + seq_length:i + seq_length + pred_length, 0])  # 预测流量

    return np.array(X), np.array(y), scaler.mean_, scaler.scale_


# 加载数据
X_full, y_full, scaler_mean, scaler_scale = load_pems_data()

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=seed, shuffle=False
)

# 帮助集（用于模型初始化）
_, X_help, _, y_help = train_test_split(
    X_test, y_test, test_size=0.1, random_state=seed, shuffle=False
)
######新增的（时间序列数据加载）#######

if args.vector_length:
    input_length = args.vector_length
else:
    ####### 原来的 原MLP参数配置：#######
    #input_length = (X_input.shape[0] + X_input.shape[1]) * len(np.unique(y_input))
    ####### 原来的 原MLP参数配置：#######
    ######新增的# LSTM参数计算（示例模型）：######
    lstm_params = TrafficLSTM(
        input_size=3,
        hidden_size=64,
        num_layers=2,
        output_size=3
    )
    total_params = sum(p.numel() for p in lstm_params.parameters())
    input_length = total_params  # 根据实际模型结构自动计算
    ######新增的# LSTM参数计算（示例模型）：######
print("input length: ", input_length)
#####原来的#####
# 移除未使用的原MLP相关数据划分
# X_train, X_test, y_train, y_test = train_test_split(X_input, y_input, \
#                                                     test_size=0.25, \
#                                                     random_state=seed)
#
# nk = floor(X_train.shape[0] / num_clients)
# n = X_train.shape[0]
#
# # correct shape parameter help
# X_test, X_help, y_test, y_help = train_test_split(X_test, y_test, \
#                                                   test_size=0.1, random_state \
#                                                       =seed)
# 移除未使用的原MLP相关数据划分
#####原来的#####

# 6. 数据加载与预处理
# 使用 fetch_data 从 PMLB 数据集存储库中获取数据。
# 使用 StandardScaler 对数据进行标准化处理。
# 根据用户指定的 --vector_length 参数或数据的形状计算输入向量的长度。
# 使用 train_test_split 函数将数据划分为训练集和测试集，以及进一步划分出帮助集。

# Randomly shuffle and split the data for training and testing.
# X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.25)

#
#
### END OF LOAD DATA SECTION


agent_types.extend(["ServiceAgent"])
agent_count += 1

### Configure a population of cooperating learning client agents.
a, b = agent_count, agent_count + num_clients

### Configure a service agent.
agents.extend([ServiceAgent(
    id=0, name="PPFL Service Agent 0",
    type="ServiceAgent",
    random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')),
    msg_fwd_delay=0,
    users=[*range(a, b)],
    iterations=num_iterations,
    round_time=pd.Timedelta(f"{round_time}s"),
    num_clients=num_clients,
    neighborhood_size=neighborhood_size,
    parallel_mode=parallel_mode,
    debug_mode=debug_mode,
    input_length=input_length,

    ######原来的####
    # 移除原MLP相关的classes、X_test、y_test、X_help、y_help、nk、n参数
    # classes=np.unique(y_train),
    # X_test=X_test,
    # y_test=y_test,
    # X_help=X_help,
    # y_help=y_help,
    # nk=nk,
    # n=n,
    ######原来的####

    c=args.constant,
    m=args.multiplier,
    ####新增的####
    scaler_mean=scaler_mean,
    scaler_scale=scaler_scale,
    X_test=X_test,
    y_test=y_test
    ####新增的####
)])

client_init_start = time()

# Iterate over all client IDs.
# Client index number starts from 1.
for i in range(a, b):
    agents.append(ClientAgent(id=i,
                              name="PPFL Client Agent {}".format(i),
                              type="ClientAgent",
                              iterations=num_iterations,
                              num_clients=num_clients,
                              neighborhood_size=neighborhood_size,
                              # multiplier = accy_multiplier, X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                              # split_size = split_size, secret_scale = secret_scale,
                              debug_mode=debug_mode,
                              random_state=np.random.RandomState(
                                  seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')),
                              X_train=X_train,
                              y_train=y_train,
                              input_length=input_length,
                              #####原来的#####
                              # 移除原MLP相关的classes和nk参数
                              # classes=np.unique(y_train),
                              # nk=nk,
                              #####原来的#####
                              c=args.constant,
                              m=args.multiplier,
                              ))

agent_types.extend(["ClientAgent" for i in range(a, b)])
agent_count += num_clients

client_init_end = time()
init_seconds = client_init_end - client_init_start
td_init = timedelta(seconds=init_seconds)
print(f"Client init took {td_init}")
# 7. 代理配置
# 初始化服务端代理和客户端代理，并将它们添加到 agents 列表中。
# 记录客户端初始化的开始和结束时间，计算初始化所需的时间并打印。

### Configure a latency model for the agents.

# Get a new-style cubic LatencyModel from the networking literature.
pairwise = (len(agent_types), len(agent_types))

model_args = {'connected'  : True,

              # All in NYC.
              # Only matters for evaluating "real world" protocol duration,
              # not for accuracy, collusion, or reconstruction.
              'min_latency': np.random.uniform(low=10000000, high=100000000, size=pairwise),
              'jitter'     : 0.3,
              'jitter_clip': 0.05,
              'jitter_unit': 5,
              }

latency_model = LatencyModel(latency_model='cubic',
                             random_state=latency_rstate,
                             kwargs=model_args)
# 8. 延迟模型配置
# 具体怎么弄得？它和掉线率什么关系，怎么决定掉线率的？
# 定义延迟模型的参数，使用 LatencyModel 类初始化一个立方延迟模型。

###############################################
manages = list()
for m in range(1, args.manage_number + 1):
    manages.append(Manage(id=m,
                          name=f"manage_{m}",
                          type=None, ))
# 9. 管理对象配置
# 根据用户指定的 --manage_number 参数，初始化管理对象并添加到 manages 列表中。

# Start the kernel running.
results = kernel.runner(agents=agents,
                        manages=manages,
                        startTime=kernelStartTime,
                        stopTime=kernelStopTime,
                        agentLatencyModel=latency_model,
                        defaultComputationDelay=defaultComputationDelay,
                        skip_log=skip_log,
                        d_final_sum=args.d_final_sum,
                        e_final_sum=args.e_final_sum,
                        log_dir=log_dir)
# 10. 启动模拟
# 调用内核的 runner 方法启动模拟，传入代理列表、管理对象列表、
# 开始时间、结束时间、延迟模型等参数，并将模拟结果存储在 results 中。

# Print parameter summary and elapsed times by category for this experimental trial.
print()
print(f"######## Microbenchmarks ########")
print(f"Protocol Iterations: {num_iterations}, Clients: {num_clients}, ")

print()
print("Service Agent mean time per iteration (except setup)...")
print(f"    Report step:         {results['srv_report']}")
print(f"    Crosscheck step:     {results['srv_crosscheck']}")
print(f"    Reconstruction step: {results['srv_reconstruction']}")
print()
print("Client Agent mean time per iteration (except setup)...")
print(f"    Report step:         {results['clt_report'] / num_clients}")
print(f"    Crosscheck step:     {results['clt_crosscheck'] / param.committee_size}")
print(f"    Reconstruction step: {results['clt_reconstruction'] / param.committee_size}")
print()
# 11. 输出结果
# 打印模拟的参数总结和各个阶段的平均耗时，
# 包括服务端代理和客户端代理的报告步骤、交叉检查步骤和重建步骤的耗时。

#35011怎么来的：
# LSTM的参数计算：
#
# - LSTM层参数：4*(input_size*hidden_size + hidden_size^2 + hidden_size) * num_layers
#
# - 全连接层参数：hidden_size * output_size + output_size
#
# 假设input_size=3, hidden_size=64, num_layers=2, output_size=3：
#
# LSTM参数：2层 * 4*(3*64 + 64^2 +64) = 2*4*(192 + 4096 +64)= 2*4*4352=34816
#
# FC参数：64*3 +3 = 195
#
# 总参数：34816+195=35011。
#其实应该是default=total_params,
#######
# 示例运行命令：
# python flamingo.py \
#     -t PEMS04 \
#     -n 64 \          # 客户端数量
#     -i 2 \          # 迭代次数
#     -e 35011 \       # 根据实际模型参数调整
#     --round_time 30  # 延长轮次时间