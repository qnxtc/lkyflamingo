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
####新增的###
from torch.utils.data import Subset


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
parser.add_argument('-t', '--dataset', default='car_evaluation',
                    help='Set ML dataset')
parser.add_argument('-e', '--vector_length', type=int, default=80000,
                    help='set input vector length')
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
####新增的###
parser.add_argument('--non_iid', type=str, default='dirichlet', choices=['iid', 'dirichlet', 'pathological'],
                    help='Data partition method (IID, Dirichlet, or Pathological)')
parser.add_argument('--dir_alpha', type=float, default=0.1,
                    help='Alpha parameter for Dirichlet distribution (smaller means more heterogeneous)')
parser.add_argument('--patho_classes', type=int, default=2,
                    help='Number of classes per client in pathological non-IID')
####新增的###



args, remaining_args = parser.parse_known_args()

if args.config_help:
    parser.print_help()
    exit()






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

#####新增的#####

# 新增函数：基于Dirichlet分布的Non-IID划分
def dirichlet_split(labels, num_clients, alpha=0.5):
    num_classes = len(np.unique(labels))
    client_samples = [[] for _ in range(num_clients)]

    for class_id in range(num_classes):
        idx = np.where(labels == class_id)[0]
        np.random.shuffle(idx)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (proportions * len(idx)).astype(int)
        split_points = np.cumsum(proportions)[:-1]
        splits = np.split(idx, split_points)

        for client_id in range(num_clients):
            if len(splits) > client_id:
                client_samples[client_id].extend(splits[client_id].tolist())

    return client_samples


# 新增函数：极端标签划分（每个客户端仅有2类）
def pathological_split(labels, num_clients, num_classes_per_client=2):
    num_classes = len(np.unique(labels))
    client_indices = [[] for _ in range(num_clients)]

    for client_id in range(num_clients):
        selected_classes = np.random.choice(num_classes, num_classes_per_client, replace=False)
        for class_id in selected_classes:
            idx = np.where(labels == class_id)[0]
            np.random.shuffle(idx)
            client_indices[client_id].extend(idx[:len(idx) // num_clients].tolist())

    return client_indices

#####新增的#####




# 修改后的数据加载与划分逻辑
X_input, y_input = fetch_data(dataset, local_cache_dir="dataset", return_X_y=True)
scaler = StandardScaler()
scaler.fit(X_input)
X_input = scaler.transform(X_input)
# 修改后的数据加载与划分逻辑


if args.vector_length:
    input_length = args.vector_length
else:
    input_length = (X_input.shape[0] + X_input.shape[1]) * len(np.unique(y_input))

print("input length: ", input_length)

# 划分全局测试集
X_train, X_test, y_train, y_test = train_test_split(X_input, y_input, \
                                                    test_size=0.25, \
                                                    random_state=seed)
# 划分全局测试集

#####新增的###
# 确保 X_test 是二维数组
X_test = np.array(X_test)
if len(X_test.shape) == 1:
    X_test = X_test.reshape(1, -1)
elif len(X_test.shape) > 2:
    X_test = X_test.reshape(X_test.shape[0], -1)


# Non-IID划分逻辑
if args.non_iid == 'dirichlet':
    split_indices = dirichlet_split(y_train, num_clients=args.num_clients, alpha=args.dir_alpha)
elif args.non_iid == 'pathological':
    split_indices = pathological_split(y_train, num_clients=args.num_clients, num_classes=args.patho_classes)
else:  # IID
    split_indices = [np.random.choice(len(X_train), len(X_train) // args.num_clients) for _ in range(args.num_clients)]

###新增测试用的##
# 检查划分后的索引范围
for i, indices in enumerate(split_indices):
    # if max(indices) >= len(X_train):
    #     raise ValueError(f"Index {max(indices)} is out of bounds for X_train of size {len(X_train)} at client {i}")
    if indices:  # 检查 indices 是否为空
        if max(indices) >= len(X_train):
            raise ValueError(f"Index {max(indices)} is out of bounds for X_train of size {len(X_train)}")
    else:
        print("Warning: indices list is empty. Skipping this client.")
        # 或者根据具体情况进行其他处理，比如跳过该客户端的创建等
        continue  # 如果是在循环中，可以使用 continue 跳过本次循环

###新增测试用###



clients_data = [Subset(X_train, indices) for indices in split_indices]
#####新增的###

###新增测试用的##
# 在划分数据后添加调试信息
print(f"X_train shape: {X_train.shape}")
for i, client_data_idx in enumerate(split_indices):
    print(f"Client {i} data index range: min={min(client_data_idx)}, max={max(client_data_idx)}")
###新增测试用的##

nk = floor(X_train.shape[0] / num_clients)
n = X_train.shape[0]

# correct shape parameter help
X_test, X_help, y_test, y_help = train_test_split(X_test, y_test, \
                                                  test_size=0.1, random_state \
                                                      =seed)
###新增的###
# 再次确保 X_test 是二维数组
X_test = np.array(X_test)
if len(X_test.shape) == 1:
    X_test = X_test.reshape(1, -1)
elif len(X_test.shape) > 2:
    X_test = X_test.reshape(X_test.shape[0], -1)
###新增的###



# Randomly shuffle and split the data for training and testing.
# X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.25)

#
#
### END OF LOAD DATA SECTION

####新增的####
# 使用Dirichlet划分替换原有逻辑
# split_indices = dirichlet_split(y_train, num_clients=args.num_clients, alpha=0.1)  # alpha控制异构程度
# clients_data = [Subset(X_train, indices) for indices in split_indices]
####新增的####




agent_types.extend(["ServiceAgent"])
agent_count += 1

### Configure a population of cooperating learning client agents.
a, b = agent_count, agent_count + num_clients

### Configure a service agent.
agents.extend([ServiceAgent(
    id=0, name="PPFL Service Agent 0",
###新增的####
    clients_data=clients_data,
    y_train=y_train,
    kernel=kernel,  # 确保 kernel 对象被正确传递
###新增的####
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
    classes=np.unique(y_train),
    X_test=X_test,
    y_test=y_test,
    X_help=X_help,
    y_help=y_help,
    nk=nk,
    n=n,
    c=args.constant,
    m=args.multiplier,
)])

client_init_start = time()

####新增测试用的###
# 在划分数据后添加调试信息
print(f"X_train shape after splitting: {X_train.shape}")

# 在创建客户端代理之前添加调试信息
print(f"X_train shape before creating client agents: {X_train.shape}")
####新增测试用###



# Iterate over all client IDs.
# Client index number starts from 1.
for i in range(a, b):
    ####新增的####
    client_idx = split_indices[i - a]  # 假设split_indices是按客户端顺序生成的
    ####新增的####
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
                              ###原有的###
                              # X_train=X_train,
                              # y_train=y_train,
                              ###原有的###
                              input_length=input_length,
                              ####新增的###
                              client_data_idx=client_idx,  # 新增参数
                              ####新增的###
                              classes=np.unique(y_train),
                              ###新增的###
                              # X_train=X_train[client_idx],
                              # y_train=y_train[client_idx],
                              # 传递完整的 X_train 和 y_train
                              X_train=X_train,
                              y_train=y_train,
                              ###新增的###
                              nk=nk,
                              c=args.constant,
                              m=args.multiplier,
                              ))

agent_types.extend(["ClientAgent" for i in range(a, b)])
agent_count += num_clients

client_init_end = time()
init_seconds = client_init_end - client_init_start
td_init = timedelta(seconds=init_seconds)
print(f"Client init took {td_init}")

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

###############################################
manages = list()
for m in range(1, args.manage_number + 1):
    manages.append(Manage(id=m,
                          name=f"manage_{m}",
                          type=None, ))
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
