# other user-level crypto functions
import hashlib
import logging
import pickle
import time

#想换LSTM
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
#下面是原来的

from copy import deepcopy

import dill
import numpy as np
import pandas as pd
from Cryptodome.Cipher import AES, ChaCha20
from Cryptodome.Hash import SHA256
# pycryptodomex library functions
from Cryptodome.PublicKey import ECC
from Cryptodome.Random import get_random_bytes
from Cryptodome.Signature import DSS
# (原有的MLP依赖于这个) from sklearn.neural_network import MLPClassifier

from agent.Agent import Agent
from agent.flamingo.SA_ServiceAgent import SA_ServiceAgent as ServiceAgent
from message.Message import Message
from util import param
###############################################
from util.AesCrypto import aes_decrypt
from util.DiffieHellman import DHKeyExchange, mod_args
from util.crypto import ecchash
from util.crypto.secretsharing import secret_int_to_points


# 定义LSTM模型类（新增代码）
# 新增在文件顶部
class TrafficLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)  # out shape: (batch, seq_len, hidden)
        out = self.fc(out[:, -1, :])  # 取最后时间步
        return out

# The PPFL_TemplateClientAgent class inherits from the base Agent class.
class SA_ClientAgent(Agent):

    # Default param:
    # num of iterations = 4
    # key length = 32 bytes
    # neighbors ~ 2 * log(num per iter) 

    def __init__(self, id, name, type,
                 iterations=4,
                 key_length=32,
                 num_clients=128,
                 neighborhood_size=1,
                 debug_mode=0,
                 random_state=None,
                 X_train=None,
                 y_train=None,
                 input_length=1024,
                 classes=None,
                 nk=10,
                 c=100,
                 m=16):

        # Base class init
        super().__init__(id, name, type, random_state)

        # Iteration counter
        self.no_of_iterations = iterations
        self.current_iteration = 1
        self.current_base = 0

        # MLP inputs
        self.classes = classes
        self.nk = nk
        if (self.nk < len(self.classes)) or (self.nk >= X_train.shape[0]):
            print("nk is a bad size")
            exit(0)
        # classes 是分类任务中的类别标签。
        # nk 表示每次训练时所使用的本地数据量。
        # 代码会检查 nk 是否有效（即，nk 小于类别数量且小于训练数据的行数）。
        # 接下来的几个变量（如 global_coefs, global_int 等）用来存储全局模型的参数，
        # 这些参数在多轮迭代中会被更新。


        self.global_coefs = None
        self.global_int = None
        self.global_n_iter = None
        self.global_n_layers = None
        self.global_n_outputs = None
        self.global_t = None
        self.global_nic = None
        self.global_loss = None
        self.global_best_loss = None
        self.global_loss_curve = None
        self.c = c
        self.m = m

        # pick local training data
        self.prng = np.random.Generator(np.random.SFC64())
        obv_per_iter = self.nk  # math.floor(X_train.shape[0]/self.num_clients)
############原有的##############
        # self.trainX = [np.empty((obv_per_iter, X_train.shape[1]), dtype=X_train.dtype) for i in
        #                range(self.no_of_iterations)]
        # self.trainY = [np.empty((obv_per_iter,), dtype=X_train.dtype) for i in range(self.no_of_iterations)]
############原有的##############

############新增的##############
        # 新增时间序列参数
        self.seq_length = 12  # 输入序列长度
        self.pred_length = 3  # 预测长度
        self.feature_size = 3  # PEMS-04特征维度

        # 加载PEMS-04数据
        raw_data = np.load("pems04.npz")['data']  # shape: (16992, 307, 3)

        # 数据标准化
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(raw_data.reshape(-1, 3))

        # 构建时间序列样本
        X, y = [], []
        for i in range(len(scaled_data) - self.seq_length - self.pred_length):
            X.append(scaled_data[i:i + self.seq_length])
            y.append(scaled_data[i + self.seq_length:i + self.seq_length + self.pred_length, 0])  # 预测流量

        # 按迭代划分数据（保持原有结构）
        samples_per_iter = len(X) // self.no_of_iterations
        self.trainX = [X[i * samples_per_iter:(i + 1) * samples_per_iter] for i in range(self.no_of_iterations)]
        self.trainY = [y[i * samples_per_iter:(i + 1) * samples_per_iter] for i in range(self.no_of_iterations)]
############新增的##############
        # for i in range(self.no_of_iterations):
        #     # self.input.append(self.prng.integer(input_range));
        #     slice = self.prng.choice(range(X_train.shape[0]), size=obv_per_iter, replace=False)
        #     perm = self.prng.permutation(range(X_train.shape[0]))
        #     p = 0
        #     while (len(set(y_train[slice])) < len(self.classes)):
        #         if p >= X_train.shape[0]:
        #             print("Dataset does not have the # classes it claims")
        #             exit(0)
        #         add = [perm[p]]
        #         merge = np.concatenate((slice, add))
        #         if (len(set(y_train[merge])) > len(set(y_train[slice]))):
        #             u, c = np.unique(y_train[slice], return_counts=True)
        #             dup = u[c > 1]
        #             rm = np.where(y_train[slice] == dup[0])[0][0]
        #             slice = np.concatenate((add, np.delete(slice, rm)))
        #         p += 1
        #
        #     if (slice.size != obv_per_iter):
        #         print("n_k not going to be consistent")
        #         exit(0)
        #
        #     # Pull together the current local training set.
        #     self.trainX.append(X_train[slice].copy())
        #     self.trainY.append(y_train[slice].copy())

        # Set logger
        self.logger = logging.getLogger("Log")
        self.logger.setLevel(logging.INFO)
        if debug_mode:
            logging.basicConfig()

        """ Read keys. """
        # sk is used to establish pairwise secret with neighbors' public keys
        try:
            hdr = 'pki_files/client' + str(self.id - 1) + '.pem'
            f = open(hdr, "rt")
            self.key = ECC.import_key(f.read())
            self.secret_key = self.key.d
            f.close()
        except IOError:
            raise RuntimeError("No such file. Run setup_pki.py first.")

        # Read system-wide pk
        try:
            f = open('pki_files/system_pk.pem', "rt")
            system_key = ECC.import_key(f.read())
            f.close()
            self.system_pk = system_key.pointQ
        except IOError:
            raise RuntimeError("No such file. Run setup_pki.py first.")

        """ Set parameters. """
        self.num_clients = num_clients
        self.neighborhood_size = neighborhood_size
        # 原有的self.vector_len = input_length  # param.vector_len
        #新增的#
        self.vector_len = 256 * 256  # 根据实际参数量调整
        self.vector_dtype = np.float32  # 与PyTorch默认类型一致
        #新增的#
        self.vector_dtype = param.vector_type
        self.prime = ecchash.n
        self.key_length = key_length
        self.neighbors_list = set()  # neighbors
        self.cipher_stored = None  # Store cipher from server across steps

        """ Select committee. """
        self.user_committee = param.choose_committee(param.root_seed, param.committee_size, self.num_clients)
        self.committee_shared_sk = None
        self.committee_member_idx = None

        # If it is in the committee:
        # read pubkeys of every other client and precompute pairwise keys
        self.symmetric_keys = {}
        if self.id in self.user_committee:
            for i in range(num_clients):
                hdr = 'pki_files/client' + str(i) + '.pem'
                try:
                    f = open(hdr, "rt")
                    key = ECC.import_key(f.read())
                    pk = key.pointQ
                except IOError:
                    raise RuntimeError("No such file. Run setup_pki.py first.")

                self.symmetric_keys[i] = pk * self.secret_key  # group 
                self.symmetric_keys[i] = (int(self.symmetric_keys[i].x) & ((1 << 128) - 1)).to_bytes(16, 'big')  # bytes

        # Accumulate this client's run time information by step.
        self.elapsed_time = {'REPORT'        : pd.Timedelta(0),
                             'CROSSCHECK'    : pd.Timedelta(0),
                             'RECONSTRUCTION': pd.Timedelta(0),
                             }

        # State flag
        self.setup_complete = False

        self.numbers = 0

    # Simulation lifecycle messages.
    def kernelStarting(self, startTime):

        # Initialize custom state properties into which we will later accumulate results.
        # To avoid redundancy, we allow only the first client to handle initialization.
        if self.id == 1:
            self.kernel.custom_state['clt_report'] = pd.Timedelta(0)
            self.kernel.custom_state['clt_crosscheck'] = pd.Timedelta(0)
            self.kernel.custom_state['clt_reconstruction'] = pd.Timedelta(0)

        # Find the PPFL service agent, so messages can be directed there.
        self.serviceAgentID = self.kernel.findAgentByType(ServiceAgent)

        self.setComputationDelay(0)

        # Request a wake-up call as in the base Agent.  Noise is kept small because
        # the overall protocol duration is so short right now.  (up to one microsecond)
        super().kernelStarting(startTime +
                               pd.Timedelta(self.random_state.randint(low=0, high=1000), unit='ns'))

    def kernelStopping(self):

        # Accumulate into the Kernel's "custom state" this client's elapsed times per category.
        # Note that times which should be reported in the mean per iteration are already so computed.
        # These will be output to the config (experiment) file at the end of the simulation.

        self.kernel.custom_state['clt_report'] += (
                self.elapsed_time['REPORT'] / self.no_of_iterations)
        self.kernel.custom_state['clt_crosscheck'] += (
                self.elapsed_time['CROSSCHECK'] / self.no_of_iterations)
        self.kernel.custom_state['clt_reconstruction'] += (
                self.elapsed_time['RECONSTRUCTION'] / self.no_of_iterations)

        super().kernelStopping()

    # Simulation participation messages.
    def wakeup(self, currentTime):
        super().wakeup(currentTime)
        dt_wake_start = pd.Timestamp('now')
        self.sendVectors(currentTime)

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)

        # with signatures of other clients from the server
        if msg.body['msg'] == "COMMITTEE_SHARED_SK":
            dt_protocol_start = pd.Timestamp('now')
            self.committee_shared_sk = msg.body['sk_share']
            self.committee_member_idx = msg.body['committee_member_idx']

        elif msg.body['msg'] == "SIGN":
            if msg.body['iteration'] == self.current_iteration:
                dt_protocol_start = pd.Timestamp('now')
                self.cipher_stored = msg
                self.signSendLabels(currentTime, msg.body['labels'])
                self.recordTime(dt_protocol_start, 'CROSSCHECK')

        elif msg.body['msg'] == "DEC":
            if msg.body['iteration'] == self.current_iteration:
                dt_protocol_start = pd.Timestamp('now')

                if self.cipher_stored == None:
                    if __debug__: self.logger.info("did not recv sign")
                else:
                    if self.cipher_stored.body['iteration'] == self.current_iteration:
                        self.decryptSendShares(currentTime,
                                               self.cipher_stored.body['dec_target_pairwise'],
                                               self.cipher_stored.body['dec_target_mi'],
                                               self.cipher_stored.body['client_id_list'])

                self.cipher_stored = None
                self.recordTime(dt_protocol_start, 'RECONSTRUCTION')

        # End of the protocol / start the next iteration
        # Receiving the output from the server
        elif msg.body['msg'] == "REQ" and self.current_iteration != 0:
            start = time.time()
            #####新增的########
            # 获取全局参数并加载到模型
            global_vec = msg.body['global_weights']

            # 反量化参数
            float_vec = (global_vec / (2 ** self.m)) - self.c
            #####新增的########
            PRO = msg.body['PRO']
            final_sum = msg.body['final_sum']
            self.verify_result(PRO, final_sum)

            end = time.time()
            start_time = msg.body['start_time']

            self.kernel.handle_T2_time[self.id] = end - start
            self.kernel.handle_T3_time[self.id] = end - start_time
            ################原有的############
            # self.global_coefs = msg.body['coefs']
            # self.global_int = msg.body['ints']
            # self.global_n_iter = msg.body['n_iter']
            # self.global_n_layers = msg.body['n_layers']
            # self.global_n_outputs = msg.body['n_outputs']
            # self.global_t = msg.body['t']
            # self.global_nic = msg.body['nic']
            # self.global_loss = msg.body['loss']
            # self.global_best_loss = msg.body['best_loss']
            # self.global_loss_curve = msg.body['loss_curve']
            ################原有的############

            ################新增的############
            # 反序列化LSTM参数
            ptr = 0
            state_dict = self.model.state_dict()
            new_weights = {}
            for name, param in state_dict.items():
                size = param.numel()
                new_weights[name] = torch.FloatTensor(
                    msg.body['weights'][ptr:ptr + size].reshape(param.shape)
                )
                ptr += size
            self.global_weights = new_weights

            ################新增的############


            # End of the iteration
            # Reset temp variables for each iteration

            # Enter next iteration
            self.current_iteration += 1
            if self.current_iteration > self.no_of_iterations:
                return

            dt_protocol_start = pd.Timestamp('now')

            self.sendVectors(currentTime)
            self.recordTime(dt_protocol_start, "REPORT")

    ###################################
    # Round logics
    ###################################
    def sendVectors(self, currentTime):
        # 这部分代码进行的是本地模型的训练。
        # MLP 训练：使用 MLPClassifier 来训练本地数据。
        # 如果当前迭代不是第一次，它会加载全局模型的权重和偏置，并继续训练。
        # mlp 的训练过程包括初始化（或继续训练）并更新全局参数，如 global_coefs、global_int 等。

        dt_protocol_start = pd.Timestamp('now')

        # train local data
        #########原来的########
        # mlp = MLPClassifier()
        # # print("CURRENT ITERATION")
        # # print(self.current_iteration)
        # if self.current_iteration > 1:
        #     mlp = MLPClassifier(warm_start=True)
        #     mlp.coefs_ = self.global_coefs.copy()
        #     mlp.intercepts_ = self.global_int.copy()
        #
        #     mlp.n_iter_ = self.global_n_iter
        #     mlp.n_layers_ = self.global_n_layers
        #     mlp.n_outputs_ = self.global_n_outputs
        #     mlp.t_ = self.global_t
        #     mlp._no_improvement_count = self.global_nic
        #     mlp.loss_ = self.global_loss
        #     mlp.best_loss_ = self.global_best_loss
        #     mlp.loss_curve_ = self.global_loss_curve.copy()
        #     mlp.out_activation_ = "softmax"
        #
        # # num epochs
        # for j in range(5):
        #     mlp.partial_fit(self.trainX[self.no_of_iterations], self.trainY[self.no_of_iterations], self.classes)
        # padding = self.vector_len - 7 - ((mlp.n_layers_ - 1) * 3)  # - mlp.n_iter_
        # for z in range(mlp.n_layers_ - 1):
        #     padding = padding - mlp.coefs_[z].size
        #     padding = padding - mlp.intercepts_[z].size
        #
        # if padding < 0:
        #     print("Need more space to encode model weights, please adjust vector by:" + str(-1 * padding))
        #     exit(1)
        #
        # float_vec = np.concatenate((np.zeros(7), np.zeros((mlp.n_layers_ - 1) * 3)))
        # # ,np.array(mlp.loss_curve_).flatten()))
        # # print("fv1: ", len(float_vec))
        #
        # for z in range(mlp.n_layers_ - 1):
        #     float_vec = np.concatenate((float_vec, np.array(mlp.coefs_[z]).flatten()))
        #     # print("fv2: ", len(float_vec))
        # for z in range(mlp.n_layers_ - 1):
        #     float_vec = np.concatenate((float_vec, np.array(mlp.intercepts_[z]).flatten()))
        #     # print("fv3: ", len(float_vec))
        #
        # float_vec = np.concatenate((float_vec, np.zeros(padding)))
        # # print("fv4: ", len(float_vec))
        #
        # # vec = float_vec
        # vec = np.vectorize(lambda d: (d + self.c) * pow(2, self.m))(float_vec).astype(self.vector_dtype)
        # vec[0] = mlp.n_iter_
        # vec[1] = mlp.n_layers_
        # vec[2] = mlp.n_outputs_
        # vec[3] = mlp.t_
        # vec[4] = mlp._no_improvement_count
        # vec[5] = mlp.loss_
        # vec[6] = mlp.best_loss_
        #
        # x = 7
        # for z in range(mlp.n_layers_ - 1):
        #     # print("shape: ", z, mlp.coefs_[z].shape)
        #     vec[x] = mlp.coefs_[z].shape[0]
        #     # print(vec[x])
        #     x += 1
        #     vec[x] = mlp.coefs_[z].shape[1]
        #     # print(vec[x])
        #     x += 1
        # for z in range(mlp.n_layers_ - 1):
        #     # print("shape: ", z, mlp.intercepts_[z].shape)
        #     vec[x] = mlp.intercepts_[z].size
        #     # print(vec[x])
        #     x += 1
        #########原来的########
        #########新增的########
        self.model.train()

        # ================== 1. 从类属性中获取当前迭代数据 ==================
        current_iter_idx = self.current_iteration - 1
        X_np = np.array(self.trainX[current_iter_idx])  # shape: (samples, seq_len, features)
        y_np = np.array(self.trainY[current_iter_idx])  # shape: (samples, pred_length)

        # ================== 2. 转换为 PyTorch Tensor ==================
        X = torch.FloatTensor(X_np)  # shape: (N, seq_len, features)
        y = torch.FloatTensor(y_np)  # shape: (N, pred_length)

        # ================== 3. 初始化模型（如果是第一次迭代） ==================
        if not hasattr(self, 'model'):
            self.model = TrafficLSTM(
                input_size=X.shape[2],  # 特征维度
                hidden_size=64,
                output_size=y.shape[1]  # 预测长度
            )
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
            self.criterion = nn.MSELoss()

        # ================== 4. 加载全局权重（非首次迭代） ==================
        if self.current_iteration > 1 and self.global_weights is not None:
            self.model.load_state_dict(self.global_weights)

        # ================== 5. 训练循环 ==================
        self.model.train()
        batch_size = 32
        for epoch in range(5):
            # 打乱数据顺序
            permutation = torch.randperm(X.size(0))
            for i in range(0, X.size(0), batch_size):
                indices = permutation[i:i + batch_size]
                batch_X = X[indices]  # shape: (batch, seq, features)
                batch_y = y[indices]  # shape: (batch, pred_len)

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

        # ================== 6. 参数序列化（保持后续加密逻辑不变） ==================
        weight_vector = []
        for param in self.model.parameters():
            weight_vector.append(param.detach().numpy().flatten())
        float_vec = np.concatenate(weight_vector)

        # 后续的加密和发送逻辑保持不变...
        vec = np.vectorize(lambda d: (d + self.c) * pow(2, self.m))(float_vec).astype(self.vector_dtype)
        self.vector_len = len(vec)  # 动态调整向量长度

        #########新增的########

        #################验证聚合解码结果########################
        if self.kernel.d_final_sum:
            file_name = f"log/client-decode-{self.id}-{self.current_iteration}.pkl"
            with open(file_name, "wb") as f:
                pickle.dump(vec, f)
        ######################################################
        self.vec_n = deepcopy(vec)

        # Find this client's neighbors: parse graph from PRG(PRF(iter, root_seed))
        self.neighbors_list = param.findNeighbors(param.root_seed, self.current_iteration, self.num_clients, self.id,
                                                  self.neighborhood_size)
        if __debug__:
            self.logger.info("client indices in neighbors list starts from 0")
            self.logger.info(f"client {self.id} neighbors list: {self.neighbors_list}")

        # Download public keys of neighbors from PKI file
        # NOTE: the ABIDES framework has client id starting from 1. 
        neighbor_pubkeys = {}
        for id in self.neighbors_list:
            try:
                hdr = 'pki_files/client' + str(id) + '.pem'
                f = open(hdr, "rt")
                key = ECC.import_key(f.read())
                f.close()
            except IOError:
                raise RuntimeError("No such file. Run setup_pki.py first.")
            pk = key.pointQ
            neighbor_pubkeys[id] = pk

        # send symmetric encryption of shares of mi  
        mi_bytes = get_random_bytes(self.key_length)
        mi_number = int.from_bytes(mi_bytes, 'big')

        mi_shares = secret_int_to_points(secret_int=mi_number,
                                         point_threshold=int(param.fraction * len(self.user_committee)),
                                         num_points=len(self.user_committee), prime=self.prime)

        committee_pubkeys = {}
        for id in self.user_committee:
            try:
                hdr = 'pki_files/client' + str(id - 1) + '.pem'
                f = open(hdr, "rt")
                key = ECC.import_key(f.read())
                f.close()
            except IOError:
                raise RuntimeError("No such file. Run setup_pki.py first.")
            pk = key.pointQ
            committee_pubkeys[id] = pk

        # separately encrypt each share
        enc_mi_shares = []
        # id is the x-axis
        cnt = 0
        for id in self.user_committee:
            per_share_bytes = (mi_shares[cnt][1]).to_bytes(self.key_length, 'big')

            # can be pre-computed
            key_with_committee_group = self.secret_key * committee_pubkeys[id]
            key_with_committee_bytes = (int(key_with_committee_group.x) & ((1 << 128) - 1)).to_bytes(16, 'big')

            per_share_encryptor = AES.new(key_with_committee_bytes, AES.MODE_GCM)
            # nouce should be sent with ciphertext
            nonce = per_share_encryptor.nonce

            tmp, _ = per_share_encryptor.encrypt_and_digest(per_share_bytes)
            enc_mi_shares.append((tmp, nonce))
            cnt += 1

        # Compute mask, compute masked vector
        # PRG individual mask
        prg_mi_holder = ChaCha20.new(key=mi_bytes, nonce=param.nonce)
        data = b"secr" * self.vector_len
        prg_mi = prg_mi_holder.encrypt(data)

        # compute pairwise masks r_ij
        neighbor_pairwise_secret_group = {}  # g^{a_i a_j} = r_ij in group
        neighbor_pairwise_secret_bytes = {}

        for id in self.neighbors_list:
            neighbor_pairwise_secret_group[id] = self.secret_key * neighbor_pubkeys[id]
            # hash the g^{ai aj} to 256 bits (16 bytes)
            px = (int(neighbor_pairwise_secret_group[id].x)).to_bytes(self.key_length, 'big')
            py = (int(neighbor_pairwise_secret_group[id].y)).to_bytes(self.key_length, 'big')

            hash_object = SHA256.new(data=(px + py))
            neighbor_pairwise_secret_bytes[id] = hash_object.digest()[0:self.key_length]

        neighbor_pairwise_mask_seed_group = {}
        neighbor_pairwise_mask_seed_bytes = {}

        """Mapping group elements to bytes.
            compute h_{i, j, t} to be PRF(r_ij, t)
            map h (a binary string) to a EC group element
            encrypt the group element
            map the group element to binary string (hash the x, y coordinate)
        """
        for id in self.neighbors_list:
            round_number_bytes = self.current_iteration.to_bytes(16, 'big')

            h_ijt = ChaCha20.new(key=neighbor_pairwise_secret_bytes[id], nonce=param.nonce).encrypt(round_number_bytes)
            h_ijt = str(int.from_bytes(h_ijt[0:4], 'big') & 0xFFFF)

            # map h_ijt to a group element
            dst = ecchash.test_dst("P256_XMD:SHA-256_SSWU_RO_")
            neighbor_pairwise_mask_seed_group[id] = ecchash.hash_str_to_curve(msg=h_ijt, count=2,
                                                                              modulus=self.prime, degree=ecchash.m,
                                                                              blen=ecchash.L,
                                                                              expander=ecchash.XMDExpander(dst,
                                                                                                           hashlib.sha256,
                                                                                                           ecchash.k))

            px = (int(neighbor_pairwise_mask_seed_group[id].x)).to_bytes(self.key_length, 'big')
            py = (int(neighbor_pairwise_mask_seed_group[id].y)).to_bytes(self.key_length, 'big')

            hash_object = SHA256.new(data=(px + py))
            neighbor_pairwise_mask_seed_bytes[id] = hash_object.digest()[0:self.key_length]

        prg_pairwise = {}
        for id in self.neighbors_list:
            prg_pairwise_holder = ChaCha20.new(key=neighbor_pairwise_mask_seed_bytes[id], nonce=param.nonce)
            data = b"secr" * self.vector_len
            prg_pairwise[id] = prg_pairwise_holder.encrypt(data)

        """Client inputs.
            For machine learning, replace it with model weights.
            For testing, set to unit vector.
        """
        # vec = np.ones(self.vector_len, dtype=self.vector_dtype)

        # vectorize bytes: 32 bit integer, 4 bytes per component
        vec_prg_mi = np.frombuffer(prg_mi, dtype=self.vector_dtype)
        if len(vec_prg_mi) != self.vector_len:
            raise RuntimeError("vector length error")

        vec += vec_prg_mi
        vec_prg_pairwise = {}

        for id in self.neighbors_list:
            vec_prg_pairwise[id] = np.frombuffer(prg_pairwise[id], dtype=self.vector_dtype)

            if len(vec_prg_pairwise[id]) != self.vector_len:
                raise RuntimeError("vector length error")
            if self.id - 1 < id:
                vec = vec + vec_prg_pairwise[id]
            elif self.id - 1 > id:
                vec = vec - vec_prg_pairwise[id]
            else:
                raise RuntimeError("self id - 1 =", self.id - 1, " should not appear in neighbor_list",
                                   self.neighbors_list)

        # compute encryption of H(t)^{r_ij} (already a group element), only for < relation
        cipher_msg = {}

        for id in self.neighbors_list:
            # NOTE the set sent to the server is indexed from 0
            cipher_msg[(self.id - 1, id)] = self.elgamal_enc_group(self.system_pk,
                                                                   neighbor_pairwise_mask_seed_group[id])

        if __debug__:
            client_comp_delay = pd.Timestamp('now') - dt_protocol_start
            self.logger.info(f"client {self.id} computation delay for vector: {client_comp_delay}")
            self.logger.info(f"client {self.id} sends vector at {currentTime + client_comp_delay}")

        #################验证聚合编码结果########################
        if self.kernel.e_final_sum:
            file_name = f"log/client-encode-{self.id}-{self.current_iteration}.pkl"
            with open(file_name, "wb") as f:
                pickle.dump(vec, f)
        ######################################################
        self.en_vec_n = vec
        # Send the vector to the server
        self.serviceAgentID = 0
        ############原来的###############
        # self.sendMessage(self.serviceAgentID,
        #                  Message({"msg"          : "VECTOR",
        #                           "iteration"    : self.current_iteration,
        #                           "sender"       : self.id,
        #                           "vector"       : vec,
        #                           "enc_mi_shares": enc_mi_shares,
        #                           "enc_pairwise" : cipher_msg,
        #                           "layers"       : mlp.n_layers_,
        #                           "iter"         : mlp.n_iter_,
        #                           "out"          : mlp.n_outputs_,
        #                           }),
        #                  tag="comm_key_generation")
        ############原来的###############

        ############新增的###############
        self.sendMessage(self.serviceAgentID,
                         Message({"msg"          : "VECTOR",
                                  "iteration"    : self.current_iteration,
                                  "sender"       : self.id,
                                  "vector"       : vec,
                                  "enc_mi_shares": enc_mi_shares,
                                  "enc_pairwise" : cipher_msg,
                                  # 移除 layers, iter, out 参数
                                  "model_config" : {
                                      "model_type"  : "LSTM",
                                      "input_size" : self.model.input_size,
                                      "hidden_size": self.model.hidden_size,
                                      "num_layers" : self.model.num_layers,
                                      "output_size": self.model.output_size
                                  }
                                  }),
                         tag="comm_key_generation")
        ############新增的###############
        # print serialization size
        if __debug__:
            tmp_cipher_pairwise = {}
            for i in cipher_msg:
                tmp_cipher_pairwise[i] = (int(cipher_msg[i][0].x), int(cipher_msg[i][0].y),
                                          int(cipher_msg[i][1].x), int(cipher_msg[i][1].y))

            self.logger.info(
                f"[Client] communication for vectors at report step: {len(dill.dumps(vec)) + len(dill.dumps(enc_mi_shares)) + len(dill.dumps(tmp_cipher_pairwise))}")

    def signSendLabels(self, currentTime, msg_to_sign):
        dt_protocol_start = pd.Timestamp('now')

        msg_to_sign = dill.dumps(msg_to_sign)
        hash_container = SHA256.new(msg_to_sign)
        signer = DSS.new(self.key, 'fips-186-3')
        signature = signer.sign(hash_container)
        client_signed_labels = (msg_to_sign, signature)

        self.sendMessage(self.serviceAgentID,
                         Message({"msg"                 : "SIGN",
                                  "iteration"           : self.current_iteration,
                                  "sender"              : self.id,
                                  "signed_labels"       : client_signed_labels,
                                  "committee_member_idx": self.committee_member_idx,
                                  "signed_labels"       : client_signed_labels,
                                  }),
                         tag="comm_sign_client")

        if __debug__:
            self.logger.info(f"[Decryptor] communication for crosscheck step: {len(dill.dumps(client_signed_labels))}")

    def decryptSendShares(self, currentTime, dec_target_pairwise, dec_target_mi, client_id_list):

        dt_protocol_start = pd.Timestamp('now')

        if self.committee_shared_sk == None:
            if __debug__:
                self.logger.info(
                    f"Decryptor {self.committee_member_idx} is asked to decrypt, but does not have sk share.")
            self.sendMessage(self.serviceAgentID,
                             Message({"msg"                 : "NO_SK_SHARE",
                                      "iteration"           : self.current_iteration,
                                      "sender"              : self.id,
                                      "shared_result"       : None,
                                      "committee_member_idx": None,
                                      }),
                             tag="no_sk_share")
            return

            # CHECK SIGNATURES

        """Compute decryption of pairwise secrets.
            dec_target is a matrix
            just need to mult sk with each of the entry
            needs elliptic curve ops
        """
        dec_shares_pairwise = []
        dec_target_list_pairwise = list(dec_target_pairwise.values())

        for i in range(len(dec_target_list_pairwise)):
            c0 = dec_target_list_pairwise[i][0]
            dec_shares_pairwise.append(self.committee_shared_sk[1] * c0)

        """Compute decryption for mi shares.
            dec_target_mi is a list of AES ciphertext (with nonce)
            decrypt each entry of dec_target_mi
        """
        dec_shares_mi = []
        cnt = 0
        for id in client_id_list:
            sym_key = self.symmetric_keys[id - 1]
            dec_entry = dec_target_mi[cnt]
            nonce = dec_entry[1]
            cipher_holder = AES.new(sym_key, AES.MODE_GCM, nonce=nonce)
            plaintext = cipher_holder.decrypt(dec_entry[0])
            plaintext = int.from_bytes(plaintext, 'big')
            dec_shares_mi.append(plaintext)
            cnt += 1

        clt_comp_delay = pd.Timestamp('now') - dt_protocol_start

        if __debug__:
            self.logger.info(f"[Decryptor] run time for reconstruction step: {clt_comp_delay}")

        self.sendMessage(self.serviceAgentID,
                         Message({"msg"                   : "SHARED_RESULT",
                                  "iteration"             : self.current_iteration,
                                  "sender"                : self.id,
                                  "shared_result_pairwise": dec_shares_pairwise,
                                  "shared_result_mi"      : dec_shares_mi,
                                  "committee_member_idx"  : self.committee_member_idx,
                                  }),
                         tag="comm_secret_sharing")

        # print serialization cost
        if __debug__:
            tmp_msg_pairwise = {}
            for i in range(len(dec_shares_pairwise)):
                tmp_msg_pairwise[i] = (int(dec_shares_pairwise[i].x), int(dec_shares_pairwise[i].y))

            self.logger.info(
                f"[Decryptor] communication for reconstruction step: {len(dill.dumps(dec_shares_mi)) + len(dill.dumps(tmp_msg_pairwise))}")

    def elgamal_enc_group(self, system_pk, ptxt_point):
        # the order of secp256r1
        n = ecchash.n

        # ptxt is in ECC group
        enc_randomness_bytes = get_random_bytes(32)
        enc_randomness = (int.from_bytes(enc_randomness_bytes, 'big')) % n

        # base point in secp256r1
        base_point = ECC.EccPoint(ecchash.Gx, ecchash.Gy)

        c0 = enc_randomness * base_point
        c1 = ptxt_point + (system_pk * enc_randomness)
        return (c0, c1)

    # ======================== UTIL ========================

    def recordTime(self, startTime, categoryName):
        dt_protocol_end = pd.Timestamp('now')
        self.elapsed_time[categoryName] += dt_protocol_end - startTime

    def verify_init(self):
        self.manages_info = dict()
        self.kc = 0
        self.big_alpha = 0
        self.big_K = 0

        # # 创建一个用于 Diffie-Hellman 密钥交换的对象。
        self.dh_key_obj = DHKeyExchange(mod_args.q, mod_args.g)

    def send_public_to_manage(self):
        send_data = []
        for mange in self.kernel.manages:
            send_data.append({
                "c_id"    : self.id,
                "m_id"    : mange.id,
                "c_public": self.dh_key_obj.public_key
            })

        return send_data

    def generate_public(self, m_data: dict):
        if m_data["m_id"] not in self.manages_info.keys():
            self.manages_info[m_data["m_id"]] = dict()
        self.manages_info[m_data["m_id"]]["m_public"] = m_data["m_public"]
        km = self.dh_key_obj.compute_shared_secret(m_data["m_public"], mod_args.q)
        self.manages_info[m_data["m_id"]]["km"] = km

    def decrypt_big_k_alpha(self, c_data: dict):
        km = self.manages_info[c_data["m_id"]]["km"]
        decrypt_text = aes_decrypt(c_data["en_text"], km)
        big_K, alpha = map(lambda x: int(x), decrypt_text.split("&&"))
        self.manages_info[c_data["m_id"]]["big_K"] = big_K
        self.manages_info[c_data["m_id"]]["alpha"] = alpha

    def handle_km_alpha(self):
        for manage in self.manages_info.values():
            self.kc += manage["km"]
            self.big_alpha += manage["alpha"]
            self.big_K += manage["big_K"]

    def count_pro_c(self):
        self.pro_c = self.kc + self.big_alpha * self.vec_n
        self.kernel.clients_pro_len[self.id] = self.pro_c.nbytes
        return self.pro_c

    def verify_result(self, PRO, final_sum):
        #  aggregation_pro-K-α×final_sum=0
        result = PRO - self.big_K - self.big_alpha * final_sum
        if not np.all(result == 0):
            print(f"client {self.id} verify result 校验不通过！")
            exit(-1)
        else:
            # print(f"client {self.id} verify result 校验通过！")
            pass

        ############新增的#############
        # 在验证部分添加评估指标
        with torch.no_grad():
            test_X = torch.FloatTensor(self.scaler.transform(self.testX))
            preds = self.model(test_X).numpy()

        # 反标准化
        preds = self.scaler.inverse_transform(
            np.concatenate([preds, np.zeros((len(preds), 2))], axis=1)
        )[:, 0]
        true = self.scaler.inverse_transform(self.testY)[:, 0]

        mae = np.mean(np.abs(preds - true))
        print(f"Client {self.id} MAE: {mae:.2f}")
        ############新增的#############
