# **************************************
# --*-- coding: utf-8 --*--
# @Time    : 2024-10-21
# @Author  : white
# @FileName: SA_Manage.py
# @Software: PyCharm
# **************************************
import random

from Cryptodome.PublicKey import ECC

from message.Message import MessageType
from message.new_msg import ReqMsg
from util.AesCrypto import aes_encrypt
from util.DiffieHellman import DHKeyExchange, mod_args
import time

class SA_Manage:
    def __init__(self, id, name, type, key_length=32):
        self.id = id
        self.name = name
        try:
            f = open(f'pki_files/manager{self.id}.pem', "rt")
            self.key = ECC.import_key(f.read())
            f.close()
        except IOError:
            raise RuntimeError("No such file. Run setup_pki.py first.")
        # 把对应的私钥赋值给self.secret_key
        self.secret_key = self.key.d

        # self.private_key = self.key.export_key(format='PEM')  # 以 PEM 格式导出私钥。
        # self.private_key = random.randint(10, 20)  # 以 PEM 格式导出私钥。
        self.public_key = self.key.public_key().export_key(format='PEM')  # 以 PEM 格式导出公钥。

    # 聚合Km,n,生成Km
    def aggregation_clients_public_key(self):
        # 条件检查 self.clients_dh_key 中的公钥数量是否与 self.kernel.all_clients 中的客户端数量相等。
        # self.clients_dh_key就是管理者根据客户端公钥生成的共享密钥
        # 打印日志信息，表示开始聚合客户端的公钥。文件路径、谁在执行操作、日志信息
        if __debug__:
            print(__file__, "\t", self.id, "\t开始聚合所有客户端key！")
        for key in self.clients_dh_key.values():
            self.agg_clients_keys += key["client_key"]
        if __debug__:
            print(f"{self.id} 聚合结果:{self.agg_clients_keys}")

    def send_dh_public_key(self, client_id, client_dh_key):
        # 管理者在 self.clients_dh_key 中保存收到的客户端公钥，键为 client_id，值为客户端的 Diffie-Hellman 公钥。
        # 计算管理者与客户端之间的共享密钥。
        shared_key = self.dh_key_obj.compute_shared_secret(client_dh_key, mod_args.q)
        # 管理者将计算出的共享密钥保存到 self.dh_key_obj.shared_key 中。
        self.clients_dh_key[client_id] = {
            "client_key": client_dh_key,
            "shared_key": shared_key,
        }

        # 将一轮训练里产生的每一个共享密钥都保存起来。
        # self.dh_key_obj.shared_keys = getattr(self.dh_key_obj, 'shared_keys', []) + [shared_key]

        # 管理者对共享密钥求和生成K管
        # self.Manage_sum_key = getattr(self, 'Manage_dh_key', 0) + shared_key
        self.kernel.prove_queue.put((
            MessageType.MANAGE_SWITCH_PUBLIC,
            # 管理者接收到客户端的公钥后，计算双方的共享密钥，并将自己的公钥发送回客户端。
            ReqMsg(manage_id=self.id,  # 管理者的 ID。
                   dh_public_key=self.dh_key_obj.public_key,  # 管理者的 Diffie-Hellman 公钥。
                   client_id=client_id)  # 客户端的 ID。
        ))

    def kernelInitializing(self, kernel):
        self.kernel = kernel

    def kernelStarting(self, time):
        pass

    def wakeup(self, currentTime):
        pass

    def receiveMessage(self, currentTime, msg):
        pass

    ###############################################
    def verify_init(self, c_public_keys: list):
        # 创建一个 DHKeyExchange 对象，用于 Diffie-Hellman 密钥交换，使用的参数为 mod_args.q 和 mod_args.g，以及实例的私钥。
        self.dh_key_obj = DHKeyExchange(mod_args.q, mod_args.g)

        self.big_K = 0
        self.alpha = random.randint(10, 100)
        self.clients_info = dict()

        send_data = list()
        for s_data in c_public_keys:
            for m_data in s_data:
                if self.id != m_data["m_id"]: continue
                if m_data["c_id"] not in self.clients_info.keys():
                    self.clients_info[m_data["c_id"]] = dict()
                start = time.time()
                self.clients_info[m_data["c_id"]]["c_public"] = m_data["c_public"]
                km = self.dh_key_obj.compute_shared_secret(m_data["c_public"], mod_args.q)
                self.clients_info[m_data["c_id"]]["km"] = km
                send_data.append({
                    "m_id"    : self.id,
                    "c_id"    : m_data["c_id"],
                    "m_public": self.dh_key_obj.public_key,
                })
                end = time.time()
                self.kernel.handle_T1_time[m_data["c_id"]].append(end - start)

        return send_data

    def count_km(self):
        for client in self.clients_info.values():
            self.big_K += client["km"]

    # 发送同态加密后的密文ct。ct包括Km和manage_alpha
    def send_cipher_text(self, line_clients: list):
        e_data = list()
        for c_id in line_clients:
            start = time.time()
            encrypt_text = f"{self.big_K}&&{self.alpha}"
            km = self.clients_info[c_id]["km"]
            encrypt_result = aes_encrypt(encrypt_text, km)
            e_data.append({
                "m_id"   : self.id,
                "c_id"   : c_id,
                "en_text": encrypt_result
            })
            end = time.time()
            self.kernel.handle_T1_time[c_id].append(end - start)

        return e_data
