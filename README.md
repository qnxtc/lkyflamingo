[//]: # (火烈鸟（Flamingo）：多轮单服务器安全聚合及其在私有联邦学习中的应用)

[//]: # (火烈鸟是一个为私有联邦学习构建的安全聚合系统。)

[//]: # (此实现与我们的论文（https://eprint.iacr.org/2023/486）相关，该论文由马一平、杰西・伍兹、塞巴斯蒂安・安吉尔、安蒂戈尼・波利赫罗尼亚多和塔尔・拉宾撰写，发表于 2023 年的 IEEE S&P（奥克兰）会议。)

[//]: # (警告：这是一个学术概念验证原型，尚未达到可投入生产的程度。)

[//]: # (概述)

[//]: # (我们将代码集成到了 ABIDES（https://github.com/jpmorganchase/abides-jpmc-public）中，这是一个开源的高保真模拟器，专为金融市场（如股票交易所）的人工智能研究而设计。)

[//]: # (该模拟器支持数万个客户端与服务器进行交互以促进交易（在我们的案例中是进行求和计算）。)

[//]: # (它还支持可配置的成对网络延迟。)

[//]: # (火烈鸟协议按步骤（即往返）运行。)

[//]: # (一个步骤包括等待和处理消息。)

[//]: # (等待时间根据网络延迟分布和目标掉线率来设置。)

[//]: # (更多细节请参阅我们论文的第 8 节。)

[//]: # (main分支包含私有求和协议的代码，fedlearn分支包含机器学习模型私有训练的代码。)

[//]: # (安装说明)

[//]: # (我们建议使用Miniconda来设置环境。)

[//]: # (你可以通过以下命令下载 Miniconda：)

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

[//]: # (要安装 Miniconda，运行以下命令：)
```
bash Miniconda3-latest-Linux-x86_64.sh
```

[//]: # (如果你使用的是 bash，那么运行：)
```
source ~/.bashrc
```

[//]: # (现在创建一个包含 Python 3.9.12 的环境，然后激活它。)
```
conda create --name flamingo-v0 python=3.9.12
conda activate flamingo-v0
```

[//]: # (使用 pip 安装所需的包。)
```
pip install -r requirements.txt
```

[//]: # (私有求和)

[//]: # (代码位于main分支。)

[//]: # (首先进入pki_files文件夹，然后运行：)
```
python setup_pki.py
```

[//]: # (我们的程序有多个配置选项。)
```
-c [protocol name] 
-n [number of clients (power of 2)]
-i [number of iterations] 
-p [parallel or not] 
-o [neighborhood size (multiplicative factor of 2logn)] 
-d [debug mode, if on then print all info]
```

[//]: # (火烈鸟支持客户端数量为 128 及以上的 2 的幂次方，例如 128、256、512。)

[//]: # (示例命令：)
```
python abides.py -c flamingo -n 128 -i 1 -p 1 
```

[//]: # (如果你想打印出每个代理的信息，在上述命令中添加-d True。)
[//]: # (机器学习应用)

[//]: # (代码位于fedlearn分支。)

[//]: # (本仓库中使用的机器学习模型是一个多层感知器分类器（sklearn中的MLPClassfier），可以从pmlb 网站获取各种不同的数据集。用户可能希望自己实现更复杂的模型。)

[//]: # (除了上述配置选项外，我们还提供以下机器学习训练配置选项。)
```
-t [dataset name]
-s [random seed (optional)]
-e [input vector length]
-x [float-as-int encoding constant (optional)]
-y [float-as-int multiplier (optional)]
```

[//]: # (示例命令)
```
python abides.py -c flamingo -n 128 -i 5 -p 1 -t mnist 
```

[//]: # (附加信息)

[//]: # (服务器等待时间根据目标掉线率（1%）在util/param.py中设置。)

[//]: # (具体来说，对于一个目标掉线率，我们根据网络延迟来设置等待时间（见model/LatencyModel.py）。对于每次迭代，服务器总时间 = 服务器等待时间 + 服务器计算时间。)
## Acknowledgement
We thank authors of [MicroFedML](https://eprint.iacr.org/2022/714.pdf) for providing an example template of ABIDES framework.