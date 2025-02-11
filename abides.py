import argparse
import importlib

import numpy as np

np.errstate(divide='ignore', invalid='ignore')
# 这行代码使用 numpy 的 errstate 上下文管理器来设置在进行数值计算时遇到除法为零和无效操作（如 NaN 参与运算）的处理方式为忽略。
# 也就是说，当出现这些情况时，程序不会抛出异常，而是继续执行后续代码。

if __name__ == '__main__':
    # Print system banner.
    system_name = "ABIDES: Agent-Based Interactive Discrete Event Simulation"

    print("=" * (len(system_name) + 2))
    print(" " + system_name)
    print("=" * (len(system_name) + 2))
    print()
    #这段执行后就是这样：
    # ==========================================================
    #  ABIDES: Agent-Based Interactive Discrete Event Simulation
    # ==========================================================

    # Test command line parameters.  Only peel off the config file.
    # Anything else should be left FOR the config file to consume as agent
    # or experiment parameterization.
    parser = argparse.ArgumentParser(description='Simulation configuration.')
    parser.add_argument('-c', '--config', required=True,
                        help='Name of config file to execute')
    parser.add_argument('--config-help', action='store_true',
                        help='Print argument options for the specific config file.')
    # 5. 定义命令行参数解析器
    # argparse.ArgumentParser：创建一个命令行参数解析器对象，用于解析用户输入的命令行参数。description
    # 参数用于提供解析器的描述信息，在用户请求帮助时会显示。
    # parser.add_argument：添加具体的命令行参数。
    # -c 或 - -config：这是一个必需的参数，用于指定要执行的配置文件的名称。
    # --config - help：这是一个布尔标志参数，如果用户指定了该参数，则会打印特定配置文件的参数选项帮助信息。


    args, config_args = parser.parse_known_args()

    # 6. 解析命令行参数
    # 使用 parse_known_args 方法解析命令行参数。
    # args 包含解析后的已知参数，即我们在 add_argument 中定义的参数；
    # config_args 包含未被解析的额外参数，这些参数将留给配置文件去处理，可能用于配置代理或实验的参数。


    # First parameter supplied is config file.
    config_file = args.config
    # 7. 获取配置文件名称
    # 从解析后的参数 args 中获取用户指定的配置文件名称。

    config = importlib.import_module('config.{}'.format(config_file),
                                     package=None)
# 8. 动态导入配置模块
# 使用 importlib.import_module 函数动态导入配置模块。'config.{}'.format(config_file) 构造了要导入的模块的完整名称，假设配置文件都位于 config 包下。
# 导入后，config 变量将引用该模块，可以通过它来访问配置文件中定义的各种配置参数和函数。