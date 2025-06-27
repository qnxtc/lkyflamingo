import pandas as pd
import os
from glob import glob


def read_summary_log():
    """读取并显示summary_log.bz2中的带宽统计信息"""
    # 指定日志目录（请修改为实际路径）
    # 方法1：使用原始字符串避免转义问题
    log_dir = r"F:\DATA\pycharm\lkyflamingo\log\1751065439"

    # 验证目录是否存在
    if not os.path.exists(log_dir):
        print(f"错误：目录不存在 - {log_dir}")
        return

    summary_file = os.path.join(log_dir, "summary_log.bz2")

    # 验证文件是否存在
    if not os.path.exists(summary_file):
        print(f"错误：未找到摘要日志文件 - {summary_file}")
        print("提示：请确认目录中存在summary_log.bz2文件")
        return

    try:
        # 读取摘要日志
        df = pd.read_pickle(summary_file)

        print("\n===== 摘要日志中的带宽统计 =====")
        # 查找系统级带宽统计行
        bandwidth_stats = df[(df['AgentID'] == -1) &
                             (df['AgentStrategy'] == 'System') &
                             (df['EventType'] == 'BandwidthSummary')]

        if not bandwidth_stats.empty:
            print(bandwidth_stats.iloc[0]['Event'])
        else:
            print("未找到带宽统计数据")

        print("\n===== 丢包统计 =====")
        # 查找丢包统计行
        loss_stats = df[(df['AgentID'] == -1) &
                        (df['AgentStrategy'] == 'System') &
                        (df['EventType'] == 'PacketLossSummary')]

        if not loss_stats.empty:
            print(loss_stats.iloc[0]['Event'])
        else:
            print("未找到丢包统计数据")

    except Exception as e:
        print(f"读取文件时出错: {e}")
        print("提示：可能是文件格式不兼容或损坏")


if __name__ == "__main__":
    read_summary_log()


###运行方式：修改要解压的文件路径，然后在”终端“处输入”python read_logs.py“即可