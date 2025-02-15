import numpy as np
import matplotlib.pyplot as plt

# MAE 和 RMSE 数据
mae = [33255409.333333332, 82653344.66666667, 317356819.3333333, 198244005.95897436, 190010051.25128207, 165815512.58461538, 195971427.08717948, 180002906.30769232, 187891288.66666666, 182616974.9846154]
rmse = [33255409.33529726, 82653347.19373198, 317359034.64819527, 198313737.38006824, 190209462.71933645, 166067332.86469916, 196249814.0094036, 180076092.66190922, 187950383.75606135, 182696085.46099985]
rounds = np.arange(1, 11)

# 计算 MAE 的线性趋势线
mae_fit = np.polyfit(rounds, mae, 1)
mae_trend = np.poly1d(mae_fit)(rounds)

# 计算 RMSE 的线性趋势线
rmse_fit = np.polyfit(rounds, rmse, 1)
rmse_trend = np.poly1d(rmse_fit)(rounds)

# 绘制 MAE 和 RMSE 及其趋势线
plt.figure(figsize=(10, 6))
plt.plot(rounds, mae, label='MAE', marker='o')
plt.plot(rounds, mae_trend, label='MAE Trend', linestyle='--')
# plt.plot(rounds, rmse, label='RMSE', marker='s')
# plt.plot(rounds, rmse_trend, label='RMSE Trend', linestyle='--')
plt.xlabel('Rounds')
plt.ylabel('Error Value')
plt.title('MAE and RMSE over 10 Rounds')
plt.legend()
plt.grid(True)
plt.show()

# 输出趋势线的斜率
print(f"MAE 趋势线斜率: {mae_fit[0]}")
print(f"RMSE 趋势线斜率: {rmse_fit[0]}")