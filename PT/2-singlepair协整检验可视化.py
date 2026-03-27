import akshare as ak
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
import matplotlib.pyplot as plt

# 定义股票代码
stock1 = "601398"  # 丰乐种业
stock2 = "601077"  # 万向德龙

# 定义获取数据的时间范围
start_date = "20230101"  # 起始日期
end_date = "20241231"    # 结束日期

# 获取股票1的历史数据
stock1_data = ak.stock_zh_a_hist(symbol=stock1, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
stock1_data.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume'}, inplace=True)
stock1_data['date'] = pd.to_datetime(stock1_data['date'])
stock1_data.set_index('date', inplace=True)

# 获取股票2的历史数据
stock2_data = ak.stock_zh_a_hist(symbol=stock2, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
stock2_data.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume'}, inplace=True)
stock2_data['date'] = pd.to_datetime(stock2_data['date'])
stock2_data.set_index('date', inplace=True)

# 合并数据
merged_data = pd.DataFrame({
    'close_1': stock1_data['close'],
    'close_2': stock2_data['close']
}).dropna()

# 对数价格序列
merged_data['log_close_1'] = np.log(merged_data['close_1'])
merged_data['log_close_2'] = np.log(merged_data['close_2'])

# 输出数据
print("股票1 (000713) 的历史收盘价：")
print(stock1_data[['close']].head())
print("\n股票2 (600371) 的历史收盘价：")
print(stock2_data[['close']].head())
print("\n合并后的数据:")
print(merged_data.head())

# 计算相关性
correlation = merged_data['log_close_1'].corr(merged_data['log_close_2'])
print(f"\n相关性: {correlation:.4f}")

# ADF检验
adf_result_1 = adfuller(merged_data['log_close_1'])
adf_result_2 = adfuller(merged_data['log_close_2'])

print(f"\nADF检验结果(资产1): p-值 = {adf_result_1[1]:.4f}")
print(f"ADF检验结果(资产2): p-值 = {adf_result_2[1]:.4f}")

if adf_result_1[1] < 0.05 and adf_result_2[1] < 0.05:
    print("两个资产的对数收盘价都是平稳的")
else:
    print("至少有一个资产的对数收盘价不是平稳的")

# 协整性检验
coint_result = coint(merged_data['log_close_1'], merged_data['log_close_2'])
coint_t, p_value, _ = coint_result

print(f"\n协整检验 t-统计量: {coint_t:.4f}")
print(f"协整检验 p-值: {p_value:.4f}")

if p_value < 0.05:
    print("两个资产存在协整关系")
else:
    print("两个资产不存在协整关系")


# 可视化
plt.figure(figsize=(14, 10))

# 对数价格序列可视化
plt.subplot(3, 1, 1)
merged_data['log_close_1'].plot(label=f'{stock1} Log Price')
merged_data['log_close_2'].plot(label=f'{stock2} Log Price')
plt.title('Log Price Series (Formation Period)')
plt.legend()

# 差分对数价格序列可视化
plt.subplot(3, 1, 2)
merged_data['log_close_1'].diff().plot(label=f'{stock1} Log Price Diff')
merged_data['log_close_2'].diff().plot(label=f'{stock2} Log Price Diff')
plt.title('Differenced Log Price Series (Formation Period)')
plt.legend()

# 价差序列可视化
plt.subplot(3, 1, 3)
spread = merged_data['log_close_1'] - merged_data['log_close_2']
spread.plot(label='Log Spread')
mu = spread.mean()
sd = spread.std()
plt.axhline(y=mu, color='black', linestyle='--', label='Mean')
plt.axhline(y=mu + 2 * sd, color='red', linestyle='--', label='Upper Bound')
plt.axhline(y=mu - 2 * sd, color='red', linestyle='--', label='Lower Bound')
plt.axhline(y=mu + 1.5 * sd, color='blue', linestyle='--', label='Upper Bound')
plt.axhline(y=mu - 1.5 * sd, color='blue', linestyle='--', label='Lower Bound')
plt.axhline(y=mu + 1 * sd, color='green', linestyle='--', label='Upper Bound')
plt.axhline(y=mu - 1 * sd, color='green', linestyle='--', label='Lower Bound')
plt.title('Log Spread Series (Formation Period)')
plt.legend()

plt.tight_layout()
plt.show()


"""

import akshare as ak
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
import matplotlib.pyplot as plt

# 定义股票代码
stock1 = "000713"  # 丰乐种业
stock2 = "600371"  # 万向德龙

# 定义获取数据的时间范围
start_date = "20230101"  # 起始日期
end_date = "20241231"    # 结束日期

# 获取股票1的历史数据
stock1_data = ak.stock_zh_a_hist(symbol=stock1, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
stock1_data.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume'}, inplace=True)
stock1_data['date'] = pd.to_datetime(stock1_data['date'])
stock1_data.set_index('date', inplace=True)

# 获取股票2的历史数据
stock2_data = ak.stock_zh_a_hist(symbol=stock2, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
stock2_data.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume'}, inplace=True)
stock2_data['date'] = pd.to_datetime(stock2_data['date'])
stock2_data.set_index('date', inplace=True)

# 合并数据
merged_data = pd.DataFrame({
    'close_1': stock1_data['close'],
    'close_2': stock2_data['close']
}).dropna()

# 输出数据
print("股票1 (601328) 的历史收盘价：")
print(stock1_data[['close']].head())
print("\n股票2 (601998) 的历史收盘价：")
print(stock2_data[['close']].head())
print("\n合并后的数据:")
print(merged_data.head())

# 计算相关性
correlation = merged_data['close_1'].corr(merged_data['close_2'])
print(f"\n相关性: {correlation:.4f}")

# 协整性检验
coint_result = coint(merged_data['close_1'], merged_data['close_2'])
coint_t, p_value, _ = coint_result

print(f"\n协整检验 t-统计量: {coint_t:.4f}")
print(f"协整检验 p-值: {p_value:.4f}")

if p_value < 0.05:
    print("两个资产存在协整关系")
else:
    print("两个资产不存在协整关系")

# ADF检验
adf_result_1 = adfuller(merged_data['close_1'])
adf_result_2 = adfuller(merged_data['close_2'])

print(f"\nADF检验结果(资产1): p-值 = {adf_result_1[1]:.4f}")
print(f"ADF检验结果(资产2): p-值 = {adf_result_2[1]:.4f}")

if adf_result_1[1] < 0.05 and adf_result_2[1] < 0.05:
    print("两个资产的收盘价都是平稳的")
else:
    print("至少有一个资产的收盘价不是平稳的")

# 可视化
plt.figure(figsize=(12, 6))
plt.plot(merged_data['close_1'], label='Asset1 (601328)')
plt.plot(merged_data['close_2'], label='Asset2 (601998)')
plt.title('Close Trend')
plt.xlabel('date')
plt.ylabel('close')
plt.legend()
plt.show()
"""