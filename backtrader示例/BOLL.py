"""
基于股票的BOLL策略(第一个不考虑手续费和滑点,第二个考虑手续费和滑点)

import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 1. 确定股票池和回测时间
stock_code = "000002"  # 示例股票代码，平安银行
start_date = "20210514"
end_date = "20211018"

# 获取股票数据
df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
df['日期'] = pd.to_datetime(df['日期'])
df.set_index('日期', inplace=True)

# 2. 计算布林带指标
window = 20
k = 2
df['MA'] = df['收盘'].rolling(window=window).mean()
df['STD'] = df['收盘'].rolling(window=window).std()
df['Upper'] = df['MA'] + k * df['STD']
df['Lower'] = df['MA'] - k * df['STD']

# 3. 确定买卖信号
df['Signal'] = 0
df.loc[df['收盘'] > df['Upper'], 'Signal'] = 1  # 价格突破上轨，买入信号
df.loc[df['收盘'] < df['Lower'], 'Signal'] = -1  # 价格突破下轨，卖出信号

# 4. 回测
initial_capital = 100000  # 初始资金
capital = initial_capital
positions = 0  # 持仓数量
df['Portfolio'] = capital  # 组合价值
df['Benchmark'] = df['收盘'] / df['收盘'].iloc[0] * initial_capital  # 基准指数的收益率

for i in range(1, len(df)):
    if df['Signal'].iloc[i] == 1:  # 买入信号
        positions = capital / df['收盘'].iloc[i]
        capital = 0
    elif df['Signal'].iloc[i] == -1:  # 卖出信号
        capital = positions * df['收盘'].iloc[i]
        positions = 0
    df.at[df.index[i], 'Portfolio'] = capital + positions * df['收盘'].iloc[i]

# 计算收益指标
df['Strategy_Return'] = df['Portfolio'].pct_change()
df['Benchmark_Return'] = df['Benchmark'].pct_change()

# 年化收益率
annualized_return = (df['Portfolio'].iloc[-1] / df['Portfolio'].iloc[0]) ** (252 / len(df)) - 1
benchmark_annualized_return = (df['Benchmark'].iloc[-1] / df['Benchmark'].iloc[0]) ** (252 / len(df)) - 1

# 阿尔法、贝塔
beta = df['Strategy_Return'].cov(df['Benchmark_Return']) / df['Benchmark_Return'].var()
alpha = df['Strategy_Return'].mean() - beta * df['Benchmark_Return'].mean()

# 夏普比率
sharpe_ratio = df['Strategy_Return'].mean() / df['Strategy_Return'].std() * np.sqrt(252)

# 胜率、盈亏比
win_rate = (df['Strategy_Return'] > 0).sum() / len(df['Strategy_Return'])
profit_loss_ratio = df['Strategy_Return'][df['Strategy_Return'] > 0].mean() / df['Strategy_Return'][df['Strategy_Return'] < 0].mean()

# 收益波动率
volatility = df['Strategy_Return'].std() * np.sqrt(252)

# 最大回撤
roll_max = df['Portfolio'].cummax()
daily_drawdown = df['Portfolio'] / roll_max - 1.0
max_drawdown = daily_drawdown.min()

# 输出收益概况
print(f"收益率: {df['Portfolio'].iloc[-1] / df['Portfolio'].iloc[0] - 1:.2%}")
print(f"年化收益率: {annualized_return:.2%}")
print(f"基准收益率: {benchmark_annualized_return:.2%}")
print(f"阿尔法: {alpha:.4f}")
print(f"贝塔: {beta:.4f}")
print(f"夏普比率: {sharpe_ratio:.2f}")
print(f"胜率: {win_rate:.2%}")
print(f"盈亏比: {profit_loss_ratio:.2f}")
print(f"收益波动率: {volatility:.2%}")
print(f"最大回撤: {max_drawdown:.2%}")

# 绘制收益率图像
plt.figure(figsize=(14, 7))
plt.plot(df['Portfolio'] / df['Portfolio'].iloc[0] - 1, label='Strategy Return')
plt.plot(df['Benchmark'] / df['Benchmark'].iloc[0] - 1, label='Benchmark Return')
plt.plot((df['Portfolio'] / df['Portfolio'].iloc[0] - df['Benchmark'] / df['Benchmark'].iloc[0]), label='Relative Return')
plt.legend()
plt.title('Strategy Performance')
plt.xlabel('Date')
plt.ylabel('Return')
plt.show()
"""

"""
考虑手续费【买入手续费（buy_cost）: 0.0003（即0.03%）
卖出手续费（sell_cost）: 0.0013（即0.13%）
最小手续费（min_cost）: 5（即每笔交易的最低手续费为5单位货币）】滑点设为0.1%
"""

"""
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 1. 确定期货代码和回测时间
stock_code = "000001"  # 示例股票代码，平安银行
start_date = "20210101"
end_date = "20241231"

# 获取期货数据
df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
df['日期'] = pd.to_datetime(df['日期'])
df.set_index('日期', inplace=True)

# 2. 计算布林带指标
window = 20
k = 2
df['MA'] = df['收盘'].rolling(window=window).mean()
df['STD'] = df['收盘'].rolling(window=window).std()
df['Upper'] = df['MA'] + k * df['STD']
df['Lower'] = df['MA'] - k * df['STD']

# 3. 确定买卖信号
df['Signal'] = 0
df.loc[df['收盘'] > df['Upper'], 'Signal'] = 1  # 价格突破上轨，买入信号
df.loc[df['收盘'] < df['Lower'], 'Signal'] = -1  # 价格突破下轨，卖出信号

# 4. 回测
initial_capital = 100000  # 初始资金
capital = initial_capital
positions = 0  # 持仓数量
df['Portfolio'] = capital  # 组合价值
df['Benchmark'] = df['收盘'] / df['收盘'].iloc[0] * initial_capital  # 基准指数的收益率

# 交易成本参数
buy_cost = 0.0003  # 买入手续费
sell_cost = 0.0013  # 卖出手续费
min_cost = 5  # 最小手续费
slippage = 0.001  # 滑点

for i in range(1, len(df)):
    if df['Signal'].iloc[i] == 1:  # 买入信号
        # 计算买入成本
        buy_price = df['收盘'].iloc[i] * (1 + slippage)
        cost = max(buy_price * positions * buy_cost, min_cost)
        positions = (capital - cost) / buy_price
        capital = 0
    elif df['Signal'].iloc[i] == -1:  # 卖出信号
        # 计算卖出成本
        sell_price = df['收盘'].iloc[i] * (1 - slippage)
        cost = max(sell_price * positions * sell_cost, min_cost)
        capital = positions * sell_price - cost
        positions = 0
    df.at[df.index[i], 'Portfolio'] = capital + positions * df['收盘'].iloc[i]

# 计算收益指标
df['Strategy_Return'] = df['Portfolio'].pct_change()
df['Benchmark_Return'] = df['Benchmark'].pct_change()

# 年化收益率
annualized_return = (df['Portfolio'].iloc[-1] / df['Portfolio'].iloc[0]) ** (252 / len(df)) - 1
benchmark_annualized_return = (df['Benchmark'].iloc[-1] / df['Benchmark'].iloc[0]) ** (252 / len(df)) - 1

# 阿尔法、贝塔
beta = df['Strategy_Return'].cov(df['Benchmark_Return']) / df['Benchmark_Return'].var()
alpha = df['Strategy_Return'].mean() - beta * df['Benchmark_Return'].mean()

# 夏普比率
sharpe_ratio = df['Strategy_Return'].mean() / df['Strategy_Return'].std() * np.sqrt(252)

# 胜率、盈亏比
win_rate = (df['Strategy_Return'] > 0).sum() / len(df['Strategy_Return'])
profit_loss_ratio = df['Strategy_Return'][df['Strategy_Return'] > 0].mean() / df['Strategy_Return'][df['Strategy_Return'] < 0].mean()

# 收益波动率
volatility = df['Strategy_Return'].std() * np.sqrt(252)

# 最大回撤
roll_max = df['Portfolio'].cummax()
daily_drawdown = df['Portfolio'] / roll_max - 1.0
max_drawdown = daily_drawdown.min()

# 输出收益概况
print(f"收益率: {df['Portfolio'].iloc[-1] / df['Portfolio'].iloc[0] - 1:.2%}")
print(f"年化收益率: {annualized_return:.2%}")
print(f"基准收益率: {benchmark_annualized_return:.2%}")
print(f"阿尔法: {alpha:.4f}")
print(f"贝塔: {beta:.4f}")
print(f"夏普比率: {sharpe_ratio:.2f}")
print(f"胜率: {win_rate:.2%}")
print(f"盈亏比: {profit_loss_ratio:.2f}")
print(f"收益波动率: {volatility:.2%}")
print(f"最大回撤: {max_drawdown:.2%}")

# 绘制收益率图像
plt.figure(figsize=(14, 7))
plt.plot(df['Portfolio'] / df['Portfolio'].iloc[0] - 1, label='Strategy Return')
plt.plot(df['Benchmark'] / df['Benchmark'].iloc[0] - 1, label='Benchmark Return')
plt.plot((df['Portfolio'] / df['Portfolio'].iloc[0] - df['Benchmark'] / df['Benchmark'].iloc[0]), label='Relative Return')
plt.legend()
plt.title('Strategy Performance')
plt.xlabel('Date')
plt.ylabel('Return')
plt.show()

"""

"""
# 修改点说明：
买入成本计算：
考虑滑点：buy_price = df['收盘'].iloc[i] * (1 + slippage)
计算手续费：cost = max(buy_price * positions * buy_cost, min_cost)
更新资本和持仓：positions = (capital - cost) / buy_price
卖出成本计算：
考虑滑点：sell_price = df['收盘'].iloc[i] * (1 - slippage)
计算手续费：cost = max(sell_price * positions * sell_cost, min_cost)
更新资本：capital = positions * sell_price - cost
组合价值更新：
每日的组合价值由资本和持仓价值组成：df.at[df.index[i], 'Portfolio'] = capital + positions * df['收盘'].iloc[i]
"""

"""
基于期货的BOLL策略，股票不能做空，需要调整上面的策略的交易逻辑(因为期货交易涉及开多仓、开空仓、平多仓和平空仓)
"""
"""
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 1. 确定期货代码和回测时间
futures_code = "JM0"  # 焦煤期货代码
start_date = "20210101"
end_date = "20241231"

# 获取期货主力合约的日线数据
futures_data = ak.futures_main_sina(symbol=futures_code, start_date=start_date, end_date=end_date)
futures_data['日期'] = pd.to_datetime(futures_data['日期'])
futures_data.set_index('日期', inplace=True)

# 2. 计算布林带指标
window = 20
k = 2
futures_data['MA'] = futures_data['收盘价'].rolling(window=window).mean()
futures_data['STD'] = futures_data['收盘价'].rolling(window=window).std()
futures_data['Upper'] = futures_data['MA'] + k * futures_data['STD']
futures_data['Lower'] = futures_data['MA'] - k * futures_data['STD']

# 3. 确定买卖信号
futures_data['Signal'] = 0
futures_data.loc[futures_data['收盘价'] > futures_data['Upper'], 'Signal'] = 1  # 价格突破上轨，买入信号
futures_data.loc[futures_data['收盘价'] < futures_data['Lower'], 'Signal'] = -1  # 价格突破下轨，卖出信号

# 4. 回测
initial_capital = 100000  # 初始资金
capital = initial_capital
positions = 0  # 持仓数量
futures_data['Portfolio'] = capital  # 组合价值
futures_data['Benchmark'] = futures_data['收盘价'] / futures_data['收盘价'].iloc[0] * initial_capital  # 基准指数的收益率

for i in range(1, len(futures_data)):
    if futures_data['Signal'].iloc[i] == 1:  # 买入信号
        positions = capital / futures_data['收盘价'].iloc[i]
        capital = 0
    elif futures_data['Signal'].iloc[i] == -1:  # 卖出信号
        capital = positions * futures_data['收盘价'].iloc[i]
        positions = 0
    futures_data.at[futures_data.index[i], 'Portfolio'] = capital + positions * futures_data['收盘价'].iloc[i]

# 计算收益指标
futures_data['Strategy_Return'] = futures_data['Portfolio'].pct_change()
futures_data['Benchmark_Return'] = futures_data['Benchmark'].pct_change()

# 年化收益率
annualized_return = (futures_data['Portfolio'].iloc[-1] / futures_data['Portfolio'].iloc[0]) ** (252 / len(futures_data)) - 1
benchmark_annualized_return = (futures_data['Benchmark'].iloc[-1] / futures_data['Benchmark'].iloc[0]) ** (252 / len(futures_data)) - 1

# 阿尔法、贝塔
beta = futures_data['Strategy_Return'].cov(futures_data['Benchmark_Return']) / futures_data['Benchmark_Return'].var()
alpha = futures_data['Strategy_Return'].mean() - beta * futures_data['Benchmark_Return'].mean()

# 夏普比率
sharpe_ratio = futures_data['Strategy_Return'].mean() / futures_data['Strategy_Return'].std() * np.sqrt(252)

# 胜率、盈亏比
win_rate = (futures_data['Strategy_Return'] > 0).sum() / len(futures_data['Strategy_Return'])
profit_loss_ratio = futures_data['Strategy_Return'][futures_data['Strategy_Return'] > 0].mean() / futures_data['Strategy_Return'][futures_data['Strategy_Return'] < 0].mean()

# 收益波动率
volatility = futures_data['Strategy_Return'].std() * np.sqrt(252)

# 最大回撤
roll_max = futures_data['Portfolio'].cummax()
daily_drawdown = futures_data['Portfolio'] / roll_max - 1.0
max_drawdown = daily_drawdown.min()

# 输出收益概况
print(f"收益率: {futures_data['Portfolio'].iloc[-1] / futures_data['Portfolio'].iloc[0] - 1:.2%}")
print(f"年化收益率: {annualized_return:.2%}")
print(f"基准收益率: {benchmark_annualized_return:.2%}")
print(f"阿尔法: {alpha:.4f}")
print(f"贝塔: {beta:.4f}")
print(f"夏普比率: {sharpe_ratio:.2f}")
print(f"胜率: {win_rate:.2%}")
print(f"盈亏比: {profit_loss_ratio:.2f}")
print(f"收益波动率: {volatility:.2%}")
print(f"最大回撤: {max_drawdown:.2%}")

# 绘制收益率图像
plt.figure(figsize=(14, 7))
plt.plot(futures_data['Portfolio'] / futures_data['Portfolio'].iloc[0] - 1, label='Strategy Return')
plt.plot(futures_data['Benchmark'] / futures_data['Benchmark'].iloc[0] - 1, label='Benchmark Return')
plt.plot((futures_data['Portfolio'] / futures_data['Portfolio'].iloc[0] - futures_data['Benchmark'] / futures_data['Benchmark'].iloc[0]), label='Relative Return')
plt.legend()
plt.title('Strategy Performance')
plt.xlabel('Date')
plt.ylabel('Return')
plt.show()
"""

"""复现bigquant中的BOLL策略，有平多开仓和平空开仓功能"""
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 1. 确定期货代码和时间区间
futures_code = "JM0"  # 焦煤期货代码
data_start_date = "20210101"  # 数据抽取开始日期
data_end_date = "20241231"  # 数据抽取结束日期
backtest_start_date = "20210514"  # 回测开始日期
backtest_end_date = "20211018"  # 回测结束日期

# 尝试获取期货主力合约的日线数据
try:
    futures_data = ak.futures_main_sina(symbol=futures_code, start_date=data_start_date, end_date=data_end_date)
    print("数据获取成功，列名如下：")
    print(futures_data.columns)
except Exception as e:
    print(f"数据获取失败：{e}")

# 确保索引是 DatetimeIndex 类型
futures_data['日期'] = pd.to_datetime(futures_data['日期'])
futures_data.set_index('日期', inplace=True)

# 数据清洗和预处理
futures_data = futures_data.dropna()  # 删除缺失值

# 2. 计算布林带指标
window = 20
k = 2
futures_data['MA'] = futures_data['收盘价'].rolling(window=window).mean()
futures_data['STD'] = futures_data['收盘价'].rolling(window=window).std()
futures_data['Upper'] = futures_data['MA'] + k * futures_data['STD']
futures_data['Lower'] = futures_data['MA'] - k * futures_data['STD']

# 3. 确定买卖信号
futures_data['Signal'] = 0
futures_data.loc[futures_data['收盘价'] > futures_data['Upper'], 'Signal'] = 1  # 价格突破上轨，买入信号
futures_data.loc[futures_data['收盘价'] < futures_data['Lower'], 'Signal'] = -1  # 价格突破下轨，卖出信号

# 4. 回测
initial_capital = 1000000  # 初始资金
capital = initial_capital
positions = 0  # 持仓数量
futures_data['Portfolio'] = capital  # 组合价值
futures_data['Benchmark'] = futures_data['收盘价'] / futures_data['收盘价'].iloc[0] * initial_capital  # 基准指数的收益率

# 交易成本和滑点
buy_cost = 0.0003  # 买入手续费
sell_cost = 0.0013  # 卖出手续费
slippage = 0.001  # 滑点

# 交易逻辑
for i in range(1, len(futures_data)):
    if futures_data.index[i] < pd.to_datetime(backtest_start_date) or futures_data.index[i] > pd.to_datetime(backtest_end_date):
        continue
    if futures_data['Signal'].iloc[i] == 1:  # 买入信号
        if positions <= 0:  # 如果当前没有多头持仓或有空头持仓
            if positions < 0:
                # 平空仓
                capital += positions * futures_data['收盘价'].iloc[i] * (1 - slippage - sell_cost)
            # 开多仓
            positions = capital / futures_data['收盘价'].iloc[i] * (1 - slippage - buy_cost)
            capital = 0
            print(futures_data.index[i], '开多仓')
    elif futures_data['Signal'].iloc[i] == -1:  # 卖出信号
        if positions >= 0:  # 如果当前没有空头持仓或有多头持仓
            if positions > 0:
                # 平多仓
                capital = positions * futures_data['收盘价'].iloc[i] * (1 - slippage - sell_cost)
            # 开空仓
            positions = -capital / futures_data['收盘价'].iloc[i] * (1 - slippage - buy_cost)
            capital = 0
            print(futures_data.index[i], '开空仓')
    # 更新组合价值
    futures_data.at[futures_data.index[i], 'Portfolio'] = capital + positions * futures_data['收盘价'].iloc[i]


# 5. 计算收益指标
futures_data['Strategy_Return'] = futures_data['Portfolio'].pct_change()
futures_data['Benchmark_Return'] = futures_data['Benchmark'].pct_change()

# 年化收益率
annualized_return = (futures_data['Portfolio'].iloc[-1] / futures_data['Portfolio'].iloc[0]) ** (252 / len(futures_data)) - 1
benchmark_annualized_return = (futures_data['Benchmark'].iloc[-1] / futures_data['Benchmark'].iloc[0]) ** (252 / len(futures_data)) - 1

# 阿尔法、贝塔
beta = futures_data['Strategy_Return'].cov(futures_data['Benchmark_Return']) / futures_data['Benchmark_Return'].var()
alpha = futures_data['Strategy_Return'].mean() - beta * futures_data['Benchmark_Return'].mean()

# 夏普比率
sharpe_ratio = futures_data['Strategy_Return'].mean() / futures_data['Strategy_Return'].std() * np.sqrt(252)

# 胜率、盈亏比
win_rate = (futures_data['Strategy_Return'] > 0).sum() / len(futures_data['Strategy_Return'])
profit_loss_ratio = futures_data['Strategy_Return'][futures_data['Strategy_Return'] > 0].mean() / futures_data['Strategy_Return'][futures_data['Strategy_Return'] < 0].mean()

# 收益波动率
volatility = futures_data['Strategy_Return'].std() * np.sqrt(252)

# 最大回撤
roll_max = futures_data['Portfolio'].cummax()
daily_drawdown = futures_data['Portfolio'] / roll_max - 1.0
max_drawdown = daily_drawdown.min()

# 输出收益概况
print(f"收益率: {futures_data['Portfolio'].iloc[-1] / futures_data['Portfolio'].iloc[0] - 1:.2%}")
print(f"年化收益率: {annualized_return:.2%}")
print(f"基准收益率: {benchmark_annualized_return:.2%}")
print(f"阿尔法: {alpha:.4f}")
print(f"贝塔: {beta:.4f}")
print(f"夏普比率: {sharpe_ratio:.2f}")
print(f"胜率: {win_rate:.2%}")
print(f"盈亏比: {profit_loss_ratio:.2f}")
print(f"收益波动率: {volatility:.2%}")
print(f"最大回撤: {max_drawdown:.2%}")

# 绘制收益率图像
plt.figure(figsize=(14, 7))
plt.plot(futures_data['Portfolio'] / futures_data['Portfolio'].iloc[0] - 1, label='Strategy Return')
plt.plot(futures_data['Benchmark'] / futures_data['Benchmark'].iloc[0] - 1, label='Benchmark Return')
plt.plot((futures_data['Portfolio'] / futures_data['Portfolio'].iloc[0] - futures_data['Benchmark'] / futures_data['Benchmark'].iloc[0]), label='Relative Return')
plt.legend()
plt.title('Strategy Performance')
plt.xlabel('Date')
plt.ylabel('Return')
plt.show()