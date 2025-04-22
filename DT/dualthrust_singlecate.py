"""
Dual Thrust是一个趋势跟踪系统
计算前N天的最高价-收盘价和收盘价-最低价。然后取这2N个价差的最大值,乘以k值。把结果称为触发值。
在今天的开盘，记录开盘价，然后在价格超过上轨（开盘＋触发值）时马上买入，或者价格低于下轨（开盘－触发值）时马上卖空。
没有明确止损。这个系统是反转系统，也就是说，如果在价格超过（开盘＋触发值）时手头有空单，则平空开多。
同理，如果在价格低于（开盘－触发值）时手上有多单，则平多开空。
选用了SHFE的rb2010 在2020-02-07 15:00:00 到 2020-04-15 15:00:00' 进行回测。
注意： 
1:为回测方便,本策略使用了on_bar的一分钟来计算,实盘中可能需要使用on_tick。
2:实盘中,如果在收盘的那一根bar或tick触发交易信号,需要自行处理,实盘可能不会成交
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import akshare as aks

# 获取所有期货主力合约的 symbol 映射关系
futures_symbols = aks.futures_display_main_sina()
print(futures_symbols)

# 获取期货主力合约的历史数据
data = aks.futures_main_sina(symbol="C0", start_date="20200101")  # C0 表示玉米主力合约
print(data.columns)  # 打印列名

# 重命名列名（如果需要）
data.rename(columns={'日期':'datetime','开盘价': 'open', '最高价': 'high', '最低价': 'low', '收盘价': 'close'}, inplace=True)

# 确保索引是日期类型
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)
data.sort_index(inplace=True)

# 参数设置
N = 14  # 前N天的价格数据
K1 = 0.5  # 上轨系数
K2 = 0.5  # 下轨系数

# 计算关键价位
data['HH'] = data['high'].rolling(window=N).max()
data['LL'] = data['low'].rolling(window=N).min()
data['HC'] = data['close'].rolling(window=N).max()
data['LC'] = data['close'].rolling(window=N).min()

# 计算上下轨
data['Range'] = np.maximum(data['HH'] - data['LC'], data['HC'] - data['LL'])
data['BuyLine'] = data['open'] + K1 * data['Range']
data['SellLine'] = data['open'] - K2 * data['Range']

# 将上下轨向前移动一天
data['BuyLine'] = data['BuyLine'].shift(1)
data['SellLine'] = data['SellLine'].shift(1)

# 生成交易信号
data['Signal'] = 0
data.loc[data['close'] > data['BuyLine'], 'Signal'] = 1  # 买入信号
data.loc[data['close'] < data['SellLine'], 'Signal'] = -1  # 卖出信号

# 计算策略收益
data['Position'] = data['Signal'].shift(1).fillna(0)
data['StrategyReturns'] = data['Position'] * data['close'].pct_change()
data['StrategyCumulative'] = (1 + data['StrategyReturns']).cumprod()

# 计算策略性能指标
annualized_return = (data['StrategyCumulative'].iloc[-1]) ** (252 / len(data)) - 1
print(f"年化收益率: {annualized_return:.2%}")

roll_max = data['StrategyCumulative'].cummax()
daily_drawdown = data['StrategyCumulative'] / roll_max - 1.0
max_drawdown = daily_drawdown.min()
print(f"最大回撤: {max_drawdown:.2%}")

risk_free_rate = 0.02  # 假设无风险利率为 2%
sharpe_ratio = (annualized_return - risk_free_rate) / (data['StrategyReturns'].std() * np.sqrt(252))
print(f"夏普比率: {sharpe_ratio:.2f}")

data['Profit'] = data['StrategyReturns'] > 0
data['Loss'] = data['StrategyReturns'] < 0
win_rate = data['Profit'].mean()
profit_loss_ratio = data[data['Profit']]['StrategyReturns'].mean() / abs(data[data['Loss']]['StrategyReturns'].mean())
print(f"胜率: {win_rate:.2%}")
print(f"盈亏比: {profit_loss_ratio:.2f}")

trade_count = data['Signal'].diff().abs().sum() / 2  # 交易次数
average_trade_return = data['StrategyReturns'].mean()  # 平均交易收益
print(f"交易次数: {trade_count}")
print(f"平均交易收益: {average_trade_return:.2%}")

# 绘制策略表现
plt.figure(figsize=(14, 7))
plt.plot(data['StrategyCumulative'], label='Strategy Cumulative Returns')
plt.plot((1 + data['close'].pct_change()).cumprod(), label='Buy and Hold Returns')
plt.plot(roll_max, label='Rolling Max')
plt.plot(roll_max * (1 - max_drawdown), label='Max Drawdown')
plt.legend()
plt.title('Dual Thrust Strategy Performance')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.show()