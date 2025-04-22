import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import akshare as aks

# 获取所有期货主力合约的 symbol 映射关系
futures_symbols = aks.futures_display_main_sina()
print(futures_symbols)

# 参数设置
N = 14  # 前N天的价格数据
K1 = 0.5  # 上轨系数
K2 = 0.5  # 下轨系数
start_date = "20200101"  # 数据起始日期

# 定义计算策略性能指标的函数
def calculate_strategy_performance(data):
    data['HH'] = data['high'].rolling(window=N).max()
    data['LL'] = data['low'].rolling(window=N).min()
    data['HC'] = data['close'].rolling(window=N).max()
    data['LC'] = data['close'].rolling(window=N).min()
    data['Range'] = np.maximum(data['HH'] - data['LC'], data['HC'] - data['LL'])
    data['BuyLine'] = data['open'] + K1 * data['Range']
    data['SellLine'] = data['open'] - K2 * data['Range']
    data['BuyLine'] = data['BuyLine'].shift(1)
    data['SellLine'] = data['SellLine'].shift(1)
    data['Signal'] = 0
    data.loc[data['close'] > data['BuyLine'], 'Signal'] = 1
    data.loc[data['close'] < data['SellLine'], 'Signal'] = -1
    data['Position'] = data['Signal'].shift(1).fillna(0)
    data['StrategyReturns'] = data['Position'] * data['close'].pct_change()
    data['StrategyCumulative'] = (1 + data['StrategyReturns']).cumprod()
    annualized_return = (data['StrategyCumulative'].iloc[-1]) ** (252 / len(data)) - 1
    roll_max = data['StrategyCumulative'].cummax()
    daily_drawdown = data['StrategyCumulative'] / roll_max - 1.0
    max_drawdown = daily_drawdown.min()
    sharpe_ratio = (annualized_return - 0.02) / (data['StrategyReturns'].std() * np.sqrt(252))
    win_rate = (data['StrategyReturns'] > 0).mean()
    profit_loss_ratio = data[data['StrategyReturns'] > 0]['StrategyReturns'].mean() / abs(data[data['StrategyReturns'] < 0]['StrategyReturns'].mean())
    trade_count = data['Signal'].diff().abs().sum() / 2
    average_trade_return = data['StrategyReturns'].mean()
    return {
        '年化收益率': annualized_return,
        '最大回撤': max_drawdown,
        '夏普比率': sharpe_ratio,
        '胜率': win_rate,
        '盈亏比': profit_loss_ratio,
        '交易次数': trade_count,
        '平均交易收益': average_trade_return
    }

# 获取所有期货主力合约的历史数据并计算年收益率
futures_performance = []
for symbol in futures_symbols['symbol']:
    try:
        data = aks.futures_main_sina(symbol=symbol, start_date=start_date)
        data.rename(columns={'日期': 'datetime', '开盘价': 'open', '最高价': 'high', '最低价': 'low', '收盘价': 'close'}, inplace=True)
        data['datetime'] = pd.to_datetime(data['datetime'])
        data.set_index('datetime', inplace=True)
        performance = calculate_strategy_performance(data)
        futures_performance.append({'symbol': symbol, '年化收益率': performance['年化收益率']})
    except Exception as e:
        print(f"Error processing {symbol}: {e}")

# 将年收益率排名前10的品种存入列表
futures_performance_df = pd.DataFrame(futures_performance)
top_10_futures = futures_performance_df.nlargest(10, '年化收益率')

# 打印这10个品种的策略性能指标
print("Top 10 Futures Performance:")
print(top_10_futures)

# 构建组合策略
portfolio_performance = {'年化收益率': 0, '最大回撤': 0, '夏普比率': 0, '胜率': 0, '盈亏比': 0, '交易次数': 0, '平均交易收益': 0}
for index, row in top_10_futures.iterrows():
    symbol = row['symbol']
    data = aks.futures_main_sina(symbol=symbol, start_date=start_date)
    data.rename(columns={'日期': 'datetime', '开盘价': 'open', '最高价': 'high', '最低价': 'low', '收盘价': 'close'}, inplace=True)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    performance = calculate_strategy_performance(data)
    for key in portfolio_performance.keys():
        portfolio_performance[key] += performance[key] / 10

# 打印组合策略的策略性能指标
print("\nPortfolio Performance:")
for key, value in portfolio_performance.items():
    if isinstance(value, float):
        print(f"{key}: {value:.2%}")
    else:
        print(f"{key}: {value}")

# 绘制组合策略表现
plt.figure(figsize=(14, 7))
for index, row in top_10_futures.iterrows():
    symbol = row['symbol']
    data = aks.futures_main_sina(symbol=symbol, start_date=start_date)
    data.rename(columns={'日期': 'datetime', '开盘价': 'open', '最高价': 'high', '最低价': 'low', '收盘价': 'close'}, inplace=True)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    performance = calculate_strategy_performance(data)
    plt.plot(data['StrategyCumulative'], label=symbol)

plt.plot((1 + data['close'].pct_change()).cumprod(), label='Buy and Hold Returns')
plt.legend()
plt.title('Portfolio Strategy Performance')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.show()