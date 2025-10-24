import pandas as pd
import numpy as np
import statsmodels.api as sm
import akshare as ak
import matplotlib.pyplot as plt

# 模拟 BigQuant 的上下文和数据结构
class Context:
    def __init__(self, instruments, data, benchmark_data):
        self.instruments = instruments
        self.data = data
        self.benchmark_data = benchmark_data
        self.portfolio = Portfolio()
        self.symbol = lambda x: x  # 简化 symbol 函数
        self.order_target_percent = self._order_target_percent
        self.returns = []  # 用于存储每日收益率
        self.benchmark_returns = []  # 用于存储基准收益率
        self.benchmark_price_yesterday = benchmark_data['close'].iloc[0]
        self.current_position = None  # 当前持仓状态，None 表示未持仓，1 表示持有 symbol_1，2 表示持有 symbol_2

    def _order_target_percent(self, symbol, percent, today):
        print(f"Order target percent for {symbol}: {percent}")
        
        # 获取当前持仓数量和总价值
        current_position = self.portfolio[symbol].amount
        total_value = self.portfolio.total_value
        
        # 计算目标持仓价值
        target_value = total_value * percent
        
        # 获取当前价格
        current_price = self.data[(self.data['date'] == today) & (self.data['instrument'] == symbol)]['close'].values[0]
        
        # 计算目标持仓数量
        target_amount = target_value / current_price
        
        # 计算需要买入或卖出的数量
        delta_amount = target_amount - current_position
        
        # 更新持仓数量
        self.portfolio[symbol].amount = target_amount
        
        # 更新现金
        self.portfolio.cash -= delta_amount * current_price
        
        # 打印交易信息
        if delta_amount > 0:
            print(f"买入 {delta_amount} 股 {symbol}，价格为 {current_price}")
        elif delta_amount < 0:
            print(f"卖出 {-delta_amount} 股 {symbol}，价格为 {current_price}")

class Portfolio:
    def __init__(self):
        self.positions = {}
        self.cash = 1000000  # 初始资金
        self.total_value = 1000000  # 初始总价值

    def __getitem__(self, key):
        if key not in self.positions:
            self.positions[key] = Position()
        return self.positions[key]

class Position:
    def __init__(self):
        self.amount = 0

class Data:
    def __init__(self, current_dt):
        self.current_dt = current_dt

    def can_trade(self, symbol):
        return True

# 初始化函数
def initialize(context):
    # 模拟设置交易手续费
    print("Setting commission: buy_cost=0.0003, sell_cost=0.0013, min_cost=5")
    
    # 初始化交易日志数据框
    context.trading_log = pd.DataFrame(columns=['date', 'zscore', 'position_1', 'position_2', 'action'])

    # 获取股票列表
    stocklist = context.instruments
    # 从 context.data 中提取股票价格数据
    prices_df = pd.pivot_table(context.data, values='close', index=['date'], columns=['instrument'])
    prices_df = prices_df.ffill()  # 替换 fillna 为 ffill
    context.x = prices_df[stocklist[0]]  # 股票1
    context.y = prices_df[stocklist[1]]  # 股票2
    # 检查并清理数据
    context.y = context.y.replace([np.inf, -np.inf], np.nan).dropna()
    context.x = context.x.replace([np.inf, -np.inf], np.nan).dropna()
   
    # 对齐索引
    context.y, context.x = context.y.align(context.x, join='inner', axis=0)

# 设置止损点
stop_loss = -0.20 # 10%的回撤
buffer_period = 20 # 设置20个交易日的缓冲期

# 每个交易日的处理函数
def handle_data(context, data_obj):
    # 线性回归两个股票的股价 y = ax + b
    X = sm.add_constant(context.x)  # 添加常数项
    # 转换为数值类型
    context.y = context.y.astype(float) 
    X = X.astype(float)
    # 进行回归分析
    model = sm.OLS(context.y, X).fit()
    
    def zscore(series):
        return (series - series.mean()) / np.std(series)

    # 计算 y - a*x 序列的 zscore 值序列
    residuals = context.y - model.predict(X)
    zscore_calcu = zscore(residuals)
    context.zscore = zscore_calcu
    
    # 确保 today 是 Timestamp 类型
    today = pd.to_datetime(data_obj.current_dt.strftime("%Y-%m-%d"))
    
    # 检查 context.zscore 是否包含 today
    if today not in context.zscore.index:
        print(f"Skipping {today}, no data available.")
        return
    
    zscore_today = context.zscore.loc[today]
    
    stocklist = context.instruments
    symbol_1 = context.symbol(stocklist[0])
    symbol_2 = context.symbol(stocklist[1])
    
    # 持仓
    cur_position_1 = context.portfolio[symbol_1].amount
    cur_position_2 = context.portfolio[symbol_2].amount

    # 打印调试信息
    print(f"{today} zscore: {zscore_today}, position 1: {cur_position_1}, position 2: {cur_position_2}")
    
     # 将交易日志写入文件
    log_entry = {
        'date': today,
        'zscore': zscore_today,
        'position_1': cur_position_1,
        'position_2': cur_position_2,
        'action': 'No action'
    }

    # 计算当日收益
    current_value = context.portfolio.cash
    for symbol, pos in context.portfolio.positions.items():
        current_price = context.data[(context.data['date'] == today) & (context.data['instrument'] == symbol)]['close']
        if not current_price.empty:
            current_value += pos.amount * current_price.values[0]

    if len(context.returns) > 0:
        daily_return = (current_value - context.portfolio.total_value) / context.portfolio.total_value
        context.returns.append(daily_return)
    else:
        context.returns.append(0)  # 初始收益为0

    context.portfolio.total_value = current_value

    # 计算基准收益
    benchmark_price_today = context.benchmark_data[context.benchmark_data['date'] == today]['close']
    if not benchmark_price_today.empty:
        if len(context.benchmark_returns) > 0:
            benchmark_return_today = (benchmark_price_today.values[0] - context.benchmark_price_yesterday) / context.benchmark_price_yesterday
            context.benchmark_returns.append(benchmark_return_today)
        else:
            context.benchmark_returns.append(0)  # 初始基准收益为0
    context.benchmark_price_yesterday = benchmark_price_today.values[0] if not benchmark_price_today.empty else context.benchmark_price_yesterday

    # 计算累计收益率
    cumulative_returns = (1 + pd.Series(context.returns)).cumprod() - 1
    roll_max = cumulative_returns.cummax()
    daily_drawdown = cumulative_returns / roll_max - 1.0
    max_drawdown = daily_drawdown.min()

    # 检查是否达到止损点
    if len(context.returns) > buffer_period and max_drawdown <= stop_loss:
        print(f"Stop loss triggered at {today}, max drawdown: {max_drawdown:.2%}")
        # 平仓操作
        context.order_target_percent(symbol_1, 0, today)
        context.order_target_percent(symbol_2, 0, today)
        context.current_position = None
        log_entry['action'] = 'Stop loss triggered'
        log_entry['position_1'] = 0
        log_entry['position_2'] = 0
    
    else:
        # 交易逻辑
        if zscore_today > 0.5 and context.current_position != 1 and data_obj.can_trade(symbol_1) and data_obj.can_trade(symbol_2):
            context.order_target_percent(symbol_2, 0, today) # 传递today参数
            context.order_target_percent(symbol_1, 1, today) # 传递today参数
            print(f"{today} 全仓买入：{stocklist[1]}，卖出全部：{stocklist[0]}")
            context.current_position = 1  # 更新当前持仓状态
            log_entry['action'] = f'Buy {stocklist[1]}, Sell {stocklist[0]}'
        elif zscore_today < -0.5 and context.current_position != 2 and data_obj.can_trade(symbol_1) and data_obj.can_trade(symbol_2):
            context.order_target_percent(symbol_1, 0, today) # 传递today参数
            context.order_target_percent(symbol_2, 1, today) # 传递today参数
            print(f"{today} 全仓买入：{stocklist[0]}，卖出全部：{stocklist[1]}")
            context.current_position = 2  # 更新当前持仓状态
            log_entry['action'] = f'Buy {stocklist[0]}, Sell {stocklist[1]}'
        else:
            print(f"{today} 无交易操作，当前持仓状态：{context.current_position}")
            log_entry['action'] = 'No action'
    
    # 将日志条目追加到交易日志数据框
    log_entry_df = pd.DataFrame([log_entry])
    if not hasattr(context, 'trading_log'):
        context.trading_log = log_entry_df
    else:
        context.trading_log = pd.concat([context.trading_log, log_entry_df], ignore_index=True)
 

    # 保存到 CSV 文件
    context.trading_log.to_csv('tradingdiary.csv', index=False)

# 模拟主函数
def main():
    # 从文件路径加载股票数据
    stock1_path = r"C:\Users\hz\Desktop\test\002978.xlsx"
    stock2_path = r"C:\Users\hz\Desktop\test\600459.xlsx"
    
    stock1_data = pd.read_excel(stock1_path)
    stock2_data = pd.read_excel(stock2_path)
    
    # 修正列名
    stock1_data.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume'}, inplace=True)
    stock2_data.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume'}, inplace=True)
    
    # 确保日期列是 datetime 类型
    stock1_data['date'] = pd.to_datetime(stock1_data['date'])
    stock2_data['date'] = pd.to_datetime(stock2_data['date'])
    
    # 添加 instrument 列
    stock1_data['instrument'] = "301488"
    stock2_data['instrument'] = "300745"
    
    # 合并数据
    data = pd.concat([stock1_data, stock2_data])
    
    # 获取基准数据
    benchmark_data = ak.stock_zh_index_daily_em(symbol="sh000300", start_date="20221230", end_date="20241231")
    benchmark_data.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume'}, inplace=True)
    benchmark_data['date'] = pd.to_datetime(benchmark_data['date'])
    
    
    # 初始化上下文
    instruments = ["301488", "300745"]
    context = Context(instruments, data, benchmark_data)
    initialize(context)
    
    # 获取实际交易日
    trading_dates = data['date'].unique()
    trading_dates = pd.Series(trading_dates).sort_values()
    
    # 确保基准数据的日期范围与交易日的日期范围一致
    benchmark_dates = benchmark_data['date'].unique()
    benchmark_dates = pd.Series(benchmark_dates).sort_values()
    
    print("Benchmark Dates:", benchmark_dates)
    print("Length of Benchmark Dates:", len(benchmark_dates))
    
    if len(benchmark_dates) < len(trading_dates):
        print("Warning: Benchmark data has fewer dates than trading dates.")
        trading_dates = trading_dates[trading_dates.isin(benchmark_dates)]
        print("Updated Trading Dates:", trading_dates)
        print("Updated Length of Trading Dates:", len(trading_dates))
    

    for date in trading_dates:
        data_obj = Data(date)
        handle_data(context, data_obj)
    

    
    # 计算收益表现
    returns = pd.Series(context.returns)
    cumulative_returns = (1 + returns).cumprod() - 1
    benchmark_returns = pd.Series(context.benchmark_returns)
    cumulative_benchmark_returns = (1 + benchmark_returns).cumprod() - 1
    annualized_return = (1 + cumulative_returns.iloc[-1]) ** (252 / len(returns)) - 1
    annualized_benchmark_return = (1 + cumulative_benchmark_returns.iloc[-1]) ** (252 / len(benchmark_returns)) - 1
    alpha = annualized_return - annualized_benchmark_return
    beta = returns.cov(benchmark_returns) / benchmark_returns.var()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
    win_rate = (returns > 0).mean()
    profit_loss_ratio = returns[returns > 0].mean() / abs(returns[returns < 0].mean())
    volatility = returns.std() * np.sqrt(252)
    information_ratio = (returns.mean() - benchmark_returns.mean()) / returns.std() * np.sqrt(252)
    # 计算最大回撤
    roll_max = cumulative_returns.cummax()
    daily_drawdown = (cumulative_returns - roll_max) / roll_max
    max_drawdown = daily_drawdown.min()
    # max_drawdown = (cumulative_returns / cumulative_returns.cummax().replace(0, np.nan) - 1).min()



    # 检查是否有负值或异常值
    if (cumulative_returns < 0).any():
        print("Warning: Negative cumulative returns detected.")
        print(cumulative_returns[cumulative_returns < 0])

    # 打印收益表现
    print(f"累计收益率: {cumulative_returns.iloc[-1]:.2%}")
    print(f"年化收益率: {annualized_return:.6%}")
    print(f"基准收益率: {annualized_benchmark_return:.2%}")
    print(f"阿尔法: {alpha:.2%}")
    print(f"贝塔: {beta:.2f}")
    print(f"夏普比率: {sharpe_ratio:.2f}")
    print(f"胜率: {win_rate:.2%}")
    print(f"盈亏比: {profit_loss_ratio:.2f}")
    print(f"收益波动率: {volatility:.2%}")
    print(f"信息比率: {information_ratio:.2f}")
    print(f"最大回撤: {max_drawdown:.2%}")

    # 绘制累计收益率图像
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # 第一个子图：累计收益和基准收益
    ax1.plot(trading_dates, cumulative_returns, label='Cumulative Returns')
    ax1.plot(trading_dates, cumulative_benchmark_returns, label='Benchmark Returns')
    ax1.set_title('Cumulative Returns Over Time')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Cumulative Returns')
    ax1.legend()

    # 第二个子图：两只股票的收盘价走势
    stock1_close = stock1_data.set_index('date').reindex(trading_dates).reset_index()['close']
    stock2_close = stock2_data.set_index('date').reindex(trading_dates).reset_index()['close']

    ax2.plot(trading_dates, stock1_close, label='Stock1 Close')
    ax2.plot(trading_dates, stock2_close, label='Stock2 Close')
    ax2.set_title('Stocks Close Price Over Time')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Close Price')
    ax2.legend()


    # 自动格式化 x 轴的时间标签，使其更易读
    fig.autofmt_xdate()

    # 显示图形
    plt.tight_layout()  # 自动调整子图布局，避免重叠
    plt.show()

if __name__ == "__main__":
    main()