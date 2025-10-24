import backtrader as bt
import pandas as pd

# 创建策略
class DualMAStrategy(bt.Strategy):
    params = (
        ('short_window', 10),
        ('long_window', 30),
    )

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.sma_short = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.short_window)
        self.sma_long = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.long_window)
        self.order = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price}, Cost: {order.executed.value}, Comm: {order.executed.comm}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price}, Cost: {order.executed.value}, Comm: {order.executed.comm}')
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f'OPERATION PROFIT, GROSS: {trade.pnl}, NET: {trade.pnlcomm}')

    def next(self):
        self.log(f'Close, {self.dataclose[0]}')
        if self.order:
            return
        if not self.position:
            if self.sma_short > self.sma_long:
                self.log(f'BUY CREATE, {self.dataclose[0]}')
                self.order = self.buy()
        else:
            if self.sma_short < self.sma_long:
                self.log(f'SELL CREATE, {self.dataclose[0]}')
                self.order = self.sell()

# 创建 Cerebro 实例
cerebro = bt.Cerebro()

# 加载本地 Excel 数据
data_path = r"C:\Users\hz\Desktop\test\600459.xlsx"
data_df = pd.read_excel(data_path)

# 确保日期列是 datetime 类型，并设置为索引
data_df['日期'] = pd.to_datetime(data_df['日期'])
data_df.set_index('日期', inplace=True)

# 重命名列名以符合 Backtrader 的要求
data_df.rename(columns={
    '开盘': 'Open',
    '最高': 'High',
    '最低': 'Low',
    '收盘': 'Close',
    '成交量': 'Volume'
}, inplace=True)

# 将数据转换为 PandasData 对象
data_feed = bt.feeds.PandasData(dataname=data_df)
cerebro.adddata(data_feed)

# 添加策略
cerebro.addstrategy(DualMAStrategy)

# 设置初始资金
cerebro.broker.setcash(100000.0)

# 设置佣金
cerebro.broker.setcommission(commission=0.001)

# 添加性能分析器
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')

# 运行回测
strat = cerebro.run()[0]

# 输出分析结果
print('Sharpe Ratio:', strat.analyzers.sharpe.get_analysis())
print('Drawdown:', strat.analyzers.drawdown.get_analysis())
print('Time Return:', strat.analyzers.timereturn.get_analysis())

# 绘制结果
cerebro.plot()