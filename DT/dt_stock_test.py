import requests
import backtrader as bt
import pandas as pd

# 获取历史价格数据
def get_historical_data(symbol, api_key, start_date=None, end_date=None):
    # 构造 API 请求 URL
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&apikey=F88JQAWBI8FCITIF&outputsize=full"
    
    # 发送请求并获取响应
    response = requests.get(url)
    data = response.json()
    
    # 检查 API 是否返回了错误信息
    if 'Error Message' in data:
        raise ValueError(f"API Error: {data['Error Message']}")
    
    # 检查是否获取到了数据
    if 'Time Series (Daily)' not in data:
        raise ValueError("No historical data found in the API response")
    
    # 获取 'Time Series (Daily)' 中的数据
    time_series_data = data['Time Series (Daily)']
    
    # 将数据转换为 DataFrame
    df = pd.DataFrame.from_dict(time_series_data, orient='index')
    
    # 重命名列名，去掉数字前缀
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    
    # 将数据类型转换为浮点数
    df = df.astype(float)
    
    # 将索引转换为日期格式
    df.index = pd.to_datetime(df.index)
    
    # 如果指定了起始日期和结束日期，则筛选数据
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]
    
    return df

# Dual Thrust策略
class DualThrustStrategy(bt.Strategy):
    params = (
        ('period', 14),  # 前N天的价格数据
        ('k1', 0.5),    # 上轨系数
        ('k2', 0.5),    # 下轨系数
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        self.dataopen = self.datas[0].open

    def next(self):
        # 确保有足够的数据
        if len(self.dataclose) < self.p.period:
            return
        
        # 计算上下轨
        HH = max(self.datahigh.get(size=self.p.period))
        LL = min(self.datalow.get(size=self.p.period))
        HC = max(self.dataclose.get(size=self.p.period))
        LC = min(self.dataclose.get(size=self.p.period))
        Range = max(HH - LC, HC - LL)
        BuyLine = self.dataopen[0] + self.p.k1 * Range
        SellLine = self.dataopen[0] - self.p.k2 * Range

        # 交易逻辑
        if self.dataclose[0] > BuyLine:
            self.buy()
        elif self.dataclose[0] < SellLine:
            self.sell()

# 主程序
if __name__ == '__main__':
    # API密钥和股票代码
    api_key = "F88JQAWBI8FCITIF"  # 替换为你的API密钥
    symbol = 'IBM'  # 替换为你想要回测的股票代码
    start_date = '2020-02-07'  # 替换为回测的开始日期
    end_date = '2024-04-15'  # 替换为回测的结束日期

    # 获取历史价格数据
    historical_data = get_historical_data(symbol, api_key, start_date, end_date)

    # 将数据转换为Backtrader的格式
    data = bt.feeds.PandasData(dataname=historical_data)

    # 创建Cerebro引擎
    cerebro = bt.Cerebro()
    cerebro.addstrategy(DualThrustStrategy)
    cerebro.adddata(data)

    # 设置初始资金
    cerebro.broker.setcash(100000.0)

    # 运行回测
    cerebro.run()
    cerebro.plot()