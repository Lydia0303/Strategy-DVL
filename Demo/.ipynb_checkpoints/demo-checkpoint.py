# 加载模块
import requests

# 获取数据
url = "http://api.finance.ifeng.com/akdaily/?code=sh600036&type=last"
r = requests.get(url)
data = r.json()

# 查看数据内容
# 前面字段为date，open，high，close，low，volume
print(data["record"][0])

# 封装函数
from datetime import datetime

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData
from vnpy.trader.database import DB_TZ, database_manager

def download_stock_daily_data(symbol: str, exchange: Exchange):
    """基于代码和交易所下载数据"""
    exchange_map = {
        Exchange.SSE: "sh",
        Exchange.SZSE: "sz",
    }
    exchange_str = exchange_map[exchange]

    url = f"http://api.finance.ifeng.com/akdaily/?code={exchange_str}{symbol}&type=last"
    r = requests.get(url)
    record = r.json()["record"]

    bar_data = []
    for rd in record:
        dt = datetime.strptime(rd[0], "%Y-%m-%d")
        bar = BarData(
        symbol=symbol,
        exchange=exchange,
        datetime=DB_TZ.localize(dt),
        interval=Interval.DAILY,
        open_peice=rd[1],
        high_price=rd[2],
        low_price=rd[3],
        close_price=rd[4],
        volume=rd[5],
        gateway_name="IFENG"
        )
        bar_data.append(bar)
        return bar_data

# 尝试下载
bar_data = download_stock_daily_data("600036", Exchange.SSE)

# 查看数据
bar_data[0]

# 转换DataFrame
import pandas as pd
df = pd.DataFrame.from_records([bar._dict_for bar in bar_data])
df.to_csv("demo.csv")