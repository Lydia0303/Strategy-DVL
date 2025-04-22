"""
"from alpha_vantage.timeseries import TimeSeries
import pandas as pd

api_key = 'F88JQAWBI8FCITIF'  # 替换为你的API Key
ts = TimeSeries(key=api_key, output_format='pandas')

# 获取期货数据，例如标普500指数期货（ES=F）
data, meta_data = ts.get_intraday(symbol='AAPL', interval='1min', outputsize='full')
print(data)"
"""
import akshare as aks

# 获取所有期货主力合约的 symbol 映射关系
futures_symbols = aks.futures_display_main_sina()
print(futures_symbols)

# 获取期货主力合约的 symbol 映射关系


# 下载特定期货主力合约的历史数据
# data = aks.futures_main_sina(symbol="C0", start_date="20200101")  # C0 表示玉米主力合约
# print(data)