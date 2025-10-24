"""
可转债策略主入口
"""
import pandas as pd
from data.get_data import download_data
from features import add_features, add_label
from backtest import rolling_backtest
from evaluate import calc_port_ret, performance

# 1. 数据
raw = download_data()

# 2. 特征 & 标签
df = add_features(raw)
df = add_label(df)

# 3. 回测
portfolio = rolling_backtest(df, hold_n=20, premium_max=50, rating_min=3)

# 4. 绩效
portfolio = calc_port_ret(portfolio, df)
returns = portfolio.set_index("month_end")["ret"]
returns.index = pd.to_datetime(returns.index)
performance(returns)

# 5. 保存结果
portfolio.to_csv("cb_portfolio.csv", index=False, encoding="utf_8_sig")
print("结果已保存：cb_portfolio.csv  &  equity_curve.png")