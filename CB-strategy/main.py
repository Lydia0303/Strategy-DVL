"""
可转债策略主入口
"""
from data.get_data import download_data
from data.get_data import fetch_real_history
from features import add_features, add_label
from backtest import rolling_backtest
from evaluate import performance

def main():
    raw = download_data()
    raw = fetch_real_history(raw)
    df = add_features(raw)
    df = add_label(df, hold_days=1)  # 5 日标签
    print("price 20 日变化描述:\n", (df["future_price"] / df["price"] - 1).describe())
    print("premium 20 日变化描述:\n", df["label"].describe())
    print("unique trade_date 数量:", df["trade_date"].nunique())
    print("label=0 比例:", (df["label"].abs() < 1e-6).mean())
    port = rolling_backtest(df, hold_n=10, premium_max=0.3, rating_min=2)
    print(port.head())          # 看有没有选中券
    print(port["ret"].describe())   # 看 ret 分布
    performance(port)
    port.to_csv("cb_portfolio.csv", index=False, encoding="utf_8_sig")
    print("结果已保存：cb_portfolio.csv  &  equity_curve.png")

if __name__ == "__main__":
    main()

"""
import pandas as pd
from data.get_data import download_data
from features import add_features, add_label
from backtest import rolling_backtest
from evaluate import calc_port_ret, performance

# main.py
import os
if os.path.exists("cb_data.pkl"):
    os.remove("cb_data.pkl")   # 强制删缓存
             

# 1. 抓原始数据
raw = download_data()          

# 2. 特征 & 标签
df = add_features(raw)         # 当场做特征
print(df.columns.tolist())
print("【1】原始 trade_date 范围:", df["trade_date"].min(), "→", df["trade_date"].max())
print("【2】month_end 唯一值:", df["month_end"].unique())
df = add_label(df)
print("[标签后] 列名:", df.columns.tolist())   # 如果这里消失 → 问题在 add_label
print("[标签后] 第一行数据:")
print(df.iloc[0].to_dict())  # 转为字典格式，更清晰展示每行的"列名:值"

# 3. 回测
portfolio = rolling_backtest(df, hold_n=20, premium_max=50, rating_min=2)

# 4. 绩效
portfolio = calc_port_ret(portfolio, df)
returns = portfolio.set_index("month_end")["ret"]
returns.index = pd.to_datetime(returns.index)
performance(returns)

# 5. 保存结果
portfolio.to_csv("cb_portfolio.csv", index=False, encoding="utf_8_sig")
print("结果已保存：cb_portfolio.csv  &  equity_curve.png")
"""

