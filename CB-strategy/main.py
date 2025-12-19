"""
可转债策略主入口
"""
from data.get_data import download_data, fetch_real_history
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
