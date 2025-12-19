"""
构造因子 + 打标签
"""
import numpy as np
import pandas as pd
import akshare as ak
# from datetime import timedelta

COL_MAP = {
    "代码": "bond_code",
    "转债名称": "bond_name",
    "现价": "price",
    "转股价值": "convert_value",
    "转股溢价率": "premium_rt",   # 溢价率源列
    "债券评级": "rating",
    "转股价": "convert_price",
    "成交额": "volume",
    "换手率": "turnover",
    "剩余规模": "balance",
    "剩余年限": "remain_years",
    "到期税前收益": "ytm",
    "到期时间": "maturity",
    "强赎触发价": "call_trigger",
}

rating_map = {'AAA': 9, 'AA+': 8, 'AA': 7, 'AA-': 6, 'A+': 5,
                  'A': 4, 'A-': 3, 'BBB+': 2, 'BBB': 1}



def add_features(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.rename(columns=COL_MAP)
    print(df["price"].head())
    print(type(df["price"]))
    print(df["price"].shape)
    df["premium"] = df["premium_rt"] / 100
    df["rating_score"] = df["rating"].map(rating_map).fillna(0).astype(int)
    df["log_cb_price"] = np.log(df["close_price"])
    df["log_premium"] = np.log1p(df["premium"])
    df["conv_ratio"] = df["convert_value"] / df["price"]
    # 月末标记
    df["month_end"] = (pd.to_datetime(df["trade_date"]) + pd.offsets.BMonthEnd(0)).dt.date
    return df

def add_label(df: pd.DataFrame, hold_days: int = 1) -> pd.DataFrame:
    df = df.sort_values(["bond_code", "trade_date"])
    # 用溢价率变化当 label
    # df["future_premium"] = df.groupby("bond_code")["premium"].shift(-hold_days)
    # df["label"] = df["future_premium"] - df["premium"]   # 负值=溢价收敛，利好转债
    df["future_price"] = df.groupby("bond_code")["close_price"].shift(-hold_days)
    print("shift 后样本:")
    print(df[['bond_code', 'trade_date', 'price', 'future_price']].head(10))
    df["label"] = df["future_price"] / df["close_price"] - 1
    df = df.dropna(subset=["label"])
    return df

FEATURES = ["log_cb_price", "log_premium", "conv_ratio", "rating_score", "ytm", "remain_years"]