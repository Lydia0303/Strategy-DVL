"""
构造因子 + 打标签
"""
import numpy as np
import pandas as pd
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
    df["premium"] = df["premium_rt"] / 100
    df["rating_score"] = df["rating"].map(rating_map).fillna(0).astype(int)
    df["log_cb_price"] = np.log(df["price"])
    df["log_premium"] = np.log1p(df["premium"])
    df["conv_ratio"] = df["convert_value"] / df["price"]
    # 月末标记
    df["month_end"] = (pd.to_datetime(df["trade_date"]) + pd.offsets.BMonthEnd(0)).dt.date
    return df
"""
def add_label(df: pd.DataFrame, hold_days: int = 5) -> pd.DataFrame:
    df = df.sort_values(["bond_code", "trade_date"])

    # 1. 构造 T+1 价格
    next_price = (df[["bond_code", "trade_date", "price"]]
                  .rename(columns={"price": "future_price"}))
    next_price["next_trade_date"] = next_price["trade_date"] + pd.Timedelta(days=1)

    # 2. 左连接（保留所有券）
    df = df.merge(next_price,
                  left_on=["bond_code", "trade_date"],
                  right_on=["bond_code", "next_trade_date"],
                  how="left")          # 不要 suffixes，直接覆盖
    # 查看日期是否正确递增（跳过周末和节假日）
    print(df[["trade_date_x", "next_trade_date", "price", "future_price"]].iloc[0:10])

    # 3. 删掉右表冗余列
    df = df.drop(columns=["next_trade_date"])

    # 4. 缺失未来价格 → 直接丢掉该行（不训练、不回测）
    df = df.dropna(subset=["future_price"])

    # 5. 计算 label
    df["label"] = df["future_price"] / df["price"] - 1

    # 6. 打印（在函数末尾，不要依赖 m）
    print(f"[标签] 未来价格缺失数: {df['future_price'].isna().sum()}")
    print(f"[标签] 未来价格示例:", df["future_price"].dropna().head())
    # 查看这些行对应的原始价格和日期，确认是否合理
    # print(df[["trade_date_x", "price", "future_price"]].iloc[0:10])
    
    return df
"""

def add_label(df: pd.DataFrame, hold_days: int = 5) -> pd.DataFrame:
    df = df.sort_values(["bond_code", "trade_date"])
    df["future_price"] = df.groupby("bond_code")["price"].shift(-hold_days)
    df["label"] = df["future_price"] / df["price"] - 1
    df = df.dropna(subset=["label"])
    return df

FEATURES = ["log_cb_price", "log_premium", "conv_ratio", "rating_score", "ytm", "remain_years"]