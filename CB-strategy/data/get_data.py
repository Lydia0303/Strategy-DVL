"""
只负责“拿数据”：akshare → DataFrame
"""
import akshare as ak
import pandas as pd
from tqdm import tqdm
import os

CACHE_NAME = "cache/cb_data.pkl"

def download_data(start: str = "20230104", end: str = "20231229") -> pd.DataFrame:
    """缓存机制下载全市场转债日行情"""
    if os.path.exists(CACHE_NAME):
        print("【缓存命中】直接读取")
        return pd.read_pickle(CACHE_NAME)
    os.makedirs("cache", exist_ok=True)
    dates = pd.bdate_range(start, end).strftime("%Y%m%d").tolist()
    res = []
    
    for d in tqdm(dates, desc="下载转债数据"):
        try:
            df = ak.bond_cb_jsl(d)
            if df is not None and not df.empty:
                CODE_MAP = {"代码": "bond_code"}
                df = df.rename(columns=CODE_MAP)
                if "bond_code" not in df.columns:
                    print(df.columns)          # 打印一次，方便调试
                    raise ValueError("bond_cb_jsl 返回表缺少债券代码列，请检查字段映射")
                df["trade_date"] = pd.to_datetime(d)
                res.append(df)
            else:
                print(f"[空表跳过]{d}")
        except Exception as e:
            print(f"[异常跳过]{d}{e}")
            continue
    if not res:
        raise RuntimeError("所有日期均未抓到数据，请检查接口或日期范围！")
    df_all = pd.concat(res, ignore_index=True)
    df_all.to_pickle(CACHE_NAME)
    return df_all

def fetch_real_history(df: pd.DataFrame) -> pd.DataFrame:
    if "bond_code" not in df.columns:
        raise KeyError(f"缺少 bond_code 列，当前列名：{list(df.columns)}")
    def add_prefix(code: str) -> str:
        return ("sh" if code.startswith("11") else "sz") + code #前缀转换

    out = []
    for code in tqdm(df["bond_code"].unique(), desc="akshare 历史行情"):
        symbol = add_prefix(code)
        try:
            sub = ak.bond_zh_hs_cov_daily(symbol=symbol)
            if sub is None or sub.empty:
                print(f"{symbol} 返回空表，跳过")
                continue
            sub = sub[["date", "close"]].rename(columns={"date": "trade_date", "close": "close_price"})
            sub["bond_code"] = code
            sub["trade_date"] = pd.to_datetime(sub["trade_date"])
            out.append(sub)
        except Exception as e:
            print(f"{symbol} 异常：{e}")
            continue

    if not out:                       # 真 · 一条都没拉到再抛错
        raise RuntimeError("所有 symbol 均拉取失败，请检查 akshare 接口或网络")
    real = pd.concat(out, ignore_index=True)
    # 在 fetch_real_history 最终返回时，删除原始 price 列，统一用 close_price
    return (
        df.drop(columns=["price", "original_price"], errors="ignore")  # 删除原始价格列
        .merge(real, on=["trade_date", "bond_code"], how="left")
)
