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
