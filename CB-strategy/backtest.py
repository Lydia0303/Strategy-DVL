"""
滚动回测：每月训练 → 选 TopN
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
# from model import train_model, predict_model
from features import FEATURES
# from datetime import date 

def rolling_backtest(df: pd.DataFrame,
                     hold_n: int = 5,
                     premium_max: float = 1.0,
                     rating_min: int = 2.0,
                     balance_min: float = 0,
                     remain_min: float = 0,
                     ytm_min: float = -5,
                     hold_days: int = 20,
                     comm: float = 0.001,
                     slip: float = 0.0005):
    df["pred"] = np.nan
    # 统一转成 datetime64[ns]，去掉时分秒
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.floor("D")

    months = sorted(df["month_end"].unique())[1:]
    records = [] #确保非空

    for m in tqdm(months, desc="rolling"):
        # 训练集边界
        train_end = pd.to_datetime(m) - pd.offsets.BMonthEnd(1)
        train_start = train_end - pd.DateOffset(months=2)
        # 转成 datetime64 保持一致
        train_start = pd.to_datetime(train_start.date())
        train_end   = pd.to_datetime(train_end.date())
        
        train_df = df[(df["trade_date"] >= train_start) &
                      (df["trade_date"] <= train_end)]\
                      .dropna(subset=FEATURES + ["label"])
        if len(train_df) < 10:
             # 即使跳过也要写空记录，保证列存在
            records.append({"month_end": m, "selected": [], "ret": 0.0})
            continue
        from model import train_model
        model = train_model(train_df[FEATURES], train_df["label"], tune=False)

        # 预测集
        pred_df = df[df["month_end"] == m].copy()
        print(f"{m} 原始截面 {len(pred_df)} 条")
        for col, thresh in [('premium', premium_max), ('rating_score', rating_min), ('balance', balance_min), ('remain_years', remain_min), ('ytm', ytm_min)]:
            pred_df = pred_df[pred_df[col] >= thresh] if col in ('rating_score','ytm','remain_years','balance') \
                                           else pred_df[pred_df[col] <= thresh]
            print(f"  经过 {col} 门槛后 {len(pred_df)} 条")
        # 强赎过滤
        # call_list = pred_df[pred_df["price"] > pred_df["call_trigger"] * 0.90]["bond_code"].tolist()
        # pred_df = pred_df[~pred_df["bond_code"].isin(call_list)]
        print(f"  经过强赎门槛后（已关闭）, {len(pred_df)} , 条")
        # 基础筛选
        pred_df = pred_df[(pred_df["premium"] <= premium_max) &
                          (pred_df["rating_score"] >= rating_min) &
                          (pred_df["balance"] >= balance_min) &
                          (pred_df["remain_years"] >= remain_min) &
                          (pred_df["ytm"] >= ytm_min)].dropna(subset=FEATURES)
        # print(f"{m} 原始截面 {len(df[df['month_end']==m])} 条，"f"筛后 {len(pred_df)} 条，call_list={len(call_list)}")
        if pred_df.empty:
            records.append({"month_end": m, "selected": [], "ret": 0.0})
            continue
        pred_df["pred"] = model.predict(pred_df[FEATURES])
        print(f"  预测得分范围：{pred_df['pred'].min():.6f} ~ {pred_df['pred'].max():.6f}")
        # 预测后先按bond_code去重，再取前5
        pred_df = pred_df.drop_duplicates(subset=["bond_code"])
        selected = pred_df.nlargest(hold_n, "pred")["bond_code"].tolist()
        # 收益
        sub = df[(df["month_end"] == m) & (df["bond_code"].isin(selected))]
        ret = sub["label"].mean() if len(sub) else 0
        ret -= comm + slip
        records.append({"month_end": m, "selected": selected, "ret": ret})
    # 如果全程无记录，返回带列的空表
    if not records:
        records = [{"month_end": pd.to_datetime("2023-01-31").date(), "selected": [], "ret": 0.0}]
    return pd.DataFrame(records)

