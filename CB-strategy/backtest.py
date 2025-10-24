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
                     hold_n: int = 10,
                     premium_max: float = 0.3,
                     rating_min: int = 2,
                     balance_min: float = 1.5,
                     remain_min: float = 0.5,
                     ytm_min: float = -5,
                     hold_days: int = 5,
                     comm: float = 0.001,
                     slip: float = 0.0005):
    df["pred"] = np.nan
    # 统一转成 datetime64[ns]，去掉时分秒
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.floor("D")

    months = sorted(df["month_end"].unique())[1:]
    records = []

    for m in tqdm(months, desc="rolling"):
        # 训练集边界
        train_end = pd.to_datetime(m) - pd.offsets.BMonthEnd(1)
        train_start = train_end - pd.DateOffset(months=2)
        # 转成 datetime64 保持一致
        train_start = pd.to_datetime(train_start.date())
        train_end   = pd.to_datetime(train_end.date())
        
        train_df = df[(df["trade_date"] >= train_start.date()) &
                      (df["trade_date"] <= train_end.date())]\
                      .dropna(subset=FEATURES + ["label"])
        if len(train_df) < 100:
            continue
        from model import train_model
        model = train_model(train_df[FEATURES], train_df["label"], tune=False)

        # 预测集
        pred_df = df[df["month_end"] == m].copy()
        # 强赎过滤
        call_list = pred_df[pred_df["price"] > pred_df["call_trigger"] * 0.98]["bond_code"].tolist()
        pred_df = pred_df[~pred_df["bond_code"].isin(call_list)]
        # 基础筛选
        pred_df = pred_df[(pred_df["premium"] <= premium_max) &
                          (pred_df["rating_score"] >= rating_min) &
                          (pred_df["balance"] >= balance_min) &
                          (pred_df["remain_years"] >= remain_min) &
                          (pred_df["ytm"] >= ytm_min)].dropna(subset=FEATURES)
        if pred_df.empty:
            continue
        pred_df["pred"] = model.predict(pred_df[FEATURES])
        selected = pred_df.nlargest(hold_n, "pred")["bond_code"].tolist()
        # 收益
        sub = df[(df["month_end"] == m) & (df["bond_code"].isin(selected))]
        ret = sub["label"].mean() if len(sub) else 0
        ret -= comm + slip
        records.append({"month_end": m, "selected": selected, "ret": ret})

    return pd.DataFrame(records)

"""
def rolling_backtest(df: pd.DataFrame,
                     hold_n: int = 5,
                     premium_max: float = 50,
                     rating_min: int = 3):
    df["pred"] = None  # 新增"pred"列，用于存储模型预测的涨跌幅（初始为None）
    months = sorted(df["month_end"].unique())[1: ]  # # 取所有不重复的月末日期，跳过第一个月（留作初始训练）
    print(f"[调试] 总月份数: {len(months)}, 列表: {months}")  # 打印回测涉及的月份数量和具体日期

    records = []   # 存储每月回测结果（选中的转债、收益等）
    for m in tqdm(months, desc="滚动回测"):   # 遍历每个回测月份，tqdm显示进度条
        # 训练集
        # 训练集的结束时间：当前回测月的上一个月末（如回测月是2023-02-28，则训练截止到2023-01-31）
        train_end = m - pd.offsets.BMonthEnd(1)
        # 训练集的开始时间：训练结束时间往前推1个月（如训练截止到2023-01-31，则开始于2022-12-31）
        train_start = train_end - pd.DateOffset(months=1)
        # 转换为date类型（去掉时间戳，只保留日期）
        train_start = train_start.date()
        train_end   = train_end.date()
        # 筛选训练集：只包含train_start到train_end之间的样本
        train_df = df[(df["month_end"] >= train_start) & (df["month_end"] <= train_end)]   # 打印训练样本量
        print(f"[调试] 训练月份 {train_start} ~ {train_end} 样本数: {len(train_df)}")
        
        # 空表保护：如果训练样本少于100条，跳过本轮（避免模型训练不稳定）
        if len(train_df) < 100:
            continue
        # 用训练数据和特征列训练模型（调用之前的train_model函数）
        model = train_model(train_df, FEATURES)
        
        # 预测集
        # 1. 先拿到预测集：当前回测月的数据（如回测月是2023-02-28，则预测该月的转债表现）
        pred_df = df[df["month_end"] == m].copy()
        # 2. 基础筛选
        pred_df = pred_df[(pred_df["premium"] <= premium_max) &
                          (pred_df["rating_score"] >= rating_min) &
                          (pred_df["剩余规模"] >= 1.5 ) &
                          (pred_df["剩余年限"] >= 0.5 ) &
                          (pred_df["到期税前收益"] >= 0 )]
        # 3. 去掉未来价格缺失的行（确保能计算实际收益label）
        pred_df = pred_df.dropna(subset=["future_price"])
        # 4. 空表判断：如果筛选后没有符合条件的转债，跳过本轮
        if pred_df.empty:
            continue
        # 5. 用训练好的模型预测当月转债的涨跌幅（调用predict_model函数）
        pred_df["pred"] = predict_model(model, pred_df, FEATURES)
        # 选择预测涨跌幅最高的前hold_n只转债（如前20只）
        selected = pred_df.nlargest(hold_n, "pred")["bond_code"].tolist()

        # 提取当月选中的转债的实际收益（label列）
        sub = df[(df["month_end"] == m) & (df["bond_code"].isin(selected))]
        # 打印当月选中的转债数量和平均实际收益
        print(f"[调试] {m} 选中 {len(selected)} 只券, 平均收益: {sub['label'].mean():.4f}")
        # 记录当月结果（月末日期、选中的转债、平均收益）
        records.append({"month_end": m,
                         "selected": selected,
                         "ret": sub["label"].mean() if len(sub) else 0
                         })
    
    portfolio = pd.DataFrame(records) # 将所有月份的回测结果转换为DataFrame
    # 空表处理：如果没有任何回测结果，返回空表（保证格式正确）
    if portfolio.empty:
        print("[警告] 未选中任何月份，返回空表！")
        portfolio = pd.DataFrame(columns=["month_end", "ret"])  # 保证列存在
    return portfolio
"""
