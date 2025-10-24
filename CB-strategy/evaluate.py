"""
计算绩效 + 画图
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

def performance(port: pd.DataFrame):
    ret = pd.Series(port["ret"].values, index=pd.to_datetime(port["month_end"]))
    ann = ret.mean() * 12
    vol = ret.std() * np.sqrt(12)
    sharpe = ann / vol if vol else 0
    cum = (1 + ret).cumprod()
    maxdd = (cum.cummax() - cum).max()
    print("-------------- 绩效 --------------")
    print(f"年化收益: {ann:.2%}")
    print(f"年化波动: {vol:.2%}")
    print(f"夏普比率: {sharpe:.2f}")
    print(f"最大回撤: {maxdd:.2%}")
    cum.plot(figsize=(10, 4), title="可转债 ML 策略累计收益")
    plt.savefig("equity_curve.png")
    plt.show()

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "SimHei"        # Windows 黑体
plt.rcParams["axes.unicode_minus"] = False    # 负号正常显示

def calc_port_ret(port: pd.DataFrame, df: pd.DataFrame):
    ret = []
    for _, row in port.iterrows():
        m, codes = row["month_end"], row["selected"]
        sub = df[(df["month_end"] == m) & (df["bond_code"].isin(codes))]
        ret.append(sub["label"].mean() if len(sub) else 0)
    port["ret"] = ret
    return port

def performance(returns: pd.Series):
    ann = returns.mean() * 12
    vol = returns.std() * np.sqrt(12)
    sharpe = ann / vol if vol else 0
    cum = (1 + returns).cumprod()
    maxdd = (cum.cummax() - cum).max()
    print(f"年化收益: {ann:.2%}")
    print(f"年化波动: {vol:.2%}")
    print(f"夏普比率: {sharpe:.2f}")
    print(f"最大回撤: {maxdd:.2%}")
    cum.plot(figsize=(10, 4), title="可转债策略累计收益")
    plt.savefig("equity_curve.png")
    plt.show()
"""
