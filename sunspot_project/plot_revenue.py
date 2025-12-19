# -*- coding: utf-8 -*-
"""
产品双账户收益曲线 + 近 10 日账户盈亏
Excel 列必须包含（顺序不限）：
账户1每日余额、账户1每日盈亏、账户1累计总盈亏
账户2每日余额、账户2每日盈亏、账户2累计总盈亏
产品每日总盈亏、产品累计总盈亏、沪深300（收盘指数）
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import ticker as mtick

# ========== 1. 参数区 ==========
file_path = r'C:\Users\hz\Desktop\日常\光照项目\光哥每日盈亏.xlsx'   # 文件路径
sheet_name = '每日图'                  # 具体表名
date_col = '时间'               # 
# 列名映射
cols_map = {
    '银河账户余额': 'YH_balance',
    '银河账户盈亏': 'YH_profit_loss',
    '银河总盈亏': 'YH_cum',
    '信达账户余额': 'XD_balance',
    '信达账户盈亏': 'XD_profit_loss',
    '信达总盈亏': 'XD_cum',
    '每日总盈亏': 'stra_profit_loss',
    '策略累计总盈亏': 'stra_cum',
    '沪深300': 'CSI300_close'  
}

# ========== 2. 读数 ==========
df = pd.read_excel(file_path, sheet_name=sheet_name)
df = df.rename(columns=cols_map)
# 确保日期列是 datetime
df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(date_col).reset_index(drop=True)

# ========== 3. 计算指标 ==========
def calc_indicators(nav_pseries: pd.Series, ann_days: int = 252):
    """输入“单位净值序列”，返回常用指标 dict（此时nav_series初始值=1） dict"""
    ret = nav_pseries.pct_change().dropna()
    ann_ret = ret.mean() * ann_days
    ann_vol = ret.std() * np.sqrt(ann_days)
    sharpe = ann_ret / ann_vol if ann_vol else np.nan
    max_dd = (nav_pseries.cummax() - nav_pseries).max() / nav_pseries.iloc[0]
    calmar = ann_ret / max_dd if max_dd else np.nan
    win_rate = (ret > 0).mean()
    plr = (ret[ret > 0].mean()) / (-ret[ret < 0].mean()) if (ret < 0).any() else np.nan
    total_ret = nav_pseries.iloc[-1] / nav_pseries.iloc[0] - 1
    return {
        '累计收益': f"{total_ret:.2%}",
        '年化收益': f"{ann_ret:.2%}",
        '年化波动': f"{ann_vol:.2%}",
        '夏普': f"{sharpe:.2f}",
        '最大回撤': f"{max_dd:.2%}",
        'Calmar': f"{calmar:.2f}",
        '胜率': f"{win_rate:.1%}",
        '盈亏比': f"{plr:.2f}",
    }

# ========== 新增：计算产品净值 + 沪深300归一化净值 ==========
# 1. 产品单位净值（初始值=1，累计净值与单位净值一致）；产品净值 = (初始本金 + 累计盈亏) / 初始本金
initial_capital = 30300000  # 银河1030W 信达2000W
df['product_unit_nav'] = (initial_capital + df['stra_cum']) / initial_capital

# 用产品单位净值算指标
indicators = calc_indicators(df['product_unit_nav'])

# 2. 沪深300归一化净值（初始值=1，方便和产品净值对比）
df['CSI300_nav'] = df['CSI300_close'] / df['CSI300_close'].iloc[0]  # 归一化到初始值1

# ========== 4. 画图 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文
plt.rcParams['axes.unicode_minus'] = False

# ========== 4. 画图（核心：用gridspec规划布局） ==========
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文
plt.rcParams['axes.unicode_minus'] = False

# 初始化画布
fig = plt.figure(figsize=(14, 8))  # 调整画布大小适配表格

# 用gridspec划分区域：主图(3行) + 副图(2行)
gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 0], hspace=0.5)  # 比例可调整
ax1 = fig.add_subplot(gs[0, 0])    # 主图（第1行）
ax2 = fig.add_subplot(gs[1, 0])    # 副图（第2行）

# 4.1 主图：产品单位净值 + 沪深300基准
# 绘制产品单位净值
ax1.plot(df[date_col], df['product_unit_nav'], color='royalblue', lw=2, label='策略')
# 绘制沪深300归一化净值
ax1.plot(df[date_col], df['CSI300_nav'], color='red', lw=2, linestyle='--', label='沪深300')

ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.2f}"))
ax1.set_title('光照策略净值走势', fontsize=14, pad=20, weight='bold')
ax1.grid(alpha=0.3)
ax1.set_ylabel('单位净值')
ax1.legend(loc='upper left')  # 添加图例区分曲线

# 在图上方写指标
txt = ' | '.join([f"{k}: {v}" for k, v in indicators.items()])
ax1.text(0.5, 1.15, txt, transform=ax1.transAxes, ha='center', va='bottom',
         fontsize=11, color='black', weight='bold')

# 4.2 副图：近 10 个交易日各账户日盈亏 + 总盈亏标注 + 标注每个交易日的盈亏数据
last10 = df.tail(10)
x = np.arange(len(last10))
width = 0.25
# 绘制柱状图
stra_bars = ax2.bar(x - width, last10['stra_profit_loss'], width, label='总盈亏', color='royalblue')
yh_bars = ax2.bar(x, last10['YH_profit_loss'], width, label='银河账户', color='seagreen')
xd_bars = ax2.bar(x + width, last10['XD_profit_loss'], width, label='信达账户', color='coral')

# 给每个柱子添加盈亏数值标签
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        # 标签位置：正数在柱子上方，负数在柱子下方
        va = 'bottom' if height >= 0 else 'top'
        offset = 100 if height >= 0 else -100
        ax2.text(bar.get_x() + bar.get_width()/2., height + offset,
                 f"{height:.0f}",  # 保留整数
                 ha='center', va=va, fontsize=9)

add_labels(yh_bars)  # 银河账户标签
add_labels(xd_bars)  # 信达账户标签
add_labels(stra_bars)
# 副图样式调整
ax2.axhline(0, color='black', lw=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels(last10[date_col].dt.strftime('%m-%d'), rotation=45)
ax2.set_title('近 10 个交易日各账户日盈亏', fontsize=14, pad=15)
ax2.legend(loc='upper left')
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylabel('日盈亏（元）')
# 调整y轴范围，避免标签超出图框
y_min = last10[['YH_profit_loss', 'XD_profit_loss', 'stra_profit_loss']].min().min()
y_max = last10[['YH_profit_loss', 'XD_profit_loss', 'stra_profit_loss']].max().max()
ax2.set_ylim(y_min * 1.2, y_max * 1.2)  # 扩大1.2倍范围

# 最新盈亏余额标注
latest_date = last10[date_col].iloc[-1].strftime('%Y-%m-%d')  # 获取最新日期
profit_text = (
    f"{latest_date}\n"
    f"银河余额：{last10['YH_balance'].iloc[-1]:.2f}元\n"
    f"信达余额：{last10['XD_balance'].iloc[-1]:.2f}元\n"
    f"银河总盈亏：{last10['YH_cum'].iloc[-1]:.2f}元\n"
    f"信达总盈亏：{last10['XD_cum'].iloc[-1]:.2f}元"
)
fig.text(0.85, 0.4, profit_text, ha='left', va='center', 
         fontsize=11, bbox=dict(boxstyle='round,pad=0.8', facecolor='white', edgecolor='gray', alpha=0.8))

plt.subplots_adjust(hspace=0.3, bottom=0.1)
plt.show()




