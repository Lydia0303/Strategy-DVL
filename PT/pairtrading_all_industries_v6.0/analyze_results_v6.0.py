# === 全行业回测结果可视化分析 ===
# 配合 pairtrading_all_industries_v6.0.py 使用

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import glob
import os
import matplotlib
import warnings


# 完全关闭字体警告
warnings.filterwarnings("ignore", category=UserWarning)

# 强制设置字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'FangSong', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# 验证字体设置
print("当前字体设置:", plt.rcParams['font.sans-serif'])
print("字体缓存位置:", matplotlib.get_cachedir())

plt.style.use('seaborn-v0_8-darkgrid')

def load_latest_results():
    """加载最新的回测结果文件"""
    # 查找最新的详细结果文件
    files = glob.glob("全行业配对回测详细结果_*.csv")
    if not files:
        print("未找到回测结果文件，请先运行回测程序")
        return None

    latest_file = max(files, key=os.path.getctime)
    print(f"加载文件: {latest_file}")

    df = pd.read_csv(latest_file, encoding='utf-8-sig')
    return df

def plot_industry_ranking(df):
    """绘制各行业平均收益率排名"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    industry_perf = df.groupby('industry')['total_return'].mean().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(6, len(industry_perf) * 0.4)))

    colors = ['green' if x > 0 else 'red' for x in industry_perf.values]
    bars = ax.barh(range(len(industry_perf)), industry_perf.values * 100, color=colors, alpha=0.7)

    ax.set_yticks(range(len(industry_perf)))
    ax.set_yticklabels(industry_perf.index)
    ax.set_xlabel('Average return (%)')
    ax.set_title('Average return rankings by industry', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')

    # 添加数值标签
    for i, (idx, val) in enumerate(industry_perf.items()):
        ax.text(val * 100 + (1 if val > 0 else -1), i, f'{val*100:.2f}%', 
                va='center', ha='left' if val > 0 else 'right', fontsize=9)

    plt.tight_layout()
    plt.savefig('全行业收益率排名.png', dpi=300, bbox_inches='tight')
    print("✓ 图表已保存: 全行业收益率排名.png")
    plt.show()

def plot_return_distribution(df):
    """绘制收益率分布图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 收益率直方图
    ax1 = axes[0, 0]
    returns = df['total_return'] * 100
    ax1.hist(returns, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(returns.mean(), color='red', linestyle='--', linewidth=2, label=f'mean: {returns.mean():.2f}%')
    ax1.axvline(0, color='black', linestyle='-', linewidth=1)
    ax1.set_xlabel('return (%)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('return distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 夏普比率分布
    ax2 = axes[0, 1]
    sharpe = df['sharpe_ratio']
    ax2.hist(sharpe, bins=20, color='orange', alpha=0.7, edgecolor='black')
    ax2.axvline(sharpe.mean(), color='red', linestyle='--', linewidth=2, label=f'mean: {sharpe.mean():.2f}')
    ax2.axvline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Sharpe ratio')
    ax2.set_ylabel('Frequency')
    ax2.set_title('SR Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 收益率 vs 夏普比率散点图
    ax3 = axes[1, 0]
    scatter = ax3.scatter(df['total_return'] * 100, df['sharpe_ratio'], 
                         c=df['max_drawdown'] * 100, cmap='RdYlGn_r', s=100, alpha=0.6)
    ax3.set_xlabel('return (%)')
    ax3.set_ylabel('sharpe ratio')
    ax3.set_title('return vs SR (color=Maximum drawdown)')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax3.axvline(0, color='black', linestyle='-', linewidth=0.5)
    plt.colorbar(scatter, ax=ax3, label='Maximum drawdown (%)')

    # 4. 胜率分布
    ax4 = axes[1, 1]
    win_rate = df['win_rate'] * 100
    ax4.hist(win_rate, bins=15, color='green', alpha=0.7, edgecolor='black')
    ax4.axvline(win_rate.mean(), color='red', linestyle='--', linewidth=2, label=f'mean: {win_rate.mean():.1f}%')
    ax4.axvline(50, color='black', linestyle='--', linewidth=1, label='50%_line')
    ax4.set_xlabel('win rate (%)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Win rate distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('收益风险分布分析.png', dpi=300, bbox_inches='tight')
    print("✓ 图表已保存: 收益风险分布分析.png")
    plt.show()

def plot_top_pairs_table(df, top_n=20):
    """绘制Top N配对表格"""
    top_df = df.nlargest(top_n, 'total_return')[['industry', 'stock1', 'stock2', 
                                                  'total_return', 'sharpe_ratio', 
                                                  'max_drawdown', 'win_rate']]

    # 【新增】处理股票代码：转换为字符串并保留前导零（假设股票代码最多6位）
    top_df['stock1'] = top_df['stock1'].apply(lambda x: f"{int(x):06d}" if pd.notna(x) else x)
    top_df['stock2'] = top_df['stock2'].apply(lambda x: f"{int(x):06d}" if pd.notna(x) else x)
    
    fig, ax = plt.subplots(figsize=(14, top_n * 0.5 + 2))
    ax.axis('tight')
    ax.axis('off')

    # 格式化数据
    display_df = top_df.copy()
    display_df['total_return'] = (display_df['total_return'] * 100).round(2).astype(str) + '%'
    display_df['max_drawdown'] = (display_df['max_drawdown'] * 100).round(2).astype(str) + '%'
    display_df['win_rate'] = (display_df['win_rate'] * 100).round(1).astype(str) + '%'
    display_df['sharpe_ratio'] = display_df['sharpe_ratio'].round(2)

    table = ax.table(cellText=display_df.values,
                    colLabels=['industry', 'stock1', 'stock2', 'total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # 设置表头颜色
    for i in range(len(display_df.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 根据收益率设置行颜色
    for i in range(1, len(top_df) + 1):
        ret = top_df.iloc[i-1]['total_return']
        if ret > 0.2:
            color = '#90EE90'  # 浅绿
        elif ret > 0:
            color = '#E0F7FA'  # 浅青
        else:
            color = '#FFCCBC'  # 浅红

        for j in range(len(display_df.columns)):
            table[(i, j)].set_facecolor(color)

    plt.title(f'Top {top_n} Return ranking for pairs', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(f'Top{top_n}_配对排名表.png', dpi=300, bbox_inches='tight')
    print(f"✓ 图表已保存: Top{top_n}_配对排名表.png")
    plt.show()

def generate_summary_report(df):
    """生成文字汇总报告"""
    print("\n" + "="*80)
    print("全行业配对策略回测分析报告")
    print("="*80)

    print(f"\n【基本信息】")
    print(f"回测行业数: {df['industry'].nunique()}")
    print(f"总配对数: {len(df)}")
    print(f"数据时间: {df.columns[0] if len(df.columns) > 0 else 'N/A'}")

    print(f"\n【收益统计】")
    print(f"平均收益率: {df['total_return'].mean()*100:.2f}%")
    print(f"收益率中位数: {df['total_return'].median()*100:.2f}%")
    print(f"最高收益率: {df['total_return'].max()*100:.2f}% ({df.loc[df['total_return'].idxmax(), 'industry']})")
    print(f"最低收益率: {df['total_return'].min()*100:.2f}% ({df.loc[df['total_return'].idxmin(), 'industry']})")
    print(f"正收益配对数: {len(df[df['total_return'] > 0])} ({len(df[df['total_return'] > 0])/len(df)*100:.1f}%)")

    print(f"\n【风险指标】")
    print(f"平均夏普比率: {df['sharpe_ratio'].mean():.2f}")
    print(f"夏普比率>1的配对数: {len(df[df['sharpe_ratio'] > 1])}")
    print(f"平均最大回撤: {df['max_drawdown'].mean()*100:.2f}%")
    print(f"平均胜率: {df['win_rate'].mean()*100:.1f}%")

    print(f"\n【行业表现Top 5】")
    top_industries = df.groupby('industry')['total_return'].mean().sort_values(ascending=False).head(5)
    for i, (industry, ret) in enumerate(top_industries.items(), 1):
        count = len(df[df['industry'] == industry])
        print(f"{i}. {industry}: {ret*100:.2f}% (共{count}对)")

    print(f"\n【最佳配对Top 5】")
    top_pairs = df.nlargest(5, 'total_return')[['industry', 'stock1', 'stock2', 'total_return', 'sharpe_ratio']]
    for i, (_, row) in enumerate(top_pairs.iterrows(), 1):
        print(f"{i}. {row['industry']}: {row['stock1']}-{row['stock2']} | 收益: {row['total_return']*100:.2f}% | 夏普: {row['sharpe_ratio']:.2f}")

    print("\n" + "="*80)

if __name__ == "__main__":
    # 加载数据
    df = load_latest_results()
    if df is not None:
        # 生成报告
        generate_summary_report(df)

        # 生成图表
        print("\n生成可视化图表...")
        plot_industry_ranking(df)
        plot_return_distribution(df)
        plot_top_pairs_table(df, top_n=20)

        print("\n✓ 全部分析完成！")
