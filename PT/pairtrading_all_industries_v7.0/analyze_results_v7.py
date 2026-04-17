# === 全行业配对回测结果可视化分析 v7.0 ===
# 适配 pairtrading_all_industries_v7.0.py 输出格式
# 新增：组合层面分析、滑点影响分析、风险平价权重可视化

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import glob
import os
import warnings

# 完全关闭字体警告
warnings.filterwarnings("ignore", category=UserWarning)

# 强制设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'FangSong', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')

def load_latest_results():
    """加载最新的v7.0回测结果文件"""
    # v7.0文件名格式：组合回测结果_YYYYMMDD_HHMMSS.csv
    files = glob.glob("组合回测结果_*.csv")
    if not files:
        # 兼容v6.0格式
        files = glob.glob("全行业配对回测详细结果_*.csv")
        if not files:
            print("未找到回测结果文件，请先运行 pairtrading_all_industries_v7.0.py")
            return None, None
    
    latest_file = max(files, key=os.path.getctime)
    print(f"加载文件: {latest_file}")
    
    df = pd.read_csv(latest_file, encoding='utf-8-sig')
    
    # 检测版本
    is_v7 = 'industry_weight' in df.columns or 'slippage_impact' in df.columns
    version = "v7.0" if is_v7 else "v6.0"
    print(f"检测到版本: {version}")
    
    return df, version

def plot_industry_weights_v7(df):
    """v7.0专属：行业权重分配可视化"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    if 'industry_weight' not in df.columns:
        print("⚠️ 非v7.0格式，跳过权重图")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. 行业权重饼图
    ax1 = axes[0]
    industry_weights = df.groupby('industry')['industry_weight'].first().sort_values(ascending=False)
    colors = plt.cm.Set3(np.linspace(0, 1, len(industry_weights)))
    
    wedges, texts, autotexts = ax1.pie(
        industry_weights.values, 
        labels=industry_weights.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors
    )
    ax1.set_title('行业权重分配（风险平价）', fontsize=14, fontweight='bold')
    
    # 2. 行业资金分配条形图
    ax2 = axes[1]
    industry_capital = df.groupby('industry')['allocated_capital'].sum().sort_values(ascending=True)
    colors_bar = ['green' if x > df['allocated_capital'].mean() else 'steelblue' for x in industry_capital.values]
    
    bars = ax2.barh(range(len(industry_capital)), industry_capital.values / 1e6, color=colors_bar, alpha=0.8)
    ax2.set_yticks(range(len(industry_capital)))
    ax2.set_yticklabels(industry_capital.index)
    ax2.set_xlabel('分配资金（百万）')
    ax2.set_title('各行业资金分配', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 添加数值标签
    for i, v in enumerate(industry_capital.values):
        ax2.text(v/1e6 + 0.1, i, f'{v/1e6:.2f}M', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('v7_行业权重分配.png', dpi=300, bbox_inches='tight')
    print("✓ 图表已保存: v7_行业权重分配.png")
    plt.show()

def plot_slippage_analysis_v7(df):
    """v7.0专属：滑点影响分析"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    if 'slippage_impact' not in df.columns:
        print("⚠️ 非v7.0格式，跳过滑点分析")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 滑点影响分布
    ax1 = axes[0, 0]
    slippage_pct = df['slippage_impact'] * 100
    ax1.hist(slippage_pct, bins=15, color='coral', alpha=0.7, edgecolor='black')
    ax1.axvline(slippage_pct.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'平均: {slippage_pct.mean():.2f}%')
    ax1.set_xlabel('滑点影响 (%)')
    ax1.set_ylabel('频数')
    ax1.set_title('滑点成本分布')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 滑点 vs 收益率
    ax2 = axes[0, 1]
    scatter = ax2.scatter(df['total_return'] * 100, df['slippage_impact'] * 100,
                       c=df['num_trades'], cmap='viridis', s=100, alpha=0.6)
    ax2.set_xlabel('总收益率 (%)')
    ax2.set_ylabel('滑点影响 (%)')
    ax2.set_title('滑点影响 vs 收益率（颜色=交易次数）')
    plt.colorbar(scatter, ax=ax2, label='交易次数')
    ax2.grid(True, alpha=0.3)
    
    # 3. 行业平均滑点
    ax3 = axes[1, 0]
    industry_slippage = df.groupby('industry')['slippage_impact'].mean() * 100
    industry_slippage = industry_slippage.sort_values(ascending=True)
    colors = ['red' if x > 1 else 'orange' if x > 0.5 else 'green' for x in industry_slippage.values]
    
    bars = ax3.barh(range(len(industry_slippage)), industry_slippage.values, color=colors, alpha=0.8)
    ax3.set_yticks(range(len(industry_slippage)))
    ax3.set_yticklabels(industry_slippage.index)
    ax3.set_xlabel('平均滑点影响 (%)')
    ax3.set_title('各行业平均滑点成本')
    ax3.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='警戒线0.5%')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. 净收益率（扣除滑点）
    ax4 = axes[1, 1]
    df['net_return'] = df['total_return'] + df['slippage_impact']  # 滑点为负值
    net_returns = df.groupby('industry')['net_return'].mean() * 100
    gross_returns = df.groupby('industry')['total_return'].mean() * 100
    
    x = np.arange(len(net_returns))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, gross_returns.values, width, label='毛收益率', color='steelblue', alpha=0.8)
    bars2 = ax4.bar(x + width/2, net_returns.values, width, label='净收益率（扣滑点）', color='coral', alpha=0.8)
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(net_returns.index, rotation=45, ha='right')
    ax4.set_ylabel('收益率 (%)')
    ax4.set_title('毛收益 vs 净收益（按行业）')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('v7_滑点影响分析.png', dpi=300, bbox_inches='tight')
    print("✓ 图表已保存: v7_滑点影响分析.png")
    plt.show()

def plot_portfolio_structure_v7(df):
    """v7.0专属：组合结构可视化"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    if 'pair_weight' not in df.columns:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 配对权重分布（行业内）
    ax1 = axes[0, 0]
    pair_weights = df['pair_weight'] * 100
    ax1.hist(pair_weights, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
    ax1.axvline(pair_weights.mean(), color='red', linestyle='--', linewidth=2,
                label=f'平均: {pair_weights.mean():.2f}%')
    ax1.set_xlabel('配对权重 (%)')
    ax1.set_ylabel('频数')
    ax1.set_title('行业内配对权重分布')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 资金分配散点图（行业权重 vs 配对收益）
    ax2 = axes[0, 1]
    scatter = ax2.scatter(df['industry_weight'] * 100, df['total_return'] * 100,
                         s=df['allocated_capital'] / 1e4,  # 气泡大小=资金
                         c=df['sharpe_ratio'], cmap='RdYlGn', alpha=0.6, edgecolors='black')
    ax2.set_xlabel('行业权重 (%)')
    ax2.set_ylabel('配对收益率 (%)')
    ax2.set_title('行业权重 vs 收益（气泡=资金，颜色=夏普）')
    plt.colorbar(scatter, ax=ax2, label='夏普比率')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 3. 行业-配对矩阵热力图
    ax3 = axes[1, 0]
    pivot_data = df.pivot_table(values='total_return', index='industry', 
                                columns='stock1', aggfunc='first')
    # 简化：只显示部分
    if len(pivot_data) > 0:
        sns.heatmap(pivot_data.fillna(0) * 100, annot=True, fmt='.1f', cmap='RdYlGn', 
                   center=0, ax=ax3, cbar_kws={'label': '收益率(%)'})
        ax3.set_title('行业-股票1 收益率矩阵')
        ax3.set_xlabel('股票1代码')
        ax3.set_ylabel('行业')
    
    # 4. 累计资金分配瀑布图
    ax4 = axes[1, 1]
    sorted_df = df.sort_values('allocated_capital', ascending=False)
    cumulative = np.cumsum(sorted_df['allocated_capital'].values) / 1e6
    x_pos = np.arange(len(sorted_df))
    
    ax4.fill_between(x_pos, 0, cumulative, alpha=0.3, color='steelblue')
    ax4.plot(x_pos, cumulative, color='steelblue', linewidth=2, marker='o', markersize=4)
    ax4.set_xlabel('配对排名')
    ax4.set_ylabel('累计资金（百万）')
    ax4.set_title('资金分配累计图')
    ax4.grid(True, alpha=0.3)
    
    # 添加80%资金线
    total_capital = df['allocated_capital'].sum() / 1e6
    ax4.axhline(y=total_capital * 0.8, color='red', linestyle='--', alpha=0.5, label='80%资金线')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('v7_组合结构分析.png', dpi=300, bbox_inches='tight')
    print("✓ 图表已保存: v7_组合结构分析.png")
    plt.show()

def plot_return_distribution_v7(df):
    """收益率分布分析（v7.0适配）"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 收益率直方图（带滑点对比）
    ax1 = axes[0, 0]
    gross_returns = df['total_return'] * 100
    
    if 'slippage_impact' in df.columns:
        net_returns = (df['total_return'] + df['slippage_impact']) * 100
        ax1.hist(gross_returns, bins=15, alpha=0.5, label='毛收益', color='steelblue', edgecolor='black')
        ax1.hist(net_returns, bins=15, alpha=0.5, label='净收益（扣滑点）', color='coral', edgecolor='black')
        ax1.axvline(net_returns.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'净收益均值: {net_returns.mean():.2f}%')
    else:
        ax1.hist(gross_returns, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axvline(gross_returns.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'均值: {gross_returns.mean():.2f}%')
    
    ax1.set_xlabel('收益率 (%)')
    ax1.set_ylabel('频数')
    ax1.set_title('收益率分布')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 夏普比率分布
    ax2 = axes[0, 1]
    sharpe = df['sharpe_ratio']
    colors = ['green' if x > 1 else 'orange' if x > 0 else 'red' for x in sharpe]
    ax2.hist(sharpe, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(sharpe.mean(), color='red', linestyle='--', linewidth=2,
                label=f'均值: {sharpe.mean():.2f}')
    ax2.axvline(1, color='green', linestyle='--', alpha=0.5, label='夏普=1')
    ax2.set_xlabel('夏普比率')
    ax2.set_ylabel('频数')
    ax2.set_title('夏普比率分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 收益率 vs 夏普（气泡=资金规模）
    ax3 = axes[1, 0]
    sizes = (df['allocated_capital'] / df['allocated_capital'].max() * 500) if 'allocated_capital' in df.columns else 100
    scatter = ax3.scatter(df['total_return'] * 100, df['sharpe_ratio'],
                         c=df['max_drawdown'] * 100, s=sizes, cmap='RdYlGn_r', alpha=0.6, edgecolors='black')
    ax3.set_xlabel('总收益率 (%)')
    ax3.set_ylabel('夏普比率')
    ax3.set_title('收益-风险散点图（颜色=最大回撤，气泡=资金规模）')
    plt.colorbar(scatter, ax=ax3, label='最大回撤(%)')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax3.axvline(0, color='black', linestyle='-', linewidth=0.5)
    
    # 4. 胜率分布
    ax4 = axes[1, 1]
    win_rate = df['win_rate'] * 100
    ax4.hist(win_rate, bins=15, color='green', alpha=0.7, edgecolor='black')
    ax4.axvline(win_rate.mean(), color='red', linestyle='--', linewidth=2,
                label=f'均值: {win_rate.mean():.1f}%')
    ax4.axvline(50, color='black', linestyle='--', linewidth=1, label='50%线')
    ax4.set_xlabel('胜率 (%)')
    ax4.set_ylabel('频数')
    ax4.set_title('胜率分布')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('v7_收益风险分布分析.png', dpi=300, bbox_inches='tight')
    print("✓ 图表已保存: v7_收益风险分布分析.png")
    plt.show()

def plot_top_pairs_table_v7(df, top_n=20):
    """Top N配对表格（v7.0适配）"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 排序：先按行业权重，再按收益率
    if 'industry_weight' in df.columns:
        top_df = df.nlargest(top_n, ['industry_weight', 'total_return'])
    else:
        top_df = df.nlargest(top_n, 'total_return')
    
    # 处理股票代码格式
    top_df['stock1'] = top_df['stock1'].apply(lambda x: f"{int(x):06d}" if pd.notna(x) else x)
    top_df['stock2'] = top_df['stock2'].apply(lambda x: f"{int(x):06d}" if pd.notna(x) else x)
    
    fig, ax = plt.subplots(figsize=(16, top_n * 0.5 + 3))
    ax.axis('tight')
    ax.axis('off')
    
    # 选择显示列
    display_cols = ['industry', 'stock1', 'stock2', 'total_return', 'sharpe_ratio',
                   'max_drawdown', 'win_rate', 'num_trades']
    
    # v7.0专属列
    if 'industry_weight' in top_df.columns:
        display_cols.insert(3, 'industry_weight')
    if 'allocated_capital' in top_df.columns:
        display_cols.insert(4, 'allocated_capital')
    if 'slippage_impact' in top_df.columns:
        display_cols.append('slippage_impact')
    
    # 确保列存在
    display_cols = [c for c in display_cols if c in top_df.columns]
    
    display_df = top_df[display_cols].copy()
    
    # 格式化
    if 'total_return' in display_df.columns:
        display_df['total_return'] = (display_df['total_return'] * 100).round(2).astype(str) + '%'
    if 'industry_weight' in display_df.columns:
        display_df['industry_weight'] = (display_df['industry_weight'] * 100).round(1).astype(str) + '%'
    if 'allocated_capital' in display_df.columns:
        display_df['allocated_capital'] = (display_df['allocated_capital'] / 1e4).round(0).astype(int).astype(str) + '万'
    if 'max_drawdown' in display_df.columns:
        display_df['max_drawdown'] = (display_df['max_drawdown'] * 100).round(2).astype(str) + '%'
    if 'win_rate' in display_df.columns:
        display_df['win_rate'] = (display_df['win_rate'] * 100).round(1).astype(str) + '%'
    if 'slippage_impact' in display_df.columns:
        display_df['slippage_impact'] = (display_df['slippage_impact'] * 100).round(2).astype(str) + '%'
    if 'sharpe_ratio' in display_df.columns:
        display_df['sharpe_ratio'] = display_df['sharpe_ratio'].round(2)
    
    # 创建表格
    table = ax.table(cellText=display_df.values,
                    colLabels=display_cols,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.12] * len(display_cols))
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # 表头样式
    for i in range(len(display_cols)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 行颜色（按收益率）
    for i in range(1, len(top_df) + 1):
        ret = top_df.iloc[i-1]['total_return']
        if ret > 0.2:
            color = '#90EE90'
        elif ret > 0.1:
            color = '#E0F7FA'
        elif ret > 0:
            color = '#FFF9C4'
        else:
            color = '#FFCCBC'
        
        for j in range(len(display_cols)):
            table[(i, j)].set_facecolor(color)
    
    plt.title(f'Top {top_n} 配对排名（v7.0组合策略）', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(f'v7_Top{top_n}_配对排名表.png', dpi=300, bbox_inches='tight')
    print(f"✓ 图表已保存: v7_Top{top_n}_配对排名表.png")
    plt.show()

def generate_summary_report_v7(df, version):
    """生成v7.0汇总报告"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("\n" + "="*80)
    print(f"全行业配对策略回测分析报告 ({version})")
    print("="*80)
    
    print(f"\n【基本信息】")
    print(f"回测行业数: {df['industry'].nunique()}")
    print(f"总配对数: {len(df)}")
    if 'industry_weight' in df.columns:
        print(f"初始资金: {df['allocated_capital'].sum():,.0f}")
    
    print(f"\n【收益统计】")
    print(f"平均收益率: {df['total_return'].mean()*100:.2f}%")
    print(f"收益率中位数: {df['total_return'].median()*100:.2f}%")
    print(f"最高收益率: {df['total_return'].max()*100:.2f}%")
    print(f"最低收益率: {df['total_return'].min()*100:.2f}%")
    print(f"正收益配对数: {len(df[df['total_return'] > 0])} ({len(df[df['total_return'] > 0])/len(df)*100:.1f}%)")
    
    # v7.0专属：净收益统计
    if 'slippage_impact' in df.columns:
        df['net_return'] = df['total_return'] + df['slippage_impact']
        print(f"\n【滑点影响（v7.0）】")
        print(f"平均滑点成本: {df['slippage_impact'].mean()*100:.2f}%")
        print(f"平均净收益率: {df['net_return'].mean()*100:.2f}%")
        print(f"滑点成本占比: {abs(df['slippage_impact'].sum())/df['total_return'].sum()*100:.1f}%")
    
    print(f"\n【风险指标】")
    print(f"平均夏普比率: {df['sharpe_ratio'].mean():.2f}")
    print(f"夏普比率>1的配对数: {len(df[df['sharpe_ratio'] > 1])}")
    print(f"平均最大回撤: {df['max_drawdown'].mean()*100:.2f}%")
    print(f"平均胜率: {df['win_rate'].mean()*100:.1f}%")
    
    # v7.0专属：权重统计
    if 'industry_weight' in df.columns:
        print(f"\n【资金分配（v7.0）】")
        industry_stats = df.groupby('industry').agg({
            'industry_weight': 'first',
            'allocated_capital': 'sum',
            'total_return': 'mean'
        }).sort_values('industry_weight', ascending=False)
        
        for idx, (industry, row) in enumerate(industry_stats.head(5).iterrows(), 1):
            print(f"{idx}. {industry}: 权重{row['industry_weight']*100:.1f}%, "
                  f"资金{row['allocated_capital']:,.0f}, 平均收益{row['total_return']*100:.2f}%")
    
    print(f"\n【最佳配对Top 5】")
    top_pairs = df.nlargest(5, 'total_return')
    for i, (_, row) in enumerate(top_pairs.iterrows(), 1):
        extra_info = ""
        if 'slippage_impact' in row:
            extra_info = f" | 滑点:{row['slippage_impact']*100:.2f}%"
        print(f"{i}. {row['industry']}: {row['stock1']}-{row['stock2']} | "
              f"收益:{row['total_return']*100:.2f}% | 夏普:{row['sharpe_ratio']:.2f}{extra_info}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    # 加载数据
    df, version = load_latest_results()
    if df is None:
        exit()
    
    # 生成报告
    generate_summary_report_v7(df, version)
    
    # 生成图表
    print("\n生成可视化图表...")
    
    if version == "v7.0":
        # v7.0专属图表
        print("\n[v7.0专属] 行业权重分配...")
        plot_industry_weights_v7(df)
        
        print("\n[v7.0专属] 滑点影响分析...")
        plot_slippage_analysis_v7(df)
        
        print("\n[v7.0专属] 组合结构分析...")
        plot_portfolio_structure_v7(df)
    
    # 通用图表（v6.0/v7.0兼容）
    print("\n收益风险分布分析...")
    plot_return_distribution_v7(df)
    
    print("\nTop配对排名表...")
    plot_top_pairs_table_v7(df, top_n=20)
    
    print("\n✓ 全部分析完成！")
    if version == "v7.0":
        print("v7.0专属图表：")
        print("  - v7_行业权重分配.png")
        print("  - v7_滑点影响分析.png")
        print("  - v7_组合结构分析.png")
    print("通用图表：")
    print("  - v7_收益风险分布分析.png")
    print(f"  - v7_Top20_配对排名表.png")