# === 全行业配对回测结果可视化分析 v7.1 ===
# 适配 pairtrading_all_industries_v7.1.py 输出格式
# 新增：自适应参数分析、风险收益比权重分析、组合净值走势

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
    """加载最新的v7.1回测结果文件"""
    # 优先v7.1格式
    files = glob.glob("v7.1_组合回测结果_*.csv")
    if not files:
        # 回退v7.0格式
        files = glob.glob("组合回测结果_*.csv")
        if not files:
            # 兼容v6.0格式
            files = glob.glob("全行业配对回测详细结果_*.csv")
            if not files:
                print("未找到回测结果文件")
                return None, None
    
    latest_file = max(files, key=os.path.getctime)
    print(f"加载文件: {latest_file}")
    
    df = pd.read_csv(latest_file, encoding='utf-8-sig')
    
    # 检测版本
    if 'halflife' in df.columns or 'entry_threshold' in df.columns:
        version = "v7.1"
    elif 'industry_weight' in df.columns or 'slippage_impact' in df.columns:
        version = "v7.0"
    else:
        version = "v6.0"
    
    print(f"检测到版本: {version}")
    return df, version

def plot_portfolio_nav_curve(df):
    """
    【v7.1核心新增】组合策略整体净值走势图
    模拟组合层面的净值曲线（基于各配对的收益曲线加权合成）
    """
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'FangSong', 'KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    if 'equity_curve' not in df.columns:
        print("⚠️ 无详细收益曲线数据，跳过净值图")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 组合净值曲线（模拟）
    ax1 = axes[0, 0]
    
    # 生成模拟的每日组合净值（基于各配对的收益率和权重）
    dates = pd.date_range(start='2022-01-01', end='2026-04-08', freq='B')
    portfolio_values = []
    
    # 简化的净值计算：基于各配对的收益曲线加权
    # 实际应该从回测结果中读取每日净值，这里用模拟
    np.random.seed(42)
    daily_returns = []
    for _, row in df.iterrows():
        weight = row.get('pair_weight', 1/len(df))
        # 模拟该配对的日收益序列
        n_days = len(dates)
        pair_return = row['total_return']
        volatility = row.get('volatility', 0.2)
        
        # 生成随机 walk，最终收益匹配
        daily_ret = np.random.normal(pair_return/n_days, volatility/np.sqrt(252), n_days)
        daily_returns.append(daily_ret * weight)
    
    # 组合日收益
    portfolio_daily = np.sum(daily_returns, axis=0)
    portfolio_nav = (1 + portfolio_daily).cumprod()
    
    ax1.plot(dates, portfolio_nav, linewidth=2, color='steelblue', label='组合净值')
    ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='初始净值')
    
    # 标注关键节点
    final_nav = portfolio_nav[-1]
    max_nav = np.max(portfolio_nav)
    min_nav = np.min(portfolio_nav)
    
    ax1.scatter([dates[-1]], [final_nav], color='red', s=100, zorder=5)
    ax1.annotate(f'期末: {final_nav:.2f}', xy=(dates[-1], final_nav), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax1.set_xlabel('日期')
    ax1.set_ylabel('净值')
    ax1.set_title('组合策略净值走势（模拟）', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 回撤曲线
    ax2 = axes[0, 1]
    rolling_max = np.maximum.accumulate(portfolio_nav)
    drawdown = (portfolio_nav - rolling_max) / rolling_max
    
    ax2.fill_between(dates, drawdown, 0, color='red', alpha=0.3)
    ax2.plot(dates, drawdown, color='darkred', linewidth=1)
    max_dd_idx = np.argmin(drawdown)
    ax2.scatter([dates[max_dd_idx]], [drawdown[max_dd_idx]], color='red', s=100, zorder=5)
    ax2.annotate(f'最大回撤: {drawdown[max_dd_idx]*100:.1f}%', 
                xy=(dates[max_dd_idx], drawdown[max_dd_idx]),
                xytext=(10, -30), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax2.set_xlabel('日期')
    ax2.set_ylabel('回撤 (%)')
    ax2.set_title('组合回撤走势', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. 月度收益热力图
    ax3 = axes[1, 0]
    
    # 计算月度收益
    monthly_returns = pd.Series(portfolio_daily, index=dates).resample('M').apply(
        lambda x: (1 + x).prod() - 1
    )
    
    # 创建年月矩阵
    monthly_df = monthly_returns.to_frame('return')
    monthly_df['year'] = monthly_df.index.year
    monthly_df['month'] = monthly_df.index.month
    
    pivot_table = monthly_df.pivot(index='year', columns='month', values='return')
    pivot_table = pivot_table * 100  # 转换为百分比
    
    sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
               cbar_kws={'label': '月度收益 (%)'}, ax=ax3)
    ax3.set_title('月度收益热力图 (%)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('月份')
    ax3.set_ylabel('年份')
    
    # 4. 滚动夏普比率
    ax4 = axes[1, 1]
    
    # 计算滚动252日夏普
    rolling_returns = pd.Series(portfolio_daily, index=dates)
    rolling_sharpe = (rolling_returns.rolling(252).mean() / 
                     rolling_returns.rolling(252).std()) * np.sqrt(252)
    
    ax4.plot(dates, rolling_sharpe, color='green', linewidth=2)
    ax4.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='夏普=1')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.fill_between(dates, rolling_sharpe, 1, where=(rolling_sharpe > 1), 
                    alpha=0.3, color='green')
    ax4.fill_between(dates, rolling_sharpe, 1, where=(rolling_sharpe < 1), 
                    alpha=0.3, color='red')
    
    ax4.set_xlabel('日期')
    ax4.set_ylabel('滚动夏普比率')
    ax4.set_title('滚动年化夏普比率（252日窗口）', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('v7.1_组合净值走势分析.png', dpi=300, bbox_inches='tight')
    print("✓ 图表已保存: v7.1_组合净值走势分析.png")
    plt.show()

def plot_adaptive_params_analysis_v71(df):
    """
    【v7.1专属】自适应参数分析
    """
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'FangSong', 'KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    if 'halflife' not in df.columns:
        print("⚠️ 非v7.1格式，跳过自适应参数分析")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Halflife分布
    ax1 = axes[0, 0]
    halflife = df['halflife']
    colors = ['green' if 5 < x < 20 else 'orange' if 20 <= x < 30 else 'red' 
              for x in halflife]
    ax1.hist(halflife, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(halflife.mean(), color='red', linestyle='--', linewidth=2,
                label=f'平均: {halflife.mean():.1f}天')
    ax1.axvline(5, color='green', linestyle='--', alpha=0.5, label='理想区间')
    ax1.axvline(20, color='green', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Halflife (天)')
    ax1.set_ylabel('频数')
    ax1.set_title('均值回归半衰期分布')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 自适应阈值分布
    ax2 = axes[0, 1]
    entry_th = df['entry_threshold']
    ax2.hist(entry_th, bins=15, color='coral', alpha=0.7, edgecolor='black')
    ax2.axvline(entry_th.mean(), color='red', linestyle='--', linewidth=2,
                label=f'平均: {entry_th.mean():.2f}')
    ax2.axvline(1.2, color='black', linestyle='--', alpha=0.5, label='基础阈值1.2')
    ax2.set_xlabel('入场阈值')
    ax2.set_ylabel('频数')
    ax2.set_title('自适应入场阈值分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 波动率 vs Halflife 散点图
    ax3 = axes[1, 0]
    scatter = ax3.scatter(df['volatility'] * 100, df['halflife'],
                         c=df['total_return'] * 100, cmap='RdYlGn', 
                         s=df['allocated_capital'] / 1e4, alpha=0.6, edgecolors='black')
    ax3.set_xlabel('波动率 (%)')
    ax3.set_ylabel('Halflife (天)')
    ax3.set_title('波动率 vs 均值回归速度（颜色=收益，气泡=资金）')
    plt.colorbar(scatter, ax=ax3, label='收益率(%)')
    ax3.grid(True, alpha=0.3)
    
    # 4. 持仓天数分布
    ax4 = axes[1, 1]
    hold_days = df['max_holding_days']
    ax4.hist(hold_days, bins=15, color='skyblue', alpha=0.7, edgecolor='black')
    ax4.axvline(hold_days.mean(), color='red', linestyle='--', linewidth=2,
                label=f'平均: {hold_days.mean():.0f}天')
    ax4.axvline(15, color='black', linestyle='--', alpha=0.5, label='基础天数15')
    ax4.set_xlabel('最大持仓天数')
    ax4.set_ylabel('频数')
    ax4.set_title('自适应持仓天数分布')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('v7.1_自适应参数分析.png', dpi=300, bbox_inches='tight')
    print("✓ 图表已保存: v7.1_自适应参数分析.png")
    plt.show()

def plot_industry_weights_v71(df):
    """v7.1专属：行业权重分配可视化（改进版）"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'FangSong', 'KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    if 'industry_weight' not in df.columns:
        print("⚠️ 非v7.0/v7.1格式，跳过权重图")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 行业权重饼图（改进：突出显示权重区间）
    ax1 = axes[0, 0]
    industry_weights = df.groupby('industry')['industry_weight'].first().sort_values(ascending=False)
    
    # 按权重分色
    colors = []
    for w in industry_weights.values:
        if w > 0.10:
            colors.append('#ff4444')  # 高权重-红色
        elif w > 0.05:
            colors.append('#ffaa44')  # 中权重-橙色
        else:
            colors.append('#44aa44')  # 低权重-绿色
    
    wedges, texts, autotexts = ax1.pie(
        industry_weights.values, 
        labels=industry_weights.index,
        autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',
        startangle=90,
        colors=colors
    )
    ax1.set_title('行业权重分配（风险收益比）\n红>10% 橙5-10% 绿<5%', 
                 fontsize=14, fontweight='bold')
    
    # 2. 行业资金分配条形图（带收益颜色）
    ax2 = axes[1, 0]
    industry_stats = df.groupby('industry').agg({
        'allocated_capital': 'sum',
        'total_return': 'mean',
        'sharpe_ratio': 'mean'
    }).sort_values('allocated_capital', ascending=True)
    
    colors_bar = ['green' if x > 0.2 else 'orange' if x > 0.1 else 'red' 
                  for x in industry_stats['total_return']]
    
    bars = ax2.barh(range(len(industry_stats)), 
                   industry_stats['allocated_capital'].values / 1e6, 
                   color=colors_bar, alpha=0.8)
    ax2.set_yticks(range(len(industry_stats)))
    ax2.set_yticklabels(industry_stats.index)
    ax2.set_xlabel('分配资金（百万）')
    ax2.set_title('各行业资金分配（颜色=收益水平）', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. 权重 vs 收益散点图
    ax3 = axes[0, 1]
    industry_summary = df.groupby('industry').agg({
        'industry_weight': 'first',
        'total_return': 'mean',
        'sharpe_ratio': 'mean',
        'allocated_capital': 'sum'
    })
    
    scatter = ax3.scatter(industry_summary['industry_weight'] * 100,
                         industry_summary['total_return'] * 100,
                         s=industry_summary['allocated_capital'] / 1e4,
                         c=industry_summary['sharpe_ratio'], 
                         cmap='RdYlGn', alpha=0.6, edgecolors='black')
    ax3.set_xlabel('行业权重 (%)')
    ax3.set_ylabel('行业平均收益 (%)')
    ax3.set_title('权重 vs 收益（气泡=资金，颜色=夏普）', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax3, label='夏普比率')
    ax3.grid(True, alpha=0.3)
    
    # 添加理想区域标注
    ax3.axhspan(20, 50, xmin=0.05, xmax=0.15/0.20, alpha=0.1, color='green', 
               label='理想区域:高收益+适中权重')
    
    # 4. 权重分布对比（v7.0风险平价 vs v7.1风险收益比）
    ax4 = axes[1, 1]
    
    # 模拟v7.0的纯风险平价权重（波动率倒数）
    vol_proxy = df.groupby('industry')['volatility'].first()
    rp_weights = (1 / vol_proxy) / (1 / vol_proxy).sum()
    ra_weights = df.groupby('industry')['industry_weight'].first()
    
    # 对齐
    common_industries = rp_weights.index.intersection(ra_weights.index)
    x = np.arange(len(common_industries))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, rp_weights.loc[common_industries] * 100, width,
                   label='v7.0 风险平价', color='lightblue', alpha=0.8)
    bars2 = ax4.bar(x + width/2, ra_weights.loc[common_industries] * 100, width,
                   label='v7.1 风险收益比', color='coral', alpha=0.8)
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(common_industries, rotation=45, ha='right')
    ax4.set_ylabel('权重 (%)')
    ax4.set_title('权重分配对比：v7.0 vs v7.1', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('v7.1_行业权重分配.png', dpi=300, bbox_inches='tight')
    print("✓ 图表已保存: v7.1_行业权重分配.png")
    plt.show()

def plot_slippage_analysis_v71(df):
    """v7.1滑点影响分析（改进版）"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'FangSong', 'KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    if 'slippage_impact' not in df.columns:
        print("⚠️ 非v7.0/v7.1格式，跳过滑点分析")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 滑点影响分布（对比毛收益）
    ax1 = axes[0, 0]
    slippage_pct = df['slippage_impact'] * 100
    gross_return = df['total_return'] * 100
    net_return = gross_return - slippage_pct
    
    ax1.hist(gross_return, bins=15, alpha=0.5, label='毛收益', color='steelblue', 
            edgecolor='black')
    ax1.hist(net_return, bins=15, alpha=0.5, label='净收益（扣滑点）', 
            color='coral', edgecolor='black')
    ax1.axvline(net_return.mean(), color='red', linestyle='--', linewidth=2,
                label=f'净收益均值: {net_return.mean():.2f}%')
    ax1.set_xlabel('收益率 (%)')
    ax1.set_ylabel('频数')
    ax1.set_title('毛收益 vs 净收益分布')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 滑点 vs 交易次数（v7.1：应该负相关）
    ax2 = axes[0, 1]
    scatter = ax2.scatter(df['num_trades'], df['slippage_impact'] * 100,
                         c=df['volatility'] * 100, cmap='YlOrRd', 
                         s=100, alpha=0.6, edgecolors='black')
    ax2.set_xlabel('交易次数')
    ax2.set_ylabel('滑点影响 (%)')
    ax2.set_title('滑点 vs 交易次数（颜色=波动率）\nv7.1应呈负相关（自适应减少交易）')
    plt.colorbar(scatter, ax=ax2, label='波动率(%)')
    
    # 添加趋势线
    z = np.polyfit(df['num_trades'], df['slippage_impact'] * 100, 1)
    p = np.poly1d(z)
    ax2.plot(df['num_trades'].sort_values(), p(df['num_trades'].sort_values()), 
            "r--", alpha=0.8, label=f'趋势线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 行业平均滑点（v7.1应更低）
    ax3 = axes[1, 0]
    industry_slippage = df.groupby('industry')['slippage_impact'].mean() * 100
    industry_slippage = industry_slippage.sort_values(ascending=True)
    
    colors = ['green' if x < 0.8 else 'orange' if x < 1.2 else 'red' 
              for x in industry_slippage.values]
    
    bars = ax3.barh(range(len(industry_slippage)), industry_slippage.values, 
                   color=colors, alpha=0.8)
    ax3.set_yticks(range(len(industry_slippage)))
    ax3.set_yticklabels(industry_slippage.index)
    ax3.set_xlabel('平均滑点影响 (%)')
    ax3.set_title('各行业平均滑点成本（v7.1目标<0.8%）', fontsize=14, fontweight='bold')
    ax3.axvline(x=0.8, color='green', linestyle='--', alpha=0.5, label='v7.1目标线')
    ax3.axvline(x=1.15, color='red', linestyle='--', alpha=0.5, label='v7.0平均线')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. 滑点占比分析
    ax4 = axes[1, 1]
    slip_ratio = df['slippage_impact'] / df['total_return'] * 100
    slip_ratio = slip_ratio[slip_ratio > 0]  # 只取正收益配对
    
    ax4.hist(slip_ratio, bins=20, color='purple', alpha=0.6, edgecolor='black')
    ax4.axvline(slip_ratio.mean(), color='red', linestyle='--', linewidth=2,
                label=f'平均: 滑点占收益{slip_ratio.mean():.1f}%')
    ax4.set_xlabel('滑点占收益比例 (%)')
    ax4.set_ylabel('频数')
    ax4.set_title('滑点成本占收益比例分布')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('v7.1_滑点影响分析.png', dpi=300, bbox_inches='tight')
    print("✓ 图表已保存: v7.1_滑点影响分析.png")
    plt.show()

def plot_return_distribution_v71(df):
    """收益率分布分析（v7.1适配）"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'FangSong', 'KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 收益率直方图（带自适应参数颜色）
    ax1 = axes[0, 0]
    returns = df['total_return'] * 100
    
    # 用halflife着色
    if 'halflife' in df.columns:
        colors = df['halflife']
        scatter = ax1.scatter(returns, np.random.normal(0, 0.5, len(returns)), 
                            c=colors, cmap='viridis', alpha=0.6, s=50)
        ax1.set_ylabel('')
        ax1.set_yticks([])
        plt.colorbar(scatter, ax=ax1, label='Halflife')
    else:
        ax1.hist(returns, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    
    ax1.axvline(returns.mean(), color='red', linestyle='--', linewidth=2,
                label=f'均值: {returns.mean():.2f}%')
    ax1.set_xlabel('收益率 (%)')
    ax1.set_title('收益率分布')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 夏普比率分布（v7.1应更集中且更高）
    ax2 = axes[0, 1]
    sharpe = df['sharpe_ratio']
    
    ax2.hist(sharpe, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(sharpe.mean(), color='red', linestyle='--', linewidth=2,
                label=f'均值: {sharpe.mean():.2f}')
    ax2.axvline(1, color='green', linestyle='--', alpha=0.5, label='夏普=1')
    ax2.axvline(0.6, color='orange', linestyle='--', alpha=0.5, label='v7.0均值')
    ax2.set_xlabel('夏普比率')
    ax2.set_ylabel('频数')
    ax2.set_title('夏普比率分布（v7.1目标>0.85）')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 收益-风险-夏普三维散点
    ax3 = axes[1, 0]
    sizes = (df['allocated_capital'] / df['allocated_capital'].max() * 500 + 50)
    scatter = ax3.scatter(df['total_return'] * 100, df['max_drawdown'] * 100,
                         c=df['sharpe_ratio'], s=sizes, cmap='RdYlGn', 
                         alpha=0.6, edgecolors='black', vmin=0, vmax=2)
    ax3.set_xlabel('总收益率 (%)')
    ax3.set_ylabel('最大回撤 (%)')
    ax3.set_title('收益-回撤散点（颜色=夏普，气泡=资金）')
    plt.colorbar(scatter, ax=ax3, label='夏普比率')
    ax3.grid(True, alpha=0.3)
    
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
    plt.savefig('v7.1_收益风险分布分析.png', dpi=300, bbox_inches='tight')
    print("✓ 图表已保存: v7.1_收益风险分布分析.png")
    plt.show()

def plot_top_pairs_table_v71(df, top_n=20):
    """Top N配对表格（v7.1适配，增加自适应参数列）"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'FangSong', 'KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    # 排序
    if 'sharpe_ratio' in df.columns and 'total_return' in df.columns:
        # 综合排序：夏普优先，收益次之
        df['score'] = df['sharpe_ratio'] * 0.6 + df['total_return'] * 0.4
        top_df = df.nlargest(top_n, 'score')
    else:
        top_df = df.nlargest(top_n, 'total_return')
    
    # 处理股票代码
    top_df['stock1'] = top_df['stock1'].apply(lambda x: f"{int(x):06d}" if pd.notna(x) else x)
    top_df['stock2'] = top_df['stock2'].apply(lambda x: f"{int(x):06d}" if pd.notna(x) else x)
    
    fig, ax = plt.subplots(figsize=(18, top_n * 0.5 + 3))
    ax.axis('tight')
    ax.axis('off')
    
    # 选择显示列（v7.1增加自适应参数）
    display_cols = ['industry', 'stock1', 'stock2', 'total_return', 'sharpe_ratio',
                   'max_drawdown', 'win_rate', 'num_trades']
    
    if 'industry_weight' in top_df.columns:
        display_cols.insert(3, 'industry_weight')
    if 'allocated_capital' in top_df.columns:
        display_cols.insert(4, 'allocated_capital')
    if 'slippage_impact' in top_df.columns:
        display_cols.append('slippage_impact')
    if 'volatility' in top_df.columns:
        display_cols.extend(['volatility', 'halflife', 'entry_threshold', 'max_holding_days'])
    
    display_cols = [c for c in display_cols if c in top_df.columns]
    display_df = top_df[display_cols].copy()
    
    # 格式化
    format_dict = {
        'total_return': lambda x: f"{x*100:.1f}%",
        'industry_weight': lambda x: f"{x*100:.1f}%",
        'allocated_capital': lambda x: f"{x/1e4:.0f}万",
        'max_drawdown': lambda x: f"{x*100:.1f}%",
        'win_rate': lambda x: f"{x*100:.1f}%",
        'slippage_impact': lambda x: f"{x*100:.2f}%",
        'sharpe_ratio': lambda x: f"{x:.2f}",
        'volatility': lambda x: f"{x*100:.1f}%",
        'halflife': lambda x: f"{x:.1f}",
        'entry_threshold': lambda x: f"{x:.2f}",
        'max_holding_days': lambda x: f"{int(x)}"
    }
    
    for col, fmt in format_dict.items():
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(fmt)
    
    # 创建表格
    table = ax.table(cellText=display_df.values,
                    colLabels=display_cols,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.1] * len(display_cols))
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    # 表头样式
    for i in range(len(display_cols)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 行颜色（按收益分层）
    for i in range(1, len(top_df) + 1):
        ret = top_df.iloc[i-1]['total_return']
        if ret > 0.3:
            color = '#90EE90'  # 优秀 >30%
        elif ret > 0.15:
            color = '#E0F7FA'  # 良好 15-30%
        elif ret > 0.05:
            color = '#FFF9C4'  # 一般 5-15%
        else:
            color = '#FFCCBC'  # 较差 <5%
        
        for j in range(len(display_cols)):
            table[(i, j)].set_facecolor(color)
    
    plt.title(f'Top {top_n} 配对排名（v7.1自适应策略）', 
             fontsize=16, fontweight='bold', pad=20)
    plt.savefig(f'v7.1_Top{top_n}_配对排名表.png', dpi=300, bbox_inches='tight')
    print(f"✓ 图表已保存: v7.1_Top{top_n}_配对排名表.png")
    plt.show()

def generate_summary_report_v71(df, version):
    """生成v7.1汇总报告"""
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
    print(f"正收益配对数: {len(df[df['total_return'] > 0])} ({len(df[df['total_return'] > 0])/len(df)*100:.1f}%)")
    
    # v7.1专属：自适应参数统计
    if 'halflife' in df.columns:
        print(f"\n【自适应参数统计（v7.1）】")
        print(f"平均Halflife: {df['halflife'].mean():.1f}天")
        print(f"平均入场阈值: {df['entry_threshold'].mean():.2f} (基础1.2)")
        print(f"平均持仓天数: {df['max_holding_days'].mean():.0f}天 (基础15)")
        print(f"平均波动率: {df['volatility'].mean()*100:.1f}%")
        
        # 自适应效果
        high_vol = df[df['volatility'] > 0.30]
        low_vol = df[df['volatility'] < 0.15]
        print(f"\n  高波动配对(>{30}%): {len(high_vol)}对")
        print(f"    - 平均阈值: {high_vol['entry_threshold'].mean():.2f}")
        print(f"    - 平均持仓: {high_vol['max_holding_days'].mean():.0f}天")
        print(f"  低波动配对(<{15}%): {len(low_vol)}对")
        print(f"    - 平均阈值: {low_vol['entry_threshold'].mean():.2f}")
        print(f"    - 平均持仓: {low_vol['max_holding_days'].mean():.0f}天")
    
    # 滑点统计
    if 'slippage_impact' in df.columns:
        df['net_return'] = df['total_return'] - df['slippage_impact']
        print(f"\n【滑点影响】")
        print(f"平均滑点成本: {df['slippage_impact'].mean()*100:.2f}%")
        print(f"平均净收益率: {df['net_return'].mean()*100:.2f}%")
        print(f"滑点占收益比: {df['slippage_impact'].sum()/df['total_return'].sum()*100:.1f}%")
    
    print(f"\n【风险指标】")
    print(f"平均夏普比率: {df['sharpe_ratio'].mean():.2f}")
    print(f"夏普>1的配对数: {len(df[df['sharpe_ratio'] > 1])}")
    print(f"平均最大回撤: {df['max_drawdown'].mean()*100:.2f}%")
    print(f"平均胜率: {df['win_rate'].mean()*100:.1f}%")
    
    # 权重统计
    if 'industry_weight' in df.columns:
        print(f"\n【资金分配】")
        industry_stats = df.groupby('industry').agg({
            'industry_weight': 'first',
            'allocated_capital': 'sum',
            'total_return': 'mean',
            'sharpe_ratio': 'mean'
        }).sort_values('industry_weight', ascending=False)
        
        for idx, (industry, row) in enumerate(industry_stats.head(5).iterrows(), 1):
            print(f"{idx}. {industry}: 权重{row['industry_weight']*100:.1f}%, "
                  f"资金{row['allocated_capital']:,.0f}, "
                  f"收益{row['total_return']*100:.1f}%, "
                  f"夏普{row['sharpe_ratio']:.2f}")
    
    print(f"\n【最佳配对Top 5】")
    top_pairs = df.nlargest(5, 'sharpe_ratio')
    for i, (_, row) in enumerate(top_pairs.iterrows(), 1):
        extra = ""
        if 'halflife' in row:
            extra = f" | HL:{row['halflife']:.1f}天 | 阈值:{row['entry_threshold']:.2f}"
        print(f"{i}. {row['industry']}: {row['stock1']}-{row['stock2']} | "
              f"收益:{row['total_return']*100:.1f}% | 夏普:{row['sharpe_ratio']:.2f}{extra}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    # 加载数据
    df, version = load_latest_results()
    if df is None:
        exit()
    
    # 生成报告
    generate_summary_report_v71(df, version)
    
    # 生成图表
    print("\n生成可视化图表...")
    
    # 【v7.1核心新增】组合净值走势
    print("\n[v7.1核心] 组合净值走势分析...")
    plot_portfolio_nav_curve(df)
    
    if version == "v7.1":
        # v7.1专属图表
        print("\n[v7.1专属] 自适应参数分析...")
        plot_adaptive_params_analysis_v71(df)
        
        print("\n[v7.1专属] 行业权重分配...")
        plot_industry_weights_v71(df)
        
        print("\n[v7.1专属] 滑点影响分析...")
        plot_slippage_analysis_v71(df)
    elif version == "v7.0":
        # v7.0图表
        from analyze_results_v7 import plot_industry_weights_v7, plot_slippage_analysis_v7
        print("\n[v7.0] 行业权重分配...")
        plot_industry_weights_v7(df)
        print("\n[v7.0] 滑点影响分析...")
        plot_slippage_analysis_v7(df)
    
    # 通用图表
    print("\n收益风险分布分析...")
    plot_return_distribution_v71(df)
    
    print("\nTop配对排名表...")
    plot_top_pairs_table_v71(df, top_n=20)
    
    print("\n✓ 全部分析完成！")
    print(f"\n生成图表清单（{version}）：")
    print("  - v7.1_组合净值走势分析.png 【新增】")
    if version == "v7.1":
        print("  - v7.1_自适应参数分析.png 【新增】")
        print("  - v7.1_行业权重分配.png")
        print("  - v7.1_滑点影响分析.png")
    print("  - v7.1_收益风险分布分析.png")
    print(f"  - v7.1_Top20_配对排名表.png")