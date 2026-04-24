# === 全行业配对回测结果可视化分析 v8.1 ===
# 适配 pairtrading_all_industries_v8.1.py 输出格式
# 支持滚动窗口回测、实盘回测、单次回测三种模式
# 完全保留v7.2.1所有图表功能

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime, timedelta
import glob
import os
import warnings
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import json
import ast
import zipfile

# 完全关闭字体警告
warnings.filterwarnings("ignore", category=UserWarning)

# 强制设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'FangSong', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')

def load_latest_results_v81():
    """加载最新的v8.1回测结果文件"""
    # 查找v8.1格式的结果文件
    files = glob.glob("v8.1_组合回测结果_*.csv")
    if not files:
        print("未找到v8.1回测结果文件")
        return None, "v8.1"
    
    latest_file = max(files, key=os.path.getctime)
    print(f"加载文件: {latest_file}")
    
    try:
        df = pd.read_csv(latest_file, encoding='utf-8-sig')
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return None, "v8.1"
    
    # 检测列名并重命名（适配v8.1的列名变化）
    column_mapping = {}
    for col in df.columns:
        if '行业' in col:
            column_mapping[col] = '行业'
        elif '股票1' in col:
            column_mapping[col] = '股票1'
        elif '股票2' in col:
            column_mapping[col] = '股票2'
        elif '分配资金' in col:
            column_mapping[col] = '分配资金'
        elif '总收益率' in col:
            column_mapping[col] = '总收益率'
        elif '夏普比率' in col:
            column_mapping[col] = '夏普比率'
        elif '最大回撤' in col:
            column_mapping[col] = '最大回撤'
        elif '胜率' in col:
            column_mapping[col] = '胜率'
        elif '交易次数' in col:
            column_mapping[col] = '交易次数'
    
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    print(f"数据形状: {df.shape}")
    print(f"行业数量: {df['行业'].nunique() if '行业' in df.columns else '未知'}")
    print(f"配对数量: {len(df)}")
    
    # 尝试加载汇总报告
    report_files = glob.glob("v8.1_汇总报告_*.json")
    if report_files:
        report_file = max(report_files, key=os.path.getctime)
        print(f"加载汇总报告: {report_file}")
        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                summary_report = json.load(f)
        except:
            summary_report = None
    else:
        summary_report = None
    
    return df, "v8.1", summary_report

def load_equity_data_from_zip():
    """从ZIP文件加载净值数据"""
    zip_files = glob.glob("v8.1_净值数据_*.zip")
    if not zip_files:
        print("未找到净值数据ZIP文件")
        return None
    
    latest_zip = max(zip_files, key=os.path.getctime)
    print(f"加载净值数据ZIP: {latest_zip}")
    
    try:
        # 解压ZIP文件到临时目录
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(latest_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # 读取所有净值CSV文件
        equity_data = {}
        equity_files = glob.glob(os.path.join(temp_dir, "*净值.csv"))
        
        for eq_file in equity_files:
            try:
                # 从文件名提取行业和股票信息
                filename = os.path.basename(eq_file)
                parts = filename.replace('净值.csv', '').split('_')
                
                if len(parts) >= 3:
                    industry = parts[0]
                    stock1 = parts[1]
                    stock2 = parts[2]
                    
                    df = pd.read_csv(eq_file, encoding='utf-8-sig')
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                    
                    pair_key = f"{stock1}_{stock2}"
                    equity_data[pair_key] = {
                        'industry': industry,
                        'stock1': stock1,
                        'stock2': stock2,
                        'data': df
                    }
            except Exception as e:
                print(f"  读取净值文件{os.path.basename(eq_file)}失败: {e}")
                continue
        
        # 清理临时目录
        shutil.rmtree(temp_dir)
        
        print(f"  成功加载{len(equity_data)}个净值文件")
        return equity_data
        
    except Exception as e:
        print(f"解压ZIP文件失败: {e}")
        return None

def convert_percentage_columns(df):
    """将百分比列转换为小数"""
    percentage_cols = ['总收益率', '最大回撤', '胜率', '滑点成本占比', '波动率', 
                      '行业权重', '配对权重', '单次交易平均收益']
    
    for col in percentage_cols:
        if col in df.columns:
            try:
                # 移除百分号并转换为数字
                df[col] = df[col].astype(str).str.replace('%', '').astype(float) / 100
            except:
                # 如果转换失败，尝试直接转换为数值
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
    
    return df

def plot_portfolio_nav_curve_v81(df, equity_data=None):
    """
    v8.1组合策略整体净值走势图
    支持多种回测模式的净值数据
    """
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'FangSong', 'KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 如果提供了净值数据，使用实际净值数据
    if equity_data and len(equity_data) > 0:
        print("  使用净值数据计算组合净值...")
        
        # 收集所有净值曲线
        all_curves = []
        all_weights = []
        
        for _, pair_info in df.iterrows():
            stock1 = str(pair_info.get('股票1', ''))
            stock2 = str(pair_info.get('股票2', ''))
            pair_key = f"{stock1}_{stock2}"
            
            if pair_key in equity_data:
                eq_data = equity_data[pair_key]['data']
                if 'net_value' in eq_data.columns or 'value' in eq_data.columns:
                    # 确定净值列名
                    nav_col = 'net_value' if 'net_value' in eq_data.columns else 'value'
                    
                    # 计算归一化净值
                    if eq_data[nav_col].iloc[0] > 0:
                        nav_series = eq_data[nav_col] / eq_data[nav_col].iloc[0]
                        all_curves.append(pd.DataFrame({'date': nav_series.index, 'nav': nav_series.values}))
                        
                        # 获取权重
                        weight = pair_info.get('配对权重', 1/len(df))
                        if isinstance(weight, str):
                            weight = float(weight.replace('%', '')) / 100
                        all_weights.append(weight)
        
        if all_curves:
            # 合并所有净值曲线
            merged_nav = None
            for curve, weight in zip(all_curves, all_weights):
                curve_df = curve.copy()
                curve_df['date'] = pd.to_datetime(curve_df['date'])
                curve_df.set_index('date', inplace=True)
                
                if merged_nav is None:
                    merged_nav = curve_df['nav'] * weight
                else:
                    # 对齐日期
                    aligned = curve_df['nav'].reindex(merged_nav.index, method='ffill').fillna(method='ffill')
                    merged_nav = merged_nav.add(aligned * weight, fill_value=0)
            
            portfolio_nav = merged_nav
        else:
            portfolio_nav = None
    else:
        portfolio_nav = None
    
    # 2. 如果没有净值数据，使用简化估算
    if portfolio_nav is None or len(portfolio_nav) < 10:
        print("  无详细净值数据，使用收益率估算...")
        
        # 生成模拟日期
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 12, 31)
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # 基于收益率生成模拟净值
        np.random.seed(42)
        daily_returns = []
        
        for _, row in df.iterrows():
            try:
                # 获取收益率
                if '总收益率' in row:
                    total_return = row['总收益率']
                    if isinstance(total_return, str):
                        total_return = float(total_return.replace('%', '')) / 100
                else:
                    total_return = 0.1
                
                # 获取权重
                weight = row.get('配对权重', 1/len(df))
                if isinstance(weight, str):
                    weight = float(weight.replace('%', '')) / 100
                
                # 生成随机walk
                n_days = len(dates)
                if n_days > 0:
                    # 年化收益率分配到日
                    daily_mean = (1 + total_return) ** (1/252) - 1
                    daily_vol = 0.02  # 假设2%的日波动率
                    
                    # 生成随机收益序列
                    random_returns = np.random.normal(daily_mean, daily_vol, n_days)
                    daily_returns.append(random_returns * weight)
            except:
                continue
        
        if daily_returns:
            portfolio_daily = np.sum(daily_returns, axis=0)
            portfolio_nav = pd.Series((1 + portfolio_daily).cumprod(), index=dates)
        else:
            portfolio_nav = pd.Series(np.ones(len(dates)), index=dates)
    
    # 绘制组合净值曲线
    ax1 = axes[0, 0]
    ax1.plot(portfolio_nav.index, portfolio_nav.values, linewidth=2, color='steelblue', label='组合净值')
    ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='初始净值')
    
    # 标注关键节点
    final_nav = portfolio_nav.iloc[-1]
    max_nav = portfolio_nav.max()
    min_nav = portfolio_nav.min()
    
    ax1.scatter([portfolio_nav.index[-1]], [final_nav], color='red', s=100, zorder=5)
    ax1.annotate(f'期末: {final_nav:.2f}', xy=(portfolio_nav.index[-1], final_nav), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # 标注最大净值点
    max_idx = portfolio_nav.idxmax()
    ax1.scatter([max_idx], [max_nav], color='green', s=80, zorder=5, marker='^')
    ax1.annotate(f'最高: {max_nav:.2f}', xy=(max_idx, max_nav), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    
    ax1.set_xlabel('日期')
    ax1.set_ylabel('净值')
    ax1.set_title('v8.1组合策略净值走势', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 回撤曲线
    ax2 = axes[0, 1]
    rolling_max = portfolio_nav.cummax()
    drawdown = (portfolio_nav - rolling_max) / rolling_max
    
    ax2.fill_between(portfolio_nav.index, drawdown, 0, color='red', alpha=0.3)
    ax2.plot(portfolio_nav.index, drawdown, color='darkred', linewidth=1)
    
    if len(drawdown) > 0:
        max_dd_idx = drawdown.idxmin()
        max_dd_val = drawdown.min()
        ax2.scatter([max_dd_idx], [max_dd_val], color='red', s=100, zorder=5)
        ax2.annotate(f'最大回撤: {max_dd_val*100:.1f}%', 
                    xy=(max_dd_idx, max_dd_val),
                    xytext=(10, -30), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax2.set_xlabel('日期')
    ax2.set_ylabel('回撤 (%)')
    ax2.set_title('组合回撤走势', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. 月度收益热力图
    ax3 = axes[1, 0]
    
    # 计算月度收益
    monthly_returns = portfolio_nav.resample('M').last().pct_change().dropna()
    
    if len(monthly_returns) > 0:
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
    else:
        ax3.text(0.5, 0.5, '无足够月度数据', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('月度收益热力图 (无数据)', fontsize=14, fontweight='bold')
    
    # 4. 滚动夏普比率
    ax4 = axes[1, 1]
    
    # 计算日收益率
    daily_returns = portfolio_nav.pct_change().dropna()
    
    if len(daily_returns) >= 252:
        rolling_sharpe = (daily_returns.rolling(252).mean() / 
                         daily_returns.rolling(252).std()) * np.sqrt(252)
        
        ax4.plot(rolling_sharpe.index, rolling_sharpe.values, color='green', linewidth=2)
        ax4.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='夏普=1')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.fill_between(rolling_sharpe.index, rolling_sharpe.values, 1, 
                        where=(rolling_sharpe.values > 1), 
                        alpha=0.3, color='green', interpolate=True)
        ax4.fill_between(rolling_sharpe.index, rolling_sharpe.values, 1, 
                        where=(rolling_sharpe.values < 1), 
                        alpha=0.3, color='red', interpolate=True)
        
        ax4.set_xlabel('日期')
        ax4.set_ylabel('滚动夏普比率')
        ax4.set_title('滚动年化夏普比率（252日窗口）', fontsize=14, fontweight='bold')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, f'数据不足({len(daily_returns)}天<252天)', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('滚动夏普比率 (数据不足)', fontsize=14, fontweight='bold')
    
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('v8.1_组合净值走势分析.png', dpi=300, bbox_inches='tight')
    print("✓ 图表已保存: v8.1_组合净值走势分析.png")
    plt.show()
    
    return portfolio_nav

def plot_adaptive_params_analysis_v81(df):
    """v8.1自适应参数分析"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'FangSong', 'KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 检查是否有自适应参数
    adaptive_cols = ['半衰期', '入场阈值', '最大持仓天数', '波动率']
    has_adaptive = any(col in df.columns for col in adaptive_cols)
    
    if not has_adaptive:
        print("⚠️ 无自适应参数数据，跳过自适应参数分析")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Halflife分布
    ax1 = axes[0, 0]
    if '半衰期' in df.columns:
        halflife = pd.to_numeric(df['半衰期'], errors='coerce')
        halflife = halflife.dropna()
        
        if len(halflife) > 0:
            colors = ['green' if 5 < x < 20 else 'orange' if 20 <= x < 30 else 'red' 
                     for x in halflife if pd.notna(x)]
            ax1.hist(halflife, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
            ax1.axvline(halflife.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'平均: {halflife.mean():.1f}天')
            ax1.axvline(5, color='green', linestyle='--', alpha=0.5, label='理想区间')
            ax1.axvline(20, color='green', linestyle='--', alpha=0.5)
            ax1.set_xlabel('半衰期 (天)')
            ax1.set_ylabel('频数')
            ax1.set_title('均值回归半衰期分布')
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, '无半衰期数据', ha='center', va='center', transform=ax1.transAxes)
    else:
        ax1.text(0.5, 0.5, '无半衰期数据', ha='center', va='center', transform=ax1.transAxes)
    
    ax1.grid(True, alpha=0.3)
    
    # 2. 自适应阈值分布
    ax2 = axes[0, 1]
    if '入场阈值' in df.columns:
        entry_th = pd.to_numeric(df['入场阈值'], errors='coerce')
        entry_th = entry_th.dropna()
        
        if len(entry_th) > 0:
            ax2.hist(entry_th, bins=15, color='coral', alpha=0.7, edgecolor='black')
            ax2.axvline(entry_th.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'平均: {entry_th.mean():.2f}')
            ax2.axvline(1.2, color='black', linestyle='--', alpha=0.5, label='基础阈值1.2')
            ax2.set_xlabel('入场阈值')
            ax2.set_ylabel('频数')
            ax2.set_title('自适应入场阈值分布')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, '无入场阈值数据', ha='center', va='center', transform=ax2.transAxes)
    else:
        ax2.text(0.5, 0.5, '无入场阈值数据', ha='center', va='center', transform=ax2.transAxes)
    
    ax2.grid(True, alpha=0.3)
    
    # 3. 波动率 vs 半衰期散点图
    ax3 = axes[1, 0]
    if '波动率' in df.columns and '半衰期' in df.columns:
        volatility = pd.to_numeric(df['波动率'], errors='coerce')
        halflife = pd.to_numeric(df['半衰期'], errors='coerce')
        total_return = pd.to_numeric(df['总收益率'], errors='coerce')
        allocated_capital = pd.to_numeric(df['分配资金'], errors='coerce')
        
        # 合并有效数据
        valid_idx = volatility.notna() & halflife.notna() & total_return.notna() & allocated_capital.notna()
        
        if valid_idx.sum() > 0:
            scatter = ax3.scatter(volatility[valid_idx] * 100, halflife[valid_idx],
                                c=total_return[valid_idx] * 100, cmap='RdYlGn', 
                                s=allocated_capital[valid_idx] / 1e4, 
                                alpha=0.6, edgecolors='black')
            ax3.set_xlabel('波动率 (%)')
            ax3.set_ylabel('半衰期 (天)')
            ax3.set_title('波动率 vs 均值回归速度（颜色=收益，气泡=资金）')
            plt.colorbar(scatter, ax=ax3, label='收益率(%)')
        else:
            ax3.text(0.5, 0.5, '无有效数据', ha='center', va='center', transform=ax3.transAxes)
    else:
        ax3.text(0.5, 0.5, '无波动率或半衰期数据', ha='center', va='center', transform=ax3.transAxes)
    
    ax3.grid(True, alpha=0.3)
    
    # 4. 持仓天数分布
    ax4 = axes[1, 1]
    if '最大持仓天数' in df.columns:
        hold_days = pd.to_numeric(df['最大持仓天数'], errors='coerce')
        hold_days = hold_days.dropna()
        
        if len(hold_days) > 0:
            ax4.hist(hold_days, bins=15, color='skyblue', alpha=0.7, edgecolor='black')
            ax4.axvline(hold_days.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'平均: {hold_days.mean():.0f}天')
            ax4.axvline(15, color='black', linestyle='--', alpha=0.5, label='基础天数15')
            ax4.set_xlabel('最大持仓天数')
            ax4.set_ylabel('频数')
            ax4.set_title('自适应持仓天数分布')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, '无持仓天数数据', ha='center', va='center', transform=ax4.transAxes)
    else:
        ax4.text(0.5, 0.5, '无持仓天数数据', ha='center', va='center', transform=ax4.transAxes)
    
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('v8.1_自适应参数分析.png', dpi=300, bbox_inches='tight')
    print("✓ 图表已保存: v8.1_自适应参数分析.png")
    plt.show()

def plot_industry_weights_v81(df):
    """v8.1行业权重分配可视化"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'FangSong', 'KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    
    if '行业权重' not in df.columns or '行业' not in df.columns:
        print("⚠️ 无行业权重数据，跳过权重图")
        return
    
    # 处理单行业情况
    n_industries = df['行业'].nunique()
    
    if n_industries == 1:
        # 单行业时简化为2x1布局
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        ax1 = axes[0]
        ax2 = axes[1]
        
        # 1. 行业权重显示（单行业100%）
        industry_name = df['行业'].iloc[0]
        
        ax1.pie([1], labels=[industry_name], autopct='100%', startangle=90, colors=['#ff4444'])
        ax1.set_title(f'行业权重分配\n（仅单行业: {industry_name}）', fontsize=14, fontweight='bold')
        
        # 2. 该行业配对收益分布
        returns = pd.to_numeric(df['总收益率'], errors='coerce') * 100
        returns = returns.dropna()
        
        if len(returns) > 0:
            ax2.bar(range(len(returns)), returns, color='steelblue', alpha=0.7)
            ax2.axhline(y=returns.mean(), color='red', linestyle='--', 
                       label=f'平均: {returns.mean():.1f}%')
            ax2.set_xlabel('配对编号')
            ax2.set_ylabel('收益率 (%)')
            ax2.set_title(f'{industry_name} - 各配对收益率', fontsize=14, fontweight='bold')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, '无收益率数据', ha='center', va='center', transform=ax2.transAxes)
        
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('v8.1_行业权重分配.png', dpi=300, bbox_inches='tight')
        print("✓ 图表已保存: v8.1_行业权重分配.png（单行业模式）")
        plt.show()
        return
    
    # 多行业情况
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 行业权重饼图
    ax1 = axes[0, 0]
    try:
        # 处理行业权重数据
        industry_weights = {}
        for _, row in df.iterrows():
            industry = row['行业']
            weight_str = row['行业权重']
            
            if isinstance(weight_str, str):
                weight = float(weight_str.replace('%', '')) / 100
            else:
                weight = float(weight_str)
            
            if industry not in industry_weights:
                industry_weights[industry] = weight
        
        if industry_weights:
            industries = list(industry_weights.keys())
            weights = list(industry_weights.values())
            
            # 排序
            sorted_idx = np.argsort(weights)[::-1]
            industries = [industries[i] for i in sorted_idx]
            weights = [weights[i] for i in sorted_idx]
            
            colors = []
            for w in weights:
                if w > 0.10:
                    colors.append('#ff4444')
                elif w > 0.05:
                    colors.append('#ffaa44')
                else:
                    colors.append('#44aa44')
            
            wedges, texts, autotexts = ax1.pie(
                weights, 
                labels=industries,
                autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',
                startangle=90,
                colors=colors
            )
            ax1.set_title('行业权重分配（风险收益比）\n红>10% 橙5-10% 绿<5%', 
                         fontsize=14, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, '无行业权重数据', ha='center', va='center', transform=ax1.transAxes)
    except Exception as e:
        ax1.text(0.5, 0.5, f'绘制错误: {e}', ha='center', va='center', transform=ax1.transAxes)
    
    # 2. 行业资金分配条形图
    ax2 = axes[1, 0]
    try:
        industry_stats = {}
        for _, row in df.iterrows():
            industry = row['行业']
            
            # 处理分配资金
            capital_str = row['分配资金']
            if isinstance(capital_str, str):
                capital = float(capital_str.replace(',', ''))
            else:
                capital = float(capital_str)
            
            # 处理收益率
            return_str = row['总收益率']
            if isinstance(return_str, str):
                ret = float(return_str.replace('%', '')) / 100
            else:
                ret = float(return_str)
            
            if industry not in industry_stats:
                industry_stats[industry] = {
                    'capital': 0,
                    'return': [],
                    'sharpe': []
                }
            
            industry_stats[industry]['capital'] += capital
            industry_stats[industry]['return'].append(ret)
            
            # 处理夏普比率
            if '夏普比率' in row:
                sharpe_str = row['夏普比率']
                if isinstance(sharpe_str, str):
                    sharpe = float(sharpe_str)
                else:
                    sharpe = float(sharpe_str)
                industry_stats[industry]['sharpe'].append(sharpe)
        
        if industry_stats:
            # 转换为DataFrame
            stats_list = []
            for industry, data in industry_stats.items():
                stats_list.append({
                    'industry': industry,
                    'capital': data['capital'],
                    'avg_return': np.mean(data['return']) if data['return'] else 0,
                    'avg_sharpe': np.mean(data['sharpe']) if data['sharpe'] else 0
                })
            
            stats_df = pd.DataFrame(stats_list)
            stats_df = stats_df.sort_values('capital', ascending=True)
            
            colors_bar = ['green' if x > 0.2 else 'orange' if x > 0.1 else 'red' 
                         for x in stats_df['avg_return']]
            
            bars = ax2.barh(range(len(stats_df)), 
                           stats_df['capital'].values / 1e6, 
                           color=colors_bar, alpha=0.8)
            ax2.set_yticks(range(len(stats_df)))
            ax2.set_yticklabels(stats_df['industry'])
            ax2.set_xlabel('分配资金（百万）')
            ax2.set_title('各行业资金分配（颜色=收益水平）', fontsize=14, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, '无行业统计数据', ha='center', va='center', transform=ax2.transAxes)
    except Exception as e:
        ax2.text(0.5, 0.5, f'绘制错误: {e}', ha='center', va='center', transform=ax2.transAxes)
    
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. 权重 vs 收益散点图
    ax3 = axes[0, 1]
    try:
        if 'industry_stats' in locals() and industry_stats:
            # 计算行业平均指标
            industry_summary = []
            for industry, data in industry_stats.items():
                # 获取行业权重
                weight = industry_weights.get(industry, 0)
                
                industry_summary.append({
                    'industry': industry,
                    'weight': weight * 100,  # 转换为百分比
                    'avg_return': np.mean(data['return']) * 100 if data['return'] else 0,
                    'avg_sharpe': np.mean(data['sharpe']) if data['sharpe'] else 0,
                    'capital': data['capital']
                })
            
            summary_df = pd.DataFrame(industry_summary)
            
            if len(summary_df) > 0:
                scatter = ax3.scatter(summary_df['weight'],
                                     summary_df['avg_return'],
                                     s=summary_df['capital'] / 1e4,
                                     c=summary_df['avg_sharpe'], 
                                     cmap='RdYlGn', alpha=0.6, edgecolors='black')
                ax3.set_xlabel('行业权重 (%)')
                ax3.set_ylabel('行业平均收益 (%)')
                ax3.set_title('权重 vs 收益（气泡=资金，颜色=夏普）', fontsize=14, fontweight='bold')
                plt.colorbar(scatter, ax=ax3, label='夏普比率')
                
                # 添加理想区域标注
                ax3.axhspan(20, 50, xmin=0.05, xmax=0.15/0.20, alpha=0.1, color='green', 
                           label='理想区域:高收益+适中权重')
            else:
                ax3.text(0.5, 0.5, '无汇总数据', ha='center', va='center', transform=ax3.transAxes)
        else:
            ax3.text(0.5, 0.5, '无行业统计数据', ha='center', va='center', transform=ax3.transAxes)
    except Exception as e:
        ax3.text(0.5, 0.5, f'绘制错误: {e}', ha='center', va='center', transform=ax3.transAxes)
    
    ax3.grid(True, alpha=0.3)
    
    # 4. 行业分布统计
    ax4 = axes[1, 1]
    try:
        if '行业' in df.columns:
            industry_counts = df['行业'].value_counts().sort_values(ascending=False)
            
            if len(industry_counts) > 0:
                colors_count = ['steelblue' if i < 5 else 'lightblue' for i in range(len(industry_counts))]
                bars = ax4.bar(range(len(industry_counts)), industry_counts.values, color=colors_count)
                ax4.set_xticks(range(len(industry_counts)))
                ax4.set_xticklabels(industry_counts.index, rotation=45, ha='right')
                ax4.set_ylabel('配对数量')
                ax4.set_title('各行业入选配对数量', fontsize=14, fontweight='bold')
            else:
                ax4.text(0.5, 0.5, '无行业数据', ha='center', va='center', transform=ax4.transAxes)
        else:
            ax4.text(0.5, 0.5, '无行业列', ha='center', va='center', transform=ax4.transAxes)
    except Exception as e:
        ax4.text(0.5, 0.5, f'绘制错误: {e}', ha='center', va='center', transform=ax4.transAxes)
    
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('v8.1_行业权重分配.png', dpi=300, bbox_inches='tight')
    print("✓ 图表已保存: v8.1_行业权重分配.png")
    plt.show()

def plot_slippage_analysis_v81(df):
    """v8.1滑点影响分析"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'FangSong', 'KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    
    if '滑点成本占比' not in df.columns:
        print("⚠️ 无滑点数据，跳过滑点分析")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 滑点影响分布
    ax1 = axes[0, 0]
    try:
        slippage_pct = pd.to_numeric(df['滑点成本占比'], errors='coerce') * 100
        gross_return = pd.to_numeric(df['总收益率'], errors='coerce') * 100
        net_return = gross_return - slippage_pct
        
        valid_idx = slippage_pct.notna() & gross_return.notna()
        
        if valid_idx.sum() > 0:
            ax1.hist(gross_return[valid_idx], bins=15, alpha=0.5, label='毛收益', 
                    color='steelblue', edgecolor='black')
            ax1.hist(net_return[valid_idx], bins=15, alpha=0.5, label='净收益（扣滑点）', 
                    color='coral', edgecolor='black')
            ax1.axvline(net_return[valid_idx].mean(), color='red', linestyle='--', linewidth=2,
                       label=f'净收益均值: {net_return[valid_idx].mean():.2f}%')
            ax1.set_xlabel('收益率 (%)')
            ax1.set_ylabel('频数')
            ax1.set_title('毛收益 vs 净收益分布')
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, '无有效数据', ha='center', va='center', transform=ax1.transAxes)
    except Exception as e:
        ax1.text(0.5, 0.5, f'绘制错误: {e}', ha='center', va='center', transform=ax1.transAxes)
    
    ax1.grid(True, alpha=0.3)
    
    # 2. 滑点 vs 交易次数
    ax2 = axes[0, 1]
    try:
        slippage = pd.to_numeric(df['滑点成本占比'], errors='coerce') * 100
        num_trades = pd.to_numeric(df['交易次数'], errors='coerce')
        volatility = pd.to_numeric(df.get('波动率', pd.Series([0.2]*len(df))), errors='coerce') * 100
        
        valid_idx = slippage.notna() & num_trades.notna() & volatility.notna()
        
        if valid_idx.sum() > 0:
            scatter = ax2.scatter(num_trades[valid_idx], slippage[valid_idx],
                                c=volatility[valid_idx], cmap='YlOrRd', 
                                s=100, alpha=0.6, edgecolors='black')
            ax2.set_xlabel('交易次数')
            ax2.set_ylabel('滑点影响 (%)')
            ax2.set_title('滑点 vs 交易次数（颜色=波动率）')
            plt.colorbar(scatter, ax=ax2, label='波动率(%)')
            
            # 线性拟合
            if len(num_trades[valid_idx]) > 1:
                z = np.polyfit(num_trades[valid_idx], slippage[valid_idx], 1)
                p = np.poly1d(z)
                sorted_trades = np.sort(num_trades[valid_idx])
                ax2.plot(sorted_trades, p(sorted_trades), "r--", alpha=0.8, label=f'趋势线')
                ax2.legend()
        else:
            ax2.text(0.5, 0.5, '无有效数据', ha='center', va='center', transform=ax2.transAxes)
    except Exception as e:
        ax2.text(0.5, 0.5, f'绘制错误: {e}', ha='center', va='center', transform=ax2.transAxes)
    
    ax2.grid(True, alpha=0.3)
    
    # 3. 行业平均滑点
    ax3 = axes[1, 0]
    try:
        if '行业' in df.columns and '滑点成本占比' in df.columns:
            # 计算行业平均滑点
            industry_slippage = {}
            for _, row in df.iterrows():
                industry = row['行业']
                slippage_str = row['滑点成本占比']
                
                if isinstance(slippage_str, str):
                    slippage_val = float(slippage_str.replace('%', ''))
                else:
                    slippage_val = float(slippage_str) * 100
                
                if industry not in industry_slippage:
                    industry_slippage[industry] = []
                industry_slippage[industry].append(slippage_val)
            
            # 计算平均值
            industry_avg = {ind: np.mean(vals) for ind, vals in industry_slippage.items()}
            industry_series = pd.Series(industry_avg).sort_values(ascending=True)
            
            if len(industry_series) > 0:
                colors = ['green' if x < 0.8 else 'orange' if x < 1.2 else 'red' 
                         for x in industry_series.values]
                
                bars = ax3.barh(range(len(industry_series)), industry_series.values, 
                               color=colors, alpha=0.8)
                ax3.set_yticks(range(len(industry_series)))
                ax3.set_yticklabels(industry_series.index)
                ax3.set_xlabel('平均滑点影响 (%)')
                ax3.set_title('各行业平均滑点成本（目标<0.8%）', fontsize=14, fontweight='bold')
                ax3.axvline(x=0.8, color='green', linestyle='--', alpha=0.5, label='目标线')
                ax3.axvline(x=1.15, color='red', linestyle='--', alpha=0.5, label='基准线')
                ax3.legend()
            else:
                ax3.text(0.5, 0.5, '无行业滑点数据', ha='center', va='center', transform=ax3.transAxes)
        else:
            ax3.text(0.5, 0.5, '无行业或滑点数据', ha='center', va='center', transform=ax3.transAxes)
    except Exception as e:
        ax3.text(0.5, 0.5, f'绘制错误: {e}', ha='center', va='center', transform=ax3.transAxes)
    
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. 滑点占比分析
    ax4 = axes[1, 1]
    try:
        slippage_pct = pd.to_numeric(df['滑点成本占比'], errors='coerce')
        total_return = pd.to_numeric(df['总收益率'], errors='coerce')
        
        valid_idx = slippage_pct.notna() & total_return.notna() & (total_return > 0)
        
        if valid_idx.sum() > 0:
            slip_ratio = (slippage_pct[valid_idx] / total_return[valid_idx]) * 100
            
            if len(slip_ratio) > 0:
                ax4.hist(slip_ratio, bins=20, color='purple', alpha=0.6, edgecolor='black')
                ax4.axvline(slip_ratio.mean(), color='red', linestyle='--', linewidth=2,
                           label=f'平均: 滑点占收益{slip_ratio.mean():.1f}%')
                ax4.set_xlabel('滑点占收益比例 (%)')
                ax4.set_ylabel('频数')
                ax4.set_title('滑点成本占收益比例分布')
                ax4.legend()
            else:
                ax4.text(0.5, 0.5, '无有效数据', ha='center', va='center', transform=ax4.transAxes)
        else:
            ax4.text(0.5, 0.5, '无有效数据', ha='center', va='center', transform=ax4.transAxes)
    except Exception as e:
        ax4.text(0.5, 0.5, f'绘制错误: {e}', ha='center', va='center', transform=ax4.transAxes)
    
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('v8.1_滑点影响分析.png', dpi=300, bbox_inches='tight')
    print("✓ 图表已保存: v8.1_滑点影响分析.png")
    plt.show()

def plot_return_distribution_v81(df):
    """v8.1收益率分布分析"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'FangSong', 'KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 收益率直方图
    ax1 = axes[0, 0]
    try:
        returns = pd.to_numeric(df['总收益率'], errors='coerce') * 100
        
        if returns.notna().sum() > 0:
            if '半衰期' in df.columns:
                halflife = pd.to_numeric(df['半衰期'], errors='coerce')
                valid_idx = returns.notna() & halflife.notna()
                
                if valid_idx.sum() > 0:
                    scatter = ax1.scatter(returns[valid_idx], 
                                        np.random.normal(0, 0.5, valid_idx.sum()), 
                                        c=halflife[valid_idx], cmap='viridis', 
                                        alpha=0.6, s=50)
                    ax1.set_ylabel('')
                    ax1.set_yticks([])
                    plt.colorbar(scatter, ax=ax1, label='半衰期')
                else:
                    ax1.hist(returns.dropna(), bins=20, color='steelblue', 
                            alpha=0.7, edgecolor='black')
            else:
                ax1.hist(returns.dropna(), bins=20, color='steelblue', 
                        alpha=0.7, edgecolor='black')
            
            if len(returns.dropna()) > 0:
                ax1.axvline(returns.mean(), color='red', linestyle='--', linewidth=2,
                           label=f'均值: {returns.mean():.2f}%')
                ax1.set_xlabel('收益率 (%)')
                ax1.set_title('收益率分布')
                ax1.legend()
        else:
            ax1.text(0.5, 0.5, '无收益率数据', ha='center', va='center', transform=ax1.transAxes)
    except Exception as e:
        ax1.text(0.5, 0.5, f'绘制错误: {e}', ha='center', va='center', transform=ax1.transAxes)
    
    ax1.grid(True, alpha=0.3)
    
    # 2. 夏普比率分布
    ax2 = axes[0, 1]
    try:
        if '夏普比率' in df.columns:
            sharpe = pd.to_numeric(df['夏普比率'], errors='coerce')
            sharpe = sharpe.dropna()
            
            if len(sharpe) > 0:
                ax2.hist(sharpe, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
                ax2.axvline(sharpe.mean(), color='red', linestyle='--', linewidth=2,
                           label=f'均值: {sharpe.mean():.2f}')
                ax2.axvline(1, color='green', linestyle='--', alpha=0.5, label='夏普=1')
                ax2.axvline(0.6, color='orange', linestyle='--', alpha=0.5, label='基准线')
                ax2.set_xlabel('夏普比率')
                ax2.set_ylabel('频数')
                ax2.set_title('夏普比率分布（目标>0.85）')
                ax2.legend()
            else:
                ax2.text(0.5, 0.5, '无夏普数据', ha='center', va='center', transform=ax2.transAxes)
        else:
            ax2.text(0.5, 0.5, '无夏普比率列', ha='center', va='center', transform=ax2.transAxes)
    except Exception as e:
        ax2.text(0.5, 0.5, f'绘制错误: {e}', ha='center', va='center', transform=ax2.transAxes)
    
    ax2.grid(True, alpha=0.3)
    
    # 3. 收益-风险散点图
    ax3 = axes[1, 0]
    try:
        if ('总收益率' in df.columns and '最大回撤' in df.columns and 
            '行业' in df.columns and '分配资金' in df.columns):
            
            returns = pd.to_numeric(df['总收益率'], errors='coerce') * 100
            drawdown = pd.to_numeric(df['最大回撤'], errors='coerce') * 100
            capital = pd.to_numeric(df['分配资金'], errors='coerce')
            industries = df['行业']
            
            valid_idx = returns.notna() & drawdown.notna() & capital.notna() & industries.notna()
            
            if valid_idx.sum() > 0:
                # 为每个行业分配颜色
                unique_industries = industries[valid_idx].unique()
                industry_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_industries)))
                color_map = dict(zip(unique_industries, industry_colors))
                
                for industry in unique_industries:
                    mask = (industries == industry) & valid_idx
                    subset_returns = returns[mask]
                    subset_drawdown = drawdown[mask]
                    subset_capital = capital[mask]
                    
                    if len(subset_returns) > 0:
                        ax3.scatter(subset_returns, subset_drawdown,
                                   s=subset_capital / 1e4,
                                   c=[color_map[industry]], alpha=0.6, 
                                   edgecolors='black', label=industry)
                
                ax3.set_xlabel('总收益率 (%)')
                ax3.set_ylabel('最大回撤 (%)')
                ax3.set_title('收益-回撤散点（气泡=资金，颜色=行业）')
                if len(unique_industries) <= 10:
                    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            else:
                ax3.text(0.5, 0.5, '无有效数据', ha='center', va='center', transform=ax3.transAxes)
        else:
            ax3.text(0.5, 0.5, '缺少必要列', ha='center', va='center', transform=ax3.transAxes)
    except Exception as e:
        ax3.text(0.5, 0.5, f'绘制错误: {e}', ha='center', va='center', transform=ax3.transAxes)
    
    ax3.grid(True, alpha=0.3)
    
    # 4. 胜率分布
    ax4 = axes[1, 1]
    try:
        if '胜率' in df.columns:
            win_rate = pd.to_numeric(df['胜率'], errors='coerce') * 100
            win_rate = win_rate.dropna()
            
            if len(win_rate) > 0:
                ax4.hist(win_rate, bins=15, color='green', alpha=0.7, edgecolor='black')
                ax4.axvline(win_rate.mean(), color='red', linestyle='--', linewidth=2,
                           label=f'均值: {win_rate.mean():.1f}%')
                ax4.axvline(50, color='black', linestyle='--', linewidth=1, label='50%线')
                ax4.set_xlabel('胜率 (%)')
                ax4.set_ylabel('频数')
                ax4.set_title('胜率分布')
                ax4.legend()
            else:
                ax4.text(0.5, 0.5, '无胜率数据', ha='center', va='center', transform=ax4.transAxes)
        else:
            ax4.text(0.5, 0.5, '无胜率列', ha='center', va='center', transform=ax4.transAxes)
    except Exception as e:
        ax4.text(0.5, 0.5, f'绘制错误: {e}', ha='center', va='center', transform=ax4.transAxes)
    
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('v8.1_收益风险分布分析.png', dpi=300, bbox_inches='tight')
    print("✓ 图表已保存: v8.1_收益风险分布分析.png")
    plt.show()

def plot_top_pairs_table_v81(df, top_n=20):
    """Top N配对表格（v8.1版）"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'FangSong', 'KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建副本避免修改原数据
    display_df = df.copy()
    
    # 转换为数值
    numeric_cols = ['总收益率', '夏普比率', '最大回撤', '胜率', '分配资金', 
                   '行业权重', '配对权重', '交易次数']
    
    for col in numeric_cols:
        if col in display_df.columns:
            try:
                if display_df[col].dtype == 'object':
                    display_df[col] = display_df[col].astype(str).str.replace('%', '').str.replace(',', '')
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
            except:
                pass
    
    # 计算排序分数
    if '夏普比率' in display_df.columns and '总收益率' in display_df.columns:
        display_df['score'] = display_df['夏普比率'] * 0.6 + display_df['总收益率'] * 0.4
        top_df = display_df.nlargest(top_n, 'score')
    else:
        top_df = display_df.nlargest(top_n, '总收益率')
    
    # 格式化股票代码
    if '股票1' in top_df.columns:
        top_df['股票1'] = top_df['股票1'].apply(lambda x: f"{int(x):06d}" if pd.notna(x) and str(x).isdigit() else x)
    if '股票2' in top_df.columns:
        top_df['股票2'] = top_df['股票2'].apply(lambda x: f"{int(x):06d}" if pd.notna(x) and str(x).isdigit() else x)
    
    fig, ax = plt.subplots(figsize=(20, top_n * 0.6 + 3))
    ax.axis('tight')
    ax.axis('off')
    
    # 确定显示的列
    display_cols = ['行业', '股票1', '股票2', '总收益率', '夏普比率',
                   '最大回撤', '胜率', '交易次数']
    
    optional_cols = ['行业权重', '配对权重', '分配资金', '滑点成本占比',
                    '波动率', '半衰期', '入场阈值', '最大持仓天数',
                    '平均持仓天数', '单次交易平均收益']
    
    for col in optional_cols:
        if col in top_df.columns and col in df.columns:
            display_cols.append(col)
    
    # 确保列存在
    display_cols = [c for c in display_cols if c in top_df.columns]
    
    # 提取显示数据
    display_data = top_df[display_cols].copy()
    
    # 格式化显示
    format_dict = {
        '总收益率': lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A",
        '行业权重': lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A",
        '配对权重': lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A",
        '分配资金': lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A",
        '最大回撤': lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A",
        '胜率': lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A",
        '滑点成本占比': lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A",
        '夏普比率': lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
        '波动率': lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A",
        '半衰期': lambda x: f"{x:.1f}" if pd.notna(x) else "N/A",
        '入场阈值': lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
        '最大持仓天数': lambda x: f"{int(x)}" if pd.notna(x) else "N/A",
        '平均持仓天数': lambda x: f"{x:.1f}" if pd.notna(x) else "N/A",
        '单次交易平均收益': lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
    }
    
    for col in display_cols:
        if col in format_dict:
            display_data[col] = display_data[col].apply(format_dict[col])
    
    table = ax.table(cellText=display_data.values,
                    colLabels=display_cols,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.08] * len(display_cols))
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)
    
    # 设置表头样式
    for i in range(len(display_cols)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置行颜色
    for i in range(1, len(top_df) + 1):
        if '总收益率' in top_df.columns:
            ret = top_df.iloc[i-1]['总收益率']
            if pd.notna(ret):
                if ret > 0.3:
                    color = '#90EE90'  # 浅绿
                elif ret > 0.15:
                    color = '#E0F7FA'  # 浅蓝
                elif ret > 0.05:
                    color = '#FFF9C4'  # 浅黄
                else:
                    color = '#FFCCBC'  # 浅橙
            else:
                color = 'white'
        else:
            color = 'white'
        
        for j in range(len(display_cols)):
            table[(i, j)].set_facecolor(color)
    
    plt.title(f'Top {top_n} 配对排名（v8.1多模式回测策略）', 
             fontsize=16, fontweight='bold', pad=20)
    plt.savefig(f'v8.1_Top{top_n}_配对排名表.png', dpi=300, bbox_inches='tight')
    print(f"✓ 图表已保存: v8.1_Top{top_n}_配对排名表.png")
    plt.show()

def plot_industry_correlation_heatmap_cluster_v81(df, equity_data=None):
    """v8.1行业收益相关性热力图 + 群落聚类图"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'FangSong', 'KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    
    if '行业' not in df.columns:
        print("⚠️ 无行业数据，跳过相关性分析")
        return
    
    # 1. 构建行业收益数据
    try:
        # 如果有净值数据，使用净值计算相关性
        if equity_data and len(equity_data) > 0:
            print("  使用净值数据计算行业相关性...")
            
            # 按行业分组净值数据
            industry_curves = {}
            for pair_key, data in equity_data.items():
                industry = data.get('industry', '未知行业')
                nav_data = data['data']
                
                if 'net_value' in nav_data.columns or 'value' in nav_data.columns:
                    nav_col = 'net_value' if 'net_value' in nav_data.columns else 'value'
                    
                    if industry not in industry_curves:
                        industry_curves[industry] = []
                    
                    # 计算归一化净值
                    if nav_data[nav_col].iloc[0] > 0:
                        nav_series = nav_data[nav_col] / nav_data[nav_col].iloc[0]
                        industry_curves[industry].append(nav_series)
            
            # 合并行业内净值曲线
            industry_combined = {}
            for industry, curves in industry_curves.items():
                if curves:
                    # 对齐日期并取平均
                    aligned_curves = []
                    for curve in curves:
                        if isinstance(curve, pd.Series):
                            aligned_curves.append(curve)
                    
                    if aligned_curves:
                        combined = pd.concat(aligned_curves, axis=1).mean(axis=1)
                        industry_combined[industry] = combined
            
            if len(industry_combined) >= 2:
                # 构建DataFrame
                curve_df = pd.DataFrame(industry_combined)
                curve_df = curve_df.resample('D').ffill()  # 填充到日频
                daily_returns = curve_df.pct_change().dropna()
                
                if len(daily_returns) >= 20:  # 至少20个交易日
                    corr_matrix = daily_returns.corr()
                else:
                    corr_matrix = None
            else:
                corr_matrix = None
        else:
            corr_matrix = None
        
        # 如果净值相关性计算失败，使用基础统计量
        if corr_matrix is None or corr_matrix.isna().all().all():
            print("  使用基础统计量计算行业相关性...")
            
            # 计算行业平均指标
            industry_stats = df.groupby('行业').agg({
                '总收益率': lambda x: pd.to_numeric(x, errors='coerce').mean(),
                '夏普比率': lambda x: pd.to_numeric(x, errors='coerce').mean(),
                '最大回撤': lambda x: pd.to_numeric(x, errors='coerce').mean(),
                '胜率': lambda x: pd.to_numeric(x, errors='coerce').mean(),
                '波动率': lambda x: pd.to_numeric(x, errors='coerce').mean() if '波动率' in df.columns else None
            }).dropna()
            
            if len(industry_stats) >= 2:
                corr_matrix = industry_stats.corr()
            else:
                print("  行业数据不足，无法计算相关性")
                return
    except Exception as e:
        print(f"  构建相关性矩阵失败: {e}")
        return
    
    if corr_matrix is None or len(corr_matrix) < 2:
        print("  无法计算相关性矩阵")
        return
    
    # 2. 绘制热力图+聚类图
    fig = plt.figure(figsize=(18, 10))
    
    try:
        # 计算聚类链接
        corr_dist = 1 - corr_matrix  # 转换为距离矩阵
        corr_dist = corr_dist.fillna(1)  # 填充NaN为最大距离
        
        linkage_matrix = linkage(squareform(corr_dist), method='ward')
        
        # 子图1：聚类树状图
        ax1 = plt.subplot2grid((1, 10), (0, 0), colspan=1)
        dendro = dendrogram(linkage_matrix, labels=corr_matrix.index, ax=ax1, orientation='left')
        ax1.set_xticks([])
        ax1.set_title('行业聚类', fontsize=12, fontweight='bold')
        ax1.grid(False)
        
        # 子图2：相关性热力图（按聚类排序）
        ax2 = plt.subplot2grid((1, 10), (0, 1), colspan=9)
        # 按聚类结果排序
        sorted_industries = dendro['ivl']
        corr_sorted = corr_matrix.loc[sorted_industries, sorted_industries]
        
        # 绘制热力图
        im = ax2.imshow(corr_sorted, cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='auto')
        
        # 设置刻度和标签
        ax2.set_xticks(range(len(sorted_industries)))
        ax2.set_yticks(range(len(sorted_industries)))
        ax2.set_xticklabels(sorted_industries, rotation=45, ha='right', fontsize=9)
        ax2.set_yticklabels(sorted_industries, fontsize=9)
        
        # 添加数值标注
        for i in range(len(sorted_industries)):
            for j in range(len(sorted_industries)):
                value = corr_sorted.iloc[i, j]
                if not np.isnan(value):
                    text = ax2.text(j, i, f'{value:.2f}',
                                   ha="center", va="center", color="black", fontsize=8)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
        cbar.set_label('收益相关性系数', fontsize=12)
        
        # 添加群落划分
        try:
            cluster_labels = fcluster(linkage_matrix, t=0.5, criterion='distance')
            cluster_colors = plt.cm.Set3(cluster_labels / cluster_labels.max())
            
            # 在热力图边缘标记群落
            for i, (industry, cluster) in enumerate(zip(sorted_industries, cluster_labels)):
                ax2.axhline(y=i-0.5, color=cluster_colors[cluster-1], linewidth=3)
                ax2.axvline(x=i-0.5, color=cluster_colors[cluster-1], linewidth=3)
        except:
            pass
        
        ax2.set_title('v8.1行业收益相关性热力图 + 群落聚类', fontsize=14, fontweight='bold')
        
    except Exception as e:
        print(f"  绘制聚类图失败: {e}")
        # 简化的相关性热力图
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(corr_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
        
        ax.set_xticks(range(len(corr_matrix)))
        ax.set_yticks(range(len(corr_matrix)))
        ax.set_xticklabels(corr_matrix.index, rotation=45, ha='right')
        ax.set_yticklabels(corr_matrix.index)
        
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                value = corr_matrix.iloc[i, j]
                if not np.isnan(value):
                    ax.text(j, i, f'{value:.2f}', ha="center", va="center", color="black")
        
        plt.colorbar(im, ax=ax, label='收益相关性系数')
        ax.set_title('v8.1行业收益相关性热力图', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('v8.1_行业相关性聚类分析.png', dpi=300, bbox_inches='tight')
    print("✓ 图表已保存: v8.1_行业相关性聚类分析.png")
    plt.show()
    
    # 输出聚类结果
    try:
        if 'cluster_labels' in locals() and 'sorted_industries' in locals():
            print("\n【行业聚类结果】")
            clusters = {}
            for industry, cluster in zip(sorted_industries, cluster_labels):
                if cluster not in clusters:
                    clusters[cluster] = []
                clusters[cluster].append(industry)
            
            for cluster_id, industries in clusters.items():
                print(f"群落 {cluster_id}: {', '.join(industries)}")
    except:
        pass

def generate_summary_report_v81(df, summary_report=None):
    """生成v8.1汇总报告"""
    print("\n" + "="*80)
    print("全行业配对策略回测分析报告 (v8.1 - 多模式回测)")
    print("="*80)
    
    # 从汇总报告中提取信息
    if summary_report and 'summary' in summary_report:
        summary = summary_report['summary']
        print(f"\n【基本信息】")
        print(f"回测模式: {summary.get('backtest_mode', '未知')}")
        print(f"总配对数: {summary.get('total_pairs', len(df))}")
        if 'portfolio_return' in summary:
            print(f"组合预期收益率: {summary.get('portfolio_return', 0)*100:.2f}%")
        if 'portfolio_sharpe' in summary:
            print(f"组合夏普比率: {summary.get('portfolio_sharpe', 0):.2f}")
    
    print(f"\n【收益统计】")
    try:
        returns = pd.to_numeric(df['总收益率'], errors='coerce')
        if returns.notna().sum() > 0:
            print(f"平均收益率: {returns.mean()*100:.2f}%")
            print(f"收益率中位数: {returns.median()*100:.2f}%")
            pos_returns = returns[returns > 0]
            print(f"正收益配对数: {len(pos_returns)} ({len(pos_returns)/len(returns)*100:.1f}%)")
    except:
        pass
    
    print(f"\n【风险指标】")
    try:
        if '夏普比率' in df.columns:
            sharpe = pd.to_numeric(df['夏普比率'], errors='coerce')
            if sharpe.notna().sum() > 0:
                print(f"平均夏普比率: {sharpe.mean():.2f}")
                print(f"夏普>1的配对数: {len(sharpe[sharpe > 1])}")
    except:
        pass
    
    try:
        if '最大回撤' in df.columns:
            drawdown = pd.to_numeric(df['最大回撤'], errors='coerce')
            if drawdown.notna().sum() > 0:
                print(f"平均最大回撤: {drawdown.mean()*100:.2f}%")
    except:
        pass
    
    try:
        if '胜率' in df.columns:
            win_rate = pd.to_numeric(df['胜率'], errors='coerce')
            if win_rate.notna().sum() > 0:
                print(f"平均胜率: {win_rate.mean()*100:.1f}%")
    except:
        pass
    
    # 滑点影响分析
    try:
        if '滑点成本占比' in df.columns:
            # 确保有收益数据
            returns = pd.Series(dtype=float)
            if '总收益率' in df.columns:
                returns = pd.to_numeric(df['总收益率'], errors='coerce')
            
            slippage = pd.to_numeric(df['滑点成本占比'], errors='coerce')
            if slippage.notna().sum() > 0 and returns.notna().sum() > 0:
                # 计算净收益
                df['net_return'] = returns - slippage
                print(f"\n【滑点影响】")
                print(f"平均滑点成本: {slippage.mean()*100:.2f}%")
                
                # 计算平均净收益率
                net_return_series = df['net_return']
                if net_return_series.notna().sum() > 0:
                    print(f"平均净收益率: {net_return_series.mean()*100:.2f}%")
                
                # 计算滑点占总收益比例
                if returns.sum() > 0:
                    slip_ratio = (slippage.sum() / returns.sum()) * 100
                    print(f"滑点占收益比: {slip_ratio:.1f}%")
    except Exception as e:
        print(f"滑点分析出错: {e}")
    
    # 自适应参数统计
    adaptive_stats = []
    try:
        if '半衰期' in df.columns:
            halflife = pd.to_numeric(df['半衰期'], errors='coerce')
            if halflife.notna().sum() > 0:
                adaptive_stats.append(f"平均半衰期: {halflife.mean():.1f}天")
    except:
        pass
    
    try:
        if '入场阈值' in df.columns:
            entry_th = pd.to_numeric(df['入场阈值'], errors='coerce')
            if entry_th.notna().sum() > 0:
                adaptive_stats.append(f"平均入场阈值: {entry_th.mean():.2f}")
    except:
        pass
    
    try:
        if '波动率' in df.columns:
            volatility = pd.to_numeric(df['波动率'], errors='coerce')
            if volatility.notna().sum() > 0:
                adaptive_stats.append(f"平均波动率: {volatility.mean()*100:.1f}%")
    except:
        pass
    
    if adaptive_stats:
        print(f"\n【自适应参数统计（v8.1）】")
        for stat in adaptive_stats:
            print(f"  {stat}")
        
        # 高/低波动配对分析
        try:
            if '波动率' in df.columns and '入场阈值' in df.columns and '最大持仓天数' in df.columns:
                volatility = pd.to_numeric(df['波动率'], errors='coerce')
                entry_th = pd.to_numeric(df['入场阈值'], errors='coerce')
                hold_days = pd.to_numeric(df['最大持仓天数'], errors='coerce')
                
                high_vol_mask = volatility > 0.30
                low_vol_mask = volatility < 0.15
                
                if high_vol_mask.any():
                    high_vol_count = high_vol_mask.sum()
                    print(f"\n  高波动配对(>30%): {high_vol_count}对")
                    if entry_th[high_vol_mask].notna().any():
                        print(f"    - 平均阈值: {entry_th[high_vol_mask].mean():.2f}")
                    if hold_days[high_vol_mask].notna().any():
                        print(f"    - 平均持仓: {hold_days[high_vol_mask].mean():.0f}天")
                
                if low_vol_mask.any():
                    low_vol_count = low_vol_mask.sum()
                    print(f"  低波动配对(<15%): {low_vol_count}对")
                    if entry_th[low_vol_mask].notna().any():
                        print(f"    - 平均阈值: {entry_th[low_vol_mask].mean():.2f}")
                    if hold_days[low_vol_mask].notna().any():
                        print(f"    - 平均持仓: {hold_days[low_vol_mask].mean():.0f}天")
        except:
            pass
    
    # 资金分配
    if '行业权重' in df.columns and '行业' in df.columns:
        print(f"\n【资金分配（行业）】")
        try:
            industry_stats = {}
            for _, row in df.iterrows():
                industry = row['行业']
                
                # 处理权重
                weight_str = row['行业权重']
                if isinstance(weight_str, str):
                    weight = float(weight_str.replace('%', '')) / 100
                else:
                    weight = float(weight_str)
                
                # 处理资金
                capital_str = row['分配资金']
                if isinstance(capital_str, str):
                    capital = float(capital_str.replace(',', ''))
                else:
                    capital = float(capital_str)
                
                # 处理收益率
                return_str = row['总收益率']
                if isinstance(return_str, str):
                    ret = float(return_str.replace('%', '')) / 100
                else:
                    ret = float(return_str)
                
                if industry not in industry_stats:
                    industry_stats[industry] = {
                        'weight': weight,
                        'capital': 0,
                        'returns': [],
                        'sharpe': []
                    }
                
                industry_stats[industry]['capital'] += capital
                industry_stats[industry]['returns'].append(ret)
                
                # 处理夏普
                if '夏普比率' in row:
                    sharpe_str = row['夏普比率']
                    if isinstance(sharpe_str, str):
                        sharpe = float(sharpe_str)
                    else:
                        sharpe = float(sharpe_str)
                    industry_stats[industry]['sharpe'].append(sharpe)
            
            # 按权重排序
            sorted_industries = sorted(industry_stats.items(), 
                                      key=lambda x: x[1]['weight'], reverse=True)
            
            for idx, (industry, stats) in enumerate(sorted_industries[:5], 1):
                avg_return = np.mean(stats['returns']) if stats['returns'] else 0
                avg_sharpe = np.mean(stats['sharpe']) if stats['sharpe'] else 0
                print(f"{idx}. {industry}: 权重{stats['weight']*100:.1f}%, "
                      f"资金{stats['capital']:,.0f}, "
                      f"收益{avg_return*100:.1f}%, "
                      f"夏普{avg_sharpe:.2f}")
        except Exception as e:
            print(f"  解析行业数据失败: {e}")
    
    # 最佳配对
    print(f"\n【最佳配对Top 5】")
    try:
        # 计算评分
        df_copy = df.copy()
        if '夏普比率' in df_copy.columns and '总收益率' in df_copy.columns:
            sharpe = pd.to_numeric(df_copy['夏普比率'], errors='coerce')
            returns = pd.to_numeric(df_copy['总收益率'], errors='coerce')
            
            valid_idx = sharpe.notna() & returns.notna()
            if valid_idx.any():
                df_copy['score'] = 0
                df_copy.loc[valid_idx, 'score'] = sharpe[valid_idx] * 0.6 + returns[valid_idx] * 0.4
                top_pairs = df_copy.nlargest(5, 'score')
            else:
                top_pairs = df_copy.nlargest(5, '总收益率')
        else:
            top_pairs = df_copy.nlargest(5, '总收益率')
        
        for i, (_, row) in enumerate(top_pairs.iterrows(), 1):
            industry = row.get('行业', '未知行业')
            stock1 = row.get('股票1', '')
            stock2 = row.get('股票2', '')
            
            # 获取收益率
            return_val = row.get('总收益率', 0)
            if isinstance(return_val, str):
                return_pct = float(return_val.replace('%', ''))
            else:
                return_pct = float(return_val) * 100
            
            # 获取夏普
            sharpe_val = row.get('夏普比率', 0)
            if isinstance(sharpe_val, str):
                sharpe_str = sharpe_val
            else:
                sharpe_str = f"{float(sharpe_val):.2f}"
            
            # 附加信息
            extra = ""
            if '半衰期' in row and pd.notna(row['半衰期']):
                hl = row['半衰期']
                if isinstance(hl, str):
                    hl_val = float(hl)
                else:
                    hl_val = float(hl)
                extra += f" | HL:{hl_val:.1f}天"
            
            if '入场阈值' in row and pd.notna(row['入场阈值']):
                th = row['入场阈值']
                if isinstance(th, str):
                    th_val = float(th)
                else:
                    th_val = float(th)
                extra += f" | 阈值:{th_val:.2f}"
            
            print(f"{i}. {industry}: {stock1}-{stock2} | "
                  f"收益:{return_pct:.1f}% | 夏普:{sharpe_str}{extra}")
    except Exception as e:
        print(f"  解析最佳配对失败: {e}")
    
    print("\n" + "="*80)

def plot_backtest_mode_comparison(df, equity_data=None, summary_report=None):
    """v8.1回测模式对比分析（如果有多模式数据）"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'FangSong', 'KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 检查是否有回测模式信息
    if summary_report is None or 'summary' not in summary_report:
        print("无回测模式信息，跳过模式对比分析")
        return
    
    summary = summary_report['summary']
    backtest_mode = summary.get('backtest_mode', 'unknown')
    
    print(f"\n[v8.1专属] 回测模式分析: {backtest_mode}")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 回测模式说明
    ax1 = axes[0, 0]
    ax1.axis('off')
    
    mode_info = {
        'rolling_window': '滚动窗口回测\n• 样本内筛选，样本外验证\n• 多窗口稳定性评估\n• 避免前视偏差',
        'real_time': '实盘回测\n• 每天用历史数据生成信号\n• 第二天交易\n• 最接近实盘',
        'single_debug': '单次回测（调试）\n• 保持v8.0逻辑\n• 存在前视偏差\n• 仅用于快速调试'
    }
    
    description = mode_info.get(backtest_mode, f"未知模式: {backtest_mode}")
    
    ax1.text(0.1, 0.9, f'当前回测模式: {backtest_mode}', 
            fontsize=16, fontweight='bold', transform=ax1.transAxes)
    ax1.text(0.1, 0.7, description, fontsize=12, transform=ax1.transAxes,
            verticalalignment='top')
    
    # 添加模式特点
    features = {
        'rolling_window': '✓ 多时间窗口验证\n✓ 基于稳定性筛选\n✓ 避免前视偏差\n✓ 适合策略验证',
        'real_time': '✓ 完全模拟实盘\n✓ 定期重新学习\n✓ 无未来信息\n✓ 适合实盘准备',
        'single_debug': '✓ 计算速度快\n✓ 便于调试\n✗ 存在前视偏差\n✗ 不用于实盘验证'
    }
    
    if backtest_mode in features:
        ax1.text(0.1, 0.4, '模式特点:', fontsize=12, fontweight='bold', 
                transform=ax1.transAxes)
        ax1.text(0.1, 0.3, features[backtest_mode], fontsize=11, 
                transform=ax1.transAxes, verticalalignment='top')
    
    ax1.set_title('v8.1回测模式分析', fontsize=14, fontweight='bold')
    
    # 2. 模式性能指标
    ax2 = axes[0, 1]
    ax2.axis('off')
    
    if summary:
        performance_text = "回测性能指标:\n\n"
        
        if 'portfolio_return' in summary:
            performance_text += f"组合收益率: {summary['portfolio_return']*100:.2f}%\n"
        
        if 'portfolio_sharpe' in summary:
            performance_text += f"组合夏普比率: {summary['portfolio_sharpe']:.2f}\n"
        
        if 'total_pairs' in summary:
            performance_text += f"有效配对数: {summary['total_pairs']}\n"
        
        if 'slippage_impact' in summary:
            performance_text += f"滑点影响: {summary['slippage_impact']*100:.2f}%\n"
        
        if 'net_return' in summary:
            performance_text += f"净收益率: {summary['net_return']*100:.2f}%\n"
        
        ax2.text(0.1, 0.8, performance_text, fontsize=12, transform=ax2.transAxes,
                verticalalignment='top')
    
    # 3. 模式优势分析
    ax3 = axes[1, 0]
    ax3.axis('off')
    
    advantages = {
        'rolling_window': '滚动窗口优势:\n\n• 多时间点验证策略稳定性\n• 避免单一时间段的偶然性\n• 基于历史表现筛选稳健配对\n• 更适合长期投资',
        'real_time': '实盘回测优势:\n\n• 完全模拟真实交易环境\n• 无未来信息，结果更可信\n• 定期调仓适应市场变化\n• 直接用于实盘交易',
        'single_debug': '调试模式用途:\n\n• 快速验证策略逻辑\n• 调试参数设置\n• 初步筛选配对\n• 不用于最终决策'
    }
    
    if backtest_mode in advantages:
        ax3.text(0.1, 0.8, advantages[backtest_mode], fontsize=12, 
                transform=ax3.transAxes, verticalalignment='top')
    
    # 4. 建议
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    recommendations = {
        'rolling_window': '建议:\n\n1. 关注配对在多个窗口的表现\n2. 优先选择稳定性高的配对\n3. 注意市场风格变化的影响\n4. 定期重新评估策略有效性',
        'real_time': '建议:\n\n1. 可直接用于实盘交易\n2. 注意控制重新学习频率\n3. 监控配对关系稳定性\n4. 准备应急平仓机制',
        'single_debug': '注意:\n\n1. 此模式存在前视偏差\n2. 不应用于实盘决策\n3. 仅用于策略逻辑验证\n4. 实际表现可能差异较大'
    }
    
    if backtest_mode in recommendations:
        ax4.text(0.1, 0.8, recommendations[backtest_mode], fontsize=12,
                transform=ax4.transAxes, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('v8.1_回测模式分析.png', dpi=300, bbox_inches='tight')
    print("✓ 图表已保存: v8.1_回测模式分析.png")
    plt.show()

def main():
    """主函数"""
    print("="*80)
    print("全行业配对策略回测可视化分析 v8.1")
    print("适配 pairtrading_all_industries_v8.1.py 输出格式")
    print("="*80)
    
    # 1. 加载数据
    print("\n[1/4] 加载回测结果数据...")
    df, version, summary_report = load_latest_results_v81()
    
    if df is None or len(df) == 0:
        print("✗ 无法加载数据，程序退出")
        return
    
    # 转换百分比列
    df = convert_percentage_columns(df)
    
    # 2. 加载净值数据
    print("\n[2/4] 加载净值数据...")
    equity_data = load_equity_data_from_zip()
    
    # 3. 生成汇总报告
    print("\n[3/4] 生成分析报告...")
    generate_summary_report_v81(df, summary_report)
    
    # 4. 生成图表
    print("\n[4/4] 生成可视化图表...")
    
    print("\n[v8.1核心] 组合净值走势分析...")
    portfolio_nav = plot_portfolio_nav_curve_v81(df, equity_data)
    
    print("\n[v8.1专属] 自适应参数分析...")
    plot_adaptive_params_analysis_v81(df)
    
    print("\n[v8.1专属] 行业权重分配...")
    plot_industry_weights_v81(df)
    
    print("\n[v8.1专属] 滑点影响分析...")
    plot_slippage_analysis_v81(df)
    
    print("\n收益风险分布分析...")
    plot_return_distribution_v81(df)
    
    print("\n[v8.1新增] 行业收益相关性聚类分析...")
    plot_industry_correlation_heatmap_cluster_v81(df, equity_data)
    
    print("\nTop配对排名表...")
    plot_top_pairs_table_v81(df, top_n=20)
    
    print("\n[v8.1新增] 回测模式对比分析...")
    plot_backtest_mode_comparison(df, equity_data, summary_report)
    
    print("\n" + "="*80)
    print("✓ v8.1全部分析完成！")
    print("\n生成图表清单:")
    print("  - v8.1_组合净值走势分析.png")
    print("  - v8.1_自适应参数分析.png")
    print("  - v8.1_行业权重分配.png")
    print("  - v8.1_滑点影响分析.png")
    print("  - v8.1_收益风险分布分析.png")
    print("  - v8.1_行业相关性聚类分析.png")
    print("  - v8.1_Top20_配对排名表.png")
    print("  - v8.1_回测模式分析.png")
    print("\n注意事项:")
    print("  1. 确保已运行pairtrading_all_industries_v8.1.py生成结果文件")
    print("  2. 净值数据从ZIP文件加载，请勿删除相关文件")
    print("  3. 支持滚动窗口、实盘、单次三种回测模式分析")
    print("="*80)

if __name__ == "__main__":
    main()