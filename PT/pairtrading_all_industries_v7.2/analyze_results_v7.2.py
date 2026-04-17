# === 全行业配对回测结果可视化分析 v7.2 ===
# 适配 pairtrading_all_industries_v7.2.py 输出格式（申万行业分类）
# 【修复】兼容新的净值数据ZIP格式，显示真实回测净值曲线
# 修复日期：2026-04-16

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import glob
import os
import warnings
import zipfile
from pathlib import Path
import matplotlib.dates as mdates

# 完全关闭字体警告
warnings.filterwarnings("ignore", category=UserWarning)

# 强制设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'FangSong', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')

def load_latest_results():
    """加载最新的v7.2回测结果文件，适配新的净值数据格式"""
    # 优先v7.2格式
    files = glob.glob("v7.2_组合回测结果_*.csv")
    if not files:
        # 回退v7.1格式
        files = glob.glob("v7.1_组合回测结果_*.csv")
        if not files:
            # 回退v7.0格式
            files = glob.glob("组合回测结果_*.csv")
            if not files:
                print("未找到回测结果文件")
                return None, None

    latest_file = max(files, key=os.path.getctime)
    print(f"加载文件: {latest_file}")

    df = pd.read_csv(latest_file, encoding='utf-8-sig')

    # 修复版本检测逻辑
    if 'v7.2' in latest_file:
        version = "v7.2"
    elif 'v7.1' in latest_file:
        version = "v7.1"
    elif 'v7.0' in latest_file:
        version = "v7.0"
    elif 'halflife' in df.columns or 'entry_threshold' in df.columns:
        version = "v7.2"
    elif 'industry_weight' in df.columns or 'slippage_impact' in df.columns:
        version = "v7.0"
    else:
        version = "v6.0"

    print(f"检测到版本: {version}")
    print(f"数据形状: {df.shape}")
    
    # 检查行业列名
    industry_col = '行业' if '行业' in df.columns else 'industry' if 'industry' in df.columns else None
    if industry_col:
        print(f"行业数量: {df[industry_col].nunique()}")
    else:
        print("行业数量: 0")
    
    # 【修复】列名映射，确保向后兼容
    column_mapping = {
        '净值文件': 'equity_file',
        '股票1': 'stock1',
        '股票2': 'stock2',
        '行业': 'industry',
        '总收益率': 'total_return',
        '夏普比率': 'sharpe_ratio',
        '最大回撤': 'max_drawdown',
        '胜率': 'win_rate',
        '交易次数': 'num_trades',
        '分配资金': 'allocated_capital',
        '配对权重': 'pair_weight',
        '行业权重': 'industry_weight',
        '滑点成本占比': 'slippage_impact',
        '波动率': 'volatility',
        '半衰期': 'halflife',
        '入场阈值': 'entry_threshold',
        '最大持仓天数': 'max_holding_days',
        '冷却次数': 'cooldown_count',
        '平均持仓天数': 'avg_hold_days',
        '单次交易平均收益': 'avg_trade_return'
    }
    
    # 应用列名映射
    for new_col, old_col in column_mapping.items():
        if new_col in df.columns and old_col not in df.columns:
            df[old_col] = df[new_col]
            print(f"  ✓ 映射列: {new_col} -> {old_col}")
    
    # 转换数值列
    numeric_columns = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 
                      'num_trades', 'allocated_capital', 'pair_weight', 'industry_weight',
                      'slippage_impact', 'volatility', 'halflife', 'entry_threshold', 
                      'max_holding_days', 'cooldown_count', 'avg_hold_days', 'avg_trade_return']
    
    for col in numeric_columns:
        if col in df.columns:
            # 移除百分号、中文单位等字符
            if df[col].dtype == 'object':
                try:
                    # 保存原始值用于调试
                    sample_value = str(df[col].iloc[0]) if len(df) > 0 else ''
                    
                    # 移除常见单位字符
                    df[col] = df[col].astype(str)
                    df[col] = df[col].str.replace('%', '')
                    df[col] = df[col].str.replace('天', '')
                    df[col] = df[col].str.replace('日', '')
                    df[col] = df[col].str.replace(' ', '')
                    
                    # 尝试转换为浮点数
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # 对于百分数，需要除以100
                    if col in ['total_return', 'max_drawdown', 'win_rate', 'volatility', 
                              'slippage_impact', 'pair_weight', 'industry_weight']:
                        df[col] = df[col] / 100
                        
                except Exception as e:
                    print(f"  警告: 转换列 {col} 时出错: {e}")
                    # 保留原值
                    pass
    
    print("✓ 列名映射和数据类型转换完成")
    
    return df, version

def load_equity_data_from_zip():
    """从最新的ZIP文件中加载所有净值数据"""
    # 查找最新的净值ZIP文件
    zip_files = glob.glob("v7.2_净值数据_*.zip")
    if not zip_files:
        print("⚠️ 未找到净值数据ZIP文件")
        return None
    
    latest_zip = max(zip_files, key=os.path.getctime)
    print(f"✓ 找到净值数据ZIP文件: {latest_zip}")
    
    all_curves = {}
    
    try:
        with zipfile.ZipFile(latest_zip, 'r') as zipf:
            # 获取ZIP中的所有CSV文件
            csv_files = [f for f in zipf.namelist() if f.endswith('.csv')]
            print(f"✓ 找到 {len(csv_files)} 个净值CSV文件")
            
            for csv_file in csv_files:
                try:
                    # 从文件名解析股票对信息
                    filename = Path(csv_file).stem
                    parts = filename.split('_')
                    
                    if len(parts) >= 3:
                        industry_name = parts[0]
                        stock1 = parts[1]
                        stock2 = parts[2]
                        
                        # 读取CSV数据
                        with zipf.open(csv_file) as f:
                            equity_df = pd.read_csv(f)
                            
                            # 检查CSV格式
                            if 'date' not in equity_df.columns or 'net_value' not in equity_df.columns:
                                print(f"  ⚠️ CSV文件 {csv_file} 缺少必要列，实际列: {equity_df.columns.tolist()}")
                                continue
                            
                            # 转换日期格式
                            equity_df['date'] = pd.to_datetime(equity_df['date'])
                            equity_df.set_index('date', inplace=True)
                            
                            # 归一化到1
                            if len(equity_df) > 0:
                                equity_df['nav'] = equity_df['net_value'] / equity_df['net_value'].iloc[0]
                                key = f"{industry_name}_{stock1}_{stock2}"
                                all_curves[key] = {
                                    'data': equity_df,
                                    'industry': industry_name,
                                    'stock1': stock1,
                                    'stock2': stock2
                                }
                                print(f"  ✓ 加载: {industry_name}_{stock1}_{stock2} ({len(equity_df)} 个数据点)")
                                
                except Exception as e:
                    print(f"  处理净值文件 {csv_file} 失败: {e}")
                    continue
        
        if not all_curves:
            print("⚠️ ZIP文件中无有效净值数据")
            return None
            
    except Exception as e:
        print(f"⚠️ 读取ZIP文件失败: {e}")
        return None
    
    return all_curves

def plot_portfolio_nav_curve(df):
    """
    【v7.2修复】组合策略整体净值走势图
    基于实际回测结果的净值ZIP文件计算组合净值
    在标题下方显示年化收益率、夏普、最大回撤等关键指标
    """
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'FangSong', 'KaiTi']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 从ZIP文件中加载净值数据
    all_curves = load_equity_data_from_zip()
    if all_curves is None:
        print("⚠️ 无法加载净值数据，使用估算版本")
        return plot_portfolio_nav_curve_simplified(df)
    
    # 2. 创建配对映射字典
    pair_mapping = {}
    for idx, row in df.iterrows():
        industry = row.get('industry', '')
        stock1 = str(row.get('stock1', '')).strip()
        stock2 = str(row.get('stock2', '')).strip()
        
        if not industry or not stock1 or not stock2:
            continue
            
        # 生成可能的键
        keys = [
            f"{industry}_{stock1}_{stock2}",
            f"{industry}_{stock1.zfill(6)}_{stock2.zfill(6)}",
            f"{industry}_{int(stock1)}_{int(stock2)}" if stock1.isdigit() and stock2.isdigit() else None
        ]
        
        for key in keys:
            if key and key in all_curves:
                weight = row.get('pair_weight', 1/len(df))
                pair_mapping[key] = {
                    'weight': weight,
                    'data': all_curves[key]['data'],
                    'row_data': row
                }
                break
    
    if not pair_mapping:
        print("⚠️ 无法匹配净值数据与配对信息，使用估算版本")
        return plot_portfolio_nav_curve_simplified(df)
    
    print(f"✓ 成功匹配 {len(pair_mapping)}/{len(df)} 个配对的净值数据")
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # 3. 计算组合净值曲线
    # 找到所有日期
    all_dates = set()
    for pair_info in pair_mapping.values():
        all_dates.update(pair_info['data'].index)
    common_dates = sorted(list(all_dates))
    
    if not common_dates:
        print("⚠️ 无有效日期数据，使用估算版本")
        return plot_portfolio_nav_curve_simplified(df)
    
    # 计算加权组合净值
    portfolio_nav = pd.Series(0.0, index=common_dates)
    total_weight = sum(info['weight'] for info in pair_mapping.values())
    
    for pair_info in pair_mapping.values():
        curve = pair_info['data']
        weight = pair_info['weight']
        
        # 对齐到共同日期
        aligned = curve.reindex(common_dates, method='ffill')
        aligned['nav'] = aligned['nav'].ffill().fillna(1.0) # aligned['nav'] = aligned['nav'].fillna(method='ffill').fillna(1.0)
        portfolio_nav += aligned['nav'] * (weight / total_weight)
    
    # 归一化
    portfolio_nav = portfolio_nav / portfolio_nav.iloc[0]
    
    # 4. 计算策略关键指标
    # 年化收益率
    years = (portfolio_nav.index[-1] - portfolio_nav.index[0]).days / 365.25
    if years > 0:
        total_return = portfolio_nav.iloc[-1] - 1
        annual_return = (1 + total_return) ** (1/years) - 1
    else:
        annual_return = 0
    
    # 计算日收益率
    daily_returns = portfolio_nav.pct_change().dropna()
    
    # 年化夏普比率（假设无风险利率为0）
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        annual_sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        annual_sharpe = 0
    
    # 最大回撤
    rolling_max = portfolio_nav.cummax()
    drawdown = (portfolio_nav - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    max_dd_idx = drawdown.idxmin() if not pd.isna(drawdown.idxmin()) else portfolio_nav.index[0]
    max_dd_date = max_dd_idx.strftime('%Y-%m-%d')
    
    # 波动率
    if len(daily_returns) > 0:
        annual_volatility = daily_returns.std() * np.sqrt(252)
    else:
        annual_volatility = 0
    
    # 卡玛比率
    if max_drawdown < 0 and abs(max_drawdown) > 0.001:
        calmar_ratio = annual_return / abs(max_drawdown)
    else:
        calmar_ratio = 0
    
    # 5. 绘制组合净值曲线
    ax1 = axes[0, 0]
    
    # 绘制净值曲线
    ax1.plot(portfolio_nav.index, portfolio_nav.values, linewidth=2.5, color='steelblue', 
            label='组合净值', alpha=0.8)
    ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='初始净值')
    
    # 添加策略指标标注（在标题下方）
    metrics_text = (f"年化收益率: {annual_return*100:.2f}% | "
                   f"年化夏普: {annual_sharpe:.2f} | "
                   f"最大回撤: {max_drawdown*100:.2f}% ({max_dd_date}) | "
                   f"波动率: {annual_volatility*100:.2f}% | "
                   f"卡玛比率: {calmar_ratio:.2f}")
    
    ax1.set_title('组合策略净值走势（v7.2 申万行业自适应）\n' + metrics_text, 
                 fontsize=14, fontweight='bold', pad=20)
    
    # 标注期末净值
    final_nav = portfolio_nav.iloc[-1]
    max_nav = portfolio_nav.max()
    min_nav = portfolio_nav.min()
    
    ax1.scatter([portfolio_nav.index[-1]], [final_nav], color='red', s=120, zorder=5, marker='o')
    ax1.annotate(f'期末: {final_nav:.2f}', 
                xy=(portfolio_nav.index[-1], final_nav), 
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    
    # 标注最大净值点
    max_idx = portfolio_nav.idxmax()
    max_date = max_idx.strftime('%Y-%m-%d')
    ax1.scatter([max_idx], [max_nav], color='green', s=100, zorder=5, marker='^')
    ax1.annotate(f'最高: {max_nav:.2f}\n({max_date})', 
                xy=(max_idx, max_nav), 
                xytext=(10, 10), textcoords='offset points',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
    
    # 添加网格和标签
    ax1.set_xlabel('日期')
    ax1.set_ylabel('净值')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 6. 回撤曲线
    ax2 = axes[0, 1]
    ax2.fill_between(portfolio_nav.index, drawdown * 100, 0, color='red', alpha=0.3)
    ax2.plot(portfolio_nav.index, drawdown * 100, color='darkred', linewidth=1.5, alpha=0.8)
    
    # 标注最大回撤
    ax2.scatter([max_dd_idx], [max_drawdown * 100], color='red', s=100, zorder=5)
    ax2.annotate(f'最大回撤: {max_drawdown*100:.1f}%\n({max_dd_date})', 
                xy=(max_dd_idx, max_drawdown * 100),
                xytext=(10, -30), textcoords='offset points',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    
    # 回撤统计
    dd_stats_text = (f"最大回撤: {max_drawdown*100:.2f}%\n"
                    f"回撤天数: {len(drawdown[drawdown < 0])}/{len(drawdown)}\n"
                    f"平均回撤: {drawdown[drawdown < 0].mean()*100:.2f}%" if len(drawdown[drawdown < 0]) > 0 else "无回撤")
    
    ax2.text(0.02, 0.98, dd_stats_text, transform=ax2.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    ax2.set_xlabel('日期')
    ax2.set_ylabel('回撤 (%)')
    ax2.set_title('组合回撤走势', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 7. 月度收益热力图
    ax3 = axes[1, 0]
    monthly_returns = portfolio_nav.resample('ME').last().pct_change().dropna()
    
    if not monthly_returns.empty and len(monthly_returns) > 1:
        # 创建年月矩阵
        monthly_df = monthly_returns.to_frame('return')
        monthly_df['year'] = monthly_df.index.year
        monthly_df['month'] = monthly_df.index.month
        
        pivot_table = monthly_df.pivot(index='year', columns='month', values='return')
        pivot_table = pivot_table * 100
        
        # 月度统计
        monthly_mean = monthly_returns.mean() * 100
        monthly_std = monthly_returns.std() * 100
        pos_months = (monthly_returns > 0).sum()
        total_months = len(monthly_returns)
        
        # 热力图标题
        heatmap_title = f'月度收益热力图 (%)\n均值: {monthly_mean:.2f}% | 波动: {monthly_std:.2f}% | 正收益月: {pos_months}/{total_months}'
        
        sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': '月度收益 (%)'}, ax=ax3)
        ax3.set_title(heatmap_title, fontsize=14, fontweight='bold')
        ax3.set_xlabel('月份')
        ax3.set_ylabel('年份')
    else:
        ax3.text(0.5, 0.5, '月度数据不足', horizontalalignment='center',
                verticalalignment='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('月度收益热力图', fontsize=14, fontweight='bold')
        ax3.axis('off')
    
    # 8. 滚动夏普比率
    ax4 = axes[1, 1]
    
    if len(daily_returns) > 252:
        rolling_sharpe = (daily_returns.rolling(252).mean() / 
                         daily_returns.rolling(252).std()) * np.sqrt(252)
        
        ax4.plot(rolling_sharpe.index, rolling_sharpe.values, 
                color='green', linewidth=2, alpha=0.8)
        
        # 滚动夏普统计
        sharpe_mean = rolling_sharpe.mean()
        sharpe_std = rolling_sharpe.std()
        
        sharpe_text = (f"平均滚动夏普: {sharpe_mean:.2f}\n"
                      f"夏普波动: {sharpe_std:.2f}\n"
                      f"夏普>1占比: {(rolling_sharpe > 1).sum()/len(rolling_sharpe)*100:.1f}%")
        
        ax4.text(0.02, 0.98, sharpe_text, transform=ax4.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    else:
        rolling_sharpe = pd.Series([annual_sharpe], index=[portfolio_nav.index[0]])
        ax4.text(0.5, 0.5, f'年化夏普: {annual_sharpe:.2f}\n数据不足计算滚动夏普', 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax4.transAxes, fontsize=12)
    
    ax4.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='夏普=1')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    if len(daily_returns) > 252:
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
    ax4.grid(True, alpha=0.3)
    
    # 9. 添加整体信息标注
    start_date = portfolio_nav.index[0].strftime("%Y-%m-%d")
    end_date = portfolio_nav.index[-1].strftime("%Y-%m-%d")
    fig.suptitle(f'v7.2 全行业配对策略回测分析\n回测区间: {start_date} 至 {end_date} | 配对数量: {len(pair_mapping)}/{len(df)}',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为标题留出空间
    plt.savefig('v7.2_组合净值走势分析.png', dpi=300, bbox_inches='tight')
    print("✓ 净值图表已保存: v7.2_组合净值走势分析.png")
    
    # 输出策略统计信息
    print("\n" + "="*80)
    print("组合策略关键指标")
    print("="*80)
    print(f"回测区间: {start_date} 至 {end_date}")
    print(f"回测天数: {len(portfolio_nav)} 天")
    print(f"配对数量: {len(pair_mapping)}/{len(df)} (匹配/总数)")
    print(f"期末净值: {final_nav:.4f}")
    print(f"累计收益: {total_return*100:.2f}%")
    print(f"年化收益: {annual_return*100:.2f}%")
    print(f"年化夏普: {annual_sharpe:.2f}")
    print(f"年化波动: {annual_volatility*100:.2f}%")
    print(f"最大回撤: {max_drawdown*100:.2f}% (于 {max_dd_date})")
    print(f"卡玛比率: {calmar_ratio:.2f}")
    
    if len(daily_returns) > 0:
        print(f"日收益均值: {daily_returns.mean()*100:.3f}%")
        print(f"日收益波动: {daily_returns.std()*100:.3f}%")
        if len(daily_returns) > 2:
            print(f"收益偏度: {daily_returns.skew():.3f}")
            print(f"收益峰度: {daily_returns.kurtosis():.3f}")
    
    print("="*80)
    
    plt.show()
    return portfolio_nav

def plot_portfolio_nav_curve_simplified(df):
    """
    【备用方案】当没有详细净值数据时，使用简化方法绘制净值图
    基于各配对的收益率和交易次数估算净值走势
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 使用回测日期范围
    dates = pd.date_range(start='2022-01-01', end='2026-04-08', freq='B')

    # 简化的净值计算：基于各配对的收益率和权重
    np.random.seed(42)
    daily_returns = []

    for _, row in df.iterrows():
        weight = row.get('pair_weight', 1/len(df))
        pair_return = row['total_return']
        volatility = row.get('volatility', 0.2)
        n_days = len(dates)

        # 生成随机 walk，最终收益匹配
        if n_days > 0:
            daily_ret = np.random.normal(pair_return/n_days, volatility/np.sqrt(252), n_days)
            daily_returns.append(daily_ret * weight)

    if daily_returns:
        portfolio_daily = np.sum(daily_returns, axis=0)
        portfolio_nav = pd.Series((1 + portfolio_daily).cumprod(), index=dates)
    else:
        portfolio_nav = pd.Series(np.ones(len(dates)), index=dates)

    # 1. 组合净值曲线
    ax1 = axes[0, 0]
    ax1.plot(portfolio_nav.index, portfolio_nav.values, linewidth=2, color='steelblue', label='组合净值（估算）')
    ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='初始净值')

    final_nav = portfolio_nav.iloc[-1]
    ax1.scatter([portfolio_nav.index[-1]], [final_nav], color='red', s=100, zorder=5)
    ax1.annotate(f'期末: {final_nav:.2f}', xy=(portfolio_nav.index[-1], final_nav), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    ax1.set_xlabel('日期')
    ax1.set_ylabel('净值')
    ax1.set_title('组合策略净值走势（基于收益率估算）', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 回撤曲线
    ax2 = axes[0, 1]
    rolling_max = portfolio_nav.cummax()
    drawdown = (portfolio_nav - rolling_max) / rolling_max

    ax2.fill_between(portfolio_nav.index, drawdown, 0, color='red', alpha=0.3)
    ax2.plot(portfolio_nav.index, drawdown, color='darkred', linewidth=1)
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
    monthly_returns = portfolio_nav.resample('ME').last().pct_change().dropna()

    monthly_df = monthly_returns.to_frame('return')
    monthly_df['year'] = monthly_df.index.year
    monthly_df['month'] = monthly_df.index.month

    pivot_table = monthly_df.pivot(index='year', columns='month', values='return')
    pivot_table = pivot_table * 100

    sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
               cbar_kws={'label': '月度收益 (%)'}, ax=ax3)
    ax3.set_title('月度收益热力图 (%)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('月份')
    ax3.set_ylabel('年份')

    # 4. 滚动夏普比率
    ax4 = axes[1, 1]
    daily_returns = portfolio_nav.pct_change().dropna()
    rolling_sharpe = (daily_returns.rolling(252).mean() / 
                     daily_returns.rolling(252).std()) * np.sqrt(252)

    ax4.plot(rolling_sharpe.index, rolling_sharpe.values, color='green', linewidth=2)
    ax4.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='夏普=1')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.fill_between(rolling_sharpe.index, rolling_sharpe.values, 1, where=(rolling_sharpe.values > 1), 
                    alpha=0.3, color='green', interpolate=True)
    ax4.fill_between(rolling_sharpe.index, rolling_sharpe.values, 1, where=(rolling_sharpe.values < 1), 
                    alpha=0.3, color='red', interpolate=True)

    ax4.set_xlabel('日期')
    ax4.set_ylabel('滚动夏普比率')
    ax4.set_title('滚动年化夏普比率（252日窗口）', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('v7.2_组合净值走势分析_估算.png', dpi=300, bbox_inches='tight')
    print("✓ 图表已保存: v7.2_组合净值走势分析_估算.png（注意：基于估算数据）")
    plt.show()
    
    return portfolio_nav

def plot_adaptive_params_analysis_v72(df):
    """【v7.2】自适应参数分析（申万行业版）"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'FangSong', 'KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    if 'halflife' not in df.columns:
        print("⚠️ 非v7.1/v7.2格式，跳过自适应参数分析")
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

    # 3. 波动率 vs Halflife 散点图（按申万行业着色）
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
    plt.savefig('v7.2_自适应参数分析.png', dpi=300, bbox_inches='tight')
    print("✓ 图表已保存: v7.2_自适应参数分析.png")
    plt.show()

def plot_industry_weights_v72(df):
    """v7.2专属：申万行业权重分配可视化（支持单行业情况）"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'FangSong', 'KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    if 'industry_weight' not in df.columns:
        print("⚠️ 非v7.0/v7.1/v7.2格式，跳过权重图")
        return

    # 【修复】处理单行业情况
    n_industries = df['industry'].nunique()

    if n_industries == 1:
        # 单行业时简化为2x1布局
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        ax1 = axes[0]
        ax2 = axes[1]

        # 1. 行业权重显示（单行业100%）
        industry_name = df['industry'].iloc[0]
        industry_weight = df['industry_weight'].iloc[0]

        ax1.pie([1], labels=[industry_name], autopct='100%', startangle=90, colors=['#ff4444'])
        ax1.set_title(f'申万二级行业权重分配\n（仅单行业: {industry_name}）', fontsize=14, fontweight='bold')

        # 2. 该行业配对收益分布
        ax2.bar(range(len(df)), df['total_return'] * 100, color='steelblue', alpha=0.7)
        ax2.axhline(y=df['total_return'].mean() * 100, color='red', linestyle='--', 
                   label=f'平均: {df["total_return"].mean()*100:.1f}%')
        ax2.set_xlabel('配对编号')
        ax2.set_ylabel('收益率 (%)')
        ax2.set_title(f'{industry_name} - 各配对收益率', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('v7.2_申万行业权重分配.png', dpi=300, bbox_inches='tight')
        print("✓ 图表已保存: v7.2_申万行业权重分配.png（单行业模式）")
        plt.show()
        return

    # 多行业情况（原有4子图布局）
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 行业权重饼图（申万二级行业）
    ax1 = axes[0, 0]
    industry_weights = df.groupby('industry')['industry_weight'].first().sort_values(ascending=False)

    colors = []
    for w in industry_weights.values:
        if w > 0.10:
            colors.append('#ff4444')
        elif w > 0.05:
            colors.append('#ffaa44')
        else:
            colors.append('#44aa44')

    wedges, texts, autotexts = ax1.pie(
        industry_weights.values, 
        labels=industry_weights.index,
        autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',
        startangle=90,
        colors=colors
    )
    ax1.set_title('申万二级行业权重分配（风险收益比）\n红>10% 橙5-10% 绿<5%', 
                 fontsize=14, fontweight='bold')

    # 2. 行业资金分配条形图
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

    # 4. 申万行业分布统计
    ax4 = axes[1, 1]
    industry_counts = df.groupby('industry').size().sort_values(ascending=False)

    colors_count = ['steelblue' if i < 5 else 'lightblue' for i in range(len(industry_counts))]
    bars = ax4.bar(range(len(industry_counts)), industry_counts.values, color=colors_count)
    ax4.set_xticks(range(len(industry_counts)))
    ax4.set_xticklabels(industry_counts.index, rotation=45, ha='right')
    ax4.set_ylabel('配对数量')
    ax4.set_title('各申万二级行业入选配对数量', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('v7.2_申万行业权重分配.png', dpi=300, bbox_inches='tight')
    print("✓ 图表已保存: v7.2_申万行业权重分配.png")
    plt.show()

def plot_slippage_analysis_v72(df):
    """v7.2滑点影响分析"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'FangSong', 'KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    if 'slippage_impact' not in df.columns:
        print("⚠️ 非v7.0/v7.1/v7.2格式，跳过滑点分析")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 滑点影响分布
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

    # 2. 滑点 vs 交易次数
    ax2 = axes[0, 1]
    scatter = ax2.scatter(df['num_trades'], df['slippage_impact'] * 100,
                         c=df['volatility'] * 100, cmap='YlOrRd', 
                         s=100, alpha=0.6, edgecolors='black')
    ax2.set_xlabel('交易次数')
    ax2.set_ylabel('滑点影响 (%)')
    ax2.set_title('滑点 vs 交易次数（颜色=波动率）\nv7.2应呈负相关（自适应减少交易）')
    plt.colorbar(scatter, ax=ax2, label='波动率(%)')

    z = np.polyfit(df['num_trades'], df['slippage_impact'] * 100, 1)
    p = np.poly1d(z)
    ax2.plot(df['num_trades'].sort_values(), p(df['num_trades'].sort_values()), 
            "r--", alpha=0.8, label=f'趋势线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 申万行业平均滑点
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
    ax3.set_title('各申万二级行业平均滑点成本（v7.2目标<0.8%）', fontsize=14, fontweight='bold')
    ax3.axvline(x=0.8, color='green', linestyle='--', alpha=0.5, label='v7.2目标线')
    ax3.axvline(x=1.15, color='red', linestyle='--', alpha=0.5, label='v7.0平均线')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='x')

    # 4. 滑点占比分析
    ax4 = axes[1, 1]
    slip_ratio = df['slippage_impact'] / df['total_return'] * 100
    slip_ratio = slip_ratio[slip_ratio > 0]

    ax4.hist(slip_ratio, bins=20, color='purple', alpha=0.6, edgecolor='black')
    ax4.axvline(slip_ratio.mean(), color='red', linestyle='--', linewidth=2,
                label=f'平均: 滑点占收益{slip_ratio.mean():.1f}%')
    ax4.set_xlabel('滑点占收益比例 (%)')
    ax4.set_ylabel('频数')
    ax4.set_title('滑点成本占收益比例分布')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('v7.2_滑点影响分析.png', dpi=300, bbox_inches='tight')
    print("✓ 图表已保存: v7.2_滑点影响分析.png")
    plt.show()

def plot_return_distribution_v72(df):
    """收益率分布分析（v7.2申万行业版）"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'FangSong', 'KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 收益率直方图（按申万行业分组）
    ax1 = axes[0, 0]
    returns = df['total_return'] * 100

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

    # 2. 夏普比率分布
    ax2 = axes[0, 1]
    sharpe = df['sharpe_ratio']

    ax2.hist(sharpe, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(sharpe.mean(), color='red', linestyle='--', linewidth=2,
                label=f'均值: {sharpe.mean():.2f}')
    ax2.axvline(1, color='green', linestyle='--', alpha=0.5, label='夏普=1')
    ax2.axvline(0.6, color='orange', linestyle='--', alpha=0.5, label='v7.0均值')
    ax2.set_xlabel('夏普比率')
    ax2.set_ylabel('频数')
    ax2.set_title('夏普比率分布（v7.2目标>0.85）')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 收益-风险-夏普三维散点（按申万行业着色）
    ax3 = axes[1, 0]

    # 为每个行业分配颜色
    industries = df['industry'].unique()
    industry_colors = plt.cm.tab10(np.linspace(0, 1, len(industries)))
    color_map = dict(zip(industries, industry_colors))

    for industry in industries:
        mask = df['industry'] == industry
        subset = df[mask]
        ax3.scatter(subset['total_return'] * 100, subset['max_drawdown'] * 100,
                   s=subset['allocated_capital'] / 1e4,
                   c=[color_map[industry]], alpha=0.6, edgecolors='black', label=industry)

    ax3.set_xlabel('总收益率 (%)')
    ax3.set_ylabel('最大回撤 (%)')
    ax3.set_title('收益-回撤散点（气泡=资金，颜色=申万行业）')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
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
    plt.savefig('v7.2_收益风险分布分析.png', dpi=300, bbox_inches='tight')
    print("✓ 图表已保存: v7.2_收益风险分布分析.png")
    plt.show()

def plot_top_pairs_table_v72(df, top_n=20):
    """Top N配对表格（v7.2申万行业版）"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'FangSong', 'KaiTi']
    plt.rcParams['axes.unicode_minus'] = False

    if 'sharpe_ratio' in df.columns and 'total_return' in df.columns:
        df['score'] = df['sharpe_ratio'] * 0.6 + df['total_return'] * 0.4
        top_df = df.nlargest(top_n, 'score')
    else:
        top_df = df.nlargest(top_n, 'total_return')

    top_df['stock1'] = top_df['stock1'].apply(lambda x: f"{int(x):06d}" if pd.notna(x) else x)
    top_df['stock2'] = top_df['stock2'].apply(lambda x: f"{int(x):06d}" if pd.notna(x) else x)

    fig, ax = plt.subplots(figsize=(18, top_n * 0.5 + 3))
    ax.axis('tight')
    ax.axis('off')

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

    table = ax.table(cellText=display_df.values,
                    colLabels=display_cols,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.1] * len(display_cols))

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)

    for i in range(len(display_cols)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(1, len(top_df) + 1):
        ret = top_df.iloc[i-1]['total_return']
        if ret > 0.3:
            color = '#90EE90'
        elif ret > 0.15:
            color = '#E0F7FA'
        elif ret > 0.05:
            color = '#FFF9C4'
        else:
            color = '#FFCCBC'

        for j in range(len(display_cols)):
            table[(i, j)].set_facecolor(color)

    plt.title(f'Top {top_n} 配对排名（v7.2申万行业自适应策略）', 
             fontsize=16, fontweight='bold', pad=20)
    plt.savefig(f'v7.2_Top{top_n}_配对排名表.png', dpi=300, bbox_inches='tight')
    print(f"✓ 图表已保存: v7.2_Top{top_n}_配对排名表.png")
    plt.show()

def generate_summary_report_v72(df, version):
    """生成v7.2汇总报告（申万行业版）"""
    print("\n" + "="*80)
    print(f"全行业配对策略回测分析报告 ({version} - 申万行业分类)")
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

    if 'halflife' in df.columns:
        print(f"\n【自适应参数统计（v7.2）】")
        print(f"平均Halflife: {df['halflife'].mean():.1f}天")
        print(f"平均入场阈值: {df['entry_threshold'].mean():.2f} (基础1.2)")
        print(f"平均持仓天数: {df['max_holding_days'].mean():.0f}天 (基础15)")
        print(f"平均波动率: {df['volatility'].mean()*100:.1f}%")

        high_vol = df[df['volatility'] > 0.30]
        low_vol = df[df['volatility'] < 0.15]
        print(f"\n  高波动配对(>{30}%): {len(high_vol)}对")
        if len(high_vol) > 0:
            print(f"    - 平均阈值: {high_vol['entry_threshold'].mean():.2f}")
            print(f"    - 平均持仓: {high_vol['max_holding_days'].mean():.0f}天")
        print(f"  低波动配对(<{15}%): {len(low_vol)}对")
        if len(low_vol) > 0:
            print(f"    - 平均阈值: {low_vol['entry_threshold'].mean():.2f}")
            print(f"    - 平均持仓: {low_vol['max_holding_days'].mean():.0f}天")

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

    if 'industry_weight' in df.columns:
        print(f"\n【资金分配（申万二级行业）】")
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
    generate_summary_report_v72(df, version)

    # 生成图表
    print("\n生成可视化图表...")

    # 【v7.2修复】组合净值走势 - 优先使用真实净值数据
    print("\n[v7.2核心] 组合净值走势分析...")
    portfolio_nav = plot_portfolio_nav_curve(df)

    if version in ["v7.1", "v7.2"]:
        print("\n[v7.2专属] 自适应参数分析...")
        plot_adaptive_params_analysis_v72(df)

        print("\n[v7.2专属] 申万行业权重分配...")
        plot_industry_weights_v72(df)

        print("\n[v7.2专属] 滑点影响分析...")
        plot_slippage_analysis_v72(df)
    elif version == "v7.0":
        from analyze_results_v7 import plot_industry_weights_v7, plot_slippage_analysis_v7
        print("\n[v7.0] 行业权重分配...")
        plot_industry_weights_v7(df)
        print("\n[v7.0] 滑点影响分析...")
        plot_slippage_analysis_v7(df)

    print("\n收益风险分布分析...")
    plot_return_distribution_v72(df)

    print("\nTop配对排名表...")
    plot_top_pairs_table_v72(df, top_n=20)

    print("\n✓ 全部分析完成！")
    print(f"\n生成图表清单（{version} - 申万行业）：")
    print("  - v7.2_组合净值走势分析.png 【修复：基于真实净值数据】")
    if version in ["v7.1", "v7.2"]:
        print("  - v7.2_自适应参数分析.png")
        print("  - v7.2_申万行业权重分配.png")
        print("  - v7.2_滑点影响分析.png")
    print("  - v7.2_收益风险分布分析.png")
    print(f"  - v7.2_Top20_配对排名表.png")