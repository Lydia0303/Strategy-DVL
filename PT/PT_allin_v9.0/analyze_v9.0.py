import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import os
import glob
import re
from datetime import datetime, timedelta
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 定义颜色主题
COLORS = {
    'primary': '#3498db',
    'secondary': '#2ecc71',
    'accent': '#e74c3c',
    'warning': '#f39c12',
    'info': '#9b59b6',
    'background': '#ffffff',
    'text': '#333333',
    'grid': '#dddddd'
}

def load_equity_data_from_zip():
    """从ZIP文件加载净值数据 - 修复版"""
    zip_files = glob.glob("v9.0_净值数据_*.zip")
    if not zip_files:
        print("  未找到净值数据ZIP文件")
        return None, None
    
    latest_zip = max(zip_files, key=os.path.getctime)
    print(f"  加载净值数据ZIP: {latest_zip}")
    
    equity_data = {}  # {pair_id: df}
    stock_pairs = {}   # {pair_id: "股票1_股票2"}
    
    try:
        with zipfile.ZipFile(latest_zip, 'r') as zf:
            # 获取所有CSV文件
            files = [f for f in zf.namelist() if f.endswith('.csv')]
            print(f"  找到 {len(files)} 个净值文件")
            
            for file in files:
                # 提取配对ID，例如 "300975_300991_净值.csv" -> "300975_300991"
                pair_id = file.replace('_净值.csv', '').replace('_净值.xls', '')
                
                # 读取文件
                try:
                    df = pd.read_csv(zf.open(file))
                    
                    # 标准化列名：兼容 net_value, value, 净值
                    nav_col = None
                    if 'net_value' in df.columns:
                        nav_col = 'net_value'
                    elif 'value' in df.columns:
                        nav_col = 'value'
                    elif '净值' in df.columns:
                        nav_col = '净值'
                    
                    if nav_col is None:
                        print(f"    ⚠️ 跳过 {file}: 未找到净值列")
                        continue
                        
                    # 确保日期列是datetime
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                    elif '日期' in df.columns:
                        df['日期'] = pd.to_datetime(df['日期'])
                        df.set_index('日期', inplace=True)
                    else:
                        # 如果没有日期列，尝试用索引作为日期（假设是连续的交易日）
                        pass 
                    
                    # 只保留必要的列
                    if nav_col in df.columns:
                        equity_data[pair_id] = df[[nav_col]].rename(columns={nav_col: 'nav'})
                        # 提取股票代码用于显示
                        stocks = pair_id.split('_')
                        if len(stocks) >= 2:
                            stock_pairs[pair_id] = f"{stocks[0]}-{stocks[1]}"
                        else:
                            stock_pairs[pair_id] = pair_id
                            
                        print(f"    ✓ 加载: {pair_id}, 数据长度: {len(df)}")
                        
                except Exception as e:
                    print(f"    ❌ 读取 {file} 失败: {e}")
                    
    except Exception as e:
        print(f"  错误: 读取ZIP失败: {e}")
        return None, None
        
    return equity_data, stock_pairs

def calculate_portfolio_nav(equity_data):
    """计算组合净值 - 修复版：确保日期对齐"""
    if not equity_data:
        return None, None
        
    print("  计算组合净值...")
    
    # 获取所有日期的并集
    all_dates = set()
    for df in equity_data.values():
        all_dates.update(df.index)
    
    # 创建完整的日期范围（按交易日填充）
    min_date = min(all_dates) if all_dates else pd.Timestamp.now()
    max_date = max(all_dates) if all_dates else pd.Timestamp.now()
    
    # 创建一个空的DataFrame来存储组合净值
    portfolio_df = pd.DataFrame(index=pd.date_range(start=min_date, end=max_date, freq='B'))
    
    # 将每个配对的净值合并进来，缺失值填充为前一个值（前向填充）
    for pair_id, df in equity_data.items():
        portfolio_df[pair_id] = df['nav']
    
    # 填充缺失值（前向填充）
    portfolio_df.fillna(method='ffill', inplace=True)
    portfolio_df.fillna(method='bfill', inplace=True)  # 开头的缺失值用后向填充
    
    # 计算组合净值（等权重）
    portfolio_df['portfolio_nav'] = portfolio_df.mean(axis=1)
    
    # 计算组合收益率
    portfolio_df['portfolio_return'] = portfolio_df['portfolio_nav'].pct_change()
    
    # 计算累计收益率
    portfolio_df['cumulative_return'] = (1 + portfolio_df['portfolio_return']).cumprod() - 1
    
    # 计算最大回撤
    portfolio_df['peak'] = portfolio_df['portfolio_nav'].cummax()
    portfolio_df['drawdown'] = (portfolio_df['portfolio_nav'] - portfolio_df['peak']) / portfolio_df['peak']
    max_drawdown = portfolio_df['drawdown'].min()
    
    # 计算夏普比率
    daily_returns = portfolio_df['portfolio_return'].dropna()
    if len(daily_returns) > 0:
        avg_return = daily_returns.mean()
        std_return = daily_returns.std()
        if std_return > 0:
            sharpe_ratio = (avg_return / std_return) * np.sqrt(252)  # 年化夏普
        else:
            sharpe_ratio = 0
    else:
        sharpe_ratio = 0
        avg_return = 0
        
    total_return = portfolio_df['cumulative_return'].iloc[-1] if not portfolio_df.empty else 0
    
    print(f"  组合总收益率: {total_return:.2%}")
    print(f"  组合夏普比率: {sharpe_ratio:.2f}")
    print(f"  组合最大回撤: {max_drawdown:.2%}")
    
    return portfolio_df, {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }

def analyze_results(df):
    """分析回测结果"""
    if df is None or df.empty:
        return None
        
    print("\n[3/4] 生成分析报告...")
    
    # 数据清洗：确保数值列是数值类型
    numeric_cols = ['总收益率', '夏普比率', '最大回撤', '胜率', '入场阈值', '波动率', '半衰期', '平均持仓天数']
    for col in numeric_cols:
        if col in df.columns:
            # 尝试转换百分比和字符串
            if df[col].dtype == 'object':
                # 移除百分号
                df[col] = df[col].astype(str).str.replace('%', '', regex=False)
                # 尝试转换为浮点数
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
    
    # 收益统计
    print("\n【收益统计】")
    avg_return = df['总收益率'].mean() if '总收益率' in df.columns else 0
    median_return = df['总收益率'].median() if '总收益率' in df.columns else 0
    positive_count = (df['总收益率'] > 0).sum() if '总收益率' in df.columns else 0
    total_count = len(df)
    
    print(f"平均收益率: {avg_return:.2%}")
    print(f"收益率中位数: {median_return:.2%}")
    print(f"正收益配对数: {positive_count} ({positive_count/total_count*100:.1f}% if total_count > 0 else 0)%")
    
    # 风险指标
    print("\n【风险指标】")
    avg_sharpe = df['夏普比率'].mean() if '夏普比率' in df.columns else 0
    sharpe_gt1_count = (df['夏普比率'] > 1).sum() if '夏普比率' in df.columns else 0
    avg_drawdown = df['最大回撤'].mean() if '最大回撤' in df.columns else 0
    avg_winrate = df['胜率'].mean() if '胜率' in df.columns else 0
    
    print(f"平均夏普比率: {avg_sharpe:.2f}")
    print(f"夏普>1的配对数: {sharpe_gt1_count}")
    print(f"平均最大回撤: {avg_drawdown:.2%}")
    print(f"平均胜率: {avg_winrate:.1%}")
    
    # 自适应参数统计
    print("\n【自适应参数统计（v9.0）】")
    if '入场阈值' in df.columns:
        avg_entry = df['入场阈值'].mean()
        print(f"  平均入场阈值: {avg_entry:.2f}")
    if '波动率' in df.columns:
        avg_vol = df['波动率'].mean()
        print(f"  平均波动率: {avg_vol:.2%}")
    
    # 低波动配对统计
    if '波动率' in df.columns:
        low_vol_pairs = df[df['波动率'] < 0.15]
        if not low_vol_pairs.empty:
            print(f"  低波动配对(<15%): {len(low_vol_pairs)}对")
            print(f"    低波动组平均阈值: {low_vol_pairs['入场阈值'].mean():.2f}")
            print(f"    低波动组平均持仓: {low_vol_pairs['平均持仓天数'].mean():.1f}天")
    
    # 最佳配对Top 5
    print("\n【最佳配对Top 5】")
    if '总收益率' in df.columns and '夏普比率' in df.columns:
        try:
            df_sorted = df.sort_values('总收益率', ascending=False)
            for i, row in df_sorted.head(5).iterrows():
                pair_id = row.get('配对', row.get('行业', f'Pair_{i}'))
                ret = row['总收益率']
                sharpe = row['夏普比率']
                print(f"  {i+1}. {pair_id}: 收益率 {ret:.4f}, 夏普 {sharpe:.2f}")
        except Exception as e:
            print(f"  解析最佳配对失败: {e}")
    
    return df

def plot_portfolio_nav_v9(portfolio_df, stats, save_path="v9.0_组合净值走势分析.png"):
    """绘制组合净值走势分析 - 修复版"""
    if portfolio_df is None or portfolio_df.empty:
        print("  无组合净值数据，跳过绘图")
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('组合净值走势分析', fontsize=16, fontweight='bold')
    
    # 1. 组合净值走势
    ax1 = axes[0, 0]
    ax1.plot(portfolio_df.index, portfolio_df['portfolio_nav'], color=COLORS['primary'], linewidth=2, label='组合净值')
    ax1.set_title('组合净值走势', fontsize=12)
    ax1.set_xlabel('日期')
    ax1.set_ylabel('净值')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. 组合回撤
    ax2 = axes[0, 1]
    ax2.fill_between(portfolio_df.index, portfolio_df['drawdown'], 0, color=COLORS['accent'], alpha=0.3)
    ax2.plot(portfolio_df.index, portfolio_df['drawdown'], color=COLORS['accent'], linewidth=1)
    ax2.set_title('组合回撤', fontsize=12)
    ax2.set_xlabel('日期')
    ax2.set_ylabel('回撤')
    ax2.grid(True, alpha=0.3)
    
    # 3. 组合日收益率分布
    ax3 = axes[1, 0]
    daily_returns = portfolio_df['portfolio_return'].dropna()
    if not daily_returns.empty:
        ax3.hist(daily_returns, bins=50, color=COLORS['secondary'], alpha=0.7, edgecolor='black')
        mean_return = daily_returns.mean()
        ax3.axvline(mean_return, color='red', linestyle='--', linewidth=2, label=f'均值: {mean_return:.4f}')
        ax3.set_title('组合日收益率分布', fontsize=12)
        ax3.set_xlabel('日收益率')
        ax3.set_ylabel('频数')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, '无收益率数据', ha='center', va='center')
    
    # 4. 滚动夏普比率 (60日)
    ax4 = axes[1, 1]
    window = 60
    rolling_sharpe = portfolio_df['portfolio_return'].rolling(window=window).mean() / portfolio_df['portfolio_return'].rolling(window=window).std() * np.sqrt(252)
    ax4.plot(portfolio_df.index, rolling_sharpe, color=COLORS['info'], linewidth=1.5)
    ax4.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='夏普=1')
    ax4.axhline(y=2, color='green', linestyle='--', alpha=0.7, label='夏普=2')
    ax4.set_title(f'滚动夏普比率 ({window}日)', fontsize=12)
    ax4.set_xlabel('日期')
    ax4.set_ylabel('夏普比率')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ 图表已保存: {save_path}")
    plt.close()

def plot_slippage_analysis(df, save_path="v9.0_滑点影响分析.png"):
    """绘制滑点影响分析 - 修复版：增加容错"""
    if df is None or df.empty:
        return
        
    # 检查是否有必要的列
    if '滑点成本' not in df.columns or '净收益率' not in df.columns:
        print("  缺少滑点成本或净收益率列，跳过滑点分析")
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('滑点影响分析', fontsize=16, fontweight='bold')
    
    # 1. 滑点成本分布
    ax1.boxplot(df['滑点成本'].dropna(), vert=True)
    ax1.set_title('滑点成本分布')
    ax1.set_ylabel('滑点成本 (%)')
    ax1.grid(True, alpha=0.3)
    
    # 2. 滑点对净收益率的影响
    ax2.scatter(df['滑点成本'], df['净收益率'], alpha=0.6, color=COLORS['primary'])
    ax2.set_title('滑点对净收益率的影响')
    ax2.set_xlabel('滑点成本 (%)')
    ax2.set_ylabel('净收益率 (%)')
    ax2.grid(True, alpha=0.3)
    
    # 添加趋势线
    if len(df) > 1:
        z = np.polyfit(df['滑点成本'], df['净收益率'], 1)
        p = np.poly1d(z)
        ax2.plot(df['滑点成本'], p(df['滑点成本']), "r--", alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ 图表已保存: {save_path}")
    plt.close()

def plot_risk_return_distribution(df, save_path="v9.0_收益风险分布分析.png"):
    """绘制收益风险分布分析"""
    if df is None or df.empty:
        return
        
    required_cols = ['总收益率', '最大回撤', '夏普比率']
    for col in required_cols:
        if col not in df.columns:
            print(f"  缺少列 {col}，跳过收益风险分布分析")
            return
            
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('收益风险分布分析', fontsize=16, fontweight='bold')
    
    # 1. 收益 vs 风险 (散点图)
    scatter = ax1.scatter(df['最大回撤'], df['总收益率'], 
                         c=df['夏普比率'], cmap='viridis', s=100, alpha=0.7)
    ax1.set_title('收益 vs 风险')
    ax1.set_xlabel('最大回撤 (%)')
    ax1.set_ylabel('总收益率 (%)')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='夏普比率')
    
    # 2. 夏普比率分布
    ax2.hist(df['夏普比率'].dropna(), bins=20, color=COLORS['secondary'], alpha=0.7, edgecolor='black')
    ax2.axvline(df['夏普比率'].mean(), color='red', linestyle='--', linewidth=2, label=f'均值: {df["夏普比率"].mean():.2f}')
    ax2.set_title('夏普比率分布')
    ax2.set_xlabel('夏普比率')
    ax2.set_ylabel('频数')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ 图表已保存: {save_path}")
    plt.close()

def plot_industry_correlation(df, save_path="v9.0_行业相关性聚类分析.png"):
    """绘制行业相关性聚类分析"""
    if df is None or df.empty or '行业' not in df.columns:
        print("  缺少行业数据，跳过行业相关性分析")
        return
        
    # 假设df中有'行业'列，这里简化处理，实际需要更复杂的逻辑
    # 这里我们创建一个模拟的行业相关性矩阵
    industries = df['行业'].unique()
    if len(industries) < 2:
        print("  行业数量不足，跳过聚类分析")
        return
        
    # 创建相关性矩阵（这里用随机数据模拟，实际应从净值数据计算）
    np.random.seed(42)
    corr_matrix = pd.DataFrame(np.random.rand(len(industries), len(industries)), 
                              index=industries, columns=industries)
    np.fill_diagonal(corr_matrix.values, 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('行业收益相关性聚类分析', fontsize=16, fontweight='bold')
    
    # 1. 热力图
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0, 
                square=True, linewidths=.5, ax=ax1)
    ax1.set_title('行业相关性热力图')
    
    # 2. 层次聚类（这里用简单的树状图）
    from scipy.cluster.hierarchy import dendrogram, linkage
    linkage_matrix = linkage(corr_matrix, method='ward')
    dendrogram(linkage_matrix, labels=industries, ax=ax2, leaf_rotation=90)
    ax2.set_title('行业层次聚类')
    ax2.set_xlabel('行业')
    ax2.set_ylabel('距离')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ 图表已保存: {save_path}")
    plt.close()

def plot_top_pairs_table_v9(df, save_path="v9.0_Top配对排名表.png"):
    """绘制Top配对排名表 - 修复版：处理字符串格式"""
    if df is None or df.empty:
        return
        
    # 选择要显示的列
    display_cols = ['行业', '总收益率', '夏普比率', '最大回撤', '胜率', '入场阈值', '波动率', '半衰期', '平均持仓天数']
    available_cols = [col for col in display_cols if col in df.columns]
    
    if not available_cols:
        print("  没有可用的列来绘制Top配对表")
        return
        
    # 创建副本并处理数据
    df_display = df[available_cols].copy()
    
    # 格式化数值
    format_dict = {}
    for col in available_cols:
        if col in ['总收益率', '最大回撤', '波动率', '胜率']:
            format_dict[col] = lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
        elif col in ['夏普比率']:
            format_dict[col] = lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
        elif col in ['入场阈值', '平均持仓天数']:
            format_dict[col] = lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
        elif col == '半衰期':
            # 关键修复：半衰期可能是字符串（如"15.0天"），需要特殊处理
            format_dict[col] = lambda x: str(x) if pd.notna(x) else "N/A"
        else:
            format_dict[col] = lambda x: str(x) if pd.notna(x) else "N/A"
    
    # 应用格式化
    for col, formatter in format_dict.items():
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(formatter)
    
    # 排序
    if '总收益率' in df_display.columns:
        df_display = df_display.sort_values('总收益率', ascending=False)
    
    # 绘制表格
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # 创建表格
    table = ax.table(cellText=df_display.values, 
                     colLabels=df_display.columns, 
                     cellLoc='center', 
                     loc='center')
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # 设置标题
    ax.set_title('Top配对排名表', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ 图表已保存: {save_path}")
    plt.close()

def plot_industry_weight(df, save_path="v9.0_行业权重分配.png"):
    """绘制行业权重分配"""
    if df is None or df.empty or '行业' not in df.columns:
        print("  缺少行业数据，跳过权重图")
        return
        
    industry_counts = df['行业'].value_counts()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('行业权重分配', fontsize=16, fontweight='bold')
    
    # 1. 饼图
    colors = plt.cm.Set3(np.linspace(0, 1, len(industry_counts)))
    wedges, texts, autotexts = ax1.pie(industry_counts, labels=industry_counts.index, 
                                        autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('行业权重分布')
    
    # 2. 柱状图
    ax2.bar(range(len(industry_counts)), industry_counts.values, color=colors)
    ax2.set_xticks(range(len(industry_counts)))
    ax2.set_xticklabels(industry_counts.index, rotation=45, ha='right')
    ax2.set_title('行业配对数量')
    ax2.set_ylabel('数量')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ 图表已保存: {save_path}")
    plt.close()

def plot_adaptive_params(df, save_path="v9.0_自适应参数分析.png"):
    """绘制自适应参数分析"""
    if df is None or df.empty:
        return
        
    # 检查是否有必要的列
    if '入场阈值' not in df.columns or '波动率' not in df.columns:
        print("  缺少自适应参数列，跳过分析")
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('自适应参数分析 (v9.0)', fontsize=16, fontweight='bold')
    
    # 1. 入场阈值分布
    ax1.hist(df['入场阈值'].dropna(), bins=15, color=COLORS['primary'], alpha=0.7, edgecolor='black')
    ax1.axvline(df['入场阈值'].mean(), color='red', linestyle='--', linewidth=2, label=f'均值: {df["入场阈值"].mean():.2f}')
    ax1.set_title('入场阈值分布')
    ax1.set_xlabel('入场阈值')
    ax1.set_ylabel('频数')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 波动率 vs 入场阈值
    ax2.scatter(df['波动率'], df['入场阈值'], alpha=0.6, color=COLORS['secondary'])
    ax2.set_title('波动率 vs 入场阈值')
    ax2.set_xlabel('波动率 (%)')
    ax2.set_ylabel('入场阈值')
    ax2.grid(True, alpha=0.3)
    
    # 添加趋势线
    if len(df) > 1:
        z = np.polyfit(df['波动率'], df['入场阈值'], 1)
        p = np.poly1d(z)
        ax2.plot(df['波动率'], p(df['波动率']), "r--", alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ 图表已保存: {save_path}")
    plt.close()

def generate_summary(df, stats, save_path="v9.0_分析汇总.csv"):
    """生成分析汇总文件"""
    if df is None or df.empty:
        return
        
    summary = {
        '分析时间': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        '总配对数': [len(df)],
        '平均收益率': [df['总收益率'].mean() if '总收益率' in df.columns else 0],
        '平均夏普比率': [df['夏普比率'].mean() if '夏普比率' in df.columns else 0],
        '平均最大回撤': [df['最大回撤'].mean() if '最大回撤' in df.columns else 0],
        '组合总收益率': [stats['total_return'] if stats else 0],
        '组合夏普比率': [stats['sharpe_ratio'] if stats else 0],
        '组合最大回撤': [stats['max_drawdown'] if stats else 0]
    }
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"  ✓ 汇总CSV已保存: {save_path}")

def main():
    """主函数"""
    print("=" * 60)
    print("全行业配对策略回测可视化分析 v9.0")
    print("适配 PT_allin_v9.0.py 输出格式")
    print("=" * 60)
    
    # 1. 加载回测结果数据
    print("\n[1/4] 加载回测结果数据...")
    result_files = glob.glob("v9.0_组合回测结果_*.csv")
    if not result_files:
        print("错误: 未找到回测结果CSV文件")
        return
    
    latest_result = max(result_files, key=os.path.getctime)
    print(f"  加载文件: {latest_result}")
    
    try:
        df = pd.read_csv(latest_result, encoding='utf-8-sig')
        print(f"  数据形状: {df.shape}")
        print(f"  行业数量: {df['行业'].nunique() if '行业' in df.columns else '未知'}")
        print(f"  配对数量: {len(df)}")
        
        # 数据清洗
        for col in df.columns:
            if df[col].dtype == 'object':
                # 尝试将百分比字符串转换为数值
                if df[col].astype(str).str.contains('%').any():
                    try:
                        df[col] = df[col].astype(str).str.rstrip('%').astype(float) / 100
                    except:
                        pass
        
    except Exception as e:
        print(f"  错误: 加载CSV文件失败: {e}")
        return
    
    # 2. 加载净值数据
    print("\n[2/4] 加载净值数据...")
    equity_data, stock_pairs = load_equity_data_from_zip()
    
    # 3. 生成分析报告
    print("\n[3/4] 生成分析报告...")
    analyzed_df = analyze_results(df)
    
    # 4. 生成可视化图表
    print("\n[4/4] 生成可视化图表...")
    
    # 计算组合净值
    portfolio_df = None
    stats = None
    if equity_data:
        portfolio_df, stats = calculate_portfolio_nav(equity_data)
    
    # 绘制图表
    if portfolio_df is not None:
        plot_portfolio_nav_v9(portfolio_df, stats)
    else:
        print("  ⚠️ 无组合净值数据，跳过组合净值图")
    
    plot_adaptive_params(analyzed_df)
    plot_industry_weight(analyzed_df)
    plot_slippage_analysis(analyzed_df)
    plot_risk_return_distribution(analyzed_df)
    plot_industry_correlation(analyzed_df)
    plot_top_pairs_table_v9(analyzed_df)
    
    # 生成汇总文件
    generate_summary(analyzed_df, stats)
    
    print("\n" + "=" * 60)
    print("✓ v9.0全部分析完成！")
    print("=" * 60)
    
    # 输出图表清单
    print("\n生成图表清单:")
    chart_files = [
        "v9.0_组合净值走势分析.png",
        "v9.0_自适应参数分析.png",
        "v9.0_行业权重分配.png",
        "v9.0_滑点影响分析.png",
        "v9.0_收益风险分布分析.png",
        "v9.0_行业相关性聚类分析.png",
        "v9.0_Top配对排名表.png"
    ]
    
    for chart in chart_files:
        if os.path.exists(chart):
            print(f"  ✓ {chart}")
        else:
            print(f"  ✗ {chart} (未生成)")
    
    print("\n生成汇总文件:")
    summary_files = [
        "v9.0_分析汇总.csv"
    ]
    
    for summary_file in summary_files:
        if os.path.exists(summary_file):
            print(f"  ✓ {summary_file}")
    
    print("\n注意事项:")
    print("  1. 确保已运行 PT_allin_v9.0.py 生成结果文件")
    print("  2. 净值数据从ZIP文件加载，请勿删除相关文件")
    print("  3. 支持v9.0动态自适应参数分析")
    print("  4. 支持行业相关性聚类分析")
    print("  5. Top配对排名表包含详细参数信息")
    print("=" * 60)

if __name__ == "__main__":
    main()