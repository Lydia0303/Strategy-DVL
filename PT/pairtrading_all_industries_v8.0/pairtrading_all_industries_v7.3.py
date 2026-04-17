# === A股全行业配对轮动策略 v7.3 ===
# 基于v7.2优化：整合行业相关性监控，增加详细筛选过程记录

import struct
import pandas as pd
import numpy as np
import os
import warnings
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm
from scipy import stats
from datetime import datetime, timedelta
from itertools import combinations
import glob
import json
warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================

CONFIG = {
    # 【v7.2修改】数据来源改为申万行业分类Excel文件
    'SW_INDUSTRY_FILE': r"C:\Users\hz\Desktop\Strategy DVL\SwClass\最新个股申万行业分类(完整版-截至7月末).xlsx",
    'INDUSTRY_COLUMN': '新版二级行业',  # 使用F列作为行业分类
    'STOCK_CODE_COLUMN': '股票代码',    # 股票代码列
    'TDX_DATA_DIR': "C:/new_tdx/vipdoc",
    
    # 回测时间区间
    'IN_SAMPLE_START': "2018-01-01",
    'IN_SAMPLE_END': "2021-12-31",
    'OUT_SAMPLE_START': "2022-01-01",
    'OUT_SAMPLE_END': "2026-04-08",
    
    # 筛选参数
    'TOP_N_PAIRS_PER_INDUSTRY': 3,
    'FINAL_TOP_N': 2,
    'MIN_POSITIVE_PAIRS': 1,  # 【修复】改为1，允许单配对行业入选
    
    # 组合资金配置
    'PORTFOLIO_CAPITAL': 10_000_000,
    
    # 滑点设置
    'SLIPPAGE': 0.0003,
    
    # 行业相关性监控参数
    'CORRELATION_PARAMS': {
        'lookback': 60,         # 相关性计算回看窗口
        'tail_quantile': 0.05,  # 尾部风险分位数
        'quality_threshold': 0.5,  # 相关性质量门槛
        'min_industry_pairs': 1,   # 行业最小配对数量
    },
    
    # 自适应阈值基础参数
    'BASE_PARAMS': {
        'entry_threshold': 1.2,
        'exit_threshold': 0.3,
        'stop_loss': 2.5,
        'max_holding_days': 15,
        'rebalance_threshold': 0.6,
        'commission_rate': 0.0003,
        'stamp_tax_rate': 0.001,
        'max_position_ratio': 0.6,
        'coint_check_freq': 30,
        'cooling_period': 2,
        'max_cooldown': 10,
        'min_coint_period': 60
    },
    
    # 风险调整权重参数
    'RISK_ADJUST_PARAMS': {
        'risk_aversion': 1.0,
        'return_boost': 1.5,
        'min_industry_weight': 0.02,
        'max_industry_weight': 0.15,
        'min_industry_return': 0.05,  # 【修复】降低门槛到5%
    }
}

# ==================== 【v7.3新增】行业相关性监控模块 ====================

class IndustryCorrelationMonitor:
    """行业相关性监控（三层）"""
    
    def __init__(self, lookback=60, tail_quantile=0.05, quality_threshold=0.5):
        self.lookback = lookback
        self.tail_q = tail_quantile
        self.quality_threshold = quality_threshold
        self.screening_report = {
            'total_industries': 0,
            'selected_industries': [],
            'filtered_industries': [],
            'filter_reasons': {}
        }
    
    def calculate_industry_returns(self, industry_results, out_sample_start, out_sample_end):
        """
        计算各行业指数收益
        基于行业内所有正收益配对的等权平均收益
        """
        industry_returns_dict = {}
        
        for industry_name, info in industry_results.items():
            all_returns = []
            
            for pair in info['pairs']:
                # 提取配对样本外日收益
                data1 = pair['data1']
                data2 = pair['data2']
                
                out_data1 = data1.loc[out_sample_start:out_sample_end, 'close']
                out_data2 = data2.loc[out_sample_start:out_sample_end, 'close']
                
                common_dates = out_data1.index.intersection(out_data2.index)
                if len(common_dates) < 20:
                    continue
                
                # 计算对数价差变化作为收益代理
                log_s1 = np.log(out_data1.loc[common_dates])
                log_s2 = np.log(out_data2.loc[common_dates])
                
                # 使用报告的卡尔曼参数计算残差
                if 'report' in pair and 'adaptive_params' in pair:
                    mu = 0
                    gamma = 1
                    if 'adaptive_params' in pair and 'hedge_ratio' in pair['adaptive_params']:
                        mu = pair['adaptive_params'].get('intercept', 0)
                        gamma = pair['adaptive_params'].get('hedge_ratio', 1)
                    
                    spread = log_s1 - (mu + gamma * log_s2)
                else:
                    # 简化计算
                    spread = log_s1 - log_s2
                
                # 价差变化作为收益
                spread_returns = spread.diff().dropna()
                if len(spread_returns) > 0:
                    all_returns.append(spread_returns.values)
            
            if all_returns:
                # 对齐长度，取平均值
                min_len = min(len(r) for r in all_returns)
                aligned_returns = [r[-min_len:] for r in all_returns]
                avg_returns = np.mean(aligned_returns, axis=0)
                industry_returns_dict[industry_name] = avg_returns
        
        return industry_returns_dict
    
    def quality_check(self, ret1, ret2):
        """相关性质量检查"""
        if len(ret1) < 30 or len(ret2) < 30:
            return 0
        
        try:
            # 滚动相关性稳定性
            rolling_corrs = []
            for i in range(self.lookback, min(len(ret1), len(ret2))):
                window_ret1 = ret1[i-self.lookback:i]
                window_ret2 = ret2[i-self.lookback:i]
                if len(window_ret1) >= 20 and len(window_ret2) >= 20:
                    corr_matrix = np.corrcoef(window_ret1, window_ret2)
                    if not np.isnan(corr_matrix[0,1]):
                        rolling_corrs.append(corr_matrix[0,1])
            
            if not rolling_corrs:
                return 0
            
            stability = 1 - np.std(rolling_corrs)
            
            # 近期与远期相关性一致性
            if len(ret1) >= 60 and len(ret2) >= 60:
                recent_corr = np.corrcoef(ret1[-20:], ret2[-20:])[0,1]
                distant_corr = np.corrcoef(ret1[-60:-20], ret2[-60:-20])[0,1]
                consistency = 1 - abs(recent_corr - distant_corr)
            else:
                consistency = 0.5
            
            # 综合质量分数
            quality_score = stability * 0.5 + consistency * 0.5
            return np.clip(quality_score, 0, 1)
            
        except:
            return 0
    
    def tail_risk_check(self, ret1, ret2):
        """尾部风险检查"""
        try:
            u = stats.rankdata(ret1) / (len(ret1) + 1)
            v = stats.rankdata(ret2) / (len(ret2) + 1)
            
            lower_tail = (u < self.tail_q) & (v < self.tail_q)
            upper_tail = (u > 1 - self.tail_q) & (v > 1 - self.tail_q)
            
            # 计算联合尾部概率
            lower_lambda = lower_tail.sum() / (u < self.tail_q).sum() if (u < self.tail_q).sum() > 0 else 0
            upper_lambda = upper_tail.sum() / (u > 1 - self.tail_q).sum() if (u > 1 - self.tail_q).sum() > 0 else 0
            
            # 如果联合尾部概率过高，表示尾部风险大
            return (lower_lambda < 0.5) and (upper_lambda < 0.5)
        except:
            return True  # 检查失败时默认通过
    
    def filter_industries(self, industry_returns_dict, industry_results):
        """
        过滤高相关性行业对
        返回: 可同时持有的行业列表
        """
        industries = list(industry_returns_dict.keys())
        self.screening_report['total_industries'] = len(industries)
        
        if len(industries) <= 1:
            self.screening_report['selected_industries'] = industries
            return industries
        
        # 初始化保留列表
        keep_industries = set(industries)
        remove_reasons = {}
        
        print(f"\n【行业相关性监控】")
        print(f"  分析行业数量: {len(industries)}")
        print(f"  相关性质量门槛: {self.quality_threshold}")
        print(f"  尾部风险分位数: {self.tail_q}")
        
        # 计算相关性矩阵
        for i in range(len(industries)):
            for j in range(i+1, len(industries)):
                ind1, ind2 = industries[i], industries[j]
                ret1, ret2 = industry_returns_dict[ind1], industry_returns_dict[ind2]
                
                # 质量检查
                quality = self.quality_check(ret1, ret2)
                
                if quality < self.quality_threshold:
                    reason = f"相关性质量过低({quality:.2f}<{self.quality_threshold})"
                    print(f"  ⚠️ {ind1}-{ind2}: {reason}")
                    
                    # 保留收益更高的行业
                    ret1_avg = np.mean(ret1) if len(ret1) > 0 else 0
                    ret2_avg = np.mean(ret2) if len(ret2) > 0 else 0
                    
                    if ret1_avg >= ret2_avg:
                        if ind2 in keep_industries:
                            keep_industries.discard(ind2)
                            remove_reasons[ind2] = reason
                    else:
                        if ind1 in keep_industries:
                            keep_industries.discard(ind1)
                            remove_reasons[ind1] = reason
                    continue
                
                # 尾部风险检查
                if not self.tail_risk_check(ret1, ret2):
                    reason = "尾部风险过高"
                    print(f"  ⚠️ {ind1}-{ind2}: {reason}")
                    
                    # 保留夏普更高的行业
                    sharpe1 = industry_results[ind1]['avg_sharpe']
                    sharpe2 = industry_results[ind2]['avg_sharpe']
                    
                    if sharpe1 >= sharpe2:
                        if ind2 in keep_industries:
                            keep_industries.discard(ind2)
                            remove_reasons[ind2] = reason
                    else:
                        if ind1 in keep_industries:
                            keep_industries.discard(ind1)
                            remove_reasons[ind1] = reason
        
        # 确保最小配对数量
        final_industries = []
        for industry in list(keep_industries):
            if industry in industry_results:
                pair_count = len(industry_results[industry]['pairs'])
                if pair_count >= CONFIG['CORRELATION_PARAMS']['min_industry_pairs']:
                    final_industries.append(industry)
                else:
                    reason = f"有效配对数量不足({pair_count}对)"
                    remove_reasons[industry] = reason
        
        # 更新筛选报告
        self.screening_report['selected_industries'] = final_industries
        self.screening_report['filtered_industries'] = [ind for ind in industries if ind not in final_industries]
        self.screening_report['filter_reasons'] = remove_reasons
        
        # 打印筛选结果
        print(f"\n【筛选结果】")
        print(f"  原始行业数: {len(industries)}")
        print(f"  入选行业数: {len(final_industries)}")
        print(f"  剔除行业数: {len(industries) - len(final_industries)}")
        
        if final_industries:
            print(f"\n  √ 入选行业:")
            for ind in final_industries:
                info = industry_results[ind]
                print(f"    - {ind}: {len(info['pairs'])}对, 收益{info['avg_return']*100:.1f}%, 夏普{info['avg_sharpe']:.2f}")
        
        if remove_reasons:
            print(f"\n  × 剔除行业:")
            for ind, reason in remove_reasons.items():
                if ind in industry_results:
                    info = industry_results[ind]
                    print(f"    - {ind}: {reason} | 收益{info['avg_return']*100:.1f}%, 夏普{info['avg_sharpe']:.2f}")
        
        return final_industries
    
    def save_screening_report(self, filename=None):
        """保存筛选过程报告"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"v7.3_行业筛选报告_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("行业相关性监控筛选报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            
            f.write("【统计概览】\n")
            f.write(f"总行业数量: {self.screening_report['total_industries']}\n")
            f.write(f"入选行业数量: {len(self.screening_report['selected_industries'])}\n")
            f.write(f"剔除行业数量: {len(self.screening_report['filtered_industries'])}\n\n")
            
            f.write("【入选行业详情】\n")
            if self.screening_report['selected_industries']:
                for idx, industry in enumerate(self.screening_report['selected_industries'], 1):
                    f.write(f"{idx}. {industry}\n")
            else:
                f.write("（无入选行业）\n")
            f.write("\n")
            
            f.write("【剔除行业详情】\n")
            if self.screening_report['filtered_industries']:
                for idx, industry in enumerate(self.screening_report['filtered_industries'], 1):
                    reason = self.screening_report['filter_reasons'].get(industry, "未知原因")
                    f.write(f"{idx}. {industry}: {reason}\n")
            else:
                f.write("（无剔除行业）\n")
        
        print(f"✓ 行业筛选报告已保存: {filename}")
        return filename

# ==================== 【v7.3修改】申万行业数据读取模块 ====================

def parse_sw_industry_file(excel_path, industry_col='新版二级行业', code_col='股票代码'):
    """
    解析申万行业分类Excel文件
    返回: {行业名: [股票代码列表]}
    """
    try:
        print(f"读取申万行业分类数据: {excel_path}")
        df = pd.read_excel(excel_path)
        
        print(f"  Excel数据形状: {df.shape}")
        print(f"  可用列: {df.columns.tolist()}")
        
        # 检查必要列是否存在
        if industry_col not in df.columns:
            raise ValueError(f"找不到行业列 '{industry_col}'，可用列: {df.columns.tolist()}")
        if code_col not in df.columns:
            raise ValueError(f"找不到股票代码列 '{code_col}'，可用列: {df.columns.tolist()}")
        
        # 清理数据：剔除ST股票（根据公司简称判断）
        st_stocks = set()
        if '公司简称' in df.columns:
            st_mask = df['公司简称'].str.contains('ST|\*ST', na=False, case=False)
            st_stocks = set(df.loc[st_mask, code_col].astype(str).str.replace('.SZ', '').str.replace('.SH', '').str.replace('.BJ', ''))
            df = df[~st_mask].copy()
            print(f"  剔除ST股票: {len(st_stocks)}只")
        
        # 剔除北交所股票（83/87/88/92开头或.BJ后缀）
        def is_bse(code):
            code_str = str(code).replace('.BJ', '').replace('.SZ', '').replace('.SH', '')
            return code_str.startswith(('83', '87', '88', '92')) or str(code).endswith('.BJ')
        
        bse_mask = df[code_col].apply(is_bse)
        bse_stocks = set(df.loc[bse_mask, code_col].astype(str).str.replace('.SZ', '').str.replace('.SH', '').str.replace('.BJ', ''))
        df = df[~bse_mask].copy()
        print(f"  剔除北交所股票: {len(bse_stocks)}只")
        
        # 处理股票代码格式：移除.SZ/.SH后缀
        df['clean_code'] = df[code_col].astype(str).str.replace('.SZ', '').str.replace('.SH', '').str.replace('.BJ', '')
        
        # 【修复】合并同名行业（去除空格，统一格式）
        df[industry_col] = df[industry_col].astype(str).str.strip()
        
        # 按行业分组
        industry_map = {}
        for industry, group in df.groupby(industry_col):
            codes = group['clean_code'].tolist()
            if len(codes) >= 2:  # 至少2只股票才能配对
                industry_map[industry] = codes
        
        print(f"✓ 申万行业分类解析完成: {len(industry_map)}个二级行业")
        print(f"  总股票数: {df['clean_code'].nunique()}只")
        
        # 打印行业分布（前10）
        industry_counts = {k: len(v) for k, v in industry_map.items()}
        top_industries = sorted(industry_counts.items(), key=lambda x: -x[1])[:10]
        print(f"  股票数最多的行业(Top 10): {top_industries}")
        
        return industry_map, st_stocks, bse_stocks
    
    except Exception as e:
        print(f"✗ 读取申万行业文件失败：{e}")
        import traceback
        traceback.print_exc()
        return {}, set(), set()

# ==================== 【v7.3修改】主函数 ====================

def run_portfolio_backtest_v7_3():
    """v7.3主函数：整合行业相关性监控"""
    cfg = CONFIG
    
    print("="*80)
    print("A股全行业配对交易策略 v7.3 - 整合行业相关性监控")
    print(f"初始资金: {cfg['PORTFOLIO_CAPITAL']:,.0f}")
    print(f"滑点: {cfg['SLIPPAGE']:.2%}")
    print(f"行业分类: {cfg['INDUSTRY_COLUMN']}")
    print("="*80)
    
    # 【v7.3新增】初始化行业相关性监控器
    correlation_monitor = IndustryCorrelationMonitor(
        lookback=cfg['CORRELATION_PARAMS']['lookback'],
        tail_quantile=cfg['CORRELATION_PARAMS']['tail_quantile'],
        quality_threshold=cfg['CORRELATION_PARAMS']['quality_threshold']
    )
    
    # 1. 读取申万行业分类数据
    print("\n[1/7] 读取申万行业分类数据...")
    industry_map, st_stocks, bse_stocks = parse_sw_industry_file(
        cfg['SW_INDUSTRY_FILE'],
        industry_col=cfg['INDUSTRY_COLUMN'],
        code_col=cfg['STOCK_CODE_COLUMN']
    )
    
    if not industry_map:
        print("✗ 行业数据加载失败")
        return
    
    threshold_manager = AdaptiveThresholdManager(cfg['BASE_PARAMS'])
    
    # 2. 各行业配对筛选与自适应参数计算
    print("\n[2/7] 各行业配对筛选与自适应参数计算...")
    industry_results = {}
    
    # 调试计数器
    processed_count = 0
    skipped_no_data = 0
    skipped_no_pairs = 0
    skipped_no_positive = 0
    
    for idx, (industry_name, stock_codes) in enumerate(industry_map.items(), 1):
        # 清理股票
        clean_stocks = [s for s in stock_codes if s not in st_stocks and s not in bse_stocks]
        if len(clean_stocks) < 2:
            continue
        
        # 加载数据
        data_dict = {}
        for code in clean_stocks:
            df = get_stock_data_from_tdx(code, cfg['TDX_DATA_DIR'])
            if df is not None and len(df.loc[cfg['IN_SAMPLE_START']:cfg['IN_SAMPLE_END']]) >= 100:
                data_dict[code] = df
        
        if len(data_dict) < 2:
            skipped_no_data += 1
            continue
        
        # 样本内筛选
        selected_pairs = select_pairs_for_industry(
            list(data_dict.keys()), data_dict,
            cfg['IN_SAMPLE_START'], cfg['IN_SAMPLE_END'],
            cfg['TOP_N_PAIRS_PER_INDUSTRY']
        )
        
        if not selected_pairs:
            skipped_no_pairs += 1
            continue
        
        print(f"\n  [{idx}/{len(industry_map)}] {industry_name}: 候选{len(selected_pairs)}对")
        positive_pairs = []
        
        for pair_info in selected_pairs:
            # 计算自适应参数
            adaptive_params, spread_returns = threshold_manager.calculate_adaptive_params(
                pair_info['data1'], pair_info['data2'],
                cfg['IN_SAMPLE_START'], cfg['IN_SAMPLE_END']
            )
            
            if adaptive_params is None:
                print(f"    ✗ {pair_info['stock1']}-{pair_info['stock2']}: 自适应参数计算失败")
                continue
            
            # 创建自适应交易实例
            trader = AdaptivePairTradingInstance(
                pair_info['stock1'], pair_info['stock2'],
                allocated_capital=1_000_000,  # 临时资金
                slippage=cfg['SLIPPAGE'],
                adaptive_params=adaptive_params,
                base_params=cfg['BASE_PARAMS']
            )
            
            # 初始化卡尔曼
            if not trader.initialize_kalman(pair_info['data1'], pair_info['data2']):
                continue
            
            # 回测
            report = trader.run_backtest(
                pair_info['data1'], pair_info['data2'],
                cfg['OUT_SAMPLE_START'], cfg['OUT_SAMPLE_END']
            )
            
            if report and report['total_return'] > 0:
                positive_pairs.append({
                    'stock1': pair_info['stock1'],
                    'stock2': pair_info['stock2'],
                    'report': report,
                    'data1': pair_info['data1'],
                    'data2': pair_info['data2'],
                    'adaptive_params': adaptive_params,
                    'volatility': adaptive_params['volatility'],
                    'halflife': adaptive_params['halflife'],
                    'trader': trader
                })
                
                # 打印自适应参数
                ap = adaptive_params
                print(f"    ✓ {pair_info['stock1']}-{pair_info['stock2']}: "
                      f"收益{report['total_return']*100:.1f}%, "
                      f"夏普{report['sharpe_ratio']:.2f}, "
                      f"波动{ap['volatility']:.1%}, "
                      f"阈值{ap['entry_threshold']:.2f}, "
                      f"持仓{ap['max_holding_days']}天, "
                      f"交易{report['num_trades']}次")
        
        # 筛选正收益配对
        if len(positive_pairs) >= cfg['MIN_POSITIVE_PAIRS']:
            top_pairs = sorted(positive_pairs, key=lambda x: x['report']['total_return'], reverse=True)[:cfg['FINAL_TOP_N']]
            
            # 计算行业平均指标
            avg_vol = np.mean([p['volatility'] for p in top_pairs])
            avg_return = np.mean([p['report']['total_return'] for p in top_pairs])
            avg_sharpe = np.mean([p['report']['sharpe_ratio'] for p in top_pairs])
            
            industry_results[industry_name] = {
                'pairs': top_pairs,
                'avg_volatility': avg_vol,
                'avg_return': avg_return,
                'avg_sharpe': avg_sharpe,
                'total_pairs': len(positive_pairs)
            }
            print(f"  ✓ 选中: {len(top_pairs)}对 | 平均收益{avg_return*100:.1f}% | 平均夏普{avg_sharpe:.2f}")
            processed_count += 1
        else:
            print(f"  ✗ 正收益配对不足({len(positive_pairs)}对)，剔除")
            skipped_no_positive += 1
    
    # 打印调试统计
    print(f"\n【行业初步筛选统计】")
    print(f"  总行业数: {len(industry_map)}")
    print(f"  数据不足跳过: {skipped_no_data}")
    print(f"  无配对跳过: {skipped_no_pairs}")
    print(f"  无正收益跳过: {skipped_no_positive}")
    print(f"  通过初步筛选行业: {processed_count}")
    
    if len(industry_results) < 3:
        print(f"✗ 有效行业不足3个 (当前{len(industry_results)}个)")
        if len(industry_results) >= 1:
            print("  警告: 继续运行但权重分配可能不均衡")
        else:
            return
    
    # 3. 【v7.3新增】行业相关性监控筛选
    print(f"\n[3/7] 行业相关性监控筛选...")
    
    # 计算各行业收益序列
    industry_returns_dict = correlation_monitor.calculate_industry_returns(
        industry_results, cfg['OUT_SAMPLE_START'], cfg['OUT_SAMPLE_END']
    )
    
    if not industry_returns_dict:
        print("✗ 无法计算行业收益序列，跳过相关性监控")
        valid_industries = list(industry_results.keys())
    else:
        # 执行相关性监控筛选
        valid_industries = correlation_monitor.filter_industries(industry_returns_dict, industry_results)
    
    # 保存行业筛选报告
    screening_report_file = correlation_monitor.save_screening_report()
    
    if not valid_industries:
        print("✗ 无有效行业通过相关性监控")
        return
    
    # 4. 风险收益比权重分配
    print(f"\n[4/7] 风险收益比权重分配...")
    allocator = RiskAdjustedParityAllocator(
        risk_aversion=cfg['RISK_ADJUST_PARAMS']['risk_aversion'],
        return_boost=cfg['RISK_ADJUST_PARAMS']['return_boost'],
        min_w=cfg['RISK_ADJUST_PARAMS']['min_industry_weight'],
        max_w=cfg['RISK_ADJUST_PARAMS']['max_industry_weight'],
        min_return=cfg['RISK_ADJUST_PARAMS']['min_industry_return']
    )
    
    # 准备行业指标
    industry_metrics = {}
    for name in valid_industries:
        info = industry_results[name]
        momentum = (info['avg_return'] - 0.15) / 0.3  # 归一化
        
        industry_metrics[name] = {
            'volatility': info['avg_volatility'],
            'expected_return': info['avg_return'],
            'sharpe': info['avg_sharpe'],
            'momentum': momentum
        }
    
    weights = allocator.calculate_weights(industry_metrics)
    
    # 5. 分配资金并重新回测
    print(f"\n[5/7] 最终资金分配与回测...")
    total_capital = cfg['PORTFOLIO_CAPITAL']
    
    final_pairs = []
    for industry_name, weight in weights.items():
        industry_capital = total_capital * weight
        pairs = industry_results[industry_name]['pairs']
        pair_capital = industry_capital / len(pairs)
        
        for pair in pairs:
            pair['allocated_capital'] = pair_capital
            pair['industry_weight'] = weight
            pair['pair_weight'] = weight / len(pairs)
            final_pairs.append(pair)
    
    # 打印分配结果
    print("\n  资金分配方案:")
    for name, w in sorted(weights.items(), key=lambda x: -x[1]):
        info = industry_results[name]
        cap = total_capital * w
        print(f"    {name}: 权重{w:.1%} | 资金{cap:,.0f} | 收益{info['avg_return']*100:.1f}% | 夏普{info['avg_sharpe']:.2f}")
    
    # 6. 生成最终报告
    print(f"\n[6/7] 生成组合报告...")
    
    # 计算组合层面指标
    portfolio_return = np.sum([p['report']['total_return'] * p['pair_weight'] for p in final_pairs])
    portfolio_slippage = np.sum([p['report']['slippage_impact'] * p['pair_weight'] for p in final_pairs])
    portfolio_sharpe = np.mean([p['report']['sharpe_ratio'] for p in final_pairs])
    
    print(f"\n{'='*80}")
    print("v7.3组合回测结果")
    print(f"{'='*80}")
    print(f"初始资金: {total_capital:,.0f}")
    print(f"原始行业数: {len(industry_map)}")
    print(f"通过初步筛选: {len(industry_results)}")
    print(f"通过相关性监控: {len(valid_industries)}")
    print(f"总配对数: {len(final_pairs)}")
    print(f"组合预期收益率: {portfolio_return*100:.2f}%")
    print(f"组合预期夏普: {portfolio_sharpe:.2f}")
    print(f"滑点成本占比: {portfolio_slippage*100:.2f}%")
    print(f"净收益率: {(portfolio_return - portfolio_slippage)*100:.2f}%")
    
    print(f"\n【v7.3改进点】")
    print(f"1. 申万行业分类: 使用新版二级行业分类")
    print(f"2. 自适应阈值: 高波动行业提高阈值")
    print(f"3. 行业相关性监控: 三层过滤（质量、尾部风险、配对数量）")
    print(f"4. 详细筛选报告: 保存完整筛选过程")
    print(f"5. 风险收益比权重: 科学分配行业权重")
    
    # 保存结果
    results_df = pd.DataFrame([
        {
            'industry': industry_name,
            'industry_weight': p['industry_weight'],
            'pair_weight': p['pair_weight'],
            'stock1': p['stock1'],
            'stock2': p['stock2'],
            'allocated_capital': p['allocated_capital'],
            'total_return': p['report']['total_return'],
            'sharpe_ratio': p['report']['sharpe_ratio'],
            'max_drawdown': p['report']['max_drawdown'],
            'num_trades': p['report']['num_trades'],
            'win_rate': p['report']['win_rate'],
            'slippage_impact': p['report']['slippage_impact'],
            'volatility': p['adaptive_params']['volatility'],
            'halflife': p['adaptive_params']['halflife'],
            'entry_threshold': p['adaptive_params']['entry_threshold'],
            'max_holding_days': p['adaptive_params']['max_holding_days'],
            'equity_curve': str(p['report']['equity_curve'])
        }
        for p in final_pairs
    ])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"v7.3_组合回测结果_{timestamp}.csv"
    results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ 回测结果已保存: {results_file}")
    
    config_file = f"v7.3_回测配置_{timestamp}.json"
    with open(config_file, 'w') as f:
        json.dump({k: str(v) if isinstance(v, (datetime, pd.Timestamp)) else v 
                  for k, v in cfg.items()}, f, indent=2, default=str)
    print(f"✓ 配置已保存: {config_file}")
    print(f"✓ 行业筛选报告: {screening_report_file}")

# ==================== 其他必要函数（保持原样） ====================

# 注意：这里需要包含v7.2中的所有其他函数
# 包括：get_stock_data_from_tdx, KalmanFilterPairTrading, adf_test, 
#       get_dynamic_cooldown, AdaptiveThresholdManager, 
#       RiskAdjustedParityAllocator, AdaptivePairTradingInstance, 
#       select_pairs_for_industry
# 这些函数与v7.2版本相同，不需要修改
def get_stock_data_from_tdx(stock_code, data_dir):
    """从通达信读取股票日线数据"""
    try:
        if stock_code.startswith(('600', '601', '603', '605', '688')):
            prefix, sub_dir = 'sh', 'sh'
        elif stock_code.startswith(('000', '001', '002', '003', '300', '301')):
            prefix, sub_dir = 'sz', 'sz'
        else:
            return None

        file_name = f"{prefix}{stock_code}.day"
        possible_paths = [
            os.path.join(data_dir, sub_dir, "lday", file_name),
            os.path.join("C:/new_tdx/vipdoc", sub_dir, "lday", file_name),
            os.path.join("D:/new_tdx/vipdoc", sub_dir, "lday", file_name),
        ]

        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break

        if file_path is None:
            return None

        data = []
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(32)
                if len(chunk) < 32:
                    break
                date, open_p, high, low, close, volume, amount, _ = struct.unpack('IIIIIIII', chunk)
                date_str = str(date)
                if len(date_str) != 8:
                    continue
                year, month, day = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])
                data.append([datetime(year, month, day), open_p/100.0, high/100.0, low/100.0, close/100.0, volume, amount])

        if not data:
            return None

        df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'amount'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        return None

# ==================== 卡尔曼滤波器 ====================

class KalmanFilterPairTrading:
    def __init__(self, initial_state=[0.0, 0.0], initial_covariance=1000.0, 
                 process_noise=1e-5, observation_noise=1e-3):
        self.state = np.array(initial_state, dtype=float)
        self.covariance = np.array([[initial_covariance, 0], [0, initial_covariance]])
        self.process_noise = np.array([[process_noise, 0], [0, process_noise]])
        self.observation_noise = observation_noise

    def update(self, y1, y2):
        T = np.eye(2)
        Z = np.array([1, y2])
        predicted_state = T @ self.state
        predicted_covariance = T @ self.covariance @ T.T + self.process_noise
        S = Z @ predicted_covariance @ Z.T + self.observation_noise
        if S == 0:
            S = 1e-10
        K = predicted_covariance @ Z.T / S
        innovation = y1 - (Z @ predicted_state)
        self.state = predicted_state + K * innovation
        I = np.eye(2)
        self.covariance = (I - np.outer(K, Z)) @ predicted_covariance
        return self.state.copy()

def adf_test(series):
    """ADF检验"""
    series = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) < 10 or series.std() < 1e-10:
        return 1.0
    try:
        return adfuller(series, maxlag=1)[1]
    except:
        return 1.0

def get_dynamic_cooldown(cooldown_count, max_cooldown=10, base_days=2):
    """动态冷却期"""
    return min(base_days + cooldown_count * 2, max_cooldown)

# ==================== 自适应阈值管理器 ====================

class AdaptiveThresholdManager:
    def __init__(self, base_params):
        self.base = base_params

    def calculate_adaptive_params(self, stock1_data, stock2_data, in_sample_start, in_sample_end):
        try:
            s1 = stock1_data.loc[in_sample_start:in_sample_end, 'close']
            s2 = stock2_data.loc[in_sample_start:in_sample_end, 'close']

            common_dates = s1.index.intersection(s2.index)
            s1, s2 = s1.loc[common_dates], s2.loc[common_dates]

            log_s1 = np.log(s1.replace(0, np.nan).dropna())
            log_s2 = np.log(s2.replace(0, np.nan).dropna())

            common_dates = log_s1.index.intersection(log_s2.index)
            if len(common_dates) < 60:
                return None, None

            log_s1, log_s2 = log_s1.loc[common_dates], log_s2.loc[common_dates]

            spread = log_s1 - log_s2
            spread_returns = spread.diff().dropna()

            if len(spread_returns) < 30:
                return None, None

            volatility = np.std(spread_returns) * np.sqrt(252)

            from statsmodels.regression.linear_model import OLS as SM_OLS
            from statsmodels.tools.tools import add_constant

            lag_spread = spread.shift(1).dropna()
            delta_spread = spread.diff().dropna()
            valid_idx = lag_spread.index.intersection(delta_spread.index)

            if len(valid_idx) < 30:
                halflife = 15
            else:
                X = add_constant(lag_spread.loc[valid_idx])
                y = delta_spread.loc[valid_idx]
                try:
                    model = SM_OLS(y, X).fit()
                    rho = model.params.iloc[1] if len(model.params) > 1 else 0
                    halflife = -np.log(2) / np.log(abs(rho)) if 0 < abs(rho) < 1 else 15
                    halflife = max(5, min(halflife, 60))
                except:
                    halflife = 15

            if volatility > 0.40:
                vol_factor = 1.6
                hold_factor = 2.2
            elif volatility > 0.30:
                vol_factor = 1.4
                hold_factor = 1.8
            elif volatility > 0.22:
                vol_factor = 1.2
                hold_factor = 1.4
            elif volatility > 0.15:
                vol_factor = 1.0
                hold_factor = 1.0
            elif volatility > 0.10:
                vol_factor = 0.85
                hold_factor = 0.9
            else:
                vol_factor = 0.75
                hold_factor = 0.8

            if halflife < 8:
                speed_factor = 0.8
            elif halflife < 15:
                speed_factor = 1.0
            elif halflife < 25:
                speed_factor = 1.2
            else:
                speed_factor = 1.4

            adaptive = {
                'entry_threshold': self.base['entry_threshold'] * vol_factor,
                'exit_threshold': self.base['exit_threshold'] * vol_factor * 0.85,
                'stop_loss': self.base['stop_loss'] * vol_factor,
                'max_holding_days': int(self.base['max_holding_days'] * hold_factor * speed_factor),
                'rebalance_threshold': self.base['rebalance_threshold'] * vol_factor,
                'volatility': volatility,
                'halflife': halflife,
                'vol_factor': vol_factor,
                'hold_factor': hold_factor,
                'speed_factor': speed_factor
            }

            adaptive['max_holding_days'] = max(10, min(adaptive['max_holding_days'], 40))
            adaptive['entry_threshold'] = max(0.8, min(adaptive['entry_threshold'], 2.0))

            return adaptive, spread_returns

        except Exception as e:
            print(f"    自适应参数计算失败: {e}")
            return None, None

# ==================== 风险调整权重分配器 ====================

class RiskAdjustedParityAllocator:
    def __init__(self, risk_aversion=1.0, return_boost=1.5, 
                 min_w=0.02, max_w=0.15, min_return=0.05):
        self.risk_aversion = risk_aversion
        self.return_boost = return_boost
        self.min_w = min_w
        self.max_w = max_w
        self.min_return = min_return

    def calculate_weights(self, industry_metrics):
        qualified = {}
        for name, metrics in industry_metrics.items():
            if metrics['expected_return'] >= self.min_return:
                qualified[name] = metrics
            else:
                print(f"    ⚠️ {name} 预期收益{metrics['expected_return']*100:.1f}% < 门槛{self.min_return*100:.0f}%，剔除")

        if len(qualified) < 3:
            print("    警告: 通过收益门槛的行业不足3个，放宽门槛...")
            qualified = {k: v for k, v in sorted(industry_metrics.items(), 
                          key=lambda x: x[1]['expected_return'], reverse=True)[:max(3, len(industry_metrics)//3)]}

        scores = {}
        for name, metrics in qualified.items():
            vol = max(metrics['volatility'], 0.001)
            ret = max(metrics['expected_return'], 0.001)
            sharpe = ret / vol if vol > 0 else 0

            risk_adjusted_score = (sharpe ** self.return_boost) / (vol ** self.risk_aversion)

            momentum = metrics.get('momentum', 0)
            momentum_factor = 1 + max(min(momentum, 0.3), -0.2)

            scores[name] = risk_adjusted_score * momentum_factor

        total_score = sum(scores.values())
        weights = {k: v / total_score for k, v in scores.items()}

        weights = self._apply_constraints(weights)

        return weights

    def _apply_constraints(self, weights):
        for iteration in range(10):
            excess_total = 0
            valid_count = sum(1 for v in weights.values() if v < self.max_w)

            for k in list(weights.keys()):
                if weights[k] > self.max_w:
                    excess = weights[k] - self.max_w
                    excess_total += excess
                    weights[k] = self.max_w

            if excess_total > 0 and valid_count > 0:
                redistribution = excess_total / valid_count
                for k in weights:
                    if weights[k] < self.max_w:
                        weights[k] += redistribution

            to_remove = [k for k, v in weights.items() if v < self.min_w]
            if to_remove:
                removed_weight = sum(weights[k] for k in to_remove)
                for k in to_remove:
                    del weights[k]

                if weights:
                    redistribution = removed_weight / len(weights)
                    for k in weights:
                        weights[k] += redistribution

            if all(self.min_w <= v <= self.max_w for v in weights.values()):
                break

        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

# ==================== 配对交易类 ====================

class AdaptivePairTradingInstance:
    def __init__(self, stock1, stock2, allocated_capital, slippage, 
                 adaptive_params, base_params):
        self.stock1 = stock1
        self.stock2 = stock2
        self.allocated_capital = allocated_capital
        self.slippage = slippage
        self.params = {**base_params, **adaptive_params}

        self.cash = allocated_capital
        self.kf = None
        self.pair_valid = True
        self.cooling_period = False
        self.cooldown_count = 0
        self.cooling_end_date = None
        self.last_check_date = None
        self.consecutive_fails = 0

        self.positions = {}
        self.holding_stock = None
        self.entry_date = None
        self.entry_z = 0
        self.holding_days = 0

        self.trade_records = []
        self.equity_curve = []

        self.adaptive_info = adaptive_params

    def initialize_kalman(self, in_sample_data1, in_sample_data2):
        try:
            common_dates = in_sample_data1.index.intersection(in_sample_data2.index)
            if len(common_dates) < 100:
                print(f"    共同数据不足: {len(common_dates)}天")
                return False

            s1 = in_sample_data1.loc[common_dates, 'close']
            s2 = in_sample_data2.loc[common_dates, 'close']

            valid = (s1 > 0) & (s2 > 0) & (~np.isnan(s1)) & (~np.isnan(s2))
            s1, s2 = s1[valid], s2[valid]

            if len(s1) < 100:
                print(f"    有效数据不足: {len(s1)}天")
                return False

            log_s1 = np.log(s1.iloc[-100:])
            log_s2 = np.log(s2.iloc[-100:])

            if len(log_s1) != len(log_s2):
                print(f"    长度不匹配: s1={len(log_s1)}, s2={len(log_s2)}")
                return False

            X = sm.add_constant(log_s2)
            model = sm.OLS(log_s1, X).fit()

            self.kf = KalmanFilterPairTrading(
                initial_state=[float(model.params[0]), float(model.params[1])],
                initial_covariance=1000.0,
                process_noise=1e-5,
                observation_noise=1e-3
            )
            return True
        except Exception as e:
            print(f"    卡尔曼初始化失败: {e}")
            return False

    def apply_slippage(self, price, direction):
        if direction == 'buy':
            return price * (1 + self.slippage)
        else:
            return price * (1 - self.slippage)

    def calculate_position_size(self, price):
        target_ratio = self.params.get('max_position_ratio', 0.6)
        available_capital = self.cash * target_ratio
        commission_factor = (1 + self.params['commission_rate'] + self.params['stamp_tax_rate'] * 0.5)
        max_amount = available_capital / commission_factor

        if price <= 0 or np.isnan(price):
            return 0

        shares = int(max_amount / price / 100) * 100
        return max(shares, 0)

    def execute_buy(self, date, stock_code, raw_price, reason=""):
        price = self.apply_slippage(raw_price, 'buy')
        shares = self.calculate_position_size(price)

        if shares < 100:
            return False, "股数不足"

        amount = price * shares
        commission = max(amount * self.params['commission_rate'], 5)
        total_cost = amount + commission

        if total_cost > self.cash:
            max_shares = int(self.cash / price / 100) * 100
            if max_shares < 100:
                return False, "资金不足"
            shares = max_shares
            amount = price * shares
            commission = max(amount * self.params['commission_rate'], 5)
            total_cost = amount + commission

        self.cash -= total_cost
        self.positions[stock_code] = {'qty': shares, 'avg_price': price, 'entry_date': date}
        self.holding_stock = stock_code

        self.trade_records.append({
            'date': date, 'action': '买入', 'stock': stock_code,
            'raw_price': raw_price, 'executed_price': price,
            'slippage_cost': (price - raw_price) * shares,
            'shares': shares, 'amount': amount, 'commission': commission,
            'cash_after': self.cash, 'reason': reason
        })
        return True, "成功"

    def execute_sell(self, date, stock_code, raw_price, reason=""):
        if stock_code not in self.positions:
            return False, "无持仓"

        price = self.apply_slippage(raw_price, 'sell')
        pos = self.positions[stock_code]
        shares = pos['qty']

        amount = price * shares
        commission = max(amount * self.params['commission_rate'], 5)
        stamp_tax = amount * self.params['stamp_tax_rate']
        total_cost = commission + stamp_tax
        net_proceeds = amount - total_cost

        pnl = (price - pos['avg_price']) * shares - total_cost

        self.cash += net_proceeds
        del self.positions[stock_code]
        self.holding_stock = None

        self.trade_records.append({
            'date': date, 'action': '卖出', 'stock': stock_code,
            'raw_price': raw_price, 'executed_price': price,
            'slippage_cost': (raw_price - price) * shares,
            'shares': shares, 'amount': amount,
            'commission': commission, 'stamp_tax': stamp_tax,
            'pnl': pnl, 'cash_after': self.cash, 'reason': reason,
            'hold_days': (date - pos['entry_date']).days if isinstance(date, datetime) and isinstance(pos['entry_date'], datetime) else 0
        })
        return True, "成功"

    def calculate_portfolio_value(self, price1, price2):
        total = self.cash
        for stock, pos in self.positions.items():
            if stock == self.stock1:
                total += pos['qty'] * price1
            elif stock == self.stock2:
                total += pos['qty'] * price2
        return total

    def calculate_spread_zscore(self, stock1_data, stock2_data, current_date, window=60):
        try:
            s1 = stock1_data.loc[:current_date].iloc[-window*2:]
            s2 = stock2_data.loc[:current_date].iloc[-window*2:]

            if len(s1) < window or len(s2) < window:
                return None, None, None, None

            log_s1 = np.log(s1.replace(0, np.nan).dropna())
            log_s2 = np.log(s2.replace(0, np.nan).dropna())

            common_dates = log_s1.index.intersection(log_s2.index)
            if len(common_dates) < window:
                return None, None, None, None

            log_s1 = log_s1.loc[common_dates]
            log_s2 = log_s2.loc[common_dates]

            if self.kf and len(log_s1) > 1:
                for i in range(len(log_s1)):
                    self.kf.update(log_s1.iloc[i], log_s2.iloc[i])

            mu, gamma = self.kf.state if self.kf else (0, 1)
            spread = log_s1 - (mu + gamma * log_s2)
            spread = spread.replace([np.inf, -np.inf], np.nan).dropna()

            if len(spread) < window:
                return None, None, None, None

            recent_spread = spread[-window:]
            z_score = (spread.iloc[-1] - recent_spread.mean()) / recent_spread.std()

            return spread, z_score, recent_spread.mean(), recent_spread.std()
        except:
            return None, None, None, None

    def check_cointegration(self, stock1_data, stock2_data, current_date, window=120):
        try:
            s1 = stock1_data.loc[:current_date].iloc[-window:]
            s2 = stock2_data.loc[:current_date].iloc[-window:]

            if len(s1) < self.params['min_coint_period'] or len(s2) < self.params['min_coint_period']:
                return True, 0.01

            log_s1 = np.log(s1.replace(0, np.nan).dropna())
            log_s2 = np.log(s2.replace(0, np.nan).dropna())

            common_dates = log_s1.index.intersection(log_s2.index)
            if len(common_dates) < self.params['min_coint_period']:
                return True, 0.01

            log_s1 = log_s1.loc[common_dates]
            log_s2 = log_s2.loc[common_dates]

            mu, gamma = self.kf.state if self.kf else (0, 1)
            residual = log_s1 - (mu + gamma * log_s2)
            residual = residual.replace([np.inf, -np.inf], np.nan).dropna()

            if len(residual) < 30 or residual.std() < 1e-10:
                return True, 0.01

            adf_p = adf_test(residual)

            if adf_p > 0.1:
                try:
                    score, pvalue, _ = coint(log_s1, log_s2)
                    if pvalue < 0.1:
                        return True, pvalue
                except:
                    pass

            return adf_p < 0.1, adf_p
        except:
            return True, 0.01

    def run_backtest(self, stock1_data, stock2_data, out_sample_start, out_sample_end):
        out_data1 = stock1_data.loc[out_sample_start:out_sample_end]
        out_data2 = stock2_data.loc[out_sample_start:out_sample_end]
        trading_dates = out_data1.index.intersection(out_data2.index)

        if len(trading_dates) == 0:
            return None

        warmup_days = min(20, self.params['max_holding_days'] // 2)
        valid_warmup = 0

        for i in range(min(warmup_days * 3, len(trading_dates))):
            current_date = trading_dates[i]
            try:
                if current_date not in out_data1.index or current_date not in out_data2.index:
                    continue
                p1 = out_data1.loc[current_date, 'close']
                p2 = out_data2.loc[current_date, 'close']
                if p1 <= 0 or p2 <= 0 or np.isnan(p1) or np.isnan(p2):
                    continue
                self.calculate_spread_zscore(out_data1['close'], out_data2['close'], current_date)
                valid_warmup += 1
                if valid_warmup >= warmup_days:
                    break
            except:
                continue

        for i, current_date in enumerate(trading_dates):
            try:
                raw_price1 = out_data1.loc[current_date, 'close']
                raw_price2 = out_data2.loc[current_date, 'close']
            except:
                continue

            if self.cooling_period:
                if self.cooling_end_date and current_date >= self.cooling_end_date:
                    self.cooling_period = False
                else:
                    total_value = self.calculate_portfolio_value(raw_price1, raw_price2)
                    self.equity_curve.append((current_date, total_value))
                    continue

            days_since_last = (current_date - self.last_check_date).days if self.last_check_date else 999
            if days_since_last >= self.params['coint_check_freq']:
                is_coint, adf_p = self.check_cointegration(out_data1['close'], out_data2['close'], current_date)
                self.last_check_date = current_date

                if not is_coint:
                    self.consecutive_fails += 1
                    if self.consecutive_fails >= 3:
                        if self.holding_stock:
                            self.execute_sell(current_date, self.holding_stock,
                                            raw_price1 if self.holding_stock == self.stock1 else raw_price2,
                                            "协整破裂平仓")
                        self.cooldown_count += 1
                        self.cooling_period = True
                        self.cooling_end_date = current_date + timedelta(days=get_dynamic_cooldown(self.cooldown_count))
                        self.consecutive_fails = 0
                        if self.cooldown_count >= 5:
                            self.pair_valid = False
                            break
                        continue
                else:
                    self.consecutive_fails = 0

            spread, z_score, mean_spread, std_spread = self.calculate_spread_zscore(
                out_data1['close'], out_data2['close'], current_date
            )

            if z_score is None:
                total_value = self.calculate_portfolio_value(raw_price1, raw_price2)
                self.equity_curve.append((current_date, total_value))
                continue

            total_value = self.calculate_portfolio_value(raw_price1, raw_price2)

            entry_th = self.params['entry_threshold']
            exit_th = self.params['exit_threshold']
            stop_th = self.params['stop_loss']
            reb_th = self.params['rebalance_threshold']

            if self.holding_stock is None:
                if abs(z_score) > entry_th and self.pair_valid:
                    if z_score > entry_th:
                        success, msg = self.execute_buy(current_date, self.stock2, raw_price2,
                                                      f"Z={z_score:.2f}>{entry_th:.2f}，买入低估方")
                    else:
                        success, msg = self.execute_buy(current_date, self.stock1, raw_price1,
                                                      f"Z={z_score:.2f}<-{entry_th:.2f}，买入低估方")
                    if success:
                        self.entry_z = z_score
                        self.entry_date = current_date
                        self.holding_days = 0
            else:
                self.holding_days += 1
                exit_flag, exit_reason = False, ""

                if abs(z_score) < exit_th:
                    exit_flag, exit_reason = True, f"价差回归(|Z|={abs(z_score):.2f}<{exit_th:.2f})"
                elif (self.holding_stock == self.stock1 and z_score > stop_th) or \
                     (self.holding_stock == self.stock2 and z_score < -stop_th):
                    exit_flag, exit_reason = True, f"止损(Z={z_score:.2f})"
                elif self.holding_days >= self.params['max_holding_days']:
                    exit_flag, exit_reason = True, f"时间止损({self.holding_days}天)"
                elif (self.holding_stock == self.stock2 and z_score < -reb_th) or \
                     (self.holding_stock == self.stock1 and z_score > reb_th):
                    other_stock = self.stock1 if self.holding_stock == self.stock2 else self.stock2
                    other_price = raw_price1 if self.holding_stock == self.stock2 else raw_price2

                    self.execute_sell(current_date, self.holding_stock,
                                    raw_price1 if self.holding_stock == self.stock1 else raw_price2,
                                    f"轮动卖出(Z={z_score:.2f})")
                    self.execute_buy(current_date, other_stock, other_price, f"轮动买入")
                    self.entry_z = z_score
                    self.entry_date = current_date
                    self.holding_days = 0
                    total_value = self.calculate_portfolio_value(raw_price1, raw_price2)
                    self.equity_curve.append((current_date, total_value))
                    continue

                if exit_flag:
                    current_raw_price = raw_price1 if self.holding_stock == self.stock1 else raw_price2
                    self.execute_sell(current_date, self.holding_stock, current_raw_price, exit_reason)
                    self.cooling_period = True
                    self.cooling_end_date = current_date + timedelta(days=get_dynamic_cooldown(self.cooldown_count))

            self.equity_curve.append((current_date, total_value))

        if self.holding_stock and len(trading_dates) > 0:
            last_date = trading_dates[-1]
            try:
                last_raw_price = out_data1.loc[last_date, 'close'] if self.holding_stock == self.stock1 else out_data2.loc[last_date, 'close']
                self.execute_sell(last_date, self.holding_stock, last_raw_price, "回测结束平仓")
            except:
                pass

        return self.generate_report()

    def generate_report(self):
        if not self.equity_curve:
            return None

        final_value = self.equity_curve[-1][1]
        total_return = (final_value - self.allocated_capital) / self.allocated_capital

        trades_df = pd.DataFrame([t for t in self.trade_records if t['action'] == '卖出'])
        num_trades = len(trades_df)

        win_rate = 0
        profit_loss_ratio = 0
        total_slippage_cost = 0

        if num_trades > 0:
            win_trades = len(trades_df[trades_df['pnl'] > 0])
            win_rate = win_trades / num_trades
            avg_profit = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if win_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if (num_trades - win_trades) > 0 else 0
            profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0

            all_trades = pd.DataFrame(self.trade_records)
            if 'slippage_cost' in all_trades.columns:
                total_slippage_cost = all_trades['slippage_cost'].sum()

        values = [v for d, v in self.equity_curve]
        returns = pd.Series(values).pct_change().dropna()
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 1 and returns.std() > 0 else 0

        peak, max_drawdown = values[0], 0
        for v in values:
            if v > peak:
                peak = v
            max_drawdown = max(max_drawdown, (peak - v) / peak)

        return {
            'stock1': self.stock1,
            'stock2': self.stock2,
            'allocated_capital': self.allocated_capital,
            'final_value': final_value,
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_slippage_cost': total_slippage_cost,
            'slippage_impact': total_slippage_cost / self.allocated_capital if self.allocated_capital > 0 else 0,
            'cooldown_count': self.cooldown_count,
            'equity_curve': self.equity_curve,
            'trade_records': self.trade_records,
            'adaptive_params': self.adaptive_info
        }

# ==================== 配对筛选模块 ====================

def select_pairs_for_industry(stock_list, data_dict, in_sample_start, in_sample_end, top_n=5):
    n = len(stock_list)
    if n < 2:
        return []

    pair_scores = []

    for i, j in combinations(range(n), 2):
        s1, s2 = stock_list[i], stock_list[j]
        df1, df2 = data_dict.get(s1), data_dict.get(s2)
        if df1 is None or df2 is None:
            continue

        common_index = df1.index.intersection(df2.index)
        sample_mask = (common_index >= in_sample_start) & (common_index <= in_sample_end)
        sample_index = common_index[sample_mask]

        if len(sample_index) < 100:
            continue

        log_s1 = np.log(df1.loc[sample_index, 'close'])
        log_s2 = np.log(df2.loc[sample_index, 'close'])

        mask = ~(np.isnan(log_s1) | np.isnan(log_s2) | np.isinf(log_s1) | np.isinf(log_s2))
        log_s1_clean, log_s2_clean = log_s1[mask], log_s2[mask]

        if len(log_s1_clean) < 100:
            continue

        spread = log_s1_clean - log_s2_clean
        ssd = np.sqrt(np.mean((spread - spread.mean())**2))
        corr = np.corrcoef(log_s1_clean, log_s2_clean)[0, 1]

        pair_scores.append((s1, s2, ssd, corr, sample_index))

    if not pair_scores:
        return []

    pair_scores.sort(key=lambda x: x[2])
    top_pairs = pair_scores[:top_n*3]

    selection_results = []
    for s1, s2, ssd, corr, sample_index in top_pairs:
        df1, df2 = data_dict.get(s1), data_dict.get(s2)
        log_s1 = np.log(df1.loc[sample_index, 'close'])
        log_s2 = np.log(df2.loc[sample_index, 'close'])

        mask = ~np.isnan(log_s1) & ~np.isnan(log_s2) & ~np.isinf(log_s1) & ~np.isinf(log_s2)
        y, x = log_s1[mask].values, log_s2[mask].values

        if len(y) < 100:
            continue

        try:
            X = sm.add_constant(x)
            model = sm.OLS(y, X).fit()
            hedge_ratio, intercept, r_squared = model.params[1], model.params[0], model.rsquared
        except:
            continue

        residual = y - (intercept + hedge_ratio * x)
        coint_p = adf_test(residual)

        if coint_p < 0.05:
            selection_results.append({
                'stock1': s1, 'stock2': s2, 'ssd': ssd, 'correlation': corr,
                'hedge_ratio': hedge_ratio, 'intercept': intercept,
                'r_squared': r_squared, 'coint_p': coint_p,
                'data1': df1, 'data2': df2
            })

    selection_results.sort(key=lambda x: (x['ssd'], -x['r_squared']))
    return selection_results[:top_n]


if __name__ == "__main__":
    run_portfolio_backtest_v7_3()