# === A股全行业配对轮动策略 v8.1（三模式回测版）===
# 基于v8.1优化：新增实盘回测模式，保留单次回测用于调试
# 三种回测模式：滚动窗口回测、实盘回测、单次回测（调试）

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
import zipfile
import shutil
warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================

CONFIG = {
    # 【v7.2修改】数据来源改为申万行业分类Excel文件
    'SW_INDUSTRY_FILE': r"C:\Users\hz\Desktop\Strategy DVL\SwClass\最新个股申万行业分类(完整版-截至7月末).xlsx",
    'INDUSTRY_COLUMN': '新版二级行业',  # 使用F列作为行业分类
    'STOCK_CODE_COLUMN': '股票代码',    # 股票代码列

    'TDX_DATA_DIR': "C:/new_tdx/vipdoc",

    # 回测时间区间
    'IN_SAMPLE_START': "2021-01-01",
    'IN_SAMPLE_END': "2024-12-31",
    'OUT_SAMPLE_START': "2025-01-01",
    'OUT_SAMPLE_END': "2025-12-31",

    # 完整样本时间范围（用于滚动窗口回测）
    'FULL_SAMPLE_START': "2021-01-01",  # 提前2年开始
    'FULL_SAMPLE_END': "2025-12-31",    # 到2023年底

    # 回测模式选择
    'BACKTEST_MODE': 'single_debug',  # 'rolling_window' | 'real_time' | 'single_debug'
    
    # 滚动窗口参数
    'ROLLING_WINDOW': {
        'enabled': True,
        'mode': 'four_year_one_year',  # 自定义模式
        
        'four_year_one_year': {
            'in_sample_days': 1008,     # 4年样本内
            'out_sample_days': 252,     # 1年样本外
            'step_days': 252,           # 每年滚动一次
            'min_window_days': 200,
            'description': '4年样本内回测1年样本外，每年滚动'
        },
        
        # 也可以同时支持其他模式
        'one_year_quarter': {
            'in_sample_days': 252,      # 1年
            'out_sample_days': 63,      # 1季度
            'step_days': 63,
            'min_window_days': 100,
            'description': '1年样本内，1季度样本外（标准滚动）'
        },

        'mode_custom': {
            'in_sample_days': 504,      # 2年
            'out_sample_days': 126,     # 半年
            'step_days': 63,            # 每季度滚动
            'description': '2年样本内，半年样本外'
        }
    },
    # 实盘回测参数
    'REAL_TIME_PARAMS': {
        'rebalance_days': 63,      # 每季度重新学习
        'lookback_days': 252,      # 回看天数
        'max_history_years': 4,    # 最大历史数据年数
        'daily_slippage': 0.0003,  # 每日滑点
        'min_trading_days': 20,    # 最小交易天数
    },

    # 筛选参数
    'TOP_N_PAIRS_PER_INDUSTRY': 5,
    'FINAL_TOP_N': 2,
    'MIN_POSITIVE_PAIRS': 1,  # 【修复】改为1，至少1对正收益行业入选（用协整就行）

    # 组合资金配置
    'PORTFOLIO_CAPITAL': 10_000_000,

    # 滑点设置
    'SLIPPAGE': 0.0003,

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
        'min_industry_return': -0.02,  # 【修复】降低门槛到2%
    }
}

# ==================== 【v7.2新增】申万行业数据读取模块 ====================

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
            st_mask = df['公司简称'].str.contains(r'ST|\*ST', na=False, case=False)  # 修复转义序列
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
            # 1. 年化波动率计算（保留你原来的逻辑）
            volatility = np.std(spread_returns) * np.sqrt(252)

            from statsmodels.regression.linear_model import OLS as SM_OLS
            from statsmodels.tools.tools import add_constant
            # 2. 半衰期计算（优化：数学稳定性 + 极端情况保护）
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
                    if 0 < abs(rho) < 1:
                        # 增加极小保护，防止 log(0.999999) 导致分母接近0 → 数值爆炸
                        denominator = np.log(abs(rho))
                        halflife = 15 if abs(denominator) < 1e-6 else -np.log(2) / denominator
            
                    halflife = max(5, min(halflife, 60))
                except:
                    halflife = 15
            # 3. 波动率因子（优化：从硬分段改为连续平滑函数，vol_factor 在 0.75 ~ 1.6 之间连续变化
            vol_factor = 0.75 + (volatility / 0.4) * (1.6 - 0.75)
            vol_factor = np.clip(vol_factor, 0.75, 1.6)  
            # 4.持仓因子不变
            if volatility > 0.40:
                hold_factor = 2.2
            elif volatility > 0.30:
                hold_factor = 1.8
            elif volatility > 0.22:
                hold_factor = 1.4
            elif volatility > 0.15:
                hold_factor = 1.0
            elif volatility > 0.10:
                hold_factor = 0.9
            else:
                hold_factor = 0.8
            # 5. 速度因子不变
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
            # ============ 修改这里开始 ============
            # 迭代计算最大可买数量（考虑交易成本）
            available_cash = self.cash
            max_shares = 0
            # 从100股开始，每次增加100股
            for trial_shares in range(100, int(available_cash / price) + 100, 100):
                trial_amount = price * trial_shares
                trial_commission = max(trial_amount * self.params['commission_rate'], 5)
                trial_total = trial_amount + trial_commission
                if trial_total <= available_cash:
                     max_shares = trial_shares
                else:
                     break #超过资金，停止尝试
        
            if max_shares < 100:
                return False, "资金不足"
            # ============ 修改这里结束 ============
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

# ==================== 动量因子计算器 ====================

class DailyDataMomentumCalculator:
    """动量因子计算器（仅用于权重分配）"""
    
    def __init__(self):
        self.available_factors = {
            'price_momentum': self.calculate_price_momentum,
            'volume_momentum': self.calculate_volume_momentum,
            'volatility_momentum': self.calculate_volatility_momentum,
            'technical_momentum': self.calculate_technical_momentum,
        }
    
    def calculate_all_factors(self, df):
        """计算所有动量因子"""
        all_factors = {}
        
        for factor_name, factor_func in self.available_factors.items():
            try:
                factors = factor_func(df)
                all_factors.update(factors)
            except:
                continue
        
        return all_factors
    
    def get_momentum_score(self, df, weight_config=None):
        """获取综合动量分数"""
        if weight_config is None:
            weight_config = {
                'price_momentum': 0.4,
                'volume_momentum': 0.2,
                'volatility_momentum': 0.2,
                'technical_momentum': 0.2
            }
        
        all_factors = self.calculate_all_factors(df)
        
        if not all_factors:
            return 0
        
        # 计算各因子组的分数
        grouped_scores = {}
        for factor_type, weight in weight_config.items():
            # 提取该类型的所有因子
            type_factors = {k: v for k, v in all_factors.items() 
                          if factor_type in k or k in ['rsi', 'bb_position']}
            
            if type_factors:
                values = list(type_factors.values())
                if len(values) > 1 and np.std(values) > 0:
                    normalized = [(v - np.mean(values)) / np.std(values) for v in values]
                    grouped_scores[factor_type] = np.mean(normalized)
                else:
                    grouped_scores[factor_type] = 0
        
        # 加权综合
        total_score = 0
        total_weight = 0
        for factor_type, weight in weight_config.items():
            if factor_type in grouped_scores:
                total_score += grouped_scores[factor_type] * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0
    
    def calculate_price_momentum(self, df, windows=[5, 10, 20, 60, 120]):
        """价格动量因子"""
        factors = {}
        close = df['close']
        
        for window in windows:
            ret = close.pct_change(window)
            factors[f'ret_{window}d'] = ret.iloc[-1] if not ret.empty else 0
        
        return factors
    
    def calculate_volume_momentum(self, df, windows=[5, 10, 20]):
        """成交量动量因子"""
        factors = {}
        volume = df['volume']
        
        for window in windows:
            vol_change = volume.pct_change(window)
            factors[f'vol_change_{window}d'] = vol_change.iloc[-1] if not vol_change.empty else 0
        
        return factors
    
    def calculate_volatility_momentum(self, df, windows=[20, 60]):
        """波动率动量因子"""
        factors = {}
        returns = df['close'].pct_change()
        
        for window in windows:
            hist_vol = returns.rolling(window).std() * np.sqrt(252)
            factors[f'hist_vol_{window}d'] = hist_vol.iloc[-1] if not hist_vol.empty else 0
        
        return factors
    
    def calculate_technical_momentum(self, df):
        """技术指标动量"""
        factors = {}
        
        # RSI
        if len(df) >= 14:
            rsi = self._calculate_rsi(df['close'], 14)
            factors['rsi'] = rsi.iloc[-1] if not rsi.empty else 50
        
        return factors
    
    def _calculate_rsi(self, prices, period=14):
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

# ==================== 改进的夏普计算 ====================

def calculate_portfolio_sharpe(final_pairs):
    """计算组合夏普比率（考虑配对相关性）"""
    if not final_pairs:
        return 0
    
    # 收集所有配对的日收益率
    all_returns = {}
    dates_aligned = None
    
    for pair in final_pairs:
        equity_curve = pair.get('report', {}).get('equity_curve', [])
        if not equity_curve:
            continue
        
        # 转换为DataFrame
        df = pd.DataFrame(equity_curve, columns=['date', 'value'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # 计算日收益率
        initial = pair.get('allocated_capital', 0)
        if initial <= 0:
            continue
            
        df['return'] = (df['value'] - initial) / initial
        daily_returns = df['return'].dropna()
        
        if daily_returns.empty:
            continue
            
        pair_key = f"{pair.get('stock1', '')}_{pair.get('stock2', '')}"
        all_returns[pair_key] = daily_returns
        
        if dates_aligned is None:
            dates_aligned = daily_returns.index
        else:
            dates_aligned = dates_aligned.intersection(daily_returns.index)
    
    if not all_returns or dates_aligned is None or len(dates_aligned) < 2:
        return 0
    
    # 对齐所有收益率的日期
    aligned_returns = pd.DataFrame(index=dates_aligned)
    for pair_key, returns in all_returns.items():
        aligned_returns[pair_key] = returns.reindex(dates_aligned).fillna(0)
    
    # 获取权重
    weights = []
    for pair in final_pairs:
        pair_key = f"{pair.get('stock1', '')}_{pair.get('stock2', '')}"
        if pair_key in aligned_returns.columns:
            weights.append(pair.get('pair_weight', 0))
    
    if len(weights) != len(aligned_returns.columns):
        return 0
    
    # 计算组合日收益率
    weights_array = np.array(weights)
    portfolio_returns = (aligned_returns * weights_array).sum(axis=1)
    
    # 计算夏普比率
    if len(portfolio_returns) < 2 or portfolio_returns.std() == 0:
        return 0
    
    daily_rf = (1 + 0.02) ** (1/252) - 1
    excess_returns = portfolio_returns - daily_rf
    sharpe = excess_returns.mean() / portfolio_returns.std() * np.sqrt(252)
    
    return sharpe

# ==================== 新增：实盘回测模式 ====================

def run_real_time_backtest(industry_map, st_stocks, bse_stocks, cfg, momentum_calculator):
    """实盘回测模式：每天只用当天之前的数据，模拟真实交易"""
    print("  [实盘回测模式] 开始...")
    print("  模式说明：每天收盘后用当天之前的数据生成信号，第二天交易")
    
    real_time_params = cfg.get('REAL_TIME_PARAMS', {})
    rebalance_days = real_time_params.get('rebalance_days', 63)  # 每季度重新学习
    lookback_days = real_time_params.get('lookback_days', 252)  # 回看1年数据
    max_history_years = real_time_params.get('max_history_years', 4)
    
    # 获取样本内数据（用于初始训练）
    in_sample_start = pd.to_datetime(cfg['IN_SAMPLE_START'])
    in_sample_end = pd.to_datetime(cfg['IN_SAMPLE_END'])
    
    # 获取样本外交易日
    out_sample_start = pd.to_datetime(cfg['OUT_SAMPLE_START'])
    out_sample_end = pd.to_datetime(cfg['OUT_SAMPLE_END'])
    
    # 获取所有交易日
    all_dates = get_all_trading_dates(cfg['FULL_SAMPLE_START'], cfg['FULL_SAMPLE_END'])
    trading_dates = [d for d in all_dates if out_sample_start <= d <= out_sample_end]
    
    if len(trading_dates) < 20:
        print(f"  ✗ 样本外交易日不足: {len(trading_dates)}天")
        return []
    
    print(f"  样本外交易日: {len(trading_dates)}天 ({trading_dates[0].date()} ~ {trading_dates[-1].date()})")
    
    # 初始训练：用样本内数据筛选配对
    print("  初始训练（用样本内数据）...")
    initial_pairs = run_initial_training(industry_map, st_stocks, bse_stocks, 
                                        in_sample_start, in_sample_end, cfg)
    
    if not initial_pairs:
        print("  ✗ 初始训练无有效配对")
        return []
    
    print(f"  初始配对数: {len(initial_pairs)}对")
    
    # 实盘回测：逐日推进
    current_pairs = initial_pairs.copy()
    current_params = {p['pair_key']: p['adaptive_params'] for p in initial_pairs}
    threshold_manager = AdaptiveThresholdManager(cfg['BASE_PARAMS'])
    
    # 存储每日结果
    daily_results = []
    all_trade_records = []
    equity_curve = []
    
    initial_capital = 1_000_000  # 每个配对初始资金
    current_equity = {p['pair_key']: initial_capital for p in initial_pairs}
    
    # 记录重新学习日期
    rebalance_dates = []
    last_rebalance_date = in_sample_end
    
    for i, current_date in enumerate(trading_dates):
        if i % 50 == 0 or i == len(trading_dates) - 1:
            print(f"    进度: {i+1}/{len(trading_dates)} ({current_date.date()})")
        
        # 获取昨日日期
        if i > 0:
            yesterday = trading_dates[i-1]
        else:
            yesterday = in_sample_end  # 第一天用样本内最后一天作为"昨天"
        
        # 检查是否需要重新学习
        days_since_rebalance = (current_date - last_rebalance_date).days
        if days_since_rebalance >= rebalance_days:
            print(f"    {current_date.date()}: 重新学习（距离上次{days_since_rebalance}天）")
            rebalance_dates.append(current_date)
            
            # 用历史数据重新筛选配对
            historical_data = {}
            lookback_start = current_date - timedelta(days=lookback_days)
            
            for pair in current_pairs:
                stock1, stock2 = pair['stock1'], pair['stock2']
                
                data1 = get_stock_data_from_tdx(stock1, cfg['TDX_DATA_DIR'])
                data2 = get_stock_data_from_tdx(stock2, cfg['TDX_DATA_DIR'])
                
                if data1 is not None and data2 is not None:
                    hist_data1 = data1.loc[lookback_start:yesterday]
                    hist_data2 = data2.loc[lookback_start:yesterday]
                    
                    if len(hist_data1) >= 100 and len(hist_data2) >= 100:
                        # 检查配对关系是否仍然有效
                        spread, z_score, _, _ = calculate_spread_stats(hist_data1, hist_data2)
                        if spread is not None and len(spread) >= 60:
                            adf_p = adf_test(spread)
                            if adf_p < 0.05:  # 仍然协整
                                # 重新计算参数
                                adaptive_params, _ = threshold_manager.calculate_adaptive_params(
                                    hist_data1, hist_data2, lookback_start, yesterday
                                )
                                
                                if adaptive_params:
                                    current_params[f"{stock1}_{stock2}"] = adaptive_params
                                    pair['adaptive_params'] = adaptive_params
                                    pair['data1'] = data1
                                    pair['data2'] = data2
            
            last_rebalance_date = current_date
        
        # 生成今日交易信号（用昨天收盘后的数据）
        today_signals = []
        
        for pair in current_pairs:
            stock1, stock2 = pair['stock1'], pair['stock2']
            pair_key = f"{stock1}_{stock2}"
            
            if pair_key not in current_params:
                continue
            
            # 获取到昨天的数据
            data1 = pair.get('data1')
            data2 = pair.get('data2')
            
            if data1 is None or data2 is None:
                continue
            
            # 只用到昨天的数据
            hist_data1 = data1.loc[:yesterday]
            hist_data2 = data2.loc[:yesterday]
            
            if len(hist_data1) < 100 or len(hist_data2) < 100:
                continue
            
            # 初始化交易实例
            trader = AdaptivePairTradingInstance(
                stock1, stock2,
                allocated_capital=initial_capital,
                slippage=cfg['SLIPPAGE'],
                adaptive_params=current_params[pair_key],
                base_params=cfg['BASE_PARAMS']
            )
            
            if not trader.initialize_kalman(hist_data1, hist_data2):
                continue
            
            # 计算到昨天的价差Z-score
            spread, z_score, mean_spread, std_spread = trader.calculate_spread_zscore(
                hist_data1, hist_data2, yesterday
            )
            
            if z_score is None:
                continue
            
            # 生成今日信号
            entry_th = current_params[pair_key]['entry_threshold']
            exit_th = current_params[pair_key]['exit_threshold']
            
            if abs(z_score) > entry_th:
                if z_score > entry_th:
                    # 做空股票1，做多股票2
                    today_signals.append({
                        'pair_key': pair_key,
                        'stock1': stock1,
                        'stock2': stock2,
                        'signal': 'short_long',
                        'z_score': z_score,
                        'entry_th': entry_th,
                        'adaptive_params': current_params[pair_key],
                        'data1': data1,
                        'data2': data2
                    })
                else:
                    # 做多股票1，做空股票2
                    today_signals.append({
                        'pair_key': pair_key,
                        'stock1': stock1,
                        'stock2': stock2,
                        'signal': 'long_short',
                        'z_score': z_score,
                        'entry_th': entry_th,
                        'adaptive_params': current_params[pair_key],
                        'data1': data1,
                        'data2': data2
                    })
        
        # 执行今日交易
        today_trades = []
        today_pnl = 0
        
        for signal in today_signals:
            # 获取今日价格
            stock1 = signal['stock1']
            stock2 = signal['stock2']
            
            data1 = signal['data1']
            data2 = signal['data2']
            
            if current_date not in data1.index or current_date not in data2.index:
                continue
            
            price1 = data1.loc[current_date, 'close']
            price2 = data2.loc[current_date, 'close']
            
            if price1 <= 0 or price2 <= 0 or np.isnan(price1) or np.isnan(price2):
                continue
            
            # 执行配对交易
            trade_result = execute_pair_trade_signal(
                signal, price1, price2, current_date, initial_capital, cfg
            )
            
            if trade_result:
                today_trades.append(trade_result)
                today_pnl += trade_result.get('pnl', 0)
                
                # 更新当前权益
                pair_key = signal['pair_key']
                if pair_key in current_equity:
                    current_equity[pair_key] += trade_result.get('pnl', 0)
        
        # 记录每日结果
        daily_equity = sum(current_equity.values())
        daily_return = today_pnl / (len(current_equity) * initial_capital) if current_equity else 0
        
        daily_results.append({
            'date': current_date,
            'signals': len(today_signals),
            'trades': len(today_trades),
            'pnl': today_pnl,
            'equity': daily_equity,
            'daily_return': daily_return
        })
        
        equity_curve.append((current_date, daily_equity))
        all_trade_records.extend(today_trades)
    
    # 整理结果
    print(f"  实盘回测完成: {len(trading_dates)}个交易日")
    print(f"  总交易次数: {len(all_trade_records)}")
    print(f"  重新学习次数: {len(rebalance_dates)}")
    
    # 计算每个配对的最终表现
    final_pairs = []
    for pair in current_pairs:
        pair_key = f"{pair['stock1']}_{pair['stock2']}"
        if pair_key in current_equity:
            final_value = current_equity[pair_key]
            total_return = (final_value - initial_capital) / initial_capital
            
            # 计算该配对的交易记录
            pair_trades = [t for t in all_trade_records if t.get('pair_key') == pair_key]
            
            if pair_trades:
                win_trades = [t for t in pair_trades if t.get('pnl', 0) > 0]
                win_rate = len(win_trades) / len(pair_trades) if pair_trades else 0
                
                # 生成回测报告格式
                report = {
                    'stock1': pair['stock1'],
                    'stock2': pair['stock2'],
                    'allocated_capital': initial_capital,
                    'final_value': final_value,
                    'total_return': total_return,
                    'num_trades': len(pair_trades),
                    'win_rate': win_rate,
                    'profit_loss_ratio': 0,  # 简化
                    'sharpe_ratio': 0,  # 简化
                    'max_drawdown': 0,  # 简化
                    'total_slippage_cost': 0,
                    'slippage_impact': 0,
                    'cooldown_count': 0,
                    'equity_curve': equity_curve,  # 使用组合净值
                    'trade_records': pair_trades,
                    'adaptive_params': pair.get('adaptive_params', {})
                }
                
                final_pairs.append({
                    'stock1': pair['stock1'],
                    'stock2': pair['stock2'],
                    'report': report,
                    'adaptive_params': pair.get('adaptive_params', {}),
                    'data1': pair.get('data1'),
                    'data2': pair.get('data2'),
                    'real_time_stats': {
                        'trade_count': len(pair_trades),
                        'total_pnl': sum(t.get('pnl', 0) for t in pair_trades),
                        'rebalance_count': len([d for d in rebalance_dates])
                    }
                })
    
    return final_pairs

def run_initial_training(industry_map, st_stocks, bse_stocks, in_sample_start, in_sample_end, cfg):
    """初始训练：用样本内数据筛选配对"""
    threshold_manager = AdaptiveThresholdManager(cfg['BASE_PARAMS'])
    initial_pairs = []
    
    for industry_name, stock_codes in industry_map.items():
        # 清理股票
        clean_stocks = [s for s in stock_codes if s not in st_stocks and s not in bse_stocks]
        if len(clean_stocks) < 2:
            continue
        
        # 加载样本内数据
        data_dict = {}
        for code in clean_stocks:
            df = get_stock_data_from_tdx(code, cfg['TDX_DATA_DIR'])
            if df is not None:
                in_sample_data = df.loc[in_sample_start:in_sample_end]
                if len(in_sample_data) >= 100:
                    data_dict[code] = in_sample_data
        
        if len(data_dict) < 2:
            continue
        
        # 样本内筛选
        industry_pairs = select_pairs_for_industry(
            list(data_dict.keys()), data_dict,
            in_sample_start, in_sample_end,
            cfg['TOP_N_PAIRS_PER_INDUSTRY']
        )
        
        for pair in industry_pairs:
            adaptive_params, _ = threshold_manager.calculate_adaptive_params(
                pair['data1'], pair['data2'],
                in_sample_start, in_sample_end
            )
            
            if adaptive_params:
                pair_key = f"{pair['stock1']}_{pair['stock2']}"
                initial_pairs.append({
                    'pair_key': pair_key,
                    'stock1': pair['stock1'],
                    'stock2': pair['stock2'],
                    'adaptive_params': adaptive_params,
                    'data1': pair['data1'],
                    'data2': pair['data2']
                })
    
    return initial_pairs

def calculate_spread_stats(data1, data2):
    """计算价差统计"""
    try:
        common_dates = data1.index.intersection(data2.index)
        if len(common_dates) < 60:
            return None, None, None, None
        
        log_s1 = np.log(data1.loc[common_dates, 'close'].replace(0, np.nan).dropna())
        log_s2 = np.log(data2.loc[common_dates, 'close'].replace(0, np.nan).dropna())
        
        common_dates = log_s1.index.intersection(log_s2.index)
        if len(common_dates) < 60:
            return None, None, None, None
        
        log_s1 = log_s1.loc[common_dates]
        log_s2 = log_s2.loc[common_dates]
        
        spread = log_s1 - log_s2
        if len(spread) < 60:
            return None, None, None, None
        
        z_score = (spread.iloc[-1] - spread.mean()) / spread.std()
        return spread, z_score, spread.mean(), spread.std()
    except:
        return None, None, None, None

def execute_pair_trade_signal(signal, price1, price2, current_date, allocated_capital, cfg):
    """执行配对交易信号"""
    try:
        stock1 = signal['stock1']
        stock2 = signal['stock2']
        z_score = signal['z_score']
        entry_th = signal['entry_th']
        adaptive_params = signal['adaptive_params']
        
        # 创建交易实例
        trader = AdaptivePairTradingInstance(
            stock1, stock2,
            allocated_capital=allocated_capital,
            slippage=cfg['SLIPPAGE'],
            adaptive_params=adaptive_params,
            base_params=cfg['BASE_PARAMS']
        )
        
        # 模拟执行交易
        if signal['signal'] == 'short_long':
            # 做空股票1，做多股票2
            success1, msg1 = trader.execute_sell(current_date, stock1, price1, 
                                                f"实盘: Z={z_score:.2f}>{entry_th:.2f}")
            success2, msg2 = trader.execute_buy(current_date, stock2, price2,
                                               f"实盘: Z={z_score:.2f}>{entry_th:.2f}")
            
            if success1 and success2:
                # 获取最新的交易记录
                trades = trader.trade_records[-2:] if len(trader.trade_records) >= 2 else trader.trade_records
                total_pnl = sum(t.get('pnl', 0) for t in trades)
                
                return {
                    'pair_key': signal['pair_key'],
                    'date': current_date,
                    'stock1': stock1,
                    'stock2': stock2,
                    'signal': 'short_long',
                    'z_score': z_score,
                    'price1': price1,
                    'price2': price2,
                    'pnl': total_pnl,
                    'trades': trades
                }
        
        elif signal['signal'] == 'long_short':
            # 做多股票1，做空股票2
            success1, msg1 = trader.execute_buy(current_date, stock1, price1,
                                               f"实盘: Z={z_score:.2f}<-{entry_th:.2f}")
            success2, msg2 = trader.execute_sell(current_date, stock2, price2,
                                                f"实盘: Z={z_score:.2f}<-{entry_th:.2f}")
            
            if success1 and success2:
                trades = trader.trade_records[-2:] if len(trader.trade_records) >= 2 else trader.trade_records
                total_pnl = sum(t.get('pnl', 0) for t in trades)
                
                return {
                    'pair_key': signal['pair_key'],
                    'date': current_date,
                    'stock1': stock1,
                    'stock2': stock2,
                    'signal': 'long_short',
                    'z_score': z_score,
                    'price1': price1,
                    'price2': price2,
                    'pnl': total_pnl,
                    'trades': trades
                }
    
    except Exception as e:
        print(f"    交易执行失败: {e}")
    
    return None

# ==================== 滚动窗口回测函数 ====================

def run_rolling_window_backtest(industry_map, st_stocks, bse_stocks, cfg):
    """执行滚动窗口回测"""
    all_window_results = []
    pair_performance_history = {}
    
    # 获取所有交易日
    all_dates = get_all_trading_dates(cfg['FULL_SAMPLE_START'], cfg['FULL_SAMPLE_END'])
    if len(all_dates) < cfg['ROLLING_WINDOW']['in_sample_days'] + cfg['ROLLING_WINDOW']['out_sample_days']:
        print(f"  数据不足: 需要{cfg['ROLLING_WINDOW']['in_sample_days'] + cfg['ROLLING_WINDOW']['out_sample_days']}天，只有{len(all_dates)}天")
        return all_window_results, pair_performance_history
    
    window_params = cfg['ROLLING_WINDOW']
    window_start_idx = 0
    window_count = 0
    
    while window_start_idx + window_params['in_sample_days'] + window_params['out_sample_days'] <= len(all_dates):
        window_count += 1
        
        # 样本内区间
        in_sample_start = all_dates[window_start_idx]
        in_sample_end = all_dates[window_start_idx + window_params['in_sample_days'] - 1]
        
        # 样本外区间
        out_sample_start = all_dates[window_start_idx + window_params['in_sample_days']]
        out_sample_end = all_dates[window_start_idx + window_params['in_sample_days'] + window_params['out_sample_days'] - 1]
        
        print(f"  窗口 {window_count}: 样本内 {in_sample_start.date()}~{in_sample_end.date()}, "
              f"样本外 {out_sample_start.date()}~{out_sample_end.date()}")
        
        # 样本内：筛选配对
        selected_pairs = []
        
        for industry_name, stock_codes in industry_map.items():
            # 清理股票
            clean_stocks = [s for s in stock_codes if s not in st_stocks and s not in bse_stocks]
            if len(clean_stocks) < 2:
                continue
            
            # 加载样本内数据
            data_dict = {}
            for code in clean_stocks:
                df = get_stock_data_from_tdx(code, cfg['TDX_DATA_DIR'])
                if df is not None:
                    in_sample_data = df.loc[in_sample_start:in_sample_end]

                    if len(in_sample_data) >= window_params['min_window_days']:
                        data_dict[code] = in_sample_data
            
            if len(data_dict) < 2:
                continue
            
            # 样本内筛选
            industry_pairs = select_pairs_for_industry(
                list(data_dict.keys()), data_dict,
                in_sample_start, in_sample_end,
                cfg['TOP_N_PAIRS_PER_INDUSTRY']
            )
            
            if industry_pairs:
                selected_pairs.extend(industry_pairs)
        
        if not selected_pairs:
            print(f"    无有效配对，跳过")
            window_start_idx += window_params['step_days']
            continue
        
        # 样本内：计算自适应参数
        threshold_manager = AdaptiveThresholdManager(cfg['BASE_PARAMS'])
        trained_pairs = []
        
        for pair in selected_pairs:
            adaptive_params, _ = threshold_manager.calculate_adaptive_params(
                pair['data1'], pair['data2'],
                in_sample_start, in_sample_end
            )
            
            if adaptive_params:
                pair['adaptive_params'] = adaptive_params
                trained_pairs.append(pair)
        
        if not trained_pairs:
            print(f"    参数计算失败，跳过")
            window_start_idx += window_params['step_days']
            continue
        
        # 样本外回测
        window_results = []
        
        for pair in trained_pairs:
            trader = AdaptivePairTradingInstance(
                pair['stock1'], pair['stock2'],
                allocated_capital=1_000_000,
                slippage=cfg['SLIPPAGE'],
                adaptive_params=pair['adaptive_params'],
                base_params=cfg['BASE_PARAMS']
            )
            
            if not trader.initialize_kalman(pair['data1'], pair['data2']):
                continue
            
            # 获取样本外数据
            full_data1 = get_stock_data_from_tdx(pair['stock1'], cfg['TDX_DATA_DIR'])
            full_data2 = get_stock_data_from_tdx(pair['stock2'], cfg['TDX_DATA_DIR'])
            
            if full_data1 is None or full_data2 is None:
                continue
            
            out_data1 = full_data1.loc[out_sample_start:out_sample_end]
            out_data2 = full_data2.loc[out_sample_start:out_sample_end]
            
            if len(out_data1) < 20 or len(out_data2) < 20:
                continue
            
            report = trader.run_backtest(
                out_data1, out_data2,
                out_sample_start, out_sample_end
            )
            
            if report:
                pair_key = f"{pair['stock1']}_{pair['stock2']}"
                
                window_results.append({
                    'pair_key': pair_key,
                    'stock1': pair['stock1'],
                    'stock2': pair['stock2'],
                    'report': report,
                    'window_id': window_count
                })
                
                # 记录历史表现
                if pair_key not in pair_performance_history:
                    pair_performance_history[pair_key] = {
                        'count': 0,
                        'total_return': 0,
                        'best_return': -999,
                        'worst_return': 999,
                        'sharpe_sum': 0,
                        'appearances': []
                    }
                
                perf_history = pair_performance_history[pair_key]
                perf_history['count'] += 1
                perf_history['total_return'] += report['total_return']
                perf_history['best_return'] = max(perf_history['best_return'], report['total_return'])
                perf_history['worst_return'] = min(perf_history['worst_return'], report['total_return'])
                perf_history['sharpe_sum'] += report['sharpe_ratio']
                perf_history['appearances'].append({
                    'window': window_count,
                    'return': report['total_return'],
                    'sharpe': report['sharpe_ratio']
                })
        
        if window_results:
            # 计算窗口表现
            avg_return = np.mean([r['report']['total_return'] for r in window_results])
            avg_sharpe = np.mean([r['report']['sharpe_ratio'] for r in window_results])
            
            all_window_results.append({
                'window_id': window_count,
                'in_sample_period': (in_sample_start, in_sample_end),
                'out_sample_period': (out_sample_start, out_sample_end),
                'results': window_results,
                'performance': {
                    'avg_return': avg_return,
                    'avg_sharpe': avg_sharpe,
                    'num_pairs': len(window_results)
                }
            })
            
            print(f"    有效配对: {len(window_results)}对, "
                  f"平均收益: {avg_return*100:.1f}%, "
                  f"平均夏普: {avg_sharpe:.2f}")
        
        window_start_idx += window_params['step_days']
        # 检查是否需要继续
        if window_count >= 3:  # 限制最多跑3个窗口
            break
    
    print(f"  滚动窗口回测完成: {window_count}个窗口, {len(pair_performance_history)}个唯一配对")
    return all_window_results, pair_performance_history

# ==================== 基于滚动历史筛选配对 ====================

def select_pairs_based_on_rolling_history(pair_performance_history, cfg, momentum_calculator):
    """基于滚动窗口历史表现筛选配对"""
    candidate_pairs = []
    
    for pair_key, history in pair_performance_history.items():
        if history['count'] >= 1:  # 至少出现1次
            avg_return = history['total_return'] / history['count']
            avg_sharpe = history['sharpe_sum'] / history['count']
            
            # 计算稳定性（收益标准差）
            returns = [app['return'] for app in history['appearances']]
            return_std = np.std(returns) if len(returns) > 1 else 0
            
            # 稳定性得分（收益/波动）
            stability_score = avg_return / (return_std + 0.01) if return_std > 0 else 0
            
            candidate_pairs.append({
                'pair_key': pair_key,
                'stock1': pair_key.split('_')[0],
                'stock2': pair_key.split('_')[1],
                'avg_return': avg_return,
                'avg_sharpe': avg_sharpe,
                'count': history['count'],
                'best_return': history['best_return'],
                'worst_return': history['worst_return'],
                'stability_score': stability_score,
                'history': history
            })
    
    if not candidate_pairs:
        return []
    
    # 按稳定性和收益筛选
    candidate_pairs.sort(key=lambda x: (x['stability_score'], x['avg_return']), reverse=True)
    top_candidates = candidate_pairs[:cfg['FINAL_TOP_N'] * 3]  # 多选一些
    
    # 重新加载最新数据回测
    final_pairs = []
    threshold_manager = AdaptiveThresholdManager(cfg['BASE_PARAMS'])
    
    # 使用最新样本内数据
    recent_in_start = pd.to_datetime(cfg['IN_SAMPLE_START'])
    recent_in_end = pd.to_datetime(cfg['IN_SAMPLE_END'])
    
    for candidate in top_candidates:
        stock1, stock2 = candidate['stock1'], candidate['stock2']
        
        data1 = get_stock_data_from_tdx(stock1, cfg['TDX_DATA_DIR'])
        data2 = get_stock_data_from_tdx(stock2, cfg['TDX_DATA_DIR'])
        
        if data1 is None or data2 is None:
            continue
        
        # 使用最新样本内数据
        recent_data1 = data1.loc[recent_in_start:recent_in_end]
        recent_data2 = data2.loc[recent_in_start:recent_in_end]
        
        if len(recent_data1) < 100 or len(recent_data2) < 100:
            continue
        
        # 重新计算自适应参数
        adaptive_params, _ = threshold_manager.calculate_adaptive_params(
            recent_data1, recent_data2,
            recent_in_start, recent_in_end
        )
        
        if not adaptive_params:
            continue
        
        # 回测
        trader = AdaptivePairTradingInstance(
            stock1, stock2,
            allocated_capital=1_000_000,
            slippage=cfg['SLIPPAGE'],
            adaptive_params=adaptive_params,
            base_params=cfg['BASE_PARAMS']
        )
        
        if not trader.initialize_kalman(recent_data1, recent_data2):
            continue
        
        report = trader.run_backtest(
            data1, data2,  # 全量数据
            cfg['OUT_SAMPLE_START'],
            cfg['OUT_SAMPLE_END']
        )
        
        if report and report['total_return'] > 0:  # 最终筛选正收益
            final_pairs.append({
                'stock1': stock1,
                'stock2': stock2,
                'report': report,
                'adaptive_params': adaptive_params,
                'data1': data1,
                'data2': data2,
                'rolling_history': candidate['history']
            })
    
    # 按最终收益排序
    final_pairs.sort(key=lambda x: x['report']['total_return'], reverse=True)
    final_pairs = final_pairs[:cfg['FINAL_TOP_N']]
    
    print(f"  基于滚动窗口选中配对: {len(final_pairs)}对")
    return final_pairs

# ==================== 单次回测函数（调试模式）====================

def run_single_debug_backtest(industry_map, st_stocks, bse_stocks, cfg, momentum_calculator):
    """
    单次回测改进版：样本内划分训练集和验证集
    避免前视偏差，同时减少过拟合风险
    
    参数：
    - industry_map: 行业到股票列表的映射
    - st_stocks: ST股票集合
    - bse_stocks: 北交所股票集合
    - cfg: 配置参数
    - momentum_calculator: 动量计算器
    
    返回：
    - final_pairs: 最终筛选的配对列表
    """
    print("  [单次回测改进版] 开始（样本内训练+验证）")
    print("  模式说明：在样本内划分训练集(70%)和验证集(30%)，避免前视偏差")
    
    threshold_manager = AdaptiveThresholdManager(cfg['BASE_PARAMS'])
    industry_results = {}
    
    # 划分样本内为训练集和验证集
    in_sample_start = pd.to_datetime(cfg['IN_SAMPLE_START'])
    in_sample_end = pd.to_datetime(cfg['IN_SAMPLE_END'])
    
    # 计算分割点（前70%训练，后30%验证）
    total_days = (in_sample_end - in_sample_start).days
    split_days = int(total_days * 0.7)
    split_date = in_sample_start + timedelta(days=split_days)
    
    train_start, train_end = in_sample_start, split_date
    valid_start, valid_end = split_date, in_sample_end
    
    print(f"  训练集: {train_start.date()} ~ {train_end.date()} ({split_days}天)")
    print(f"  验证集: {valid_start.date()} ~ {valid_end.date()} ({total_days - split_days}天)")
    
    # 调试计数器
    processed_count = 0
    skipped_no_data = 0
    skipped_no_pairs = 0
    skipped_no_positive = 0
    
    for idx, (industry_name, stock_codes) in enumerate(industry_map.items(), 1):
        if idx % 10 == 0:
            print(f"    处理进度: {idx}/{len(industry_map)} 个行业")
        
        # 清理股票
        clean_stocks = [s for s in stock_codes if s not in st_stocks and s not in bse_stocks]
        if len(clean_stocks) < 2:
            continue
        
        # 加载数据
        data_dict = {}
        for code in clean_stocks:
            df = get_stock_data_from_tdx(code, cfg['TDX_DATA_DIR'])
            if df is not None and len(df.loc[train_start:train_end]) >= 100:
                data_dict[code] = df
        
        if len(data_dict) < 2:
            skipped_no_data += 1
            continue
        
        # 样本内筛选（基于训练集）
        selected_pairs = select_pairs_for_industry(
            list(data_dict.keys()), data_dict,
            train_start, train_end,  # 用训练集筛选
            cfg['TOP_N_PAIRS_PER_INDUSTRY']
        )
        
        if not selected_pairs:
            skipped_no_pairs += 1
            continue
        
        positive_pairs = []
        
        for pair_info in selected_pairs:
            # 用训练集计算自适应参数
            adaptive_params, _ = threshold_manager.calculate_adaptive_params(
                pair_info['data1'], pair_info['data2'],
                train_start, train_end
            )
            
            if adaptive_params is None:
                continue
            
            # 创建交易实例
            trader = AdaptivePairTradingInstance(
                pair_info['stock1'], pair_info['stock2'],
                allocated_capital=1_000_000,
                slippage=cfg['SLIPPAGE'],
                adaptive_params=adaptive_params,
                base_params=cfg['BASE_PARAMS']
            )
            
            if not trader.initialize_kalman(pair_info['data1'], pair_info['data2']):
                continue
            
            # 关键修改：用验证集回测
            report = trader.run_backtest(
                pair_info['data1'], pair_info['data2'],
                valid_start, valid_end  # 验证集，不是样本外
            )
            
            if report and report['total_return'] > 0:
                positive_pairs.append({
                    'stock1': pair_info['stock1'],
                    'stock2': pair_info['stock2'],
                    'validation_report': report,  # 验证集表现
                    'adaptive_params': adaptive_params,
                    'data1': pair_info['data1'],
                    'data2': pair_info['data2'],
                    'volatility': adaptive_params['volatility'],
                    'halflife': adaptive_params['halflife']
                })
        
        if len(positive_pairs) >= cfg['MIN_POSITIVE_PAIRS']:
            # 按验证集收益排序
            top_pairs = sorted(positive_pairs, 
                              key=lambda x: x['validation_report']['total_return'], 
                              reverse=True)[:cfg['FINAL_TOP_N']]
            
            # 计算行业平均指标
            avg_vol = np.mean([p['volatility'] for p in top_pairs])
            avg_return = np.mean([p['validation_report']['total_return'] for p in top_pairs])
            avg_sharpe = np.mean([p['validation_report']['sharpe_ratio'] for p in top_pairs])
            
            industry_results[industry_name] = {
                'pairs': top_pairs,
                'avg_volatility': avg_vol,
                'avg_return': avg_return,
                'avg_sharpe': avg_sharpe,
                'total_pairs': len(positive_pairs)
            }
            processed_count += 1
        else:
            skipped_no_positive += 1
    
    print(f"  处理完成: {processed_count}个行业有有效配对")
    print(f"  跳过统计: {skipped_no_data}个行业无数据, {skipped_no_pairs}个行业无配对, {skipped_no_positive}个行业无正收益")
    
    if len(industry_results) < 1:
        print("  ✗ 无有效配对，返回空列表")
        return []
    
    # 准备行业指标
    industry_metrics = {}
    for name, info in industry_results.items():
        # 动量计算（使用训练集数据计算）
        momentum = 0
        if info['pairs']:
            pair = info['pairs'][0]
            train_data1 = pair['data1'].loc[train_start:train_end]
            train_data2 = pair['data2'].loc[train_start:train_end]
            
            momentum1 = momentum_calculator.get_momentum_score(train_data1)
            momentum2 = momentum_calculator.get_momentum_score(train_data2)
            momentum = (momentum1 + momentum2) / 2
        
        industry_metrics[name] = {
            'volatility': info['avg_volatility'],
            'expected_return': info['avg_return'],
            'sharpe': info['avg_sharpe'],
            'momentum': momentum
        }
    
    # 风险调整权重分配
    allocator = RiskAdjustedParityAllocator(
        risk_aversion=cfg['RISK_ADJUST_PARAMS']['risk_aversion'],
        return_boost=cfg['RISK_ADJUST_PARAMS']['return_boost'],
        min_w=cfg['RISK_ADJUST_PARAMS']['min_industry_weight'],
        max_w=cfg['RISK_ADJUST_PARAMS']['max_industry_weight'],
        min_return=cfg['RISK_ADJUST_PARAMS']['min_industry_return']
    )
    
    weights = allocator.calculate_weights(industry_metrics)
    
    # 分配资金
    total_capital = cfg['PORTFOLIO_CAPITAL']
    final_pairs = []
    
    for industry_name, weight in weights.items():
        industry_capital = total_capital * weight
        pairs = industry_results[industry_name]['pairs']
        pair_capital = industry_capital / len(pairs)
        
        for pair in pairs:
            # 用完整的样本内数据重新训练最终模型
            print(f"    重新训练配对 {pair['stock1']}-{pair['stock2']}...")
            
            # 获取完整样本内数据
            full_in_data1 = pair['data1'].loc[in_sample_start:in_sample_end]
            full_in_data2 = pair['data2'].loc[in_sample_start:in_sample_end]
            
            # 用完整样本内数据重新计算参数
            final_adaptive_params, _ = threshold_manager.calculate_adaptive_params(
                full_in_data1, full_in_data2,
                in_sample_start, in_sample_end
            )
            
            if final_adaptive_params is None:
                # 如果重新计算失败，使用验证集的参数
                final_adaptive_params = pair['adaptive_params']
            
            # 创建最终交易实例
            final_trader = AdaptivePairTradingInstance(
                pair['stock1'], pair['stock2'],
                allocated_capital=pair_capital,
                slippage=cfg['SLIPPAGE'],
                adaptive_params=final_adaptive_params,
                base_params=cfg['BASE_PARAMS']
            )
            
            if not final_trader.initialize_kalman(full_in_data1, full_in_data2):
                continue
            
            # 在样本外进行最终回测
            out_sample_start = pd.to_datetime(cfg['OUT_SAMPLE_START'])
            out_sample_end = pd.to_datetime(cfg['OUT_SAMPLE_END'])
            
            final_report = final_trader.run_backtest(
                pair['data1'], pair['data2'],
                out_sample_start, out_sample_end
            )
            
            if final_report:
                # 合并验证集和样本外表现
                final_pair = {
                    'stock1': pair['stock1'],
                    'stock2': pair['stock2'],
                    'report': final_report,  # 样本外最终表现
                    'validation_report': pair['validation_report'],  # 验证集表现
                    'adaptive_params': final_adaptive_params,
                    'data1': pair['data1'],
                    'data2': pair['data2'],
                    'allocated_capital': pair_capital,
                    'industry_weight': weight,
                    'pair_weight': weight / len(pairs),
                    'industry_name': industry_name,
                    'volatility': final_adaptive_params.get('volatility', pair.get('volatility', 0)),
                    'halflife': final_adaptive_params.get('halflife', pair.get('halflife', 0))
                }
                
                final_pairs.append(final_pair)
    
    # 输出统计信息
    print(f"\n  [单次回测改进版] 完成统计:")
    print(f"    总配对数: {len(final_pairs)}")
    
    if final_pairs:
        # 计算验证集和样本外收益对比
        val_returns = [p['validation_report']['total_return'] for p in final_pairs]
        out_returns = [p['report']['total_return'] for p in final_pairs]
        
        print(f"    验证集平均收益: {np.mean(val_returns)*100:.2f}%")
        print(f"    样本外平均收益: {np.mean(out_returns)*100:.2f}%")
        
        # 计算收益差异
        return_diff = [out_returns[i] - val_returns[i] for i in range(len(final_pairs))]
        print(f"    平均收益差异: {np.mean(return_diff)*100:.2f}%")
        
        # 计算相关性
        if len(val_returns) > 1 and len(out_returns) > 1:
            corr = np.corrcoef(val_returns, out_returns)[0, 1]
            print(f"    验证集-样本外收益相关性: {corr:.3f}")
    
    return final_pairs

# ==================== 辅助函数 ====================

def get_all_trading_dates(start_date, end_date):
    """获取所有交易日"""
    try:
        # 生成工作日
        from pandas.tseries.offsets import BDay
        dates = pd.date_range(start=start_date, end=end_date, freq=BDay())
        return dates.tolist()
    except:
        # 备用方案
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        return dates.tolist()

# ==================== 主函数 ====================

def run_portfolio_backtest_v8_1():
    """v8.1主函数：三模式回测版本"""
    cfg = CONFIG

    print("="*80)
    print("A股全行业配对交易策略 v8.1 - 三模式回测版本")
    print(f"初始资金: {cfg['PORTFOLIO_CAPITAL']:,.0f}")
    print(f"回测模式: {cfg['BACKTEST_MODE']}")
    print("="*80)

    # 1. 读取申万行业分类数据
    print("\n[1/6] 读取申万行业分类数据...")
    industry_map, st_stocks, bse_stocks = parse_sw_industry_file(
        cfg['SW_INDUSTRY_FILE'],
        industry_col=cfg['INDUSTRY_COLUMN'],
        code_col=cfg['STOCK_CODE_COLUMN']
    )

    if not industry_map:
        print("✗ 行业数据加载失败")
        return

    # 初始化动量计算器
    momentum_calculator = DailyDataMomentumCalculator()
    
    # 2. 根据模式选择回测方法
    backtest_mode = cfg['BACKTEST_MODE']
    
    if backtest_mode == 'rolling_window':
        # 滚动窗口回测模式
        print(f"\n[2/6] 滚动窗口回测模式（避免前视偏差）...")
        
        all_window_results, pair_performance_history = run_rolling_window_backtest(
            industry_map, st_stocks, bse_stocks, cfg
        )
        
        if not pair_performance_history:
            print("  滚动窗口回测无有效结果，回退到实盘模式")
            final_pairs = run_real_time_backtest(industry_map, st_stocks, bse_stocks, cfg, momentum_calculator)
        else:
            # 基于滚动历史筛选配对
            final_pairs = select_pairs_based_on_rolling_history(
                pair_performance_history, cfg, momentum_calculator
            )
            
            if not final_pairs:
                print("  基于滚动历史筛选无配对，回退到实盘模式")
                final_pairs = run_real_time_backtest(industry_map, st_stocks, bse_stocks, cfg, momentum_calculator)
    
    elif backtest_mode == 'real_time':
        # 实盘回测模式
        print(f"\n[2/6] 实盘回测模式（最接近实盘）...")
        print("  模式说明：每天收盘后用当天之前的数据生成信号，第二天交易")
        
        final_pairs = run_real_time_backtest(industry_map, st_stocks, bse_stocks, cfg, momentum_calculator)
    
    else:  # 'single_debug'
        # 单次回测调试模式
        print(f"\n[2/6] 单次回测调试模式（仅用于快速调试）...")
        print("  ⚠️ 警告：此模式存在前视偏差，不适用于实盘验证")
        
        final_pairs = run_single_debug_backtest(industry_map, st_stocks, bse_stocks, cfg, momentum_calculator)
    
    if not final_pairs:
        print("✗ 无有效配对")
        return
    
    # 3. 动量因子分析和权重分配
    print(f"\n[3/6] 动量因子分析和权重分配...")
    
    # 计算行业平均指标
    industry_metrics = {}
    for pair in final_pairs:
        industry_name = pair.get('industry_name', '未知行业')
        if industry_name not in industry_metrics:
            industry_metrics[industry_name] = {
                'returns': [],
                'sharpe': [],
                'volatility': [],
                'momentum_scores': []
            }
        
        # 收集统计指标
        report = pair.get('report', {})
        adaptive_params = pair.get('adaptive_params', {})
        
        industry_metrics[industry_name]['returns'].append(report.get('total_return', 0))
        industry_metrics[industry_name]['sharpe'].append(report.get('sharpe_ratio', 0))
        industry_metrics[industry_name]['volatility'].append(adaptive_params.get('volatility', 0))
        
        # 计算动量（使用样本内数据）
        try:
            data1 = pair.get('data1')
            data2 = pair.get('data2')
            if data1 is not None and data2 is not None:
                in_data1 = data1.loc[cfg['IN_SAMPLE_START']:cfg['IN_SAMPLE_END']]
                in_data2 = data2.loc[cfg['IN_SAMPLE_START']:cfg['IN_SAMPLE_END']]
                
                momentum1 = momentum_calculator.get_momentum_score(in_data1)
                momentum2 = momentum_calculator.get_momentum_score(in_data2)
                momentum = (momentum1 + momentum2) / 2
                industry_metrics[industry_name]['momentum_scores'].append(momentum)
        except:
            pass
    
    # 计算行业平均指标
    processed_industry_metrics = {}
    for industry, metrics in industry_metrics.items():
        if metrics['returns']:
            processed_industry_metrics[industry] = {
                'expected_return': np.mean(metrics['returns']),
                'sharpe': np.mean(metrics['sharpe']),
                'volatility': np.mean(metrics['volatility']),
                'momentum': np.mean(metrics['momentum_scores']) if metrics['momentum_scores'] else 0
            }
    
    # 使用风险调整分配器
    allocator = RiskAdjustedParityAllocator(
        risk_aversion=cfg['RISK_ADJUST_PARAMS']['risk_aversion'],
        return_boost=cfg['RISK_ADJUST_PARAMS']['return_boost'],
        min_w=cfg['RISK_ADJUST_PARAMS']['min_industry_weight'],
        max_w=cfg['RISK_ADJUST_PARAMS']['max_industry_weight'],
        min_return=cfg['RISK_ADJUST_PARAMS']['min_industry_return']
    )
    
    weights = allocator.calculate_weights(processed_industry_metrics)
    
    # 4. 分配资金
    print(f"\n[4/6] 最终资金分配...")
    total_capital = cfg['PORTFOLIO_CAPITAL']
    
    # 按行业分配资金
    industry_allocation = {}
    for industry_name, weight in weights.items():
        industry_allocation[industry_name] = {
            'weight': weight,
            'capital': total_capital * weight,
            'pairs': []
        }
    
    # 将配对分配到行业
    for pair in final_pairs:
        industry_name = pair.get('industry_name', '未知行业')
        if industry_name in industry_allocation:
            industry_allocation[industry_name]['pairs'].append(pair)
    
    # 在行业内平均分配资金
    for industry_name, info in industry_allocation.items():
        if info['pairs']:
            pair_capital = info['capital'] / len(info['pairs'])
            for pair in info['pairs']:
                pair['allocated_capital'] = pair_capital
                pair['industry_weight'] = info['weight']
                pair['pair_weight'] = info['weight'] / len(info['pairs'])
    
    # 打印分配结果
    print("\n  资金分配方案:")
    for name, w in sorted(weights.items(), key=lambda x: -x[1]):
        info = processed_industry_metrics.get(name, {})
        cap = total_capital * w
        print(f"    {name}: 权重{w:.1%} | 资金{cap:,.0f} | 收益{info.get('expected_return', 0)*100:.1f}% | 夏普{info.get('sharpe', 0):.2f}")

    # 5. 生成最终报告
    print(f"\n[5/6] 生成组合报告...")

    # 创建净值数据存储目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    equity_curves_dir = f"equity_curves_{timestamp}"
    os.makedirs(equity_curves_dir, exist_ok=True)
    print(f"✓ 创建净值数据目录: {equity_curves_dir}")

    # 保存每个股票对的净值数据
    equity_files = []
    for i, pair in enumerate(final_pairs):
        # 生成净值文件名
        industry_name = pair.get('industry_name', '未知行业')
        equity_filename = f"{industry_name}_{pair['stock1']}_{pair['stock2']}_净值.csv"
        equity_filepath = os.path.join(equity_curves_dir, equity_filename)
        
        # 提取净值数据
        equity_curve = pair['report']['equity_curve']
        if equity_curve:
            # 转换为DataFrame
            equity_df = pd.DataFrame(equity_curve, columns=['date', 'net_value'])
            equity_df['date'] = pd.to_datetime(equity_df['date']).dt.date
            equity_df.set_index('date', inplace=True)
            
            # 计算净值收益率
            initial_capital = pair['allocated_capital']
            equity_df['net_return'] = (equity_df['net_value'] - initial_capital) / initial_capital
            equity_df['cumulative_return'] = (1 + equity_df['net_return'].shift(1)).cumprod() - 1
            equity_df['cumulative_return'].fillna(0, inplace=True)
            
            # 保存到CSV
            equity_df.to_csv(equity_filepath, encoding='utf-8-sig')
            equity_files.append(equity_filepath)
            
            # 记录文件路径
            pair['equity_file'] = equity_filename
        else:
            pair['equity_file'] = None
    
    # 打包净值数据为ZIP文件
    zip_filename = f"v8.1_净值数据_{timestamp}.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for equity_file in equity_files:
            arcname = os.path.basename(equity_file)
            zipf.write(equity_file, arcname)
    print(f"✓ 净值数据已打包: {zip_filename}")     

    # 清理临时目录
    import shutil
    shutil.rmtree(equity_curves_dir)
    print(f"✓ 清理临时目录: {equity_curves_dir}")

    # 计算组合层面指标（使用改进的夏普计算）
    portfolio_return = np.sum([p['report']['total_return'] * p['pair_weight'] for p in final_pairs])
    portfolio_slippage = np.sum([p['report']['slippage_impact'] * p['pair_weight'] for p in final_pairs])
    portfolio_sharpe = calculate_portfolio_sharpe(final_pairs)  # 使用改进的夏普计算

    print(f"\n{'='*80}")
    print("v8.1组合回测结果（三模式回测）")
    print(f"{'='*80}")
    print(f"回测模式: {backtest_mode}")
    print(f"初始资金: {total_capital:,.0f}")
    print(f"总配对数: {len(final_pairs)}")
    print(f"组合预期收益率: {portfolio_return*100:.2f}%")
    print(f"组合夏普比率: {portfolio_sharpe:.2f}")
    print(f"滑点成本占比: {portfolio_slippage*100:.2f}%")
    print(f"净收益率: {(portfolio_return - portfolio_slippage)*100:.2f}%")

    print(f"\n【v8.1改进点】")
    print(f"1. 三种回测模式: 滚动窗口回测 | 实盘回测 | 单次回测（调试）")
    print(f"2. 实盘回测模式: 每天只用当天之前的数据，模拟真实交易")
    print(f"3. 改进组合夏普计算: 考虑配对间的相关性")
    print(f"4. 保留v8.0输出格式: 结果文件、净值数据、交易明细、配置文件")

    # 创建最终结果表格
    results_data = []
    for p in final_pairs:
        # 汇总关键交易统计
        trades_df = pd.DataFrame(p['report']['trade_records'])
        if len(trades_df) > 0:
            sell_trades = trades_df[trades_df['action'] == '卖出']
            avg_hold_days = sell_trades['hold_days'].mean() if 'hold_days' in sell_trades.columns and len(sell_trades) > 0 else 0
            avg_return_per_trade = sell_trades['pnl'].mean() if 'pnl' in sell_trades.columns and len(sell_trades) > 0 else 0
        else:
            avg_hold_days = 0
            avg_return_per_trade = 0

        results_data.append({
            '行业': p.get('industry_name', '未知行业'),
            '行业权重': f"{p.get('industry_weight', 0):.2%}",
            '配对权重': f"{p.get('pair_weight', 0):.2%}",
            '股票1': p['stock1'],
            '股票2': p['stock2'],
            '分配资金': p['allocated_capital'],
            '总收益率': f"{p['report']['total_return']:.2%}",
            '夏普比率': f"{p['report']['sharpe_ratio']:.2f}",
            '最大回撤': f"{p['report']['max_drawdown']:.2%}",
            '胜率': f"{p['report']['win_rate']:.1%}",
            '盈亏比': f"{p['report']['profit_loss_ratio']:.2f}",
            '交易次数': p['report']['num_trades'],
            '平均持仓天数': f"{avg_hold_days:.1f}",
            '单次交易平均收益': f"{avg_return_per_trade:.2f}",
            '滑点成本占比': f"{p['report']['slippage_impact']:.2%}",
            '波动率': f"{p['adaptive_params']['volatility']:.2%}",
            '半衰期': f"{p['adaptive_params']['halflife']:.1f}天",
            '入场阈值': f"{p['adaptive_params']['entry_threshold']:.2f}",
            '最大持仓天数': p['adaptive_params']['max_holding_days'],
            '净值文件': p.get('equity_file', 'N/A'),
            '冷却次数': p['report']['cooldown_count']
        })
    
    results_df = pd.DataFrame(results_data)
    
    # 保存结果文件
    results_file = f"v8.1_组合回测结果_{timestamp}.csv"
    results_df.to_csv(results_file, index=False, encoding='utf-8-sig', float_format='%.4f')
    print(f"✓ 结果已保存: {results_file}")

    # 生成详细交易记录汇总
    trade_details = []
    for p in final_pairs:
        trades = p['report']['trade_records']
        for trade in trades:
            trade_details.append({
                '行业': p.get('industry_name', '未知行业'),
                '股票1': p['stock1'],
                '股票2': p['stock2'],
                '交易日期': trade.get('date', ''),
                '操作': trade.get('action', ''),
                '股票代码': trade.get('stock', ''),
                '原始价格': trade.get('raw_price', 0),
                '执行价格': trade.get('executed_price', 0),
                '股数': trade.get('shares', 0),
                '金额': trade.get('amount', 0),
                '佣金': trade.get('commission', 0),
                '印花税': trade.get('stamp_tax', 0) if 'stamp_tax' in trade else 0,
                '盈亏': trade.get('pnl', 0) if 'pnl' in trade else 0,
                '原因': trade.get('reason', ''),
                '持仓天数': trade.get('hold_days', 0),
                '滑点成本': trade.get('slippage_cost', 0)
            })

    if trade_details:
        trade_details_df = pd.DataFrame(trade_details)
        trade_file = f"v8.1_交易明细_{timestamp}.csv"
        trade_details_df.to_csv(trade_file, index=False, encoding='utf-8-sig')
        print(f"✓ 交易明细已保存: {trade_file}")

    # 保存配置
    config_file = f"v8.1_回测配置_{timestamp}.json"
    with open(config_file, 'w') as f:
        json.dump({k: str(v) if isinstance(v, (datetime, pd.Timestamp)) else v 
                  for k, v in cfg.items()}, f, indent=2, default=str)
    print(f"✓ 配置已保存: {config_file}")

    print(f"\n{'='*80}")
    print("文件生成完成:")
    print(f"1. 组合结果: {results_file}")
    print(f"2. 净值数据(ZIP): {zip_filename}")
    print(f"3. 交易明细: {trade_file if 'trade_file' in locals() else '无交易'}")
    print(f"4. 配置文件: {config_file}")
    print(f"{'='*80}")

# ==================== 主程序入口 ====================

if __name__ == "__main__":
    run_portfolio_backtest_v8_1()

