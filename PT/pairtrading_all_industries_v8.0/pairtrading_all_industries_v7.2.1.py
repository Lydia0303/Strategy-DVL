# === A股全行业配对轮动策略 v7.2 ===
# 基于v7.1优化：改用申万行业分类数据（新版二级行业）
# 修复：补全主函数逻辑、参数错误、异常处理、结果输出

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
    'TOP_N_PAIRS_PER_INDUSTRY': 5,
    'FINAL_TOP_N': 2,
    'MIN_POSITIVE_PAIRS': 1,  # 【修复】注释与值统一，至少1个正收益配对

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
        'min_industry_return': -0.02,  # 【修复】注释与值统一，门槛-2%
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
        print(f"✗ 读取股票{stock_code}数据失败: {e}")
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
                 min_industry_weight=0.02, max_industry_weight=0.15, min_industry_return=-0.02):
        self.risk_aversion = risk_aversion
        self.return_boost = return_boost
        self.min_w = min_industry_weight
        self.max_w = max_industry_weight
        self.min_return = min_industry_return

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

        total_score = sum(scores.values()) if scores else 1
        weights = {k: v / total_score for k, v in scores.items()}

        weights = self._apply_constraints(weights)

        return weights

    def _apply_constraints(self, weights):
        if not weights:
            return {}
            
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

        total = sum(weights.values()) if weights else 1
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
        except Exception as e:
            print(f"    计算Z-score失败: {e}")
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
        except Exception as e:
            print(f"    协整检验失败: {e}")
            return True, 0.01

    def run_backtest(self, stock1_data, stock2_data, out_sample_start, out_sample_end):
        out_data1 = stock1_data.loc[out_sample_start:out_sample_end]
        out_data2 = stock2_data.loc[out_sample_start:out_sample_end]
        trading_dates = out_data1.index.intersection(out_data2.index)

        if len(trading_dates) == 0:
            print("    无回测数据")
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
    """为单个行业筛选最优配对"""
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

# ==================== 主函数 ====================

def run_portfolio_backtest_v7_2():
    """v7.2主函数：使用申万行业分类数据"""
    cfg = CONFIG

    print("="*80)
    print("A股全行业配对交易策略 v7.2 - 申万行业分类数据源")
    print(f"初始资金: {cfg['PORTFOLIO_CAPITAL']:,.0f}")
    print(f"滑点: {cfg['SLIPPAGE']:.2%}")
    print(f"行业分类: {cfg['INDUSTRY_COLUMN']}")
    print("="*80)

    # 【v7.2修改】使用申万行业分类数据
    print("\n[1/6] 读取申万行业分类数据...")
    industry_map, st_stocks, bse_stocks = parse_sw_industry_file(
        cfg['SW_INDUSTRY_FILE'],
        industry_col=cfg['INDUSTRY_COLUMN'],
        code_col=cfg['STOCK_CODE_COLUMN']
    )

    if not industry_map:
        print("✗ 行业数据加载失败")
        return None

    # 加载股票数据
    print("\n[2/6] 加载通达信股票数据...")
    all_stock_codes = []
    for codes in industry_map.values():
        all_stock_codes.extend(codes)
    all_stock_codes = list(set(all_stock_codes))
    print(f"  需加载股票数: {len(all_stock_codes)}")

    data_dict = {}
    for i, code in enumerate(all_stock_codes):
        if i % 100 == 0:
            print(f"  进度: {i}/{len(all_stock_codes)}")
        df = get_stock_data_from_tdx(code, cfg['TDX_DATA_DIR'])
        if df is not None and len(df) > 365:  # 至少1年数据
            data_dict[code] = df
    print(f"✓ 成功加载 {len(data_dict)} 只股票数据")

    # 按行业筛选配对
    print("\n[3/6] 按行业筛选最优配对...")
    industry_pairs = {}
    industry_metrics = {}
    for industry, stock_list in industry_map.items():
        print(f"\n  处理行业: {industry} (股票数: {len(stock_list)})")
        
        # 过滤有数据的股票
        valid_stocks = [code for code in stock_list if code in data_dict]
        if len(valid_stocks) < 2:
            print(f"    有效股票不足，跳过")
            continue
            
        # 筛选配对
        pairs = select_pairs_for_industry(
            valid_stocks, data_dict,
            cfg['IN_SAMPLE_START'], cfg['IN_SAMPLE_END'],
            top_n=cfg['TOP_N_PAIRS_PER_INDUSTRY']
        )
        
        if not pairs:
            print(f"    无合格配对，跳过")
            continue
            
        industry_pairs[industry] = pairs
        print(f"    筛选出 {len(pairs)} 个合格配对")
        
        # 计算行业指标（取最优配对）
        best_pair = pairs[0]
        s1_data = best_pair['data1']
        s2_data = best_pair['data2']
        
        # 计算样本内收益
        in_sample_1 = s1_data.loc[cfg['IN_SAMPLE_START']:cfg['IN_SAMPLE_END'], 'close']
        in_sample_2 = s2_data.loc[cfg['IN_SAMPLE_START']:cfg['IN_SAMPLE_END'], 'close']
        common_dates = in_sample_1.index.intersection(in_sample_2.index)
        
        if len(common_dates) < 60:
            continue
            
        log_ret1 = np.log(in_sample_1.loc[common_dates]).pct_change().dropna()
        log_ret2 = np.log(in_sample_2.loc[common_dates]).pct_change().dropna()
        pair_ret = (log_ret1 - best_pair['hedge_ratio'] * log_ret2).dropna()
        
        industry_metrics[industry] = {
            'expected_return': pair_ret.mean() * 252,
            'volatility': pair_ret.std() * np.sqrt(252),
            'momentum': pair_ret[-20:].sum()
        }

    # 计算行业权重
    print("\n[4/6] 计算风险调整后的行业权重...")
    allocator = RiskAdjustedParityAllocator(**cfg['RISK_ADJUST_PARAMS'])
    industry_weights = allocator.calculate_weights(industry_metrics)
    print(f"  最终入选行业数: {len(industry_weights)}")
    for industry, weight in industry_weights.items():
        print(f"    {industry}: {weight:.1%}")

    # 运行行业配对回测
    print("\n[5/6] 运行配对交易回测...")
    all_reports = []
    total_capital = cfg['PORTFOLIO_CAPITAL']
    
    for industry, weight in industry_weights.items():
        print(f"\n  回测行业: {industry} (权重: {weight:.1%})")
        pairs = industry_pairs[industry]
        if not pairs:
            continue
            
        # 取前N个配对
        top_pairs = pairs[:cfg['FINAL_TOP_N']]
        pair_capital = total_capital * weight / len(top_pairs)
        
        for pair in top_pairs:
            print(f"    回测配对: {pair['stock1']} & {pair['stock2']}")
            
            # 计算自适应参数
            atm = AdaptiveThresholdManager(cfg['BASE_PARAMS'])
            adaptive_params, _ = atm.calculate_adaptive_params(
                pair['data1'], pair['data2'],
                cfg['IN_SAMPLE_START'], cfg['IN_SAMPLE_END']
            )
            
            if adaptive_params is None:
                print(f"      自适应参数计算失败，跳过")
                continue
                
            # 初始化配对交易实例
            trading_instance = AdaptivePairTradingInstance(
                stock1=pair['stock1'],
                stock2=pair['stock2'],
                allocated_capital=pair_capital,
                slippage=cfg['SLIPPAGE'],
                adaptive_params=adaptive_params,
                base_params=cfg['BASE_PARAMS']
            )
            
            # 初始化卡尔曼滤波器
            init_success = trading_instance.initialize_kalman(
                pair['data1'].loc[cfg['IN_SAMPLE_START']:cfg['IN_SAMPLE_END']],
                pair['data2'].loc[cfg['IN_SAMPLE_START']:cfg['IN_SAMPLE_END']]
            )
            
            if not init_success:
                print(f"      卡尔曼滤波器初始化失败，跳过")
                continue
                
            # 运行回测
            report = trading_instance.run_backtest(
                pair['data1'], pair['data2'],
                cfg['OUT_SAMPLE_START'], cfg['OUT_SAMPLE_END']
            )
            
            if report:
                report['industry'] = industry
                report['industry_weight'] = weight
                all_reports.append(report)
                print(f"      总收益: {report['total_return']:.1%}")
                print(f"      夏普比率: {report['sharpe_ratio']:.2f}")
                print(f"      最大回撤: {report['max_drawdown']:.1%}")

    # 汇总结果
    print("\n[6/6] 汇总回测结果...")
    if not all_reports:
        print("✗ 无有效回测结果")
        return None
    
    # 计算组合整体指标
    total_final_value = sum(report['final_value'] for report in all_reports)
    total_return = (total_final_value - total_capital) / total_capital
    
    # 合并权益曲线
    all_equity = {}
    for report in all_reports:
        for date, value in report['equity_curve']:
            if date not in all_equity:
                all_equity[date] = 0
            all_equity[date] += value
    
    # 排序并计算组合指标
    sorted_dates = sorted(all_equity.keys())
    equity_values = [all_equity[date] for date in sorted_dates]
    equity_returns = pd.Series(equity_values).pct_change().dropna()
    sharpe = np.sqrt(252) * equity_returns.mean() / equity_returns.std() if len(equity_returns) > 0 else 0
    
    # 计算最大回撤
    peak, max_drawdown = equity_values[0], 0
    for v in equity_values:
        if v > peak:
            peak = v
        max_drawdown = max(max_drawdown, (peak - v) / peak)
    
    # 打印最终结果
    print("="*80)
    print("回测结果汇总")
    print("="*80)
    print(f"初始资金: {total_capital:,.0f}")
    print(f"最终资金: {total_final_value:,.0f}")
    print(f"总收益率: {total_return:.1%}")
    print(f"组合夏普比率: {sharpe:.2f}")
    print(f"组合最大回撤: {max_drawdown:.1%}")
    print(f"总交易次数: {sum(report['num_trades'] for report in all_reports)}")
    print(f"平均胜率: {np.mean([report['win_rate'] for report in all_reports]):.1%}")
    
    # 保存结果到JSON
    result_summary = {
        'total_initial_capital': total_capital,
        'total_final_value': total_final_value,
        'total_return': float(total_return),
        'sharpe_ratio': float(sharpe),
        'max_drawdown': float(max_drawdown),
        'num_industries': len(industry_weights),
        'num_pairs': len(all_reports),
        'pair_reports': [{
            'industry': r['industry'],
            'stock1': r['stock1'],
            'stock2': r['stock2'],
            'return': float(r['total_return']),
            'sharpe': float(r['sharpe_ratio']),
            'max_drawdown': float(r['max_drawdown'])
        } for r in all_reports]
    }
    
    with open('pair_trading_v7.2_result.json', 'w', encoding='utf-8') as f:
        json.dump(result_summary, f, ensure_ascii=False, indent=2)
    
    print("\n✓ 结果已保存到 pair_trading_v7.2_result.json")
    return result_summary

# 执行主函数
if __name__ == "__main__":
    result = run_portfolio_backtest_v7_2()