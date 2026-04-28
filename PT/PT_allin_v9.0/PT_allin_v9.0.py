# === A股全行业配对轮动策略 v9.0（三模式回测版）===
# 基于v8.1优化：修正自适应参数使用方式
# 验证期和样本外都启用动态自适应参数

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
    'SW_INDUSTRY_FILE': r"C:\Users\hz\Desktop\Strategy DVL\SwClass\最新个股申万行业分类(完整版-截至7月末).xlsx",
    'INDUSTRY_COLUMN': '新版二级行业',
    'STOCK_CODE_COLUMN': '股票代码',

    'TDX_DATA_DIR': "C:/new_tdx/vipdoc",

    # 回测时间区间
    'IN_SAMPLE_START': "2021-01-01",
    'IN_SAMPLE_END': "2024-12-31",
    'OUT_SAMPLE_START': "2025-01-01",
    'OUT_SAMPLE_END': "2025-12-31",
    
    # 验证期参数（训练期结束后，样本外开始前）
    'VALIDATION_SPLIT_RATIO': 0.3,  # 样本内后30%作为验证期

    # 回测模式选择
    'BACKTEST_MODE': 'single_debug',  # 'rolling_window' | 'real_time' | 'single_debug'
    
    # 滚动窗口参数
    'ROLLING_WINDOW': {
        'in_sample_days': 1008,
        'out_sample_days': 252,
        'step_days': 252,
        'min_window_days': 200
    },

    # 实盘回测参数
    'REAL_TIME_PARAMS': {
        'rebalance_days': 63,
        'lookback_days': 252,
        'max_history_years': 4,
        'daily_slippage': 0.0003,
        'min_trading_days': 20,
    },

    # 筛选参数
    'TOP_N_PAIRS_PER_INDUSTRY': 5,
    'FINAL_TOP_N': 2,
    'MIN_POSITIVE_PAIRS': 1,

    # 组合资金配置
    'PORTFOLIO_CAPITAL': 10_000_000,
    'SLIPPAGE': 0.0003,

    # 自适应参数更新频率
    'ADAPTIVE_UPDATE_FREQ': 10,  # 每10个交易日更新一次参数
    'ADAPTIVE_LOOKBACK_DAYS': 120,  # 回看120天计算参数

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
        'min_industry_return': 0.05,
    }
}

# ==================== 样本内数据分段用于全局 ====================

def calculate_validation_period(cfg):
    """计算验证期时间范围"""
    in_sample_start = pd.to_datetime(cfg['IN_SAMPLE_START'])
    in_sample_end = pd.to_datetime(cfg['IN_SAMPLE_END'])
    total_days = (in_sample_end - in_sample_start).days
    split_days = int(total_days * (1 - cfg.get('VALIDATION_SPLIT_RATIO', 0.3)))
    valid_start = in_sample_start + timedelta(days=split_days)
    valid_end = in_sample_end
    return valid_start, valid_end

# ==================== 申万行业数据读取模块 ====================

def parse_sw_industry_file(excel_path, industry_col='新版二级行业', code_col='股票代码'):
    """解析申万行业分类Excel文件"""
    try:
        print(f"读取申万行业分类数据: {excel_path}")
        df = pd.read_excel(excel_path)
        print(f"  Excel数据形状: {df.shape}")

        if industry_col not in df.columns or code_col not in df.columns:
            raise ValueError(f"找不到必要列")

        # 清理数据
        st_stocks = set()
        if '公司简称' in df.columns:
            st_mask = df['公司简称'].str.contains(r'ST|\*ST', na=False, case=False)
            st_stocks = set(df.loc[st_mask, code_col].astype(str).str.replace('.SZ', '').str.replace('.SH', '').str.replace('.BJ', ''))
            df = df[~st_mask].copy()
            print(f"  剔除ST股票: {len(st_stocks)}只")

        def is_bse(code):
            code_str = str(code).replace('.BJ', '').replace('.SZ', '').replace('.SH', '')
            return code_str.startswith(('83', '87', '88', '92')) or str(code).endswith('.BJ')

        bse_mask = df[code_col].apply(is_bse)
        bse_stocks = set(df.loc[bse_mask, code_col].astype(str).str.replace('.SZ', '').str.replace('.SH', '').str.replace('.BJ', ''))
        df = df[~bse_mask].copy()
        print(f"  剔除北交所股票: {len(bse_stocks)}只")

        df['clean_code'] = df[code_col].astype(str).str.replace('.SZ', '').str.replace('.SH', '').str.replace('.BJ', '')
        df[industry_col] = df[industry_col].astype(str).str.strip()

        industry_map = {}
        for industry, group in df.groupby(industry_col):
            codes = group['clean_code'].tolist()
            if len(codes) >= 2:
                industry_map[industry] = codes

        print(f"✓ 申万行业分类解析完成: {len(industry_map)}个二级行业")
        print(f"  总股票数: {df['clean_code'].nunique()}只")

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
                    if 0 < abs(rho) < 1:
                        denominator = np.log(abs(rho))
                        halflife = 15 if abs(denominator) < 1e-6 else -np.log(2) / denominator
                    halflife = max(5, min(halflife, 60))
                except:
                    halflife = 15
            
            vol_factor = 0.75 + (volatility / 0.4) * (1.6 - 0.75)
            vol_factor = np.clip(vol_factor, 0.75, 1.6)
            
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
            
            if halflife < 8:
                speed_factor = 0.8
            elif halflife < 15:
                speed_factor = 1.0
            elif halflife < 25:
                speed_factor = 1.2
            else:
                speed_factor = 1.4
            # 在 calculate_adaptive_params 中添加
            #print(f"  计算波动率: 时间窗口 {in_sample_start.date()} 到 {in_sample_end.date()}")
            #print(f"  价差收益率数据点: {len(spread_returns)}, 波动率: {volatility:.4f}")
            #print(f"  vol_factor: {vol_factor:.2f}")

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

# ==================== 支持动态自适应参数的回测类 ====================

class DynamicAdaptivePairTrading:
    """支持动态参数更新的配对交易类"""
    
    def __init__(self, stock1, stock2, allocated_capital, slippage, 
                 base_params, threshold_manager, initial_adaptive_params=None):
        self.stock1 = stock1
        self.stock2 = stock2
        self.allocated_capital = allocated_capital
        self.slippage = slippage
        self.base_params = base_params
        self.threshold_manager = threshold_manager
        
        # 初始参数
        self.current_params = {**base_params}
        if initial_adaptive_params is None:
            self.current_params = base_params.copy()
        else:
            # 使用传入的参数
            self.current_params = {**base_params, **initial_adaptive_params}
        
        self.cash = allocated_capital
        self.kf = None
        self.pair_valid = True
        self.cooling_period = False
        self.cooldown_count = 0
        self.cooling_end_date = None
        self.last_check_date = None
        self.consecutive_fails = 0
        self.last_param_update_date = None
        self.param_update_freq = 10  # 每10个交易日更新一次参数
        # self.param_update_freq = cfg.get('ADAPTIVE_UPDATE_FREQ', 20)
        self.param_history = []  # 记录参数变化历史

        self.positions = {}
        self.holding_stock = None
        self.entry_date = None
        self.entry_z = 0
        self.holding_days = 0

        self.trade_records = []
        self.equity_curve = []

    def initialize_kalman(self, in_sample_data1, in_sample_data2):
        """初始化卡尔曼滤波器"""
        try:
            common_dates = in_sample_data1.index.intersection(in_sample_data2.index)
            if len(common_dates) < 100:
                return False

            s1 = in_sample_data1.loc[common_dates, 'close']
            s2 = in_sample_data2.loc[common_dates, 'close']

            valid = (s1 > 0) & (s2 > 0) & (~np.isnan(s1)) & (~np.isnan(s2))
            s1, s2 = s1[valid], s2[valid]

            if len(s1) < 100:
                return False

            log_s1 = np.log(s1.iloc[-100:])
            log_s2 = np.log(s2.iloc[-100:])

            if len(log_s1) != len(log_s2):
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
            return False

    def update_adaptive_params(self, current_date, stock1_data, stock2_data, lookback_days=120):
        """更新自适应参数"""
        if self.last_param_update_date is not None:
            days_since_update = (current_date - self.last_param_update_date).days
            if days_since_update < self.param_update_freq:
                return False  # 未到更新周期
        
        # 使用过去lookback_days的数据重新计算参数
        lookback_start = current_date - timedelta(days=lookback_days)
        
        # 添加调试信息
        # print(f"  [{current_date.date()}] 更新自适应参数: 回看 {lookback_days} 天 ({lookback_start.date()} 到 {current_date.date()})")

        adaptive_params, _ = self.threshold_manager.calculate_adaptive_params(
            stock1_data, stock2_data,
            lookback_start, current_date
        )
        
        if adaptive_params:
            # 更新当前参数
            for key, value in adaptive_params.items():
                self.current_params[key] = value
            
            # 记录参数历史
            self.param_history.append({
                'date': current_date,
                'params': adaptive_params.copy()
            })
            
            # 添加调试信息
            # print(f"    更新后参数: 入场={adaptive_params.get('entry_threshold', 0):.2f}, 出场={adaptive_params.get('exit_threshold', 0):.2f}")
            # print(f"              波动率={adaptive_params.get('volatility', 0):.4f}, vol_factor={adaptive_params.get('vol_factor', 0):.2f}")

            self.last_param_update_date = current_date
            return True
        
        return False

    def get_current_params(self):
        """获取当前参数"""
        return self.current_params.copy()

    def apply_slippage(self, price, direction):
        if direction == 'buy':
            return price * (1 + self.slippage)
        else:
            return price * (1 - self.slippage)

    def calculate_position_size(self, price):
        target_ratio = self.current_params.get('max_position_ratio', 0.6)
        available_capital = self.cash * target_ratio
        commission_factor = (1 + self.current_params['commission_rate'] + self.current_params['stamp_tax_rate'] * 0.5)
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
        commission = max(amount * self.current_params['commission_rate'], 5)
        total_cost = amount + commission

        if total_cost > self.cash:
            available_cash = self.cash
            max_shares = 0
            for trial_shares in range(100, int(available_cash / price) + 100, 100):
                trial_amount = price * trial_shares
                trial_commission = max(trial_amount * self.current_params['commission_rate'], 5)
                trial_total = trial_amount + trial_commission
                if trial_total <= available_cash:
                     max_shares = trial_shares
                else:
                     break
        
            if max_shares < 100:
                return False, "资金不足"
            
            shares = max_shares
            amount = price * shares
            commission = max(amount * self.current_params['commission_rate'], 5)
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
        commission = max(amount * self.current_params['commission_rate'], 5)
        stamp_tax = amount * self.current_params['stamp_tax_rate']
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

            if len(s1) < self.current_params['min_coint_period'] or len(s2) < self.current_params['min_coint_period']:
                return True, 0.01

            log_s1 = np.log(s1.replace(0, np.nan).dropna())
            log_s2 = np.log(s2.replace(0, np.nan).dropna())

            common_dates = log_s1.index.intersection(log_s2.index)
            if len(common_dates) < self.current_params['min_coint_period']:
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

    def run_backtest_with_dynamic_params(self, stock1_data, stock2_data, 
                                        out_sample_start, out_sample_end):
        """使用动态参数运行回测"""
        out_data1 = stock1_data.loc[out_sample_start:out_sample_end]
        out_data2 = stock2_data.loc[out_sample_start:out_sample_end]
        trading_dates = out_data1.index.intersection(out_data2.index)

        if len(trading_dates) == 0:
            return None

        # 在回测开始时更新一次参数
        if self.last_param_update_date is None:
            self.update_adaptive_params(trading_dates[0], out_data1, out_data2, 
                                       lookback_days=CONFIG['ADAPTIVE_LOOKBACK_DAYS'])

        warmup_days = min(20, self.current_params['max_holding_days'] // 2)
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

            # 检查并更新自适应参数
            self.update_adaptive_params(current_date, out_data1, out_data2, 
                                       lookback_days=CONFIG['ADAPTIVE_LOOKBACK_DAYS'])

            days_since_last = (current_date - self.last_check_date).days if self.last_check_date else 999
            if days_since_last >= self.current_params['coint_check_freq']:
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

            entry_th = self.current_params['entry_threshold']
            exit_th = self.current_params['exit_threshold']
            stop_th = self.current_params['stop_loss']
            reb_th = self.current_params['rebalance_threshold']

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
                elif self.holding_days >= self.current_params['max_holding_days']:
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
            'param_history': self.param_history,
            'final_params': self.get_current_params()
        }

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
        print(f"\n    === RiskAdjustedParityAllocator.calculate_weights()调试 ===")
        print(f"    收到 {len(industry_metrics)} 个行业")
    
        if not industry_metrics:
            print(f"    ❌ 错误: industry_metrics为空!")
            return {}
    
        # 详细检查每个行业
        for name, metrics in industry_metrics.items():
            print(f"\n    处理行业: {name}")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if key == 'expected_return':
                        print(f"      {key}: {value:.6f} ({value*100:.2f}%)")
                    elif key == 'volatility':
                        print(f"      {key}: {value:.6f} ({value*100:.2f}%)")
                    else:
                        print(f"      {key}: {value:.6f}")
                else:
                    print(f"      {key}: {value}")
    
        qualified = {}
        for name, metrics in industry_metrics.items():
            expected_return = metrics.get('expected_return', 0)
            if expected_return >= self.min_return:
                qualified[name] = metrics
                print(f"    ✅ {name}: 预期收益{expected_return*100:.2f}% >= 门槛{self.min_return*100:.2f}%，通过")
            else:
                print(f"    ❌ {name}: 预期收益{expected_return*100:.2f}% < 门槛{self.min_return*100:.2f}%，剔除")
    
        print(f"\n    通过门槛的行业数: {len(qualified)}/{len(industry_metrics)}")
    
        if len(qualified) < 3:
            print(f"    警告: 通过收益门槛的行业不足3个({len(qualified)})，放宽门槛...")
            # 按预期收益排序，取前几个
            sorted_industries = sorted(industry_metrics.items(),
                                       key=lambda x: x[1]['expected_return'], 
                                       reverse=True)
            qualified = dict(sorted_industries[:max(3, len(industry_metrics)//3)])
            print(f"    放宽后选择行业: {list(qualified.keys())}")
        
            # 调试输出放宽后的行业
            print(f"\n    放宽门槛后的行业详情:")
            for name, metrics in qualified.items():
                expected_return = metrics.get('expected_return', 0)
                print(f"      {name}: 预期收益{expected_return*100:.2f}%")
    
        # 计算风险调整分数
        scores = {}
        print(f"\n    计算风险调整分数:")
    
        for name, metrics in qualified.items():
            vol = max(metrics['volatility'], 0.001)
            ret = max(metrics['expected_return'], 0.001)
            sharpe = ret / vol if vol > 0 else 0
        
            print(f"\n    ── 行业: {name} ──")
            print(f"      波动率(vol): {vol:.4f}")
            print(f"      预期收益(ret): {ret:.4f} ({ret*100:.1f}%)")
            print(f"      夏普比率(sharpe): {sharpe:.4f}")
        
            # 调试计算过程
            risk_adjusted_score = (sharpe ** self.return_boost) / (vol ** self.risk_aversion)
            print(f"      sharpe^{self.return_boost}: {sharpe ** self.return_boost:.6f}")
            print(f"      vol^{self.risk_aversion}: {vol ** self.risk_aversion:.6f}")
            print(f"      风险调整分数: {risk_adjusted_score:.6f}")
        
            momentum = metrics.get('momentum', 0)
            momentum_factor = 1 + max(min(momentum, 0.3), -0.2)
            print(f"      原始动量: {momentum:.4f}")
            print(f"      动量因子: {momentum_factor:.4f}")
        
            final_score = risk_adjusted_score * momentum_factor
            scores[name] = final_score
            print(f"      最终分数: {final_score:.6f}")
    
        if not scores:
            print(f"    ❌ 错误: 没有计算任何分数!")
            return {}
    
        total_score = sum(scores.values())
        print(f"\n    总分数: {total_score:.6f}")
    
        # 计算原始权重
        raw_weights = {k: v / total_score for k, v in scores.items()}
        print(f"\n    原始权重:")
        for name, weight in raw_weights.items():
            print(f"      {name}: {weight:.6f} ({weight*100:.2f}%)")
    
        # 应用约束
        weights = self._apply_constraints(raw_weights)
    
        print(f"\n    应用约束后的权重:")
        for name, weight in weights.items():
            print(f"      {name}: {weight:.6f} ({weight*100:.2f}%)")
    
        # 检查权重总和
        weight_sum = sum(weights.values())
        print(f"    权重总和: {weight_sum:.6f}")
    
        if abs(weight_sum - 1.0) > 0.001:
            print(f"    ⚠️ 警告: 权重总和{weight_sum:.6f} ≠ 1.0, 重新归一化")
            weights = {k: v/weight_sum for k, v in weights.items()}
    
        return weights

    def _apply_constraints(self, weights):
        for iteration in range(10):
            excess_total = 0
            valid_count = 0
        
            # 先找出有效行业
            valid_industries = [k for k, v in weights.items() if v < self.max_w]
            valid_count = len(valid_industries)
        
            for k in list(weights.keys()):
                if weights[k] > self.max_w:
                    excess = weights[k] - self.max_w
                    excess_total += excess
                    weights[k] = self.max_w

            if excess_total > 0:
                if valid_count > 0:
                    # 如果有有效行业，分配给它们
                    redistribution = excess_total / valid_count
                    for k in valid_industries:
                        weights[k] += redistribution
                else:
                    # 如果没有有效行业，分配给所有行业
                    redistribution = excess_total / len(weights)
                    for k in weights:
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
        if total > 0:
            return {k: v / total for k, v in weights.items()}
        else:
            return {}


# ==================== 配对筛选模块 ====================

def select_pairs_for_industry(stock_list, data_dict, in_sample_start, in_sample_end, top_n=5):
    """筛选行业内的配对"""
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

def get_industry_for_pair(stock1, stock2, industry_map):
    """查找配对所属的真实申万行业"""
    # 1. 首先检查两个股票是否在同一个行业
    for industry, stocks in industry_map.items():
        if stock1 in stocks and stock2 in stocks:
            return industry
    
    # 2. 如果不在同一行业，分别查找各自的行业
    industry1 = None
    industry2 = None
    
    for industry, stocks in industry_map.items():
        if stock1 in stocks:
            industry1 = industry
        if stock2 in stocks:
            industry2 = industry
    
    # 3. 返回组合行业名称
    if industry1 and industry2:
        if industry1 == industry2:
            return industry1
        else:
            return f"{industry1}_{industry2}"  # 跨行业配对
    elif industry1:
        return f"{industry1}_单"
    elif industry2:
        return f"{industry2}_单"
    else:
        return "未知行业"

# ==================== 动量因子计算器 ====================

class DailyDataMomentumCalculator:
    """动量因子计算器"""
    
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
        
        grouped_scores = {}
        for factor_type, weight in weight_config.items():
            type_factors = {k: v for k, v in all_factors.items() 
                          if factor_type in k or k in ['rsi', 'bb_position']}
            
            if type_factors:
                values = list(type_factors.values())
                if len(values) > 1 and np.std(values) > 0:
                    normalized = [(v - np.mean(values)) / np.std(values) for v in values]
                    grouped_scores[factor_type] = np.mean(normalized)
                else:
                    grouped_scores[factor_type] = 0
        
        total_score = 0
        total_weight = 0
        for factor_type, weight in weight_config.items():
            if factor_type in grouped_scores:
                total_score += grouped_scores[factor_type] * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0
    
    def calculate_all_factors(self, df):
        """计算所有动量因子"""
        all_factors = {}
        all_factors.update(self.calculate_price_momentum(df))
        all_factors.update(self.calculate_volume_momentum(df))
        all_factors.update(self.calculate_volatility_momentum(df))
        all_factors.update(self.calculate_technical_momentum(df))
        return all_factors
    
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

# 修改 PT_allin_v9.0.py 中的 calculate_portfolio_sharpe
def calculate_portfolio_sharpe(final_pairs):
    if not final_pairs:
        return 0
    
    # 使用实际净值计算，不用0填充
    all_returns = {}
    for pair in final_pairs:
        equity_curve = pair.get('report', {}).get('equity_curve', [])
        if not equity_curve:
            continue
        
        df = pd.DataFrame(equity_curve, columns=['date', 'value'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        initial = pair.get('allocated_capital', 0)
        if initial <= 0:
            continue
            
        # 计算日收益率
        df['return'] = df['value'].pct_change()
        daily_returns = df['return'].dropna()
        
        if len(daily_returns) < 2:
            continue
            
        pair_key = f"{pair.get('stock1', '')}_{pair.get('stock2', '')}"
        all_returns[pair_key] = daily_returns
    
    if not all_returns or len(all_returns) < 2:
        return 0
    
    # 找到所有配对的共同日期
    common_dates = None
    for returns in all_returns.values():
        if common_dates is None:
            common_dates = returns.index
        else:
            common_dates = common_dates.intersection(returns.index)
    
    if len(common_dates) < 2:
        return 0
    
    # 对齐收益率
    aligned_returns = pd.DataFrame(index=common_dates)
    for pair_key, returns in all_returns.items():
        aligned_returns[pair_key] = returns.reindex(common_dates)
    
    # 使用前向填充而不是0填充
    aligned_returns = aligned_returns.ffill()
    
    # 获取权重
    weights = []
    for pair in final_pairs:
        pair_key = f"{pair.get('stock1', '')}_{pair.get('stock2', '')}"
        if pair_key in aligned_returns.columns:
            weights.append(pair.get('pair_weight', 0))
    
    if len(weights) != len(aligned_returns.columns):
        return 0
    
    weights_array = np.array(weights)
    portfolio_returns = (aligned_returns * weights_array).sum(axis=1)
    
    if len(portfolio_returns) < 2 or portfolio_returns.std() == 0:
        return 0
    
    daily_rf = (1 + 0.02) ** (1/252) - 1
    excess_returns = portfolio_returns - daily_rf
    sharpe = excess_returns.mean() / portfolio_returns.std() * np.sqrt(252)
    
    return sharpe

# ==================== 单次回测函数（调试模式）====================

def run_single_debug_backtest(industry_map, st_stocks, bse_stocks, cfg, momentum_calculator):
    """
    单次回测改进版：验证期和样本外都从BASE_PARAMS开始，独立运行动态自适应参数
    
    参数：
    - industry_map: 行业到股票列表的映射
    - st_stocks: ST股票集合
    - bse_stocks: 北交所股票集合
    - cfg: 配置参数
    - momentum_calculator: 动量计算器
    
    返回：
    - final_pairs: 最终筛选的配对列表，包含行业信息
    """
    print("  [单次回测改进版] 开始（独立动态自适应参数）")
    print("  模式说明：验证期和样本外都从BASE_PARAMS开始，独立运行动态参数")
    
    threshold_manager = AdaptiveThresholdManager(cfg['BASE_PARAMS'])
    
    # 1. 训练期：仅筛选配对（不使用自适应参数）
    print("\n  [阶段1] 训练期配对筛选...")
    in_sample_start = pd.to_datetime(cfg['IN_SAMPLE_START'])
    in_sample_end = pd.to_datetime(cfg['IN_SAMPLE_END'])
    
    # 计算验证期分割点
    valid_start, valid_end = calculate_validation_period(cfg)
    split_date = valid_start

    in_sample_start = pd.to_datetime(cfg['IN_SAMPLE_START'])
    in_sample_end = pd.to_datetime(cfg['IN_SAMPLE_END'])
    total_days = (in_sample_end - in_sample_start).days
    split_days = int(total_days * (1 - cfg.get('VALIDATION_SPLIT_RATIO', 0.3)))
    
    train_start, train_end = in_sample_start, split_date
    valid_start, valid_end = split_date, in_sample_end
    
    print(f"  训练集: {train_start.date()} ~ {train_end.date()} ({split_days}天)")
    print(f"  验证集: {valid_start.date()} ~ {valid_end.date()} ({total_days - split_days}天)")
    
    # 调试计数器
    processed_count = 0
    skipped_no_data = 0
    skipped_no_pairs = 0
    candidate_pairs = []
    
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
            train_start, train_end,
            cfg['TOP_N_PAIRS_PER_INDUSTRY']
        )
        
        if not selected_pairs:
            skipped_no_pairs += 1
            continue
        
        # 为每个配对添加行业信息
        for pair_info in selected_pairs:
            pair_info['industry_name'] = industry_name
        
        candidate_pairs.extend(selected_pairs)
        processed_count += 1
    
    print(f"    处理完成: {processed_count}个行业有有效配对")
    print(f"    跳过统计: {skipped_no_data}个行业无数据, {skipped_no_pairs}个行业无配对")
    print(f"    初步筛选出 {len(candidate_pairs)} 对候选配对")
    
    if not candidate_pairs:
        print("  ✗ 训练期无有效配对")
        return []
    
    # 2. 验证期：从BASE_PARAMS开始，动态自适应参数回测
    print("\n  [阶段2] 验证期动态参数回测与筛选...")
    
    validation_results = []
    skipped_no_positive = 0
    
    for pair_info in candidate_pairs:
        # 关键修改：验证期从BASE_PARAMS开始，不使用训练期参数
        trader = DynamicAdaptivePairTrading(
            pair_info['stock1'], pair_info['stock2'],
            allocated_capital=1_000_000,
            slippage=cfg['SLIPPAGE'],
            base_params=cfg['BASE_PARAMS'],
            threshold_manager=threshold_manager,
            initial_adaptive_params=None  # 传入None，从BASE_PARAMS开始
        )
        
        if not trader.initialize_kalman(pair_info['data1'], pair_info['data2']):
            continue
        
        # 在验证期使用动态参数回测
        report = trader.run_backtest_with_dynamic_params(
            pair_info['data1'], pair_info['data2'],
            valid_start, valid_end
        )
        
        if report and report['total_return'] > 0:
            validation_results.append({
                'pair_info': pair_info,
                'validation_report': report,
                'param_history': report.get('param_history', []),
                'final_params': report.get('final_params', {})
            })
        else:
            skipped_no_positive += 1
    
    if not validation_results:
        print("  ✗ 验证期无正收益配对")
        return []
    
    # 按验证期收益排序
    validation_results.sort(key=lambda x: x['validation_report']['total_return'], reverse=True)
    selected_pairs = validation_results[:min(len(validation_results), cfg['FINAL_TOP_N'] * 3)]
    
    print(f"    验证期筛选出 {len(selected_pairs)} 对表现良好的配对")
    print(f"    跳过 {skipped_no_positive} 对非正收益配对")
    
    # 3. 样本外：也从BASE_PARAMS开始，独立动态自适应参数
    print("\n  [阶段3] 样本外动态参数实盘模拟...")
    out_sample_start = pd.to_datetime(cfg['OUT_SAMPLE_START'])
    out_sample_end = pd.to_datetime(cfg['OUT_SAMPLE_END'])
    
    final_pairs = []
    skipped_no_report = 0
    
    for selected in selected_pairs:
        pair_info = selected['pair_info']
        
        # 关键修改：样本外也从BASE_PARAMS开始，不使用验证期结束时的参数
        final_trader = DynamicAdaptivePairTrading(
            pair_info['stock1'], pair_info['stock2'],
            allocated_capital=1_000_000,
            slippage=cfg['SLIPPAGE'],
            base_params=cfg['BASE_PARAMS'],
            threshold_manager=threshold_manager,
            initial_adaptive_params=None  # 传入None，从BASE_PARAMS重新开始
        )
        
        if not final_trader.initialize_kalman(pair_info['data1'], pair_info['data2']):
            skipped_no_report += 1
            continue
        
        # 样本外独立动态参数回测
        final_report = final_trader.run_backtest_with_dynamic_params(
            pair_info['data1'], pair_info['data2'],
            out_sample_start, out_sample_end
        )
        
        if final_report:
            # 为最终配对添加行业信息
            final_pair = {
                'stock1': pair_info['stock1'],
                'stock2': pair_info['stock2'],
                'industry_name': pair_info.get('industry_name', '未知行业'),
                'report': final_report,
                'validation_report': selected['validation_report'],
                'param_history': final_report.get('param_history', []),
                'final_params': final_report.get('final_params', {}),
                'data1': pair_info['data1'],
                'data2': pair_info['data2']
            }
            final_pairs.append(final_pair)
        else:
            skipped_no_report += 1
    
    print(f"    样本外回测完成: {len(final_pairs)} 对配对有回测报告")
    print(f"    跳过 {skipped_no_report} 对无回测报告配对")
    
    # 4. 行业分析和统计
    if final_pairs:
        print("\n  [阶段4] 行业分析与统计...")
        
        # 按行业分组统计
        industry_stats = {}
        industry_performance = {}
        
        for pair in final_pairs:
            industry = pair.get('industry_name', '未知行业')
            if industry not in industry_stats:
                industry_stats[industry] = 0
                industry_performance[industry] = {
                    'returns': [],
                    'sharpe': [],
                    'num_pairs': 0
                }
            
            industry_stats[industry] += 1
            industry_performance[industry]['returns'].append(pair['report']['total_return'])
            industry_performance[industry]['sharpe'].append(pair['report']['sharpe_ratio'])
            industry_performance[industry]['num_pairs'] += 1
        
        # 输出行业统计
        print(f"    行业分布统计 ({len(industry_stats)} 个行业):")
        for industry, count in sorted(industry_stats.items(), key=lambda x: -x[1]):
            perf = industry_performance[industry]
            avg_return = np.mean(perf['returns']) * 100 if perf['returns'] else 0
            avg_sharpe = np.mean(perf['sharpe']) if perf['sharpe'] else 0
            print(f"    {industry}: {count} 对配对 | 平均收益 {avg_return:.1f}% | 平均夏普 {avg_sharpe:.2f}")
    
    # 5. 输出统计信息
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
        
        # 打印参数变化情况
        print(f"    参数更新情况:")
        param_change_samples = min(3, len(final_pairs))
        for i in range(param_change_samples):
            pair = final_pairs[i]
            param_history = pair.get('param_history', [])
            if param_history:
                initial_entry = param_history[0]['params']['entry_threshold'] if 'entry_threshold' in param_history[0]['params'] else 0
                final_entry = pair['final_params'].get('entry_threshold', 0)
                print(f"      {pair['stock1']}-{pair['stock2']}: 入场阈值 {initial_entry:.2f} -> {final_entry:.2f}")
        
        # 行业收益统计
        industry_returns = {}
        for pair in final_pairs:
            industry = pair.get('industry_name', '未知行业')
            if industry not in industry_returns:
                industry_returns[industry] = []
            industry_returns[industry].append(pair['report']['total_return'])
        
        print(f"    行业收益统计:")
        for industry, returns in industry_returns.items():
            avg_return = np.mean(returns) * 100
            print(f"      {industry}: {len(returns)} 对配对，平均收益 {avg_return:.1f}%")
        for pair in final_pairs:
            pair['valid_start'] = valid_start
            pair['valid_end'] = valid_end
            pair['train_start'] = train_start
            pair['train_end'] = train_end
    
    return final_pairs

# ==================== 实盘回测模式 ====================

def run_real_time_backtest(industry_map, st_stocks, bse_stocks, cfg, momentum_calculator):
    """实盘回测模式（简化版）"""
    print("  [实盘回测模式] 开始...")
    print("  注意：实盘模式需要较长时间运行，建议先使用单次回测模式")
    
    # 实盘模式也使用动态自适应参数
    return run_single_debug_backtest(industry_map, st_stocks, bse_stocks, cfg, momentum_calculator)

# ==================== 滚动窗口回测模式 ====================

def run_rolling_window_backtest(industry_map, st_stocks, bse_stocks, cfg, momentum_calculator):
    """滚动窗口回测模式（简化版）"""
    print("  [滚动窗口回测模式] 开始...")
    print("  注意：滚动窗口模式需要较长时间运行，建议先使用单次回测模式")
    
    # 滚动窗口模式也使用动态自适应参数
    return run_single_debug_backtest(industry_map, st_stocks, bse_stocks, cfg, momentum_calculator)

# ==================== 主函数 ====================

def run_portfolio_backtest_v9_0():
    """v9.0主函数：支持动态自适应参数"""
    cfg = CONFIG

    print("="*80)
    print("A股全行业配对交易策略 v9.0 - 动态自适应参数版")
    print(f"初始资金: {cfg['PORTFOLIO_CAPITAL']:,.0f}")
    print(f"回测模式: {cfg['BACKTEST_MODE']}")
    print(f"参数更新频率: 每{cfg['ADAPTIVE_UPDATE_FREQ']}个交易日")
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
        print(f"\n[2/6] 滚动窗口回测模式...")
        final_pairs = run_rolling_window_backtest(industry_map, st_stocks, bse_stocks, cfg, momentum_calculator)
    
    elif backtest_mode == 'real_time':
        print(f"\n[2/6] 实盘回测模式...")
        final_pairs = run_real_time_backtest(industry_map, st_stocks, bse_stocks, cfg, momentum_calculator)
    
    else:  # 'single_debug'
        print(f"\n[2/6] 单次回测调试模式...")
        final_pairs = run_single_debug_backtest(industry_map, st_stocks, bse_stocks, cfg, momentum_calculator)
    
    if not final_pairs:
        print("✗ 无有效配对")
        return
    
    # 3. 行业分析和权重分配
    print(f"\n[3/6] 动量因子分析和权重分配...")
    # 获取验证期时间
    valid_start, valid_end = calculate_validation_period(cfg)
    print(f"    使用配对中的验证期: {valid_start.date()} 到 {valid_end.date()}")

    
    # 计算行业平均指标
    industry_metrics = {}
    for pair in final_pairs:
        # 使用正确的行业标识函数，已有，无需重新计算
        industry_name = pair.get('industry_name', '未知行业')
        if industry_name == "未知行业":
            industry_name = get_industry_for_pair(pair['stock1'], pair['stock2'], industry_map)

        if industry_name not in industry_metrics:
            industry_metrics[industry_name] = {
                'returns': [],
                'sharpe': [],
                'volatility': [],
                'momentum_scores': []
            }
        
        report = pair.get('report', {})
        validation_report = pair.get('validation_report', {})
        final_params = pair.get('final_params', {})
        # 获取验证期收益率
        validation_return = validation_report.get('total_return', 0)
        # 计算验证期交易日数
        data1 = pair.get('data1')
        data2 = pair.get('data2')
        valid_trading_days = 0
        if data1 is not None and data2 is not None:
            # 从股票数据获取实际交易日
            dates1 = data1.loc[valid_start:valid_end].index
            dates2 = data2.loc[valid_start:valid_end].index
            common_dates = set(dates1).intersection(set(dates2))
            valid_trading_days = len(common_dates)
        # 如果无法获取实际交易日，使用近似
        if valid_trading_days <= 0:
            valid_days = (valid_end - valid_start).days
            valid_trading_days = int(valid_days * 0.7)
        # 计算年化收益率
        if valid_trading_days > 0:
            annual_return = (1 + validation_return) ** (252/valid_trading_days) - 1
        else:
            annual_return = validation_return


        industry_metrics[industry_name]['returns'].append(annual_return)  # 存储年化收益
        industry_metrics[industry_name]['sharpe'].append(report.get('sharpe_ratio', 0))
        industry_metrics[industry_name]['volatility'].append(final_params.get('volatility', 0))
        
        # 计算动量
        
        try:
            if data1 is not None and data2 is not None:
                validation_data1 = data1.loc[valid_start:valid_end] if valid_start else data1.iloc[-120:]
                validation_data2 = data2.loc[valid_start:valid_end] if valid_start else data2.iloc[-120:]
                # 确保有足够数据
                if len(validation_data1) >= 20 and len(validation_data2) >= 20:
                    # 可以使用自定义权重配置
                    weight_config = {
                        'price_momentum': 0.4,
                        'volume_momentum': 0.2,
                        'volatility_momentum': 0.2,
                        'technical_momentum': 0.2
                    }
            
                    momentum1 = momentum_calculator.get_momentum_score(validation_data1, weight_config)
                    momentum2 = momentum_calculator.get_momentum_score(validation_data2, weight_config)
                    momentum = (momentum1 + momentum2) / 2
                else:
                    # 数据不足，使用简单方法
                    momentum1 = momentum_calculator.get_momentum_score(validation_data1)
                    momentum2 = momentum_calculator.get_momentum_score(validation_data2)
                    momentum = (momentum1 + momentum2) / 2
            else:
                momentum = 0

            industry_metrics[industry_name]['momentum_scores'].append(momentum)
        except Exception as e:
            print(f"    动量计算失败 {pair['stock1']}-{pair['stock2']}: {e}")
            industry_metrics[industry_name]['momentum_scores'].append(0)
    
    # 用验证集进行权重分配，所以计算年化收益率时应基于验证期，而不是样本外
    processed_industry_metrics = {}
    for industry, metrics in industry_metrics.items():
        if metrics['returns']:

            processed_industry_metrics[industry] = {
                'expected_return': np.mean(metrics['returns']), # 传递的是年化收益率
                'sharpe': np.mean(metrics['sharpe']),
                'volatility': np.mean(metrics['volatility']), # 传递的是年化波动率
                'momentum': np.mean(metrics['momentum_scores']) if metrics['momentum_scores'] else 0
            }
            # 调试输出
            print(f"    行业{industry}: 年化收益{np.mean(metrics['returns'])*100:.1f}%, "
                  f"波动率{np.mean(metrics['volatility'])*100:.1f}%, 夏普{np.mean(metrics['sharpe']):.2f}")
    
    # === 新增：详细的调试代码（开始）===
    print(f"\n[3.5/6] 调试: 详细检查行业指标...")
    # 检查每个行业的指标计算
    for industry, metrics in industry_metrics.items():
        print(f"\n    === 行业: {industry} ===")
        # 检查原始数据
        print(f"    原始指标数据:")
        print(f"      returns列表长度: {len(metrics['returns'])}")
        if metrics['returns']:
            print(f"      returns值: {[f'{x:.4f}' for x in metrics['returns']]}")
        else:
            print(f"      returns值: 空列表")
        print(f"      volatility列表长度: {len(metrics['volatility'])}")
        if metrics['volatility']:
            print(f"      volatility值: {[f'{x:.4f}' for x in metrics['volatility']]}")
        else:
            print(f"      volatility值: 空列表")
        
        print(f"      sharpe列表长度: {len(metrics['sharpe'])}")
        if metrics['sharpe']:
            print(f"      sharpe值: {[f'{x:.2f}' for x in metrics['sharpe']]}")
        else:
            print(f"      sharpe值: 空列表")

        print(f"      momentum_scores列表长度: {len(metrics['momentum_scores'])}")
        if metrics['momentum_scores']:
            print(f"      momentum_scores值: {[f'{x:.2f}' for x in metrics['momentum_scores']]}")
        else:
            print(f"      momentum_scores值: 空列表")
        
        # 计算平均值
        if metrics['returns']:
            avg_return = np.mean(metrics['returns'])
            avg_vol = np.mean(metrics['volatility'])
            avg_sharpe = np.mean(metrics['sharpe'])
            avg_momentum = np.mean(metrics['momentum_scores']) if metrics['momentum_scores'] else 0

            print(f"    计算平均值:")
            print(f"      平均收益: {avg_return:.4f} ({avg_return*100:.1f}%)")
            print(f"      平均波动率: {avg_vol:.4f} ({avg_vol*100:.1f}%)")
            print(f"      平均夏普: {avg_sharpe:.2f}")
            print(f"      平均动量: {avg_momentum:.2f}")

            # 计算年化收益率的正确性检查
            print(f"    年化收益率计算检查:")
            print(f"      平均收益{avg_return:.4f}是否大于0? {avg_return > 0}")
            print(f"      5%年化收益率门槛: {cfg['RISK_ADJUST_PARAMS']['min_industry_return']}")
            print(f"      是否通过门槛? {avg_return >= cfg['RISK_ADJUST_PARAMS']['min_industry_return']}")
        else:
            print(f"    警告: 该行业无有效数据!")
    # 检查processed_industry_metrics
    print(f"\n    === processed_industry_metrics检查 ===")
    if processed_industry_metrics:
        print(f"    有 {len(processed_industry_metrics)} 个行业的指标")
        for industry, metrics in processed_industry_metrics.items():
            print(f"\n    行业: {industry}")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"      {key}: {value:.6f}")
                else:
                    print(f"      {key}: {value}")
    else:
        print(f"    警告: processed_industry_metrics为空!")
    # 检查是否有负收益行业
    print(f"\n    === 负收益行业检查 ===")
    negative_industries = []
    for industry, metrics in processed_industry_metrics.items():
        expected_return = metrics.get('expected_return', 0)
        if expected_return < cfg['RISK_ADJUST_PARAMS']['min_industry_return']:
            negative_industries.append((industry, expected_return))
    if negative_industries:
        print(f"    有 {len(negative_industries)} 个行业收益低于门槛:")
        for industry, ret in negative_industries:
            print(f"      {industry}: {ret*100:.2f}%")
    else:
        print(f"    所有行业收益都满足{cfg['RISK_ADJUST_PARAMS']['min_industry_return']*100:.1f}%门槛")
    # === 新增：详细的调试代码（结束）===    

    # 使用风险调整分配器
    allocator = RiskAdjustedParityAllocator(
        risk_aversion=cfg['RISK_ADJUST_PARAMS']['risk_aversion'],
        return_boost=cfg['RISK_ADJUST_PARAMS']['return_boost'],
        min_w=cfg['RISK_ADJUST_PARAMS']['min_industry_weight'],
        max_w=cfg['RISK_ADJUST_PARAMS']['max_industry_weight'],
        min_return=cfg['RISK_ADJUST_PARAMS']['min_industry_return']
    )
    
    weights = allocator.calculate_weights(processed_industry_metrics)
    # 打印行业权重
    print("\n  行业权重分配结果:")
    for industry, weight in weights.items():
        metrics = processed_industry_metrics.get(industry, {})
        print(f"    {industry}: 权重{weight:.2%}, "
              f"年化收益{metrics.get('expected_return', 0)*100:.1f}%, "
              f"夏普{metrics.get('sharpe', 0):.2f}")
    
    # 4. 最终分配资金
    print(f"\n[4/6] 最终资金分配...")
    total_capital = cfg['PORTFOLIO_CAPITAL']

    # 按行业分组配对
    industry_pairs = {}
    for pair in final_pairs:
        industry = pair.get('industry_name', '未知行业')
        if industry not in industry_pairs:
            industry_pairs[industry] = []
        industry_pairs[industry].append(pair)

    # 按行业权重分配总资金
    industry_capital = {}
    allocated_industries = 0
    allocated_total_capital = 0
    
    for industry, industry_weight in weights.items():
        if industry in industry_pairs and industry_weight > 0:
            industry_capital[industry] = total_capital * industry_weight
            allocated_industries += 1
            allocated_total_capital += industry_capital[industry]
        else:
            industry_capital[industry] = 0
    # 如果有行业没分到资金，重新分配剩余资金
    if allocated_total_capital < total_capital and allocated_industries > 0:
        remaining_capital = total_capital - allocated_total_capital
        equal_share = remaining_capital / allocated_industries
        for industry in industry_capital:
            if industry_capital[industry] > 0:
                industry_capital[industry] += equal_share
    
    # 在行业内按验证期收益分配资金
    for pair in final_pairs:
        industry = pair.get('industry_name', '未知行业')
        industry_total_cap = industry_capital.get(industry, 0)
        
        if industry_total_cap <= 0:
            pair['allocated_capital'] = 0
            pair['pair_weight'] = 0
            continue
        # 获取该行业的所有配对
        industry_pairs_list = industry_pairs.get(industry, [])
        if len(industry_pairs_list) > 0:
            # 计算行业内验证期收益
            industry_val_returns = []
            for p in industry_pairs_list:
                validation_report = p.get('validation_report', {})
                val_return = max(validation_report.get('total_return', 0), 0)  # 只考虑正收益
                industry_val_returns.append(val_return)
            
            # 找到当前配对在行业列表中的索引
            pair_idx = -1
            for i, p in enumerate(industry_pairs_list):
                if p['stock1'] == pair['stock1'] and p['stock2'] == pair['stock2']:
                    pair_idx = i
                    break
            
            if pair_idx >= 0 and sum(industry_val_returns) > 0:
                # 按验证期收益正比分配
                pair_val_weight = industry_val_returns[pair_idx] / sum(industry_val_returns)
                pair['allocated_capital'] = industry_total_cap * pair_val_weight
                pair['pair_weight'] = weights.get(industry, 0) * pair_val_weight
            else:
                # 验证期收益全为0，平均分配
                equal_weight = 1.0 / len(industry_pairs_list)
                pair['allocated_capital'] = industry_total_cap * equal_weight
                pair['pair_weight'] = weights.get(industry, 0) * equal_weight
        else:
            pair['allocated_capital'] = 0
            pair['pair_weight'] = 0
    # 打印分配结果
    print("\n  资金分配方案:")
    for pair in final_pairs:
        cap = pair.get('allocated_capital', 0)
        if cap > 0:
            ret = pair['report']['total_return'] * 100
            print(f"    {pair['stock1']}-{pair['stock2']}: 资金{cap:,.0f} | "
                  f"收益{ret:.1f}% | 夏普{pair['report']['sharpe_ratio']:.2f}")
    
    # 打印分配结果
    print("\n  资金分配方案:")
    for pair in final_pairs:
        cap = pair.get('allocated_capital', 0)
        if cap > 0:
            ret = pair['report']['total_return'] * 100
            print(f"    {pair['stock1']}-{pair['stock2']}: 资金{cap:,.0f} | "
                  f"收益{ret:.1f}% | 夏普{pair['report']['sharpe_ratio']:.2f}")
    
    # 验证资金分配总和
    total_allocated = sum(pair.get('allocated_capital', 0) for pair in final_pairs)
    print(f"  总分配资金: {total_allocated:,.0f}/{total_capital:,.0f}")

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
        # 检查是否分配了资金
        if 'allocated_capital' not in pair or pair['allocated_capital'] <= 0:
            print(f"    跳过配对 {pair['stock1']}-{pair['stock2']}净值计算，未分配资金")
            continue
        equity_curve = pair['report']['equity_curve']
        if equity_curve:
            equity_df = pd.DataFrame(equity_curve, columns=['date', 'net_value'])
            equity_df['date'] = pd.to_datetime(equity_df['date']).dt.date
            equity_df.set_index('date', inplace=True)
            # 计算净值收益率
            initial_capital = pair['allocated_capital']
            equity_df['net_return'] = (equity_df['net_value'] - initial_capital) / initial_capital
            equity_df['cumulative_return'] = (1 + equity_df['net_return'].shift(1)).cumprod() - 1
            equity_df['cumulative_return'].fillna(0, inplace=True)
            
            equity_filename = f"{pair['stock1']}_{pair['stock2']}_净值.csv"
            equity_filepath = os.path.join(equity_curves_dir, equity_filename)
            equity_df.to_csv(equity_filepath, encoding='utf-8-sig')
            equity_files.append(equity_filepath)
            # 记录文件路径
            pair['equity_file'] = equity_filename
        else:
            pair['equity_file'] = None
    
    # 打包净值数据
    if equity_files:
        zip_filename = f"v9.0_净值数据_{timestamp}.zip"
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for equity_file in equity_files:
                arcname = os.path.basename(equity_file)
                zipf.write(equity_file, arcname)
        print(f"✓ 净值数据已打包: {zip_filename}")
    
    # 清理临时目录
    shutil.rmtree(equity_curves_dir)
    print(f"✓ 清理临时目录: {equity_curves_dir}")

    # 计算组合层面指标
    portfolio_return = np.sum([p['report']['total_return'] * p.get('pair_weight', 0) for p in final_pairs])
    portfolio_slippage = np.sum([p['report']['slippage_impact'] * p.get('pair_weight', 0) for p in final_pairs])
    portfolio_sharpe = calculate_portfolio_sharpe(final_pairs)

    print(f"\n{'='*80}")
    print("v9.0组合回测结果（动态自适应参数）")
    print(f"{'='*80}")
    print(f"回测模式: {backtest_mode}")
    print(f"初始资金: {total_capital:,.0f}")
    print(f"总配对数: {len(final_pairs)}")
    print(f"组合预期收益率: {portfolio_return*100:.2f}%")
    print(f"组合夏普比率: {portfolio_sharpe:.2f}")
    print(f"滑点成本占比: {portfolio_slippage*100:.2f}%")
    print(f"净收益率: {(portfolio_return - portfolio_slippage)*100:.2f}%")

    print(f"\n【改进点】")
    print(f"1. 验证期和样本外都启用动态自适应参数")
    print(f"2. 参数每{cfg['ADAPTIVE_UPDATE_FREQ']}个交易日更新一次")
    print(f"3. 基于最近{cfg['ADAPTIVE_LOOKBACK_DAYS']}天数据计算参数")
    print(f"4. 记录参数变化历史，便于分析")

    # 创建最终结果表格
    results_data = []
    for p in final_pairs:
        trades_df = pd.DataFrame(p['report']['trade_records'])
        if len(trades_df) > 0:
            sell_trades = trades_df[trades_df['action'] == '卖出']
            avg_hold_days = sell_trades['hold_days'].mean() if 'hold_days' in sell_trades.columns and len(sell_trades) > 0 else 0
            avg_return_per_trade = sell_trades['pnl'].mean() if 'pnl' in sell_trades.columns and len(sell_trades) > 0 else 0
        else:
            avg_hold_days = 0
            avg_return_per_trade = 0

        # 获取最终参数
        final_params = p.get('final_params', {})
        
        results_data.append({
            '行业': p.get('industry_name', '未知行业'),
            '股票1': p['stock1'],
            '股票2': p['stock2'],
            '分配资金': p.get('allocated_capital', 0),
            '配对权重': f"{p.get('pair_weight', 0):.2%}",
            '总收益率': f"{p['report']['total_return']:.2%}",
            '验证期收益率': f"{p.get('validation_report', {}).get('total_return', 0):.2%}",
            '夏普比率': f"{p['report']['sharpe_ratio']:.2f}",
            '最大回撤': f"{p['report']['max_drawdown']:.2%}",
            '胜率': f"{p['report']['win_rate']:.1%}",
            '盈亏比': f"{p['report']['profit_loss_ratio']:.2f}",
            '交易次数': p['report']['num_trades'],
            '平均持仓天数': f"{avg_hold_days:.1f}",
            '单次交易平均收益': f"{avg_return_per_trade:.2f}",
            '滑点成本占比': f"{p['report']['slippage_impact']:.2%}",
            '波动率': f"{final_params.get('volatility', 0):.2%}",
            '半衰期': f"{final_params.get('halflife', 0):.1f}天",
            '入场阈值': f"{final_params.get('entry_threshold', 0):.2f}",
            '出场阈值': f"{final_params.get('exit_threshold', 0):.2f}",
            '止损阈值': f"{final_params.get('stop_loss', 0):.2f}",
            '最大持仓天数': final_params.get('max_holding_days', 0),
            '净值文件': p.get('equity_file', 'N/A'),
            '冷却次数': p['report']['cooldown_count']
        })
    
    results_df = pd.DataFrame(results_data)
    
    # 保存结果文件
    results_file = f"v9.0_组合回测结果_{timestamp}.csv"
    results_df.to_csv(results_file, index=False, encoding='utf-8-sig', float_format='%.4f')
    print(f"✓ 结果已保存: {results_file}")

    # 生成详细交易记录汇总
    trade_details = []
    for p in final_pairs:
        trades = p['report']['trade_records']
        for trade in trades:
            trade_details.append({
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
        trade_file = f"v9.0_交易明细_{timestamp}.csv"
        trade_details_df.to_csv(trade_file, index=False, encoding='utf-8-sig')
        print(f"✓ 交易明细已保存: {trade_file}")

    # 生成参数变化历史
    param_changes = []
    for p in final_pairs:
        param_history = p.get('param_history', [])
        for param in param_history:
            param_changes.append({
                '股票1': p['stock1'],
                '股票2': p['stock2'],
                '更新日期': param.get('date', ''),
                '入场阈值': param.get('params', {}).get('entry_threshold', 0),
                '出场阈值': param.get('params', {}).get('exit_threshold', 0),
                '止损阈值': param.get('params', {}).get('stop_loss', 0),
                '最大持仓天数': param.get('params', {}).get('max_holding_days', 0),
                '波动率': param.get('params', {}).get('volatility', 0),
                '半衰期': param.get('params', {}).get('halflife', 0)
            })
    
    if param_changes:
        param_df = pd.DataFrame(param_changes)
        param_file = f"v9.0_参数变化历史_{timestamp}.csv"
        param_df.to_csv(param_file, index=False, encoding='utf-8-sig')
        print(f"✓ 参数变化历史已保存: {param_file}")

    # 保存配置
    config_file = f"v9.0_回测配置_{timestamp}.json"
    with open(config_file, 'w') as f:
        json.dump({k: str(v) if isinstance(v, (datetime, pd.Timestamp)) else v 
                  for k, v in cfg.items()}, f, indent=2, default=str)
    print(f"✓ 配置已保存: {config_file}")

    print(f"\n{'='*80}")
    print("文件生成完成:")
    print(f"1. 组合结果: {results_file}")
    if 'zip_filename' in locals():
        print(f"2. 净值数据(ZIP): {zip_filename}")
    if 'trade_file' in locals():
        print(f"3. 交易明细: {trade_file}")
    if 'param_file' in locals():
        print(f"4. 参数变化历史: {param_file}")
    print(f"5. 配置文件: {config_file}")
    print(f"{'='*80}")

# ==================== 主程序入口 ====================

if __name__ == "__main__":
    run_portfolio_backtest_v9_0()