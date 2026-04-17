# === A股配对交易策略 v5.8（完整版）===
# === 配对特异性参数优化版 ===

import struct
import pandas as pd
import numpy as np
import os
import warnings
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from itertools import combinations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# 第一部分：基础函数（从v5.5/v5.6/v5.7继承）
# =============================================================================

def get_industry_stocks_local(txt_path, target_industry):
    """从通达信数据目录读取行业成分股"""
    industry_stocks = []
    try:
        with open(txt_path, 'r', encoding='gbk') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    industry_name = parts[1]
                    stock_code = parts[2]
                    if industry_name == target_industry:
                        industry_stocks.append(stock_code)
        print(f"从本地文件获取到 {len(industry_stocks)} 只{target_industry}成分股")
        return industry_stocks
    except Exception as e:
        print(f"读取本地文件失败：{e}")
        return []


def get_stock_data_from_tdx(stock_code, data_dir):
    """从通达信数据目录读取股票日线数据"""
    try:
        if stock_code.startswith(('600', '601', '603', '605', '688')):
            prefix = 'sh'
            sub_dir = 'sh'
        elif stock_code.startswith(('000', '001', '002', '003', '300', '301')):
            prefix = 'sz'
            sub_dir = 'sz'
        elif stock_code.startswith(('9')):
            prefix = 'bj'
            sub_dir = 'bj'
        else:
            print(f"未知市场代码: {stock_code}")
            return None

        file_name = f"{prefix}{stock_code}.day"
        base_path = os.path.dirname(data_dir)
        possible_paths = [
            os.path.join(data_dir, sub_dir, "lday", file_name),
            os.path.join(data_dir, file_name),
            os.path.join(base_path, "vipdoc", sub_dir, "lday", file_name)
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
                date, open_price, high, low, close, volume, amount, _ = struct.unpack('IIIIIIII', chunk)
                date_str = str(date)
                if len(date_str) != 8:
                    continue
                year = int(date_str[:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                data.append([datetime(year, month, day), open_price / 100.0, high / 100.0, low / 100.0, close / 100.0, volume, amount])

        if not data:
            return None

        df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'amount'])
        df.set_index('date', inplace=True)
        return df
    except Exception as e:
        print(f"读取{stock_code}数据时出错: {e}")
        return None


def adf_test(series):
    """ADF检验：检验序列是否平稳（来自v5.5的严格版本）"""
    series = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()

    if len(series) < 10:
        return 1.0  # 数据不足，返回不显著

    if series.std() == 0 or np.isnan(series.std()) or series.std() < 1e-10:
        return 1.0  # 常数序列

    try:
        result = adfuller(series, maxlag=1)
        return result[1]
    except Exception as e:
        return 1.0  # 异常时返回不显著


class KalmanFilterPairTrading:
    """卡尔曼滤波器 - 修复数值稳定性（来自v5.6）"""

    def __init__(self, initial_state=[0.0, 0.0], initial_covariance=1000.0, 
                 process_noise=1e-5, observation_noise=1e-3):
        self.state = np.array(initial_state, dtype=float)
        self.covariance = np.array([[initial_covariance, 0], [0, initial_covariance]])
        self.process_noise = np.array([[process_noise, 0], [0, process_noise]])
        self.observation_noise = observation_noise

    def update(self, y1, y2):
        """执行一次卡尔曼滤波更新"""
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


# =============================================================================
# 第二部分：v5.8核心创新 - 配对特异性参数优化
# =============================================================================

@dataclass
class PairSpecificParams:
    """配对特异性参数容器"""
    entry_threshold: float = 1.2
    exit_threshold: float = 0.3
    stop_loss: float = 2.5
    max_holding_days: int = 15
    rebalance_threshold: float = 0.6
    coint_check_freq: int = 30
    coint_p_threshold: float = 0.1
    base_cooling_days: int = 2
    max_cooldown: int = 10
    max_position_ratio: float = 0.6

    # 样本内统计特征
    in_sample_volatility: float = 0.0
    in_sample_r2: float = 0.0
    in_sample_coint_p: float = 1.0
    mean_reversion_speed: float = 0.0

    def to_dict(self):
        return {
            'entry_threshold': self.entry_threshold,
            'exit_threshold': self.exit_threshold,
            'stop_loss': self.stop_loss,
            'max_holding_days': self.max_holding_days,
            'rebalance_threshold': self.rebalance_threshold,
            'coint_check_freq': self.coint_check_freq,
            'coint_p_threshold': self.coint_p_threshold,
            'base_cooling_days': self.base_cooling_days,
            'max_cooldown': self.max_cooldown,
            'max_position_ratio': self.max_position_ratio,
            'in_sample_volatility': self.in_sample_volatility,
            'in_sample_r2': self.in_sample_r2,
            'in_sample_coint_p': self.in_sample_coint_p,
            'mean_reversion_speed': self.mean_reversion_speed
        }


def calculate_mean_reversion_speed(spread_series: pd.Series) -> float:
    """
    计算均值回归速度（Ornstein-Uhlenbeck过程参数）

    使用AR(1)模型：Δy_t = α + β*y_{t-1} + ε_t
    均值回归速度 = -ln(1+β) ≈ -β （当β<0时）
    """
    spread = spread_series.dropna()
    if len(spread) < 30:
        return 0.0

    y_lag = spread.shift(1).dropna()
    y_diff = spread.diff().dropna()

    common_idx = y_lag.index.intersection(y_diff.index)
    y_lag = y_lag.loc[common_idx]
    y_diff = y_diff.loc[common_idx]

    if len(y_lag) < 20:
        return 0.0

    try:
        X = sm.add_constant(y_lag)
        model = sm.OLS(y_diff, X).fit()
        beta = model.params[1] if len(model.params) > 1 else 0

        if beta < 0:
            speed = -beta
            return min(speed, 1.0)
        return 0.0
    except:
        return 0.0


def calculate_optimal_thresholds(spread_series: pd.Series, 
                                  z_score_series: pd.Series,
                                  in_sample_coint_p: float) -> Dict[str, float]:
    """
    基于样本内数据计算最优阈值
    """
    volatility = spread_series.std()
    mr_speed = calculate_mean_reversion_speed(spread_series)

    # 基础阈值
    base_entry = 1.2
    base_exit = 0.3

    # 根据波动率调整
    vol_factor = volatility / 0.05 if volatility > 0 else 1.0

    if vol_factor > 1.5:  # 高波动
        entry_threshold = base_entry * min(vol_factor, 2.0)
        exit_threshold = base_exit * 1.2
    elif vol_factor < 0.8:  # 低波动
        entry_threshold = base_entry * 0.9
        exit_threshold = base_exit * 0.8
    else:
        entry_threshold = base_entry
        exit_threshold = base_exit

    # 根据均值回归速度调整
    if mr_speed > 0.1:  # 快速均值回归
        rebalance_threshold = 0.5
        max_holding_days = 12
    elif mr_speed < 0.03:  # 慢速均值回归
        rebalance_threshold = 0.8
        max_holding_days = 25
    else:
        rebalance_threshold = 0.6
        max_holding_days = 15

    # 根据协整显著性调整止损
    if in_sample_coint_p < 0.01:  # 高度显著
        stop_loss = 3.0
        coint_p_threshold = 0.05
    elif in_sample_coint_p < 0.05:  # 显著
        stop_loss = 2.5
        coint_p_threshold = 0.08
    else:  # 边缘显著
        stop_loss = 2.0
        coint_p_threshold = 0.12

    return {
        'entry_threshold': round(entry_threshold, 2),
        'exit_threshold': round(exit_threshold, 2),
        'stop_loss': round(stop_loss, 2),
        'max_holding_days': max_holding_days,
        'rebalance_threshold': round(rebalance_threshold, 2),
        'coint_p_threshold': coint_p_threshold,
        'mean_reversion_speed': round(mr_speed, 4)
    }


def generate_pair_specific_params(pair_info: Dict, 
                                   stock1_data: pd.DataFrame,
                                   stock2_data: pd.DataFrame,
                                   in_sample_start: str,
                                   in_sample_end: str) -> PairSpecificParams:
    """为单个配对生成特异性参数"""
    # 获取样本内数据
    s1_sample = stock1_data.loc[in_sample_start:in_sample_end, 'close']
    s2_sample = stock2_data.loc[in_sample_start:in_sample_end, 'close']

    # 计算对数价格和价差
    log_s1 = np.log(s1_sample.replace(0, np.nan).dropna())
    log_s2 = np.log(s2_sample.replace(0, np.nan).dropna())

    common_dates = log_s1.index.intersection(log_s2.index)
    log_s1 = log_s1.loc[common_dates]
    log_s2 = log_s2.loc[common_dates]

    # 使用样本内对冲比率计算价差
    hedge_ratio = pair_info['hedge_ratio']
    intercept = pair_info['intercept']
    spread = log_s1 - (intercept + hedge_ratio * log_s2)

    # 计算Z-Score序列
    z_score = (spread - spread.mean()) / spread.std()

    # 计算最优阈值
    optimal = calculate_optimal_thresholds(spread, z_score, pair_info['coint_p'])

    # 创建参数对象
    params = PairSpecificParams(
        entry_threshold=optimal['entry_threshold'],
        exit_threshold=optimal['exit_threshold'],
        stop_loss=optimal['stop_loss'],
        max_holding_days=optimal['max_holding_days'],
        rebalance_threshold=optimal['rebalance_threshold'],
        coint_check_freq=30,
        coint_p_threshold=optimal['coint_p_threshold'],
        base_cooling_days=2,
        max_cooldown=10,
        max_position_ratio=0.6,
        in_sample_volatility=round(spread.std(), 4),
        in_sample_r2=pair_info['r_squared'],
        in_sample_coint_p=pair_info['coint_p'],
        mean_reversion_speed=optimal['mean_reversion_speed']
    )

    return params


# =============================================================================
# 第三部分：v5.8策略类
# =============================================================================

class ASharePairsTradingKalmanV5_8:
    """v5.8: 配对特异性参数优化版"""

    def __init__(self, pair_params: PairSpecificParams, initial_capital=1000000,
                 commission_rate=0.0003, stamp_tax_rate=0.001):

        self.params = pair_params
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate

        # 运行时状态
        self.kf = None
        self.pair_valid = True
        self.cooling_period = False
        self.cooldown_count = 0
        self.last_check_date = None
        self.coint_check_count = 0
        self.consecutive_fails = 0

        self.positions = {}
        self.holding_stock = None
        self.entry_date = None
        self.entry_z = 0
        self.holding_days = 0

        self.trade_records = []
        self.daily_records = []
        self.equity_curve = []

        self.stock1 = None
        self.stock2 = None

    def calculate_position_size(self, price, target_ratio=None):
        if target_ratio is None:
            target_ratio = self.params.max_position_ratio

        available_capital = self.cash * target_ratio
        commission_factor = (1 + self.commission_rate + self.stamp_tax_rate * 0.5)
        max_amount = available_capital / commission_factor

        if price <= 0 or np.isnan(price):
            return 0

        shares = int(max_amount / price / 100) * 100
        return max(shares, 0)

    def execute_buy(self, date, stock_code, price, reason=""):
        shares = self.calculate_position_size(price)
        if shares < 100:
            return False, f"计算股数不足100股"

        amount = price * shares
        commission = max(amount * self.commission_rate, 5)
        total_cost = amount + commission

        if total_cost > self.cash:
            max_shares = int(self.cash / price / 100) * 100
            if max_shares < 100:
                return False, "资金不足"
            shares = max_shares
            amount = price * shares
            commission = max(amount * self.commission_rate, 5)
            total_cost = amount + commission

        self.cash -= total_cost
        self.positions[stock_code] = {
            'qty': shares,
            'avg_price': price,
            'entry_date': date
        }
        self.holding_stock = stock_code

        record = {
            'date': date,
            'action': '买入',
            'stock': stock_code,
            'price': price,
            'shares': shares,
            'amount': amount,
            'commission': commission,
            'cash_after': self.cash,
            'reason': reason
        }
        self.trade_records.append(record)
        print(f"    [交易] {date.strftime('%Y-%m-%d')} 买入 {stock_code} {shares}股 @ {price:.2f}, 原因: {reason}")
        return True, "成功"

    def execute_sell(self, date, stock_code, price, reason=""):
        if stock_code not in self.positions:
            return False, "无持仓"

        pos = self.positions[stock_code]
        shares = pos['qty']

        amount = price * shares
        commission = max(amount * self.commission_rate, 5)
        stamp_tax = amount * self.stamp_tax_rate
        total_cost = commission + stamp_tax
        net_proceeds = amount - total_cost

        pnl = (price - pos['avg_price']) * shares - total_cost

        self.cash += net_proceeds
        del self.positions[stock_code]
        self.holding_stock = None

        record = {
            'date': date,
            'action': '卖出',
            'stock': stock_code,
            'price': price,
            'shares': shares,
            'amount': amount,
            'commission': commission,
            'stamp_tax': stamp_tax,
            'pnl': pnl,
            'cash_after': self.cash,
            'reason': reason,
            'hold_days': (date - pos['entry_date']).days
        }
        self.trade_records.append(record)
        print(f"    [交易] {date.strftime('%Y-%m-%d')} 卖出 {stock_code} {shares}股 @ {price:.2f}, 盈亏: {pnl:,.2f}, 原因: {reason}")
        return True, "成功"

    def calculate_portfolio_value(self, date, price1, price2):
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

            if len(log_s1) > 1 and self.kf is not None:
                for i in range(len(log_s1)):
                    self.kf.update(log_s1.iloc[i], log_s2.iloc[i])

            mu, gamma = self.kf.state if self.kf else (0, 1)

            spread = log_s1 - (mu + gamma * log_s2)
            spread = spread.replace([np.inf, -np.inf], np.nan).dropna()

            if len(spread) < window:
                return None, None, None, None

            recent_spread = spread[-window:]
            mean_spread = recent_spread.mean()
            std_spread = recent_spread.std()

            if std_spread == 0 or np.isnan(std_spread) or std_spread < 1e-10:
                return None, None, None, None

            z_score = (spread.iloc[-1] - mean_spread) / std_spread

            return spread, z_score, mean_spread, std_spread

        except Exception as e:
            return None, None, None, None

    def check_cointegration(self, stock1_data, stock2_data, current_date, window=120):
        """使用配对特异性阈值进行协整检验"""
        try:
            s1 = stock1_data.loc[:current_date].iloc[-window:]
            s2 = stock2_data.loc[:current_date].iloc[-window:]

            if len(s1) < 60 or len(s2) < 60:
                return True, 0.01

            log_s1 = np.log(s1.replace(0, np.nan).dropna())
            log_s2 = np.log(s2.replace(0, np.nan).dropna())

            common_dates = log_s1.index.intersection(log_s2.index)
            if len(common_dates) < 60:
                return True, 0.01

            log_s1 = log_s1.loc[common_dates]
            log_s2 = log_s2.loc[common_dates]

            mu, gamma = self.kf.state if self.kf else (0, 1)
            residual = log_s1 - (mu + gamma * log_s2)
            residual = residual.replace([np.inf, -np.inf], np.nan).dropna()

            if len(residual) < 30:
                return True, 0.01

            if residual.std() < 1e-10 or np.isnan(residual.std()):
                return True, 0.01

            adf_p = adf_test(residual)

            # 使用配对特异性阈值
            is_coint = adf_p < self.params.coint_p_threshold

            return is_coint, adf_p

        except Exception as e:
            return True, 0.01

    def run_backtest(self, stock1, stock2, stock1_data, stock2_data, 
                     in_sample_start, in_sample_end, out_sample_start, out_sample_end):
        """执行回测 - 使用配对特异性参数"""
        self.stock1 = stock1
        self.stock2 = stock2

        print(f"\n{'='*60}")
        print(f"回测配对: {stock1} vs {stock2}")
        print(f"配对特异性参数:")
        print(f"  入场阈值: {self.params.entry_threshold}")
        print(f"  出场阈值: {self.params.exit_threshold}")
        print(f"  止损阈值: {self.params.stop_loss}")
        print(f"  最大持仓: {self.params.max_holding_days}天")
        print(f"  轮动阈值: {self.params.rebalance_threshold}")
        print(f"  协整阈值: {self.params.coint_p_threshold}")
        print(f"  均值回归速度: {self.params.mean_reversion_speed}")
        print(f"{'='*60}")

        # 初始化卡尔曼滤波器
        try:
            sample_data1 = stock1_data.loc[in_sample_start:in_sample_end]
            sample_data2 = stock2_data.loc[in_sample_start:in_sample_end]

            log_s1 = np.log(sample_data1['close'].iloc[-100:])
            log_s2 = np.log(sample_data2['close'].iloc[-100:])

            X = sm.add_constant(log_s2)
            model = sm.OLS(log_s1, X).fit()
            initial_mu = model.params[0]
            initial_gamma = model.params[1]

            self.kf = KalmanFilterPairTrading(
                initial_state=[float(initial_mu), float(initial_gamma)],
                initial_covariance=1000.0,
                process_noise=1e-5,
                observation_noise=1e-3
            )

        except Exception as e:
            print(f"初始化失败: {e}")
            return None

        out_data1 = stock1_data.loc[out_sample_start:out_sample_end]
        out_data2 = stock2_data.loc[out_sample_start:out_sample_end]

        trading_dates = out_data1.index.intersection(out_data2.index)

        if len(trading_dates) == 0:
            print("无交易日数据")
            return None

        print(f"样本外交易日数量: {len(trading_dates)}")

        # 预热期
        for i in range(min(20, len(trading_dates))):
            current_date = trading_dates[i]
            try:
                self.calculate_spread_zscore(out_data1['close'], out_data2['close'], current_date)
            except:
                pass

        # 主回测循环
        for i, current_date in enumerate(trading_dates):
            try:
                price1 = out_data1.loc[current_date, 'close']
                price2 = out_data2.loc[current_date, 'close']
            except:
                continue

            # 冷却期检查
            if self.cooling_period:
                if hasattr(self, 'cooling_end_date') and self.cooling_end_date and current_date >= self.cooling_end_date:
                    self.cooling_period = False
                    print(f"  [{current_date.strftime('%Y-%m-%d')}] 冷却期结束")
                else:
                    total_value = self.calculate_portfolio_value(current_date, price1, price2)
                    self.daily_records.append({
                        'date': current_date,
                        'cash': self.cash,
                        'total_value': total_value,
                        'holding': self.holding_stock,
                        'z_score': np.nan,
                        'price1': price1,
                        'price2': price2,
                        'hedge_ratio': self.kf.state[1] if self.kf else np.nan,
                        'status': 'cooling'
                    })
                    self.equity_curve.append((current_date, total_value))
                    continue

            # 协整检验（使用配对特异性频率和阈值）
            days_since_last = (current_date - self.last_check_date).days if self.last_check_date else 999
            if days_since_last >= self.params.coint_check_freq:
                is_coint, adf_p = self.check_cointegration(out_data1['close'], out_data2['close'], current_date)
                self.coint_check_count += 1
                self.last_check_date = current_date

                print(f"  [{current_date.strftime('%Y-%m-%d')}] 协整检验 p={adf_p:.4f} (阈值{self.params.coint_p_threshold}) {'✓' if is_coint else '✗'}")

                if not is_coint:
                    self.consecutive_fails += 1
                    if self.consecutive_fails >= 3:
                        print(f"    ⚠️ 连续{self.consecutive_fails}次失败，进入冷却")
                        if self.holding_stock:
                            self.execute_sell(current_date, self.holding_stock, 
                                            price1 if self.holding_stock == stock1 else price2,
                                            "协整破裂强制平仓")

                        self.cooldown_count += 1
                        self.cooling_period = True
                        cooldown_days = min(self.params.base_cooling_days + self.cooldown_count * 2, 
                                          self.params.max_cooldown)
                        self.cooling_end_date = current_date + timedelta(days=cooldown_days)
                        self.consecutive_fails = 0

                        if self.cooldown_count >= 5:
                            print(f"    ❌ 配对永久失效")
                            self.pair_valid = False
                            break
                        continue
                else:
                    self.consecutive_fails = 0

            # 计算Z-Score
            spread, z_score, mean_spread, std_spread = self.calculate_spread_zscore(
                out_data1['close'], out_data2['close'], current_date
            )

            current_hedge = self.kf.state[1] if self.kf else np.nan

            if z_score is None:
                total_value = self.calculate_portfolio_value(current_date, price1, price2)
                self.daily_records.append({
                    'date': current_date,
                    'cash': self.cash,
                    'total_value': total_value,
                    'holding': self.holding_stock,
                    'z_score': np.nan,
                    'price1': price1,
                    'price2': price2,
                    'hedge_ratio': current_hedge,
                    'status': 'no_data'
                })
                self.equity_curve.append((current_date, total_value))
                continue

            total_value = self.calculate_portfolio_value(current_date, price1, price2)

            # 使用配对特异性参数的交易逻辑
            entry_th = self.params.entry_threshold
            exit_th = self.params.exit_threshold
            stop_th = self.params.stop_loss
            max_hold = self.params.max_holding_days
            reb_th = self.params.rebalance_threshold

            if self.holding_stock is None:
                # 入场判断
                if abs(z_score) > entry_th and self.pair_valid:
                    if z_score > entry_th:
                        success, msg = self.execute_buy(current_date, stock2, price2, 
                                                      f"Z={z_score:.2f}>{entry_th}，买入低估方{stock2}")
                    else:
                        success, msg = self.execute_buy(current_date, stock1, price1,
                                                      f"Z={z_score:.2f}<-{entry_th}，买入低估方{stock1}")

                    if success:
                        self.entry_z = z_score
                        self.entry_date = current_date
                        self.holding_days = 0

            else:
                # 出场判断
                self.holding_days += 1

                exit_flag = False
                exit_reason = ""

                # 1. 价差回归
                if abs(z_score) < exit_th:
                    exit_flag = True
                    exit_reason = f"价差回归(|Z|={abs(z_score):.2f}<{exit_th})"

                # 2. 止损
                elif (self.holding_stock == stock1 and z_score > stop_th) or \
                     (self.holding_stock == stock2 and z_score < -stop_th):
                    exit_flag = True
                    exit_reason = f"止损(持仓Z={self.entry_z:.2f}, 当前Z={z_score:.2f}, 阈值{stop_th})"

                # 3. 时间止损
                elif self.holding_days >= max_hold:
                    exit_flag = True
                    exit_reason = f"时间止损(持仓{self.holding_days}天, 上限{max_hold}天)"

                # 4. 轮动
                elif (self.holding_stock == stock2 and z_score < -reb_th) or \
                     (self.holding_stock == stock1 and z_score > reb_th):

                    other_stock = stock1 if self.holding_stock == stock2 else stock2
                    other_price = price1 if self.holding_stock == stock2 else price2

                    self.execute_sell(current_date, self.holding_stock,
                                    price1 if self.holding_stock == stock1 else price2,
                                    f"轮动卖出(Z={z_score:.2f}, 阈值{reb_th})")

                    success, msg = self.execute_buy(current_date, other_stock, other_price,
                                                  f"轮动买入(Z={z_score:.2f})")
                    if success:
                        self.entry_z = z_score
                        self.entry_date = current_date
                        self.holding_days = 0

                    self.daily_records.append({
                        'date': current_date,
                        'cash': self.cash,
                        'total_value': total_value,
                        'holding': self.holding_stock,
                        'z_score': z_score,
                        'price1': price1,
                        'price2': price2,
                        'hedge_ratio': current_hedge,
                        'status': 'rotating'
                    })
                    self.equity_curve.append((current_date, total_value))
                    continue

                if exit_flag:
                    current_price = price1 if self.holding_stock == stock1 else price2
                    self.execute_sell(current_date, self.holding_stock, current_price, exit_reason)

                    self.cooling_period = True
                    cooldown_days = min(self.params.base_cooling_days + self.cooldown_count * 2,
                                      self.params.max_cooldown)
                    self.cooling_end_date = current_date + timedelta(days=cooldown_days)
                    print(f"    进入{cooldown_days}天冷却期")

            self.daily_records.append({
                'date': current_date,
                'cash': self.cash,
                'total_value': total_value,
                'holding': self.holding_stock,
                'z_score': z_score,
                'price1': price1,
                'price2': price2,
                'hedge_ratio': current_hedge,
                'status': 'holding' if self.holding_stock else 'empty'
            })
            self.equity_curve.append((current_date, total_value))

        # 平仓
        if self.holding_stock and len(trading_dates) > 0:
            last_date = trading_dates[-1]
            try:
                last_price1 = out_data1.loc[last_date, 'close']
                last_price2 = out_data2.loc[last_date, 'close']
                last_price = last_price1 if self.holding_stock == stock1 else last_price2
                self.execute_sell(last_date, self.holding_stock, last_price, "回测结束平仓")
            except:
                pass

        return self.generate_report()

    def generate_report(self):
        if not self.equity_curve:
            return None

        final_value = self.equity_curve[-1][1]
        total_return = (final_value - self.initial_capital) / self.initial_capital

        trades_df = pd.DataFrame([t for t in self.trade_records if t['action'] == '卖出'])
        num_trades = len(trades_df)

        if num_trades > 0 and 'pnl' in trades_df.columns:
            win_trades = len(trades_df[trades_df['pnl'] > 0])
            win_rate = win_trades / num_trades
            avg_profit = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if win_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if (num_trades - win_trades) > 0 else 0
            profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0
        else:
            win_rate = 0
            profit_loss_ratio = 0

        if len(self.equity_curve) > 1:
            values = [v for d, v in self.equity_curve]
            returns = pd.Series(values).pct_change().dropna()
            if len(returns) > 1 and returns.std() > 0:
                sharpe = np.sqrt(252) * returns.mean() / returns.std()
            else:
                sharpe = 0
        else:
            sharpe = 0

        values = [v for d, v in self.equity_curve]
        peak = values[0]
        max_drawdown = 0
        for v in values:
            if v > peak:
                peak = v
            drawdown = (peak - v) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        report = {
            'stock1': self.stock1,
            'stock2': self.stock2,
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'coint_check_count': self.coint_check_count,
            'cooldown_count': self.cooldown_count,
            'param_entry_threshold': self.params.entry_threshold,
            'param_exit_threshold': self.params.exit_threshold,
            'param_stop_loss': self.params.stop_loss,
            'param_max_holding_days': self.params.max_holding_days,
            'param_rebalance_threshold': self.params.rebalance_threshold,
            'param_coint_p_threshold': self.params.coint_p_threshold,
            'param_mean_reversion_speed': self.params.mean_reversion_speed,
            'param_in_sample_volatility': self.params.in_sample_volatility,
            'param_in_sample_r2': self.params.in_sample_r2
        }

        return report

    def plot_results(self, save_path=None):
        """绘制回测结果"""
        if not self.equity_curve or not self.daily_records:
            print("无数据可绘制")
            return

        df_daily = pd.DataFrame(self.daily_records)
        df_daily.set_index('date', inplace=True)

        dates = [d for d, v in self.equity_curve]
        values = [v for d, v in self.equity_curve]

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3)

        # 1. 净值曲线
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(dates, values, linewidth=2, color='#1f77b4', label='Portfolio Value')
        ax1.axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.5, label='Initial Capital')

        buys = [t for t in self.trade_records if t['action'] == '买入']
        sells = [t for t in self.trade_records if t['action'] == '卖出']

        for buy in buys:
            color = 'red' if buy['stock'] == self.stock1 else 'orange'
            label = f'Buy {buy["stock"]}' if buy == buys[0] else ""
            ax1.scatter(buy['date'], buy['cash_after'] + buy['amount'], 
                       color=color, marker='^', s=100, zorder=5, label=label)

        for sell in sells:
            label = 'Sell' if sell == sells[0] else ""
            ax1.scatter(sell['date'], sell['cash_after'], 
                       color='green', marker='v', s=100, zorder=5, label=label)

        report = self.generate_report()
        title = f'Backtest: {self.stock1} vs {self.stock2} | Return: {((values[-1]/values[0])-1)*100:.2f}% | Sharpe: {report["sharpe_ratio"]:.2f}'
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_ylabel('Value (CNY)')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # 2. Z-Score
        ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
        ax2.plot(df_daily.index, df_daily['z_score'], color='purple', alpha=0.7, label='Z-Score')
        ax2.axhline(y=self.params.entry_threshold, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=-self.params.entry_threshold, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=self.params.exit_threshold, color='g', linestyle='--', alpha=0.5)
        ax2.axhline(y=-self.params.exit_threshold, color='g', linestyle='--', alpha=0.5)
        ax2.fill_between(df_daily.index, -self.params.entry_threshold, self.params.entry_threshold, alpha=0.1, color='green')

        holding_mask = df_daily['holding'].notna()
        if holding_mask.any():
            ax2.fill_between(df_daily.index, -self.params.entry_threshold*1.5, self.params.entry_threshold*1.5, 
                           where=holding_mask, alpha=0.2, color='yellow', label='Holding Period')

        ax2.set_ylabel('Z-Score')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        # 3. 对冲比率
        ax3 = fig.add_subplot(gs[2, :], sharex=ax1)
        ax3.plot(df_daily.index, df_daily['hedge_ratio'], color='brown', label='Kalman Hedge Ratio (γ)')
        ax3.set_ylabel('Hedge Ratio')
        ax3.set_xlabel('Date')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {save_path}")

        plt.show()

    def export_trade_log(self, filename=None):
        """导出交易日志"""
        if not self.trade_records:
            print("无交易记录")
            return pd.DataFrame()

        df_trades = pd.DataFrame(self.trade_records)
        if 'pnl' in df_trades.columns:
            df_trades['cumulative_pnl'] = df_trades['pnl'].cumsum()

        if filename:
            df_trades.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"交易日志已保存: {filename}")

        return df_trades

    def export_daily_nav(self, filename=None):
        """导出每日净值"""
        if not self.daily_records:
            print("无每日记录")
            return pd.DataFrame()

        df_daily = pd.DataFrame(self.daily_records)
        df_daily.set_index('date', inplace=True)
        df_daily['daily_return'] = df_daily['total_value'].pct_change()
        df_daily['cumulative_return'] = (df_daily['total_value'] / self.initial_capital) - 1

        if filename:
            df_daily.to_csv(filename, encoding='utf-8-sig')
            print(f"每日净值已保存: {filename}")

        return df_daily


# =============================================================================
# 第四部分：批量回测与汇总
# =============================================================================

def select_and_optimize_pairs(stock_list, data_dict, in_sample_start, in_sample_end, top_n=5):
    """v5.8核心：样本内筛选 + 配对特异性参数优化"""
    print("\n" + "="*80)
    print("v5.8 样本内配对筛选与参数优化")
    print("="*80)

    n = len(stock_list)
    pair_scores = []

    for i, j in combinations(range(n), 2):
        s1 = stock_list[i]
        s2 = stock_list[j]

        df1 = data_dict.get(s1)
        df2 = data_dict.get(s2)

        if df1 is None or df2 is None:
            continue

        common_index = df1.index.intersection(df2.index)
        if len(common_index) < 500:
            continue

        sample_mask = (common_index >= in_sample_start) & (common_index <= in_sample_end)
        sample_index = common_index[sample_mask]

        if len(sample_index) < 100:
            continue

        df1_aligned = df1.loc[sample_index]
        df2_aligned = df2.loc[sample_index]

        log_s1 = np.log(df1_aligned['close'])
        log_s2 = np.log(df2_aligned['close'])

        mask = ~(np.isnan(log_s1) | np.isnan(log_s2) | np.isinf(log_s1) | np.isinf(log_s2))
        log_s1_clean = log_s1[mask]
        log_s2_clean = log_s2[mask]

        if len(log_s1_clean) < 100:
            continue

        spread = log_s1_clean - log_s2_clean
        ssd = np.sqrt(np.mean((spread - spread.mean())**2))
        corr = np.corrcoef(log_s1_clean, log_s2_clean)[0, 1]

        pair_scores.append((s1, s2, ssd, corr, sample_index))

    pair_scores.sort(key=lambda x: x[2])
    top_pairs = pair_scores[:top_n*3]

    print(f"SSD筛选: {len(pair_scores)}候选对，取前{len(top_pairs)}对进行协整检验")

    selection_results = []

    for s1, s2, ssd, corr, sample_index in top_pairs:
        df1 = data_dict.get(s1)
        df2 = data_dict.get(s2)

        log_s1 = np.log(df1.loc[sample_index, 'close'])
        log_s2 = np.log(df2.loc[sample_index, 'close'])

        mask = ~np.isnan(log_s1) & ~np.isnan(log_s2) & ~np.isinf(log_s1) & ~np.isinf(log_s2)
        y = log_s1[mask].values
        x = log_s2[mask].values

        if len(y) < 100:
            continue

        try:
            X = sm.add_constant(x)
            model = sm.OLS(y, X).fit()
            hedge_ratio = model.params[1]
            intercept = model.params[0]
            r_squared = model.rsquared
        except:
            continue

        residual = y - (intercept + hedge_ratio * x)
        coint_p = adf_test(residual)

        if coint_p < 0.05:
            pair_info = {
                'stock1': s1,
                'stock2': s2,
                'ssd': ssd,
                'correlation': corr,
                'hedge_ratio': hedge_ratio,
                'intercept': intercept,
                'r_squared': r_squared,
                'coint_p': coint_p
            }

            optimized_params = generate_pair_specific_params(
                pair_info, df1, df2, in_sample_start, in_sample_end
            )

            selection_results.append({
                **pair_info,
                'optimized_params': optimized_params
            })

            print(f"\n✓ {s1}-{s2}:")
            print(f"  基础统计: SSD={ssd:.2f}, corr={corr:.3f}, R²={r_squared:.3f}, p={coint_p:.4f}")
            print(f"  优化参数: 入场{optimized_params.entry_threshold}/出场{optimized_params.exit_threshold}/"
                  f"持仓{optimized_params.max_holding_days}天/轮动{optimized_params.rebalance_threshold}/"
                  f"均值回归速度{optimized_params.mean_reversion_speed:.4f}")

    selection_results.sort(key=lambda x: (x['ssd'], -x['r_squared']))
    final_pairs = selection_results[:top_n]

    print(f"\n✅ 完成！共{len(final_pairs)}对通过筛选并优化参数")

    return final_pairs


def run_batch_backtest_v5_8(selected_pairs, data_dict, in_sample_start, in_sample_end,
                            out_sample_start, out_sample_end, target_industry):
    """v5.8批量回测：每个配对使用其特异性参数"""
    print("\n" + "="*80)
    print("v5.8 样本外批量回测（配对特异性参数）")
    print("="*80)

    all_results = []

    for idx, pair_data in enumerate(selected_pairs):
        stock1 = pair_data['stock1']
        stock2 = pair_data['stock2']
        params = pair_data['optimized_params']

        print(f"\n{'='*60}")
        print(f"回测第 {idx+1}/{len(selected_pairs)} 对: {stock1} - {stock2}")
        print(f"{'='*60}")

        df1 = data_dict.get(stock1)
        df2 = data_dict.get(stock2)

        if df1 is None or df2 is None:
            continue

        strategy = ASharePairsTradingKalmanV5_8(pair_params=params)

        report = strategy.run_backtest(
            stock1, stock2, df1, df2,
            in_sample_start, in_sample_end,
            out_sample_start, out_sample_end
        )

        if report:
            all_results.append(report)

            strategy.plot_results(save_path=f"backtest_v58_{stock1}_{stock2}.png")
            strategy.export_trade_log(f"trades_v58_{stock1}_{stock2}.csv")

            print(f"\n  结果: 收益={report['total_return']*100:.2f}%, 夏普={report['sharpe_ratio']:.2f}, "
                  f"交易={report['num_trades']}")

    generate_v58_comparison_report(all_results, target_industry)

    return all_results


def generate_v58_comparison_report(all_results, target_industry):
    """生成v5.8特异性对比报告"""
    if not all_results:
        return

    df = pd.DataFrame(all_results)

    print("\n" + "="*80)
    print("v5.8 配对特异性参数回测汇总")
    print("="*80)

    print(f"\n总体统计:")
    print(f"  配对数量: {len(df)}")
    print(f"  平均收益率: {df['total_return'].mean()*100:.2f}%")
    print(f"  平均夏普比率: {df['sharpe_ratio'].mean():.2f}")
    print(f"  胜率中位数: {df['win_rate'].median()*100:.1f}%")

    print(f"\n参数分布:")
    print(f"  入场阈值范围: {df['param_entry_threshold'].min():.2f} - {df['param_entry_threshold'].max():.2f}")
    print(f"  出场阈值范围: {df['param_exit_threshold'].min():.2f} - {df['param_exit_threshold'].max():.2f}")
    print(f"  持仓天数范围: {df['param_max_holding_days'].min()} - {df['param_max_holding_days'].max()}天")
    print(f"  轮动阈值范围: {df['param_rebalance_threshold'].min():.2f} - {df['param_rebalance_threshold'].max():.2f}")

    df.to_csv(f"{target_industry}_v5.8_配对特异性优化结果.csv", index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存: {target_industry}_v5.8_配对特异性优化结果.csv")

    plot_v58_parameter_analysis(df, target_industry)


def plot_v58_parameter_analysis(df, target_industry):
    """分析参数设置与实际绩效的关系"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{target_industry} v5.8 参数-绩效关系分析', fontsize=16)

    # 1. 入场阈值 vs 收益率
    ax1 = axes[0, 0]
    ax1.scatter(df['param_entry_threshold'], df['total_return']*100, s=100, alpha=0.7)
    ax1.set_xlabel('Entry Threshold')
    ax1.set_ylabel('Return (%)')
    ax1.set_title('Entry Threshold vs Return')
    ax1.grid(True, alpha=0.3)

    # 2. 均值回归速度 vs 夏普比率
    ax2 = axes[0, 1]
    ax2.scatter(df['param_mean_reversion_speed'], df['sharpe_ratio'], s=100, alpha=0.7, color='green')
    ax2.set_xlabel('Mean Reversion Speed')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_title('Mean Reversion Speed vs Sharpe')
    ax2.grid(True, alpha=0.3)

    # 3. 持仓天数 vs 交易次数
    ax3 = axes[0, 2]
    ax3.scatter(df['param_max_holding_days'], df['num_trades'], s=100, alpha=0.7, color='red')
    ax3.set_xlabel('Max Holding Days')
    ax3.set_ylabel('Number of Trades')
    ax3.set_title('Holding Days vs Trade Frequency')
    ax3.grid(True, alpha=0.3)

    # 4. R² vs 实际收益
    ax4 = axes[1, 0]
    ax4.scatter(df['param_in_sample_r2'], df['total_return']*100, s=100, alpha=0.7, color='purple')
    ax4.set_xlabel('In-Sample R²')
    ax4.set_ylabel('Return (%)')
    ax4.set_title('In-Sample Fit vs Out-Sample Return')
    ax4.grid(True, alpha=0.3)

    # 5. 轮动阈值 vs 胜率
    ax5 = axes[1, 1]
    ax5.scatter(df['param_rebalance_threshold'], df['win_rate']*100, s=100, alpha=0.7, color='orange')
    ax5.set_xlabel('Rebalance Threshold')
    ax5.set_ylabel('Win Rate (%)')
    ax5.set_title('Rebalance Threshold vs Win Rate')
    ax5.grid(True, alpha=0.3)

    # 6. 波动率 vs 止损阈值
    ax6 = axes[1, 2]
    ax6.scatter(df['param_in_sample_volatility'], df['param_stop_loss'], s=100, alpha=0.7, color='brown')
    ax6.set_xlabel('In-Sample Volatility')
    ax6.set_ylabel('Stop Loss Threshold')
    ax6.set_title('Volatility vs Stop Loss Setting')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{target_industry}_v5.8_参数分析.png', dpi=300, bbox_inches='tight')
    print(f"参数分析图已保存: {target_industry}_v5.8_参数分析.png")
    plt.show()


# =============================================================================
# 第五部分：主程序
# =============================================================================

if __name__ == "__main__":
    # 配置
    LOCAL_INDUSTRY_TXT_PATH = "C:/new_tdx/T0002/export/行业板块.txt"
    TDX_DATA_DIR = "C:/new_tdx/vipdoc"

    IN_SAMPLE_START = "2018-01-01"
    IN_SAMPLE_END = "2021-12-31"
    OUT_SAMPLE_START = "2022-01-01"
    OUT_SAMPLE_END = "2026-04-08"

    TARGET_INDUSTRY = "自动化设备"
    TOP_N_PAIRS = 2

    print("="*80)
    print("A股配对交易策略 v5.8（配对特异性参数优化版）")
    print(f"目标行业: {TARGET_INDUSTRY}")
    print("核心改进：基于样本内统计特征为每个配对定制最优参数")
    print("="*80)

    # 获取数据
    industry_stocks = get_industry_stocks_local(LOCAL_INDUSTRY_TXT_PATH, TARGET_INDUSTRY)

    data_dict = {}
    valid_stocks = []
    for code in industry_stocks:
        df = get_stock_data_from_tdx(code, TDX_DATA_DIR)
        if df is not None and len(df.loc[IN_SAMPLE_START:IN_SAMPLE_END]) >= 100:
            if len(df.loc[:IN_SAMPLE_END]) >= 500:
                data_dict[code] = df
                valid_stocks.append(code)

    print(f"\n有效成分股: {len(valid_stocks)} 只")

    if len(valid_stocks) < 2:
        print("有效股票不足，程序退出")
        exit()

    # v5.8核心：筛选 + 参数优化
    selected_pairs = select_and_optimize_pairs(
        valid_stocks, data_dict,
        IN_SAMPLE_START, IN_SAMPLE_END,
        TOP_N_PAIRS
    )

    if not selected_pairs:
        print("未筛选出符合条件的配对")
        exit()

    # 保存参数配置
    params_config = {}
    for p in selected_pairs:
        params_config[f"{p['stock1']}_{p['stock2']}"] = p['optimized_params'].to_dict()

    with open(f"{TARGET_INDUSTRY}_v5.8_配对参数配置.json", 'w', encoding='utf-8') as f:
        json.dump(params_config, f, indent=2, ensure_ascii=False)

    print(f"\n参数配置已保存: {TARGET_INDUSTRY}_v5.8_配对参数配置.json")

    # 批量回测
    all_results = run_batch_backtest_v5_8(
        selected_pairs, data_dict,
        IN_SAMPLE_START, IN_SAMPLE_END,
        OUT_SAMPLE_START, OUT_SAMPLE_END,
        TARGET_INDUSTRY
    )

    print(f"\n{'='*80}")
    print("v5.8执行完成！")
    print("="*80)