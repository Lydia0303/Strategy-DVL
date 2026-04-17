# === A股全行业配对轮动策略 v6.0 ===
# 基于v5.7扩展：支持全行业批量回测 + ST剔除 + Top2筛选

import struct
import pandas as pd
import numpy as np
import os
import warnings
import re
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from itertools import combinations
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 数据读取模块 ====================

def parse_industry_file(txt_path):
    """
    解析通达信行业板块文件，返回行业->股票列表的映射
    文件格式：市场代码 行业名称 股票代码 股票名称
    """
    industry_map = {}
    st_stocks = set()

    try:
        with open(txt_path, 'r', encoding='gbk', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    market_code = parts[0]
                    industry_name = parts[1]
                    stock_code = parts[2]
                    stock_name = parts[3]

                    # 识别ST股票（名称中包含ST）
                    if 'ST' in stock_name.upper() or '*ST' in stock_name.upper():
                        st_stocks.add(stock_code)
                        continue

                    if industry_name not in industry_map:
                        industry_map[industry_name] = []

                    # 去重
                    if stock_code not in industry_map[industry_name]:
                        industry_map[industry_name].append(stock_code)

        print(f"✓ 行业文件解析完成")
        print(f"  共 {len(industry_map)} 个行业")
        print(f"  共识别 {len(st_stocks)} 只ST股票")

        return industry_map, st_stocks

    except Exception as e:
        print(f"✗ 读取行业文件失败：{e}")
        return {}, set()

def get_stock_data_from_tdx(stock_code, data_dir):
    """从通达信数据目录读取股票日线数据"""
    try:
        if stock_code.startswith(('600', '601', '603', '605', '688')):
            prefix = 'sh'
            sub_dir = 'sh'
        elif stock_code.startswith(('000', '001', '002', '003', '300', '301')):
            prefix = 'sz'
            sub_dir = 'sz'
        elif stock_code.startswith(('8', '9', '4', '43')):
            prefix = 'bj'
            sub_dir = 'bj'
        else:
            return None

        file_name = f"{prefix}{stock_code}.day"
        base_path = os.path.dirname(data_dir)
        possible_paths = [
            os.path.join(data_dir, sub_dir, "lday", file_name),
            os.path.join(data_dir, file_name),
            os.path.join(base_path, "vipdoc", sub_dir, "lday", file_name),
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
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        return None

# ==================== 卡尔曼滤波器（v5.7原版）====================

class KalmanFilterPairTrading:
    def __init__(self, initial_state=[0.0, 0.0], initial_covariance=1000.0, process_noise=1e-5, observation_noise=1e-3):
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
    if len(series) < 10:
        return 1.0
    if series.std() == 0 or np.isnan(series.std()) or series.std() < 1e-10:
        return 1.0
    try:
        result = adfuller(series, maxlag=1)
        return result[1]
    except Exception as e:
        return 1.0

def get_dynamic_cooldown(cooldown_count, max_cooldown=10, base_days=2):
    """动态冷却期计算"""
    return min(base_days + cooldown_count * 2, max_cooldown)

# ==================== 策略类（v5.7原版，简化版用于批量回测）====================

class ASharePairsTradingKalmanV6:
    """v6.0: 针对全行业回测优化的策略类"""

    def __init__(self, initial_capital=1000000, commission_rate=0.0003, stamp_tax_rate=0.001,
                 entry_threshold=1.2, exit_threshold=0.3, stop_loss=2.5, 
                 max_holding_days=20, rebalance_threshold=0.6,
                 max_position_ratio=0.6, cooling_period=2, max_cooldown=10,
                 coint_check_freq=30, risk_free_rate=0.02, min_coint_period=60):

        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate

        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss = stop_loss
        self.max_holding_days = max_holding_days
        self.rebalance_threshold = rebalance_threshold
        self.max_position_ratio = max_position_ratio

        self.coint_check_freq = coint_check_freq
        self.base_cooling_period = cooling_period
        self.max_cooldown = max_cooldown
        self.min_coint_period = min_coint_period
        self.risk_free_rate = risk_free_rate

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
        """计算仓位大小"""
        if target_ratio is None:
            target_ratio = self.max_position_ratio

        available_capital = self.cash * target_ratio
        commission_factor = (1 + self.commission_rate + self.stamp_tax_rate * 0.5)
        max_amount = available_capital / commission_factor

        if price <= 0 or np.isnan(price):
            return 0

        shares = int(max_amount / price / 100) * 100
        return max(shares, 0)

    def execute_buy(self, date, stock_code, price, reason=""):
        """执行买入"""
        shares = self.calculate_position_size(price)
        if shares < 100:
            return False, f"计算股数不足100股"

        amount = price * shares
        commission = max(amount * self.commission_rate, 5)
        total_cost = amount + commission

        if total_cost > self.cash:
            max_shares = int(self.cash / price / 100) * 100
            if max_shares < 100:
                return False, "资金不足买入100股"
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
        return True, "成功"

    def execute_sell(self, date, stock_code, price, reason=""):
        """执行卖出"""
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
            'hold_days': (date - pos['entry_date']).days if isinstance(date, datetime) and isinstance(pos['entry_date'], datetime) else 0
        }
        self.trade_records.append(record)
        return True, "成功"

    def calculate_portfolio_value(self, date, price1, price2):
        """计算组合市值"""
        total = self.cash
        for stock, pos in self.positions.items():
            if stock == self.stock1:
                total += pos['qty'] * price1
            elif stock == self.stock2:
                total += pos['qty'] * price2
        return total

    def calculate_spread_zscore(self, stock1_data, stock2_data, current_date, window=60):
        """计算价差和Z-Score"""
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
        """动态协整检验"""
        try:
            s1 = stock1_data.loc[:current_date].iloc[-window:]
            s2 = stock2_data.loc[:current_date].iloc[-window:]

            if len(s1) < self.min_coint_period or len(s2) < self.min_coint_period:
                return True, 0.01

            log_s1 = np.log(s1.replace(0, np.nan).dropna())
            log_s2 = np.log(s2.replace(0, np.nan).dropna())

            common_dates = log_s1.index.intersection(log_s2.index)
            if len(common_dates) < self.min_coint_period:
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

            if adf_p > 0.1:
                try:
                    score, pvalue, _ = coint(log_s1, log_s2)
                    if pvalue < 0.1:
                        return True, pvalue
                except:
                    pass

            return adf_p < 0.1, adf_p

        except Exception as e:
            return True, 0.01

    def run_backtest(self, stock1, stock2, stock1_data, stock2_data, 
                     in_sample_start, in_sample_end, out_sample_start, out_sample_end):
        """执行回测"""
        self.stock1 = stock1
        self.stock2 = stock2

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
            return None

        out_data1 = stock1_data.loc[out_sample_start:out_sample_end]
        out_data2 = stock2_data.loc[out_sample_start:out_sample_end]

        trading_dates = out_data1.index.intersection(out_data2.index)

        if len(trading_dates) == 0:
            return None

        # 预热期
        warmup_period = 20
        for i in range(min(warmup_period, len(trading_dates))):
            current_date = trading_dates[i]
            try:
                p1 = out_data1.loc[current_date, 'close']
                p2 = out_data2.loc[current_date, 'close']
                self.calculate_spread_zscore(
                    out_data1['close'], out_data2['close'], current_date
                )
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

            # 定期协整检验
            days_since_last = (current_date - self.last_check_date).days if self.last_check_date else 999
            if days_since_last >= self.coint_check_freq:
                is_coint, adf_p = self.check_cointegration(out_data1['close'], out_data2['close'], current_date)
                self.coint_check_count += 1
                self.last_check_date = current_date

                if not is_coint:
                    self.consecutive_fails += 1
                    if self.consecutive_fails >= 3:
                        if self.holding_stock:
                            self.execute_sell(current_date, self.holding_stock, 
                                            price1 if self.holding_stock == stock1 else price2,
                                            "协整破裂强制平仓")

                        self.cooldown_count += 1
                        self.cooling_period = True
                        cooldown_days = get_dynamic_cooldown(self.cooldown_count, self.max_cooldown, self.base_cooling_period)
                        self.cooling_end_date = current_date + timedelta(days=cooldown_days)
                        self.consecutive_fails = 0

                        if self.cooldown_count >= 5:
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

            # 交易逻辑
            if self.holding_stock is None:
                # 寻找入场机会
                if abs(z_score) > self.entry_threshold and self.pair_valid:
                    if z_score > self.entry_threshold:
                        success, msg = self.execute_buy(current_date, stock2, price2, 
                                                      f"Z={z_score:.2f}>{self.entry_threshold}，买入低估方{stock2}")
                    else:
                        success, msg = self.execute_buy(current_date, stock1, price1,
                                                      f"Z={z_score:.2f}<-{self.entry_threshold}，买入低估方{stock1}")

                    if success:
                        self.entry_z = z_score
                        self.entry_date = current_date
                        self.holding_days = 0

            else:
                # 检查出场
                self.holding_days += 1

                exit_flag = False
                exit_reason = ""

                # 1. 价差回归
                if abs(z_score) < self.exit_threshold:
                    exit_flag = True
                    exit_reason = f"价差回归(|Z|={abs(z_score):.2f}<{self.exit_threshold})"

                # 2. 止损
                elif (self.holding_stock == stock1 and z_score > self.stop_loss) or                      (self.holding_stock == stock2 and z_score < -self.stop_loss):
                    exit_flag = True
                    exit_reason = f"止损(持仓Z={self.entry_z:.2f}, 当前Z={z_score:.2f})"

                # 3. 时间止损
                elif self.holding_days >= self.max_holding_days:
                    exit_flag = True
                    exit_reason = f"时间止损(持仓{self.holding_days}天)"

                # 4. 轮动
                elif (self.holding_stock == stock2 and z_score < -self.rebalance_threshold) or                      (self.holding_stock == stock1 and z_score > self.rebalance_threshold):

                    other_stock = stock1 if self.holding_stock == stock2 else stock2
                    other_price = price1 if self.holding_stock == stock2 else price2

                    self.execute_sell(current_date, self.holding_stock,
                                    price1 if self.holding_stock == stock1 else price2,
                                    f"轮动卖出(Z={z_score:.2f})")

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
                    cooldown_days = get_dynamic_cooldown(self.cooldown_count, self.max_cooldown, self.base_cooling_period)
                    self.cooling_end_date = current_date + timedelta(days=cooldown_days)

            # 记录每日状态
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

        # 回测结束平仓
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
        """生成回测报告"""
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

        # 夏普比率
        if len(self.equity_curve) > 1:
            values = [v for d, v in self.equity_curve]
            returns = pd.Series(values).pct_change().dropna()
            if len(returns) > 1 and returns.std() > 0:
                sharpe = np.sqrt(252) * returns.mean() / returns.std()
            else:
                sharpe = 0
        else:
            sharpe = 0

        # 最大回撤
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
            'cooldown_count': self.cooldown_count
        }

        return report

# ==================== 配对筛选模块（v5.5风格）====================

def select_pairs_for_industry(stock_list, data_dict, in_sample_start, in_sample_end, top_n=5):
    """
    为单个行业筛选配对（v5.5风格）
    返回：配对列表，每个配对包含详细统计信息
    """
    n = len(stock_list)
    if n < 2:
        return []

    total_pairs = n * (n - 1) // 2

    pair_scores = []
    valid_pair_count = 0

    # 第一阶段：计算SSD距离
    for i, j in combinations(range(n), 2):
        s1 = stock_list[i]
        s2 = stock_list[j]

        df1 = data_dict.get(s1)
        df2 = data_dict.get(s2)

        if df1 is None or df2 is None:
            continue

        # 严格时间对齐
        common_index = df1.index.intersection(df2.index)

        if len(common_index) < 500:
            continue

        # 截取样本内共同数据
        sample_mask = (common_index >= in_sample_start) & (common_index <= in_sample_end)
        sample_index = common_index[sample_mask]

        if len(sample_index) < 100:
            continue

        df1_aligned = df1.loc[sample_index]
        df2_aligned = df2.loc[sample_index]

        log_s1 = np.log(df1_aligned['close'])
        log_s2 = np.log(df2_aligned['close'])

        # 清洗数据
        mask = ~(np.isnan(log_s1) | np.isnan(log_s2) | np.isinf(log_s1) | np.isinf(log_s2))
        log_s1_clean = log_s1[mask]
        log_s2_clean = log_s2[mask]

        if len(log_s1_clean) < 100:
            continue

        # 计算SSD距离
        spread = log_s1_clean - log_s2_clean
        ssd = np.sqrt(np.mean((spread - spread.mean())**2))

        # 计算相关性
        corr = np.corrcoef(log_s1_clean, log_s2_clean)[0, 1]

        pair_scores.append((s1, s2, ssd, corr, sample_index))
        valid_pair_count += 1

    if not pair_scores:
        return []

    # 按SSD排序，取前3*N对进行协整检验
    pair_scores.sort(key=lambda x: x[2])
    top_pairs = pair_scores[:top_n*3]

    # 第二阶段：严格协整检验
    selection_results = []

    for s1, s2, ssd, corr, sample_index in top_pairs:
        df1 = data_dict.get(s1)
        df2 = data_dict.get(s2)

        if df1 is None or df2 is None:
            continue

        log_s1 = np.log(df1.loc[sample_index, 'close'])
        log_s2 = np.log(df2.loc[sample_index, 'close'])

        mask = ~np.isnan(log_s1) & ~np.isnan(log_s2) & ~np.isinf(log_s1) & ~np.isinf(log_s2)
        y = log_s1[mask].values
        x = log_s2[mask].values

        if len(y) < 100:
            continue

        # OLS回归估计对冲比率
        try:
            X = sm.add_constant(x)
            model = sm.OLS(y, X).fit()
            hedge_ratio = model.params[1]
            intercept = model.params[0]
            r_squared = model.rsquared
        except Exception as e:
            continue

        # 计算残差并进行协整检验
        residual = y - (intercept + hedge_ratio * x)
        coint_p = adf_test(residual)

        if coint_p < 0.05:
            spread_std = residual.std()
            selection_results.append({
                'stock1': s1,
                'stock2': s2,
                'ssd': ssd,
                'correlation': corr,
                'hedge_ratio': hedge_ratio,
                'intercept': intercept,
                'r_squared': r_squared,
                'coint_p': coint_p,
                'spread_std': spread_std,
                'sample_size': len(y)
            })

    # 按综合评分排序（SSD + R²）
    selection_results.sort(key=lambda x: (x['ssd'], -x['r_squared']))

    # 取前N对
    final_pairs = selection_results[:top_n]

    return final_pairs

# ==================== 全行业回测主程序 ====================

def run_all_industries_backtest():
    """
    主函数：运行全行业回测
    """
    # 配置参数
    LOCAL_INDUSTRY_TXT_PATH = "C:/new_tdx/T0002/export/行业板块.txt"
    TDX_DATA_DIR = "C:/new_tdx/vipdoc"

    IN_SAMPLE_START = "2018-01-01"
    IN_SAMPLE_END = "2021-12-31"
    OUT_SAMPLE_START = "2022-01-01"
    OUT_SAMPLE_END = "2026-04-08"

    TOP_N_PAIRS_PER_INDUSTRY = 5  # 每个行业先选5对，回测后取前2
    FINAL_TOP_N = 2  # 每个行业最终保留前2

    PARAMS = {
        'initial_capital': 1000000,
        'commission_rate': 0.0003,
        'stamp_tax_rate': 0.001,
        'entry_threshold': 1.2,
        'exit_threshold': 0.3,
        'stop_loss': 2.5,
        'max_holding_days': 15,
        'rebalance_threshold': 0.6,
        'max_position_ratio': 0.6,
        'coint_check_freq': 30,
        'cooling_period': 2,
        'max_cooldown': 10,
        'min_coint_period': 60
    }

    print("="*80)
    print("A股全行业配对交易策略 v6.0")
    print(f"样本内: {IN_SAMPLE_START} ~ {IN_SAMPLE_END}")
    print(f"样本外: {OUT_SAMPLE_START} ~ {OUT_SAMPLE_END}")
    print(f"每个行业选取收益率前{FINAL_TOP_N}的配对")
    print("="*80)

    # 1. 解析行业文件
    print("\n[1/5] 解析行业板块文件...")
    industry_map, st_stocks = parse_industry_file(LOCAL_INDUSTRY_TXT_PATH)

    if not industry_map:
        print("✗ 行业文件解析失败，程序退出")
        return

    print(f"  发现 {len(industry_map)} 个行业板块")

    # 2. 遍历每个行业
    all_industry_results = []

    for idx, (industry_name, stock_codes) in enumerate(industry_map.items(), 1):
        print(f"\n{'='*80}")
        print(f"[{idx}/{len(industry_map)}] 处理行业: {industry_name}")
        print(f"{'='*80}")

        # 剔除ST股票
        clean_stocks = [s for s in stock_codes if s not in st_stocks]
        removed_count = len(stock_codes) - len(clean_stocks)

        print(f"  原始成分股: {len(stock_codes)} 只")
        if removed_count > 0:
            print(f"  剔除ST股票: {removed_count} 只")
        print(f"  有效成分股: {len(clean_stocks)} 只")

        if len(clean_stocks) < 2:
            print(f"  ⚠️ 有效股票不足2只，跳过该行业")
            continue

        # 3. 预加载该行业所有股票数据
        print(f"\n  [2/5] 加载股票数据...")
        data_dict = {}
        valid_stocks = []

        for code in clean_stocks:
            df = get_stock_data_from_tdx(code, TDX_DATA_DIR)
            if df is None:
                continue

            try:
                # 检查样本内数据
                sample_df = df.loc[IN_SAMPLE_START:IN_SAMPLE_END]
                if len(sample_df) >= 100:
                    full_history = df.loc[:IN_SAMPLE_END]
                    if len(full_history) >= 500:
                        data_dict[code] = df
                        valid_stocks.append(code)
            except:
                continue

        print(f"  有效数据股票: {len(valid_stocks)}/{len(clean_stocks)} 只")

        if len(valid_stocks) < 2:
            print(f"  ⚠️ 有效数据不足，跳过该行业")
            continue

        # 4. 样本内配对筛选
        print(f"\n  [3/5] 样本内配对筛选...")
        selected_pairs = select_pairs_for_industry(
            valid_stocks, data_dict, 
            IN_SAMPLE_START, IN_SAMPLE_END, 
            TOP_N_PAIRS_PER_INDUSTRY
        )

        if not selected_pairs:
            print(f"  ⚠️ 未筛选出符合条件的配对，跳过该行业")
            continue

        print(f"  筛选出 {len(selected_pairs)} 对候选配对")

        # 5. 样本外回测所有候选配对
        print(f"\n  [4/5] 样本外回测...")
        industry_pair_results = []

        for pair_idx, pair_info in enumerate(selected_pairs):
            stock1 = pair_info['stock1']
            stock2 = pair_info['stock2']

            df1 = data_dict.get(stock1)
            df2 = data_dict.get(stock2)

            if df1 is None or df2 is None:
                continue

            # 创建策略实例并回测
            strategy = ASharePairsTradingKalmanV6(**PARAMS)

            report = strategy.run_backtest(
                stock1, stock2, df1, df2,
                IN_SAMPLE_START, IN_SAMPLE_END,
                OUT_SAMPLE_START, OUT_SAMPLE_END
            )

            if report is None:
                continue

            # 添加行业和配对信息
            report.update({
                'industry': industry_name,
                'ssd': pair_info['ssd'],
                'correlation': pair_info['correlation'],
                'in_sample_hr': pair_info['hedge_ratio'],
                'in_sample_r2': pair_info['r_squared'],
                'in_sample_coint_p': pair_info['coint_p']
            })

            industry_pair_results.append(report)
            print(f"    配对 {pair_idx+1}/{len(selected_pairs)}: {stock1}-{stock2} | 收益: {report['total_return']*100:.2f}% | 夏普: {report['sharpe_ratio']:.2f}")

        if not industry_pair_results:
            print(f"  ⚠️ 所有配对回测失败，跳过该行业")
            continue

        # 6. 按收益率排序，取前N
        print(f"\n  [5/5] 筛选Top {FINAL_TOP_N} 配对...")
        sorted_results = sorted(industry_pair_results, key=lambda x: x['total_return'], reverse=True)
        top_pairs = sorted_results[:FINAL_TOP_N]

        print(f"  ✓ 选中配对:")
        for i, r in enumerate(top_pairs, 1):
            print(f"    {i}. {r['stock1']}-{r['stock2']}: 收益={r['total_return']*100:.2f}%, 夏普={r['sharpe_ratio']:.2f}")

        all_industry_results.extend(top_pairs)

    # 7. 生成最终汇总报告
    print(f"\n{'='*80}")
    print("全行业回测完成！生成汇总报告...")
    print(f"{'='*80}")

    if not all_industry_results:
        print("✗ 没有成功的回测结果")
        return

    # 创建汇总DataFrame
    df_summary = pd.DataFrame(all_industry_results)

    # 调整列顺序
    column_order = [
        'industry', 'stock1', 'stock2', 'total_return', 'sharpe_ratio',
        'max_drawdown', 'win_rate', 'num_trades', 'profit_loss_ratio',
        'in_sample_r2', 'in_sample_coint_p', 'correlation', 'ssd',
        'final_value', 'initial_capital', 'cooldown_count'
    ]

    # 只保留存在的列
    existing_cols = [c for c in column_order if c in df_summary.columns]
    df_summary = df_summary[existing_cols]

    # 格式化数值列
    if 'total_return' in df_summary.columns:
        df_summary['total_return_pct'] = (df_summary['total_return'] * 100).round(2)
    if 'win_rate' in df_summary.columns:
        df_summary['win_rate_pct'] = (df_summary['win_rate'] * 100).round(1)
    if 'max_drawdown' in df_summary.columns:
        df_summary['max_drawdown_pct'] = (df_summary['max_drawdown'] * 100).round(2)

    # 保存详细结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_file = f"全行业配对回测详细结果_{timestamp}.csv"
    df_summary.to_csv(detail_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ 详细结果已保存: {detail_file}")

    # 生成简洁版分析表格
    df_analysis = df_summary[['industry', 'stock1', 'stock2', 'total_return_pct', 
                              'sharpe_ratio', 'max_drawdown_pct', 'win_rate_pct', 
                              'num_trades', 'in_sample_r2', 'in_sample_coint_p']].copy()
    df_analysis.columns = ['行业', '股票1', '股票2', '收益率(%)', '夏普比率', 
                          '最大回撤(%)', '胜率(%)', '交易次数', '样本内R²', '协整P值']

    analysis_file = f"全行业配对分析表_{timestamp}.csv"
    df_analysis.to_csv(analysis_file, index=False, encoding='utf-8-sig')
    print(f"✓ 分析表格已保存: {analysis_file}")

    # 打印汇总统计
    print(f"\n{'='*80}")
    print("汇总统计")
    print(f"{'='*80}")
    print(f"成功回测行业数: {df_summary['industry'].nunique()}")
    print(f"总配对数: {len(df_summary)}")
    print(f"平均收益率: {df_summary['total_return'].mean()*100:.2f}%")
    print(f"平均夏普比率: {df_summary['sharpe_ratio'].mean():.2f}")
    print(f"平均最大回撤: {df_summary['max_drawdown'].mean()*100:.2f}%")
    print(f"胜率>50%的配对数: {len(df_summary[df_summary['win_rate'] > 0.5])}")
    print(f"正收益配对数: {len(df_summary[df_summary['total_return'] > 0])}")

    # 按行业分组统计
    print(f"\n{'='*80}")
    print("各行业表现")
    print(f"{'='*80}")
    industry_stats = df_summary.groupby('industry').agg({
        'total_return': 'mean',
        'sharpe_ratio': 'mean',
        'max_drawdown': 'mean'
    }).round(4)
    industry_stats.columns = ['平均收益率', '平均夏普', '平均回撤']
    print(industry_stats)

    # 保存行业统计
    stats_file = f"全行业统计_{timestamp}.csv"
    industry_stats.to_csv(stats_file, encoding='utf-8-sig')
    print(f"\n✓ 行业统计已保存: {stats_file}")

    print(f"\n{'='*80}")
    print("全部完成！")
    print(f"{'='*80}")

    return df_summary, df_analysis

if __name__ == "__main__":
    # 运行全行业回测
    df_detail, df_analysis = run_all_industries_backtest()
