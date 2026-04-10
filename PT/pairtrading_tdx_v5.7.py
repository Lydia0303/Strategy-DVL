# === A股适配的卡尔曼滤波配对轮动策略 v5.7 ===
# === 融合v5.5的严格样本筛选 + v5.6的可视化与A股适配 ===

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
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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

class KalmanFilterPairTrading:
    """卡尔曼滤波器 - 修复数值稳定性"""
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
    """ADF检验 - 来自v5.5的严格版本"""
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

class ASharePairsTradingKalmanV5_7:
    """
    v5.7: 融合版本
    - 样本内筛选逻辑：严格遵循v5.5（SSD + 协整检验）
    - 交易执行：v5.6的单腿轮动（A股适配）
    - 可视化：v5.6的完整图表
    """

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

        # 配对信息
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
            return False, f"计算股数不足100股 (计算得{shares}股)"

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
        print(f"    [交易] {date.strftime('%Y-%m-%d')} 买入 {stock_code} {shares}股 @ {price:.2f}, 原因: {reason}")
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
        print(f"    [交易] {date.strftime('%Y-%m-%d')} 卖出 {stock_code} {shares}股 @ {price:.2f}, 盈亏: {pnl:,.2f}, 原因: {reason}")
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

            # 更新卡尔曼滤波器
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

        # 初始化卡尔曼滤波器（使用样本内数据）
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

            print(f"\n初始化完成：{stock1}-{stock2}")
            print(f"初始对冲比率(γ): {initial_gamma:.4f}, 截距(μ): {initial_mu:.4f}")
            print(f"样本内R²: {model.rsquared:.4f}")

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
                    print(f"  [{current_date.strftime('%Y-%m-%d')}] 冷却期结束，恢复交易")
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

                print(f"  [{current_date.strftime('%Y-%m-%d')}] 协整检验 p={adf_p:.4f} {'✓' if is_coint else '✗'}")

                if not is_coint:
                    self.consecutive_fails += 1
                    if self.consecutive_fails >= 3:
                        print(f"    ⚠️ 连续{self.consecutive_fails}次协整检验失败，进入冷却")
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
                elif (self.holding_stock == stock1 and z_score > self.stop_loss) or \
                     (self.holding_stock == stock2 and z_score < -self.stop_loss):
                    exit_flag = True
                    exit_reason = f"止损(持仓Z={self.entry_z:.2f}, 当前Z={z_score:.2f})"

                # 3. 时间止损
                elif self.holding_days >= self.max_holding_days:
                    exit_flag = True
                    exit_reason = f"时间止损(持仓{self.holding_days}天)"

                # 4. 轮动
                elif (self.holding_stock == stock2 and z_score < -self.rebalance_threshold) or \
                     (self.holding_stock == stock1 and z_score > self.rebalance_threshold):

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
                    print(f"    进入{cooldown_days}天冷却期")

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

    def plot_results(self, save_path=None):
        """绘制回测结果 - 来自v5.6"""
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

        # 标记买卖点
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
        ax2.axhline(y=self.entry_threshold, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=-self.entry_threshold, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=self.exit_threshold, color='g', linestyle='--', alpha=0.5)
        ax2.axhline(y=-self.exit_threshold, color='g', linestyle='--', alpha=0.5)
        ax2.fill_between(df_daily.index, -self.entry_threshold, self.entry_threshold, alpha=0.1, color='green')

        holding_mask = df_daily['holding'].notna()
        if holding_mask.any():
            ax2.fill_between(df_daily.index, -self.entry_threshold*1.5, self.entry_threshold*1.5, 
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


def select_pairs_v5_5_style(stock_list, data_dict, in_sample_start, in_sample_end, top_n=5):
    """
    严格遵循v5.5的样本内筛选逻辑：
    1. 计算所有配对的SSD距离
    2. 按SSD排序取前N对
    3. 对这N对进行严格的协整检验（p<0.05）
    4. 估计对冲比率和统计指标
    """
    print("\n" + "="*80)
    print("第一步：样本内配对筛选（v5.5风格）")
    print("="*80)

    n = len(stock_list)
    total_pairs = n * (n - 1) // 2
    print(f"SSD筛选: 从 {total_pairs} 对中选取前{top_n*3}对进行协整检验")

    pair_scores = []
    valid_pair_count = 0

    # 第一阶段：计算SSD距离（快速筛选）
    for i, j in combinations(range(n), 2):
        s1 = stock_list[i]
        s2 = stock_list[j]

        df1 = data_dict.get(s1)
        df2 = data_dict.get(s2)

        if df1 is None or df2 is None:
            continue

        # 严格时间对齐
        common_index = df1.index.intersection(df2.index)

        if len(common_index) < 500:  # 至少2年共同历史
            continue

        # 截取样本内共同数据
        sample_mask = (common_index >= in_sample_start) & (common_index <= in_sample_end)
        sample_index = common_index[sample_mask]

        if len(sample_index) < 100:  # 样本内至少100天
            continue

        # 使用对齐数据计算
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

    print(f"有效配对候选: {valid_pair_count}/{total_pairs}")

    if not pair_scores:
        print("❌ 没有有效的配对候选")
        return []

    # 按SSD排序，取前3*N对进行协整检验
    pair_scores.sort(key=lambda x: x[2])
    top_pairs = pair_scores[:top_n*3]

    # 第二阶段：严格协整检验
    print(f"\n协整检验: 对前{len(top_pairs)}对进行详细检验 (p<0.05)...")

    selection_results = []

    for s1, s2, ssd, corr, sample_index in top_pairs:
        df1 = data_dict.get(s1)
        df2 = data_dict.get(s2)

        if df1 is None or df2 is None:
            continue

        # 复用时间索引
        log_s1 = np.log(df1.loc[sample_index, 'close'])
        log_s2 = np.log(df2.loc[sample_index, 'close'])

        # 清洗数据
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

        if coint_p < 0.05:  # 严格标准：协整显著
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
            print(f"✓ {s1}-{s2}: SSD={ssd:.2f}, corr={corr:.3f}, hr={hedge_ratio:.4f}, p={coint_p:.4f}")
        else:
            print(f"  ✗ 未通过协整检验: {s1}-{s2} (p={coint_p:.4f})")

    # 按综合评分排序（SSD + R²）
    selection_results.sort(key=lambda x: (x['ssd'], -x['r_squared']))

    # 取前N对
    final_pairs = selection_results[:top_n]

    print(f"\n✅ 样本内选股完成！共筛选出 {len(final_pairs)} 对符合条件的股票")

    return final_pairs


def run_batch_backtest(selected_pairs, data_dict, in_sample_start, in_sample_end, 
                       out_sample_start, out_sample_end, params, target_industry):
    """
    批量回测选中的配对，并生成汇总报告和可视化
    """
    print("\n" + "="*80)
    print("第二步：样本外批量回测")
    print("="*80)

    all_results = []

    for idx, pair_info in enumerate(selected_pairs):
        stock1 = pair_info['stock1']
        stock2 = pair_info['stock2']

        print(f"\n回测第 {idx+1}/{len(selected_pairs)} 对: {stock1} - {stock2}")
        print(f"样本内对冲比率: {pair_info['hedge_ratio']:.4f}, 协整p值: {pair_info['coint_p']:.4f}")

        df1 = data_dict.get(stock1)
        df2 = data_dict.get(stock2)

        if df1 is None or df2 is None:
            print(f"  ✗ 数据缺失，跳过")
            continue

        # 创建策略实例
        strategy = ASharePairsTradingKalmanV5_7(**params)

        # 运行回测
        report = strategy.run_backtest(
            stock1, stock2, df1, df2,
            in_sample_start, in_sample_end,
            out_sample_start, out_sample_end
        )

        if report is None:
            print(f"  ✗ 回测失败")
            continue

        # 添加配对信息到报告
        report.update({
            'ssd': pair_info['ssd'],
            'correlation': pair_info['correlation'],
            'in_sample_hr': pair_info['hedge_ratio'],
            'in_sample_r2': pair_info['r_squared'],
            'in_sample_coint_p': pair_info['coint_p']
        })

        all_results.append(report)

        # 生成可视化
        save_path = f"backtest_{stock1}_{stock2}.png"
        strategy.plot_results(save_path=save_path)

        # 导出数据
        strategy.export_trade_log(f"trades_{stock1}_{stock2}.csv")
        strategy.export_daily_nav(f"daily_nav_{stock1}_{stock2}.csv")

        print(f"  ✓ 回测完成: 收益率={report['total_return']*100:.2f}%, 夏普={report['sharpe_ratio']:.2f}")

    # 生成汇总报告
    generate_summary_report(all_results, target_industry, params)

    return all_results


def generate_summary_report(all_results, target_industry, params):
    """生成汇总报告和对比可视化"""
    if not all_results:
        print("\n❌ 无有效回测结果")
        return

    print("\n" + "="*80)
    print("样本外回测汇总结果（v5.7融合版）")
    print("="*80)

    df_results = pd.DataFrame(all_results)

    # 基础统计
    print(f"\n回测配对数量: {len(df_results)}")
    print(f"初始总资金: {len(df_results) * 1000000:,.0f}")
    final_total_value = df_results['final_value'].sum()
    print(f"最终总资金: {final_total_value:,.0f}")
    total_return = (final_total_value - len(df_results) * 1000000) / (len(df_results) * 1000000)
    print(f"总收益率: {total_return*100:.2f}%")
    print(f"平均收益率: {df_results['total_return'].mean()*100:.2f}%")
    print(f"平均夏普比率: {df_results['sharpe_ratio'].mean():.2f}")
    print(f"平均交易次数: {df_results['num_trades'].mean():.1f}")
    print(f"平均胜率: {df_results['win_rate'].mean()*100:.1f}%")

    # 配对表现排名
    print("\n配对表现排名:")
    sorted_results = df_results.sort_values('total_return', ascending=False)
    for i, (_, r) in enumerate(sorted_results.iterrows()):
        print(f"{i+1}. {r['stock1']}-{r['stock2']}: 收益={r['total_return']*100:.2f}%, "
              f"夏普={r['sharpe_ratio']:.2f}, 交易={r['num_trades']}, "
              f"胜率={r['win_rate']*100:.1f}%, 冷却={r['cooldown_count']}")

    # 保存CSV
    csv_filename = f"{target_industry}_配对回测汇总_v5.7_{len(df_results)}对.csv"
    df_results.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存: {csv_filename}")

    # 生成对比图表
    plot_comparison_chart(df_results, target_industry)


def plot_comparison_chart(df_results, target_industry):
    """绘制多配对对比图表"""
    if len(df_results) == 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{target_industry}行业配对策略回测对比 (v5.7)', fontsize=16, fontweight='bold')

    pair_labels = [f"{r['stock1']}-{r['stock2']}" for _, r in df_results.iterrows()]

    # 1. 收益率对比
    ax1 = axes[0, 0]
    colors = ['green' if x > 0 else 'red' for x in df_results['total_return']]
    ax1.bar(range(len(df_results)), df_results['total_return'] * 100, color=colors, alpha=0.7)
    ax1.set_title('Total Return (%)')
    ax1.set_xticks(range(len(df_results)))
    ax1.set_xticklabels(pair_labels, rotation=45, ha='right')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.grid(True, alpha=0.3)

    # 2. 夏普比率对比
    ax2 = axes[0, 1]
    colors = ['green' if x > 0 else 'red' for x in df_results['sharpe_ratio']]
    ax2.bar(range(len(df_results)), df_results['sharpe_ratio'], color=colors, alpha=0.7)
    ax2.set_title('Sharpe Ratio')
    ax2.set_xticks(range(len(df_results)))
    ax2.set_xticklabels(pair_labels, rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3)

    # 3. 胜率对比
    ax3 = axes[0, 2]
    ax3.bar(range(len(df_results)), df_results['win_rate'] * 100, color='blue', alpha=0.7)
    ax3.set_title('Win Rate (%)')
    ax3.set_xticks(range(len(df_results)))
    ax3.set_xticklabels(pair_labels, rotation=45, ha='right')
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3)

    # 4. 交易次数
    ax4 = axes[1, 0]
    ax4.bar(range(len(df_results)), df_results['num_trades'], color='purple', alpha=0.7)
    ax4.set_title('Number of Trades')
    ax4.set_xticks(range(len(df_results)))
    ax4.set_xticklabels(pair_labels, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)

    # 5. 最大回撤
    ax5 = axes[1, 1]
    ax5.bar(range(len(df_results)), df_results['max_drawdown'] * 100, color='orange', alpha=0.7)
    ax5.set_title('Max Drawdown (%)')
    ax5.set_xticks(range(len(df_results)))
    ax5.set_xticklabels(pair_labels, rotation=45, ha='right')
    ax5.grid(True, alpha=0.3)

    # 6. 样本内R² vs 样本外收益
    ax6 = axes[1, 2]
    if 'in_sample_r2' in df_results.columns:
        ax6.scatter(df_results['in_sample_r2'], df_results['total_return'] * 100, s=100, alpha=0.7)
        for i, row in df_results.iterrows():
            ax6.annotate(f"{row['stock1']}-{row['stock2']}", 
                        (row['in_sample_r2'], row['total_return'] * 100),
                        fontsize=8, alpha=0.7)
        ax6.set_xlabel('In-Sample R²')
        ax6.set_ylabel('Out-Sample Return (%)')
        ax6.set_title('In-Sample Fit vs Out-Sample Performance')
        ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{target_industry}_配对对比图_v5.7.png', dpi=300, bbox_inches='tight')
    print(f"对比图已保存: {target_industry}_配对对比图_v5.7.png")
    plt.show()


if __name__ == "__main__":
    # 配置参数
    LOCAL_INDUSTRY_TXT_PATH = "C:/new_tdx/T0002/export/行业板块.txt"
    TDX_DATA_DIR = "C:/new_tdx/vipdoc"

    IN_SAMPLE_START = "2018-01-01"
    IN_SAMPLE_END = "2023-12-31"
    OUT_SAMPLE_START = "2024-01-01"
    OUT_SAMPLE_END = "2026-04-08"

    TARGET_INDUSTRY = "房屋建设"
    TOP_N_PAIRS = 5  # 回测前N对

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
    print("A股配对交易策略 v5.7（融合版）")
    print(f"目标行业: {TARGET_INDUSTRY}")
    print(f"回测前{TOP_N_PAIRS}对筛选出的配对")
    print("="*80)

    # 获取行业成分股
    industry_stocks = get_industry_stocks_local(LOCAL_INDUSTRY_TXT_PATH, TARGET_INDUSTRY)
    if not industry_stocks:
        print("无有效成分股，程序退出！")
        exit()

    # 预加载所有股票数据
    print("\n预加载股票数据...")
    data_dict = {}
    valid_stocks = []

    for code in industry_stocks:
        df = get_stock_data_from_tdx(code, TDX_DATA_DIR)
        if df is None:
            continue

        try:
            # 检查样本内数据
            sample_df = df.loc[IN_SAMPLE_START:IN_SAMPLE_END]
            if len(sample_df) >= 100:
                # 检查历史数据长度
                full_history = df.loc[:IN_SAMPLE_END]
                if len(full_history) >= 500:
                    data_dict[code] = df
                    valid_stocks.append(code)
                    print(f"  ✓ {code}: 样本内{len(sample_df)}天, 历史{len(full_history)}天")
                else:
                    print(f"  ✗ {code}: 历史数据不足 ({len(full_history)}天 < 500天)")
            else:
                print(f"  ✗ {code}: 样本内数据不足 ({len(sample_df)}天 < 100天)")
        except Exception as e:
            print(f"  ✗ {code}: 数据读取异常 ({e})")

    print(f"\n有效成分股: {len(valid_stocks)} 只")

    if len(valid_stocks) < 2:
        print("有效股票不足2只，程序退出！")
        exit()

    # 第一步：样本内筛选配对（v5.5风格）
    selected_pairs = select_pairs_v5_5_style(
        valid_stocks, data_dict, 
        IN_SAMPLE_START, IN_SAMPLE_END, 
        TOP_N_PAIRS
    )

    if not selected_pairs:
        print("\n❌ 未筛选出符合条件的配对，程序退出！")
        exit()

    # 保存筛选结果
    df_selection = pd.DataFrame(selected_pairs)
    df_selection.to_csv(f"{TARGET_INDUSTRY}_样本内筛选结果_v5.7_{len(selected_pairs)}对.csv", 
                        index=False, encoding='utf-8-sig')

    # 第二步：批量回测（v5.6风格执行 + 批量管理）
    all_results = run_batch_backtest(
        selected_pairs, data_dict,
        IN_SAMPLE_START, IN_SAMPLE_END,
        OUT_SAMPLE_START, OUT_SAMPLE_END,
        PARAMS, TARGET_INDUSTRY
    )

    print(f"\n{'='*80}")
    print("v5.7融合版执行完成！")
    print("主要特性:")
    print("1. 样本内筛选：SSD距离 + 严格协整检验（p<0.05）- 来自v5.5")
    print("2. 交易执行：单腿轮动策略（A股适配）- 来自v5.6")
    print("3. 可视化：完整的单配对图表 + 多配对对比图 - 来自v5.6")
    print("4. 批量管理：支持前N对股票的并行回测和汇总 - 新增")
    print(f"{'='*80}")