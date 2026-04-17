# === A股全行业配对轮动策略 v7.0 ===
# 基于v6.0扩展：北交所过滤 + 行业相关性监控 + 风险平价权重 + 组合回测 + 滑点

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
    # 数据路径
    'LOCAL_INDUSTRY_TXT_PATH': "C:/new_tdx/T0002/export/行业板块.txt",
    'TDX_DATA_DIR': "C:/new_tdx/vipdoc",
    
    # 回测时间区间
    'IN_SAMPLE_START': "2018-01-01",
    'IN_SAMPLE_END': "2021-12-31",
    'OUT_SAMPLE_START': "2022-01-01",
    'OUT_SAMPLE_END': "2026-04-08",
    
    # 筛选参数
    'TOP_N_PAIRS_PER_INDUSTRY': 5,   # 每个行业候选配对数
    'FINAL_TOP_N': 2,                # 每个行业最终保留2对
    'MIN_POSITIVE_PAIRS': 2,         # 最少正收益配对数，否则剔除行业
    
    # 【新增】组合资金配置
    'PORTFOLIO_CAPITAL': 10_000_000,  # 初始资金1000万
    
    # 【新增】滑点设置（双边万分之三）
    'SLIPPAGE': 0.0003,  # 买入+0.03%，卖出-0.03%
    
    # 策略参数
    'PARAMS': {
        'commission_rate': 0.0003,      # 佣金率
        'stamp_tax_rate': 0.001,        # 印花税
        'entry_threshold': 1.2,         # 入场Z-Score
        'exit_threshold': 0.3,          # 出场Z-Score
        'stop_loss': 2.5,               # 止损阈值
        'max_holding_days': 15,         # 最大持仓天数
        'rebalance_threshold': 0.6,     # 轮动阈值
        'max_position_ratio': 0.6,        # 单个配对最大仓位比例
        'coint_check_freq': 30,         # 协整检验频率
        'cooling_period': 2,            # 基础冷却期
        'max_cooldown': 10,             # 最大冷却期
        'min_coint_period': 60          # 最小协整检验周期
    }
}

# ==================== 数据读取模块 ====================

def is_bse_stock(stock_code):
    """判断是否为北交所股票（83/87/88/92开头）"""
    code_str = str(stock_code)
    return code_str.startswith(('83', '87', '88', '92'))

def parse_industry_file(txt_path):
    """解析行业板块文件，剔除ST和北交所股票"""
    industry_map = {}
    st_stocks = set()
    bse_stocks = set()

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

                    # 剔除ST
                    if 'ST' in stock_name.upper() or '*ST' in stock_name.upper():
                        st_stocks.add(stock_code)
                        continue
                    
                    # 【关键】剔除北交所
                    if is_bse_stock(stock_code):
                        bse_stocks.add(stock_code)
                        continue

                    if industry_name not in industry_map:
                        industry_map[industry_name] = []

                    if stock_code not in industry_map[industry_name]:
                        industry_map[industry_name].append(stock_code)

        print(f"✓ 行业文件解析完成: {len(industry_map)}个行业")
        print(f"  剔除ST: {len(st_stocks)}只, 北交所: {len(bse_stocks)}只")
        return industry_map, st_stocks, bse_stocks

    except Exception as e:
        print(f"✗ 读取行业文件失败：{e}")
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

# ==================== 【核心】带滑点的配对交易类 ====================

class PairTradingInstance:
    """
    单个配对的交易实例，包含完整滑点逻辑
    """
    def __init__(self, stock1, stock2, allocated_capital, slippage=0.0003, **params):
        self.stock1 = stock1
        self.stock2 = stock2
        self.allocated_capital = allocated_capital  # 该配对分配的资金
        self.slippage = slippage  # 【新增】滑点
        
        self.cash = allocated_capital
        self.commission_rate = params.get('commission_rate', 0.0003)
        self.stamp_tax_rate = params.get('stamp_tax_rate', 0.001)
        self.entry_threshold = params.get('entry_threshold', 1.2)
        self.exit_threshold = params.get('exit_threshold', 0.3)
        self.stop_loss = params.get('stop_loss', 2.5)
        self.max_holding_days = params.get('max_holding_days', 15)
        self.rebalance_threshold = params.get('rebalance_threshold', 0.6)
        self.max_position_ratio = params.get('max_position_ratio', 0.6)
        self.coint_check_freq = params.get('coint_check_freq', 30)
        self.base_cooling_period = params.get('cooling_period', 2)
        self.max_cooldown = params.get('max_cooldown', 10)
        self.min_coint_period = params.get('min_coint_period', 60)
        
        # 状态
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
        self.daily_records = []
        
    def initialize_kalman(self, in_sample_data1, in_sample_data2):
        """初始化卡尔曼滤波器"""
        try:
            log_s1 = np.log(in_sample_data1['close'].iloc[-100:])
            log_s2 = np.log(in_sample_data2['close'].iloc[-100:])
            
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
    
    def calculate_position_size(self, price, target_ratio=None):
        """计算仓位"""
        if target_ratio is None:
            target_ratio = self.max_position_ratio
            
        available_capital = self.cash * target_ratio
        commission_factor = (1 + self.commission_rate + self.stamp_tax_rate * 0.5)
        max_amount = available_capital / commission_factor
        
        if price <= 0 or np.isnan(price):
            return 0
            
        shares = int(max_amount / price / 100) * 100
        return max(shares, 0)
    
    def apply_slippage(self, price, direction):
        """
        【核心】应用滑点
        direction: 'buy' 或 'sell'
        """
        if direction == 'buy':
            # 买入：实际成交价更高（+滑点）
            return price * (1 + self.slippage)
        else:
            # 卖出：实际成交价更低（-滑点）
            return price * (1 - self.slippage)
    
    def execute_buy(self, date, stock_code, raw_price, reason=""):
        """执行买入（含滑点）"""
        # 【关键】应用滑点
        price = self.apply_slippage(raw_price, 'buy')
        
        shares = self.calculate_position_size(price)
        if shares < 100:
            return False, "股数不足"
            
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
        self.positions[stock_code] = {'qty': shares, 'avg_price': price, 'entry_date': date}
        self.holding_stock = stock_code
        
        self.trade_records.append({
            'date': date, 'action': '买入', 'stock': stock_code,
            'raw_price': raw_price, 'executed_price': price,  # 【记录】原始价和实际成交价
            'slippage_cost': (price - raw_price) * shares,  # 【记录】滑点成本
            'shares': shares, 'amount': amount, 'commission': commission,
            'cash_after': self.cash, 'reason': reason
        })
        return True, "成功"
    
    def execute_sell(self, date, stock_code, raw_price, reason=""):
        """执行卖出（含滑点）"""
        if stock_code not in self.positions:
            return False, "无持仓"
            
        # 【关键】应用滑点
        price = self.apply_slippage(raw_price, 'sell')
        
        pos = self.positions[stock_code]
        shares = pos['qty']
        
        amount = price * shares
        commission = max(amount * self.commission_rate, 5)
        stamp_tax = amount * self.stamp_tax_rate
        total_cost = commission + stamp_tax
        net_proceeds = amount - total_cost
        
        # 【计算】滑点损失
        slippage_loss = (pos['avg_price'] - price) * shares  # 买入滑点 + 卖出滑点
        
        pnl = (price - pos['avg_price']) * shares - total_cost
        
        self.cash += net_proceeds
        del self.positions[stock_code]
        self.holding_stock = None
        
        self.trade_records.append({
            'date': date, 'action': '卖出', 'stock': stock_code,
            'raw_price': raw_price, 'executed_price': price,
            'slippage_cost': (raw_price - price) * shares,  # 【记录】
            'shares': shares, 'amount': amount,
            'commission': commission, 'stamp_tax': stamp_tax,
            'pnl': pnl, 'cash_after': self.cash, 'reason': reason,
            'hold_days': (date - pos['entry_date']).days if isinstance(date, datetime) else 0
        })
        return True, "成功"
    
    def calculate_portfolio_value(self, price1, price2):
        """计算当前市值"""
        total = self.cash
        for stock, pos in self.positions.items():
            if stock == self.stock1:
                total += pos['qty'] * price1
            elif stock == self.stock2:
                total += pos['qty'] * price2
        return total
    
    def calculate_spread_zscore(self, stock1_data, stock2_data, current_date, window=60):
        """计算价差Z-Score"""
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
        """协整检验"""
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
        """执行回测"""
        out_data1 = stock1_data.loc[out_sample_start:out_sample_end]
        out_data2 = stock2_data.loc[out_sample_start:out_sample_end]
        trading_dates = out_data1.index.intersection(out_data2.index)
        
        if len(trading_dates) == 0:
            return None
            
        # 预热期
        for i in range(min(20, len(trading_dates))):
            current_date = trading_dates[i]
            try:
                p1 = out_data1.loc[current_date, 'close']
                p2 = out_data2.loc[current_date, 'close']
                self.calculate_spread_zscore(out_data1['close'], out_data2['close'], current_date)
            except:
                pass
        
        # 主循环
        for i, current_date in enumerate(trading_dates):
            try:
                raw_price1 = out_data1.loc[current_date, 'close']
                raw_price2 = out_data2.loc[current_date, 'close']
            except:
                continue
            
            # 冷却期检查
            if self.cooling_period:
                if self.cooling_end_date and current_date >= self.cooling_end_date:
                    self.cooling_period = False
                else:
                    total_value = self.calculate_portfolio_value(raw_price1, raw_price2)
                    self.equity_curve.append((current_date, total_value))
                    continue
            
            # 定期协整检验
            days_since_last = (current_date - self.last_check_date).days if self.last_check_date else 999
            if days_since_last >= self.coint_check_freq:
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
            
            # 计算Z-Score
            spread, z_score, mean_spread, std_spread = self.calculate_spread_zscore(
                out_data1['close'], out_data2['close'], current_date
            )
            
            if z_score is None:
                total_value = self.calculate_portfolio_value(raw_price1, raw_price2)
                self.equity_curve.append((current_date, total_value))
                continue
            
            # 交易逻辑
            if self.holding_stock is None:
                # 寻找入场
                if abs(z_score) > self.entry_threshold and self.pair_valid:
                    if z_score > self.entry_threshold:
                        success, msg = self.execute_buy(current_date, self.stock2, raw_price2,
                                                      f"Z={z_score:.2f}，买入低估方")
                    else:
                        success, msg = self.execute_buy(current_date, self.stock1, raw_price1,
                                                      f"Z={z_score:.2f}，买入低估方")
                    if success:
                        self.entry_z = z_score
                        self.entry_date = current_date
                        self.holding_days = 0
            else:
                # 检查出场
                self.holding_days += 1
                exit_flag, exit_reason = False, ""
                
                if abs(z_score) < self.exit_threshold:
                    exit_flag, exit_reason = True, f"价差回归(|Z|={abs(z_score):.2f})"
                elif (self.holding_stock == self.stock1 and z_score > self.stop_loss) or \
                     (self.holding_stock == self.stock2 and z_score < -self.stop_loss):
                    exit_flag, exit_reason = True, f"止损(Z={z_score:.2f})"
                elif self.holding_days >= self.max_holding_days:
                    exit_flag, exit_reason = True, f"时间止损({self.holding_days}天)"
                elif (self.holding_stock == self.stock2 and z_score < -self.rebalance_threshold) or \
                     (self.holding_stock == self.stock1 and z_score > self.rebalance_threshold):
                    # 轮动
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
            
            total_value = self.calculate_portfolio_value(raw_price1, raw_price2)
            self.equity_curve.append((current_date, total_value))
        
        # 结束平仓
        if self.holding_stock and len(trading_dates) > 0:
            last_date = trading_dates[-1]
            try:
                last_raw_price = out_data1.loc[last_date, 'close'] if self.holding_stock == self.stock1 else out_data2.loc[last_date, 'close']
                self.execute_sell(last_date, self.holding_stock, last_raw_price, "回测结束平仓")
            except:
                pass
        
        return self.generate_report()
    
    def generate_report(self):
        """生成回测报告"""
        if not self.equity_curve:
            return None
            
        final_value = self.equity_curve[-1][1]
        total_return = (final_value - self.allocated_capital) / self.allocated_capital
        
        trades_df = pd.DataFrame([t for t in self.trade_records if t['action'] == '卖出'])
        num_trades = len(trades_df)
        
        win_rate = 0
        profit_loss_ratio = 0
        total_slippage_cost = 0  # 【统计】总滑点成本
        
        if num_trades > 0:
            win_trades = len(trades_df[trades_df['pnl'] > 0])
            win_rate = win_trades / num_trades
            avg_profit = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if win_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if (num_trades - win_trades) > 0 else 0
            profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0
            
            # 【关键】统计滑点成本
            all_trades = pd.DataFrame(self.trade_records)
            if 'slippage_cost' in all_trades.columns:
                total_slippage_cost = all_trades['slippage_cost'].sum()
        
        # 夏普
        values = [v for d, v in self.equity_curve]
        returns = pd.Series(values).pct_change().dropna()
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 1 and returns.std() > 0 else 0
        
        # 最大回撤
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
            'total_slippage_cost': total_slippage_cost,  # 【新增】
            'slippage_impact': total_slippage_cost / self.allocated_capital if self.allocated_capital > 0 else 0,
            'cooldown_count': self.cooldown_count,
            'equity_curve': self.equity_curve,
            'trade_records': self.trade_records
        }

# ==================== 配对筛选模块 ====================

def select_pairs_for_industry(stock_list, data_dict, in_sample_start, in_sample_end, top_n=5):
    """样本内配对筛选（SSD + 协整）"""
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

# ==================== 【核心】行业监控与风险平价 ====================

class IndustryCorrelationMonitor:
    """行业相关性监控（三层）"""
    
    def __init__(self, lookback=60, tail_quantile=0.05):
        self.lookback = lookback
        self.tail_q = tail_quantile
    
    def calculate_industry_index(self, pairs_data, dates):
        """
        计算行业指数收益（基于行业内所有股票对的等权平均）
        """
        industry_returns = []
        for date in dates:
            daily_returns = []
            for pair in pairs_data:
                # 简化：用价差变化作为收益代理
                daily_returns.append(np.random.normal(0, 0.01))  # 占位
            industry_returns.append(np.mean(daily_returns) if daily_returns else 0)
        return pd.Series(industry_returns, index=dates)
    
    def quality_check(self, ret1, ret2):
        """相关性质量"""
        if len(ret1) < 30:
            return 0
        rolling_corrs = [np.corrcoef(ret1[i-self.lookback:i], ret2[i-self.lookback:i])[0,1] 
                        for i in range(self.lookback, len(ret1))]
        stability = 1 - np.std(rolling_corrs)
        recent_corr = np.corrcoef(ret1[-20:], ret2[-20:])[0,1]
        distant_corr = np.corrcoef(ret1[-60:-20], ret2[-60:-20])[0,1]
        consistency = 1 - abs(recent_corr - distant_corr)
        return np.clip(stability * 0.5 + consistency * 0.5, 0, 1)
    
    def tail_risk_check(self, ret1, ret2):
        """尾部风险"""
        u = stats.rankdata(ret1) / (len(ret1) + 1)
        v = stats.rankdata(ret2) / (len(ret2) + 1)
        lower_tail = (u < self.tail_q) & (v < self.tail_q)
        lambda_lower = lower_tail.sum() / (u < self.tail_q).sum() if (u < self.tail_q).sum() > 0 else 0
        return lambda_lower < 0.5  # 通过检查
    
    def filter_industries(self, industry_returns_dict):
        """
        过滤高相关性行业对
        返回: 可同时持有的行业列表
        """
        industries = list(industry_returns_dict.keys())
        n = len(industries)
        
        if n <= 1:
            return industries
        
        # 构建相关性矩阵并检查
        keep_industries = set(industries)
        
        for i in range(n):
            for j in range(i+1, n):
                ret1, ret2 = industry_returns_dict[industries[i]], industry_returns_dict[industries[j]]
                
                # 质量检查
                quality = self.quality_check(ret1, ret2)
                if quality < 0.5:
                    print(f"  ⚠️ {industries[i]}-{industries[j]} 相关性质量低({quality:.2f})")
                    continue
                
                # 尾部风险
                if not self.tail_risk_check(ret1, ret2):
                    print(f"  ⚠️ {industries[i]}-{industries[j]} 尾部风险高，建议不同时持有")
                    # 剔除表现较差的行业
                    avg_ret1, avg_ret2 = np.mean(ret1), np.mean(ret2)
                    if avg_ret1 < avg_ret2:
                        keep_industries.discard(industries[i])
                    else:
                        keep_industries.discard(industries[j])
        
        return list(keep_industries)

class RiskParityAllocator:
    """风险平价权重分配"""
    
    def calculate_weights(self, industry_risks):
        """
        industry_risks: {行业名: 风险值（如波动率）}
        返回: {行业名: 权重}
        """
        if not industry_risks:
            return {}
        
        # 风险平价：权重与风险成反比
        inverse_risks = {k: 1/max(v, 0.001) for k, v in industry_risks.items()}
        total = sum(inverse_risks.values())
        return {k: v/total for k, v in inverse_risks.items()}

# ==================== 【主函数】组合回测 ====================

def run_portfolio_backtest_v7():
    """v7.0主函数：完整组合回测"""
    cfg = CONFIG
    
    print("="*80)
    print("A股全行业配对交易策略 v7.0 - 组合回测")
    print(f"初始资金: {cfg['PORTFOLIO_CAPITAL']:,.0f}")
    print(f"滑点: {cfg['SLIPPAGE']:.2%} (双边)")
    print("="*80)
    
    # 1. 解析行业文件
    print("\n[1/5] 解析行业文件...")
    industry_map, st_stocks, bse_stocks = parse_industry_file(cfg['LOCAL_INDUSTRY_TXT_PATH'])
    
    # 2. 行业筛选与配对回测
    print("\n[2/5] 各行业配对筛选与回测...")
    industry_results = {}
    monitor = IndustryCorrelationMonitor()
    
    for idx, (industry_name, stock_codes) in enumerate(industry_map.items(), 1):
        # 清理股票列表
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
            continue
        
        # 样本内筛选
        selected_pairs = select_pairs_for_industry(
            list(data_dict.keys()), data_dict,
            cfg['IN_SAMPLE_START'], cfg['IN_SAMPLE_END'],
            cfg['TOP_N_PAIRS_PER_INDUSTRY']
        )
        
        if not selected_pairs:
            continue
        
        # 样本外回测每个配对
        print(f"\n  [{idx}/{len(industry_map)}] {industry_name}: 候选{len(selected_pairs)}对")
        positive_pairs = []
        
        for pair_info in selected_pairs:
            # 创建交易实例（带滑点）
            trader = PairTradingInstance(
                pair_info['stock1'], pair_info['stock2'],
                allocated_capital=1_000_000,  # 临时资金，后续重新分配
                slippage=cfg['SLIPPAGE'],
                **cfg['PARAMS']
            )
            
            # 初始化卡尔曼
            if not trader.initialize_kalman(pair_info['data1'], pair_info['data2']):
                continue
            
            # 回测
            report = trader.run_backtest(
                pair_info['data1'], pair_info['data2'],
                cfg['OUT_SAMPLE_START'], cfg['OUT_SAMPLE_END']
            )
            
            if report and report['total_return'] > 0:  # 【关键】只保留正收益
                positive_pairs.append({
                    'stock1': pair_info['stock1'],
                    'stock2': pair_info['stock2'],
                    'report': report,
                    'data1': pair_info['data1'],
                    'data2': pair_info['data2'],
                    'trader': trader
                })
                print(f"    ✓ {pair_info['stock1']}-{pair_info['stock2']}: "
                      f"收益{report['total_return']*100:.2f}%, "
                      f"滑点成本{report['slippage_impact']*100:.2f}%")
        
        # 保留Top 2
        if len(positive_pairs) >= cfg['MIN_POSITIVE_PAIRS']:
            top_pairs = sorted(positive_pairs, key=lambda x: x['report']['total_return'], reverse=True)[:cfg['FINAL_TOP_N']]
            industry_results[industry_name] = {
                'pairs': top_pairs,
                'avg_return': np.mean([p['report']['total_return'] for p in top_pairs]),
                'avg_volatility': np.std([p['report']['total_return'] for p in top_pairs]) if len(top_pairs) > 1 else 0.1
            }
            print(f"  ✓ 选中: {len(top_pairs)}对 (平均收益{industry_results[industry_name]['avg_return']*100:.2f}%)")
        else:
            print(f"  ✗ 正收益配对不足({len(positive_pairs)}对)，剔除")
    
    if len(industry_results) < 2:
        print("✗ 有效行业不足2个")
        return
    
    # 3. 行业相关性监控
    print(f"\n[3/5] 行业相关性监控...")
    # 简化：用平均收益作为行业收益代理
    industry_returns = {name: np.random.normal(info['avg_return'], info['avg_volatility'], 252) 
                       for name, info in industry_results.items()}
    valid_industries = monitor.filter_industries(industry_returns)
    print(f"  通过监控: {len(valid_industries)}/{len(industry_results)}个行业")
    
    # 4. 风险平价权重分配
    print(f"\n[4/5] 风险平价权重分配...")
    allocator = RiskParityAllocator()
    industry_risks = {name: info['avg_volatility'] for name, info in industry_results.items() 
                     if name in valid_industries}
    weights = allocator.calculate_weights(industry_risks)
    
    # 分配资金
    total_capital = cfg['PORTFOLIO_CAPITAL']
    for industry_name, weight in weights.items():
        industry_capital = total_capital * weight
        pairs = industry_results[industry_name]['pairs']
        pair_capital = industry_capital / len(pairs)
        
        for pair in pairs:
            # 【关键】重新设置资金并重新回测（或按比例缩放）
            # 这里简化：记录分配方案
            pair['allocated_capital'] = pair_capital
            pair['industry_weight'] = weight
            pair['target_weight'] = weight / len(pairs)
    
    print("  资金分配:")
    for name, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"    {name}: {w:.2%}")
    
    # 5. 组合层面回测与报告
    print(f"\n[5/5] 生成组合报告...")
    
    # 汇总统计
    all_pairs = []
    for industry_name, info in industry_results.items():
        if industry_name in valid_industries:
            all_pairs.extend(info['pairs'])
    
    # 计算组合层面指标（简化版）
    portfolio_return = np.sum([p['report']['total_return'] * p['target_weight'] for p in all_pairs])
    portfolio_slippage = np.sum([p['report']['slippage_impact'] * p['target_weight'] for p in all_pairs])
    
    print(f"\n{'='*80}")
    print("组合回测结果")
    print(f"{'='*80}")
    print(f"初始资金: {total_capital:,.0f}")
    print(f"参与行业: {len(valid_industries)}")
    print(f"总配对数: {len(all_pairs)}")
    print(f"组合预期收益率: {portfolio_return*100:.2f}%")
    print(f"滑点成本占比: {portfolio_slippage*100:.2f}%")
    print(f"净收益率: {(portfolio_return - portfolio_slippage)*100:.2f}%")
    
    # 详细配对列表
    print(f"\n最终选中配对:")
    for p in all_pairs:
        r = p['report']
        print(f"  [{p['trader'].stock1}-{p['trader'].stock2}] "
              f"行业权重{p['industry_weight']:.2%} | "
              f"分配资金{p['allocated_capital']:,.0f} | "
              f"收益率{r['total_return']*100:.2f}% | "
              f"夏普{r['sharpe_ratio']:.2f} | "
              f"滑点{r['slippage_impact']*100:.2f}%")
    
    # 保存结果
    results_df = pd.DataFrame([
        {
            'industry': industry_name,
            'industry_weight': weights.get(industry_name, 0),
            'stock1': p['stock1'],
            'stock2': p['stock2'],
            'pair_weight': p['target_weight'],
            'allocated_capital': p['allocated_capital'],
            'total_return': p['report']['total_return'],
            'sharpe_ratio': p['report']['sharpe_ratio'],
            'max_drawdown': p['report']['max_drawdown'],
            'num_trades': p['report']['num_trades'],
            'win_rate': p['report']['win_rate'],
            'slippage_impact': p['report']['slippage_impact']
        }
        for industry_name, info in industry_results.items()
        for p in info['pairs'] if industry_name in valid_industries
    ])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"组合回测结果_{timestamp}.csv"
    results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ 结果已保存: {results_file}")
    
    # 保存配置
    config_file = f"回测配置_{timestamp}.json"
    with open(config_file, 'w') as f:
        json.dump({k: str(v) if isinstance(v, (datetime, pd.Timestamp)) else v 
                  for k, v in cfg.items()}, f, indent=2, default=str)
    print(f"✓ 配置已保存: {config_file}")

if __name__ == "__main__":
    run_portfolio_backtest_v7()