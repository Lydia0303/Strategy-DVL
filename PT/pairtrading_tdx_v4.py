# === 完整的优化版代码（实现动态对冲比率 + 动态协整监控）===
# === 第一部分：导入必要的库 ===
import struct
import pandas as pd
import numpy as np
import os
import warnings
import akshare as ak
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# === 第二部分：定义读取通达信数据的函数 (保持不变) ===
def read_tdx_day_file(file_path):
    """读取通达信日线数据文件(.day)"""
    with open(file_path, 'rb') as f:
        data = f.read()
    records = []
    for i in range(0, len(data), 32):
        if i + 32 > len(data):
            break
        buffer = data[i:i+32]
        date, open_price, high, low, close, amount, volume, _ = struct.unpack('IfffffII', buffer)
        year = date // 10000
        month = (date % 10000) // 100
        day = date % 100
        records.append({
            'date': f'{year}-{month:02d}-{day:02d}',
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'amount': amount
        })
    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

def get_stock_data_from_tdx(stock_code, tdx_data_dir):
    """从通达信数据目录获取股票数据"""
    if stock_code.startswith('6'):
        market = 'sh'
    elif stock_code.startswith('0') or stock_code.startswith('3'):
        market = 'sz'
    elif stock_code.startswith('688'):
        market = 'sh'
    else:
        raise ValueError(f"未知股票代码: {stock_code}")
    file_name = f'{market}{stock_code}.day'
    file_path = os.path.join(tdx_data_dir, f'{market}/lday', file_name)
    if os.path.exists(file_path):
        return read_tdx_day_file(file_path)
    else:
        print(f"文件不存在: {file_path}")
        return None

def get_industry_stocks_from_akshare(industry_name="银行", local_file_path=None):
    """从东方财富行业板块中提取指定行业的股票列表（股票代码）
       支持接口调用失败时自动回退到本地CSV文件
       参数：
       industry_name (str): 行业名称，例如 "银行", "房地产", "电力" 等
       local_file_path (str): 本地CSV文件路径，例如 "C:/Users/hz/Desktop/Strategy DVL/东方财富指定行业成分股.csv"
       返回:
       list: 股票代码列表，例如 ['601288', '601398', ...]
       """
    try:
        stock_board_industry_cons_df = ak.stock_board_industry_cons_em(symbol=industry_name)
        if stock_board_industry_cons_df is None or stock_board_industry_cons_df.empty:
            print(f"警告：未获取到行业 '{industry_name}' 的成分股数据")
            return get_industry_stocks_from_local(industry_name, local_file_path)
        
        # 提取股票代码列
        stock_codes = stock_board_industry_cons_df['代码'].astype(str).tolist()
        clean_codes = [] # 清理代码：去除后缀
        for code in stock_codes:
            if '.' in code:
                code = code.split('.')[0]
            if code.isdigit() and len(code) == 6:
                clean_codes.append(code)
            else:
                if code.isdigit():
                    clean_codes.append(code.zfill(6))
        print(f"成功获取行业 '{industry_name}' 的成分股，共 {len(clean_codes)} 只股票")
        print(f"前10只股票代码：{clean_codes[:10]}")
        return clean_codes
    except Exception as e:
        print(f"获取行业 '{industry_name}' 成分股时发生错误：{e}")
        print("正在尝试从本地文件获取...")
        # 接口调用失败，回退到本地文件
        return get_industry_stocks_from_local(industry_name, local_file_path)
def get_industry_stocks_from_local(industry_name, local_file_path=None):
    """
    从本地CSV文件获取指定行业的股票列表
    
    参数:
        industry_name (str): 行业名称
        local_file_path (str): 本地CSV文件路径
    
    返回:
        list: 股票代码列表
    """
    # 如果没有指定本地文件路径，使用默认路径
    if local_file_path is None:
        local_file_path = r"C:\Users\hz\Desktop\Strategy DVL\东方财富指定行业成分股.csv"
    
    try:
        # 检查文件是否存在
        if not os.path.exists(local_file_path):
            print(f"❌ 本地文件不存在: {local_file_path}")
            return []
        
        # 读取CSV文件
        df = pd.read_csv(local_file_path, encoding='utf-8-sig')
        
        if df is None or df.empty:
            print("❌ 本地CSV文件为空")
            return []
        
        # 打印列名以便调试
        print(f"本地CSV文件列名: {df.columns.tolist()}")
        
        # 根据CSV文件结构提取股票代码
        # 假设CSV文件有以下列：'代码', '名称', '最新价', '涨跌幅' 等
        if '代码' in df.columns:
            stock_codes = df['代码'].astype(str).tolist()
        elif '股票代码' in df.columns:
            stock_codes = df['股票代码'].astype(str).tolist()
        else:
            # 尝试找到包含"代码"的列
            code_column = [col for col in df.columns if '代码' in col]
            if code_column:
                stock_codes = df[code_column[0]].astype(str).tolist()
            else:
                print("❌ 未在本地文件中找到股票代码列")
                return []
        
        # 清理代码：去除后缀
        clean_codes = []
        for code in stock_codes:
            if pd.isna(code):
                continue
            code = str(code)
            if '.' in code:
                code = code.split('.')[0]
            if code.isdigit() and len(code) == 6:
                clean_codes.append(code)
            else:
                if code.isdigit():
                    clean_codes.append(code.zfill(6))
        
        print(f"✅ 成功从本地文件获取行业 '{industry_name}' 的成分股，共 {len(clean_codes)} 只股票")
        print(f"本地文件路径: {local_file_path}")
        print(f"前10只股票代码：{clean_codes[:10]}")
        
        return clean_codes
    
    except Exception as e:
        print(f"❌ 从本地文件获取股票列表失败: {e}")
        return []
# === 第三部分：A股适配的强弱轮动策略（优化版）===
class ASharePairsTradingStrategyOptimized:
    """A股配对交易策略（强弱轮动，不允许融券）- 优化版"""
    def __init__(self, initial_capital=1000000, commission_rate=0.0003, 
                 stamp_tax_rate=0.001, slippage=0.001, min_commission=5.0,
                 max_position_ratio=0.5, min_position_ratio=0.1,
                 hedge_ratio=1.0,  # 新增：对冲比率
                 coint_pvalue=0.05,  # 新增：样本内协整检验p值
                 coint_check_window=120,  # 新增：动态协整检验窗口长度
                 coint_check_freq=60,  # 新增：动态协整检验频率（交易日）
                 coint_threshold=0.05,  # 新增：动态协整检验阈值
                 dynamic_check=True):  # 新增：是否启用动态检验
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.slippage = slippage
        self.min_commission = min_commission
        self.max_position_ratio = max_position_ratio
        self.min_position_ratio = min_position_ratio
        
        # 新增：配对关系参数
        self.hedge_ratio = hedge_ratio  # 对冲比率 (log(S1) = hedge_ratio * log(S2) + spread)
        self.coint_pvalue = coint_pvalue  # 样本内协整检验p值
        self.coint_check_window = coint_check_window
        self.coint_check_freq = coint_check_freq
        self.coint_threshold = coint_threshold
        self.dynamic_check = dynamic_check
        
        # 新增：配对关系状态跟踪
        self.pair_valid = True  # 配对是否有效
        self.last_check_date = None  # 上次检查日期
        self.coint_check_count = 0  # 检查次数计数器
        
        # 持仓记录
        self.positions = {}
        self.trade_history = []
        self.equity_curve = []
        self.daily_records = []
        
    def calculate_trading_cost(self, price, volume, is_buy=True):
        """计算交易成本（A股规则）"""
        if is_buy:
            exec_price = price * (1 + self.slippage)
        else:
            exec_price = price * (1 - self.slippage)
        trade_amount = exec_price * volume
        commission = max(trade_amount * self.commission_rate, self.min_commission)
        stamp_tax = 0
        if not is_buy:
            stamp_tax = trade_amount * self.stamp_tax_rate
        total_cost = commission + stamp_tax
        return exec_price, total_cost
    
    def buy_stock(self, date, stock_code, price, volume, reason=""):
        """买入股票（A股只能做多）"""
        if volume <= 0:
            return False, 0, 0
        min_lot = 100
        volume = (volume // min_lot) * min_lot
        if volume <= 0:
            return False, 0, 0
        exec_price, cost = self.calculate_trading_cost(price, volume, is_buy=True)
        required_capital = exec_price * volume + cost
        if required_capital > self.capital:
            max_volume = int(self.capital / (exec_price * 1.01) / min_lot) * min_lot
            if max_volume <= 0:
                return False, 0, 0
            volume = max_volume
            required_capital = exec_price * volume + cost
        self.capital -= required_capital
        if stock_code in self.positions:
            pos = self.positions[stock_code]
            old_value = pos['qty'] * pos['avg_price']
            new_qty = pos['qty'] + volume
            new_avg_price = (old_value + exec_price * volume) / new_qty
            pos['qty'] = new_qty
            pos['avg_price'] = new_avg_price
        else:
            self.positions[stock_code] = {
                'qty': volume,
                'avg_price': exec_price,
                'entry_date': date
            }
        self.trade_history.append({
            'date': date,
            'action': '买入',
            'stock': stock_code,
            'reason': reason,
            'price': price,
            'exec_price': exec_price,
            'volume': volume,
            'amount': exec_price * volume,
            'cost': cost,
            'capital_after': self.capital
        })
        return True, exec_price, cost
    
    def sell_stock(self, date, stock_code, price, volume, reason=""):
        """卖出股票（必须有持仓）"""
        if stock_code not in self.positions:
            return False, 0, 0
        pos = self.positions[stock_code]
        if pos['qty'] <= 0:
            return False, 0, 0
        volume = min(volume, pos['qty'])
        min_lot = 100
        volume = (volume // min_lot) * min_lot
        if volume <= 0:
            return False, 0, 0
        exec_price, cost = self.calculate_trading_cost(price, volume, is_buy=False)
        sell_amount = exec_price * volume
        pnl = (exec_price - pos['avg_price']) * volume - cost
        self.capital += sell_amount - cost
        pos['qty'] -= volume
        if pos['qty'] <= 0:
            del self.positions[stock_code]
        self.trade_history.append({
            'date': date,
            'action': '卖出',
            'stock': stock_code,
            'reason': reason,
            'price': price,
            'exec_price': exec_price,
            'volume': volume,
            'amount': sell_amount,
            'cost': cost,
            'pnl': pnl,
            'capital_after': self.capital
        })
        return True, exec_price, pnl
    
    def calculate_spread_zscore(self, stock1_data, stock2_data, window=40):
        """
        计算价差的z-score序列 (优化版)
        使用训练期确定的对冲比率 hedge_ratio 构建价差: spread = log(S1) - hedge_ratio * log(S2)
        """
        common_dates = stock1_data.index.intersection(stock2_data.index)
        if len(common_dates) < window * 2:
            return None, None, None, None
        
        s1 = stock1_data.loc[common_dates]
        s2 = stock2_data.loc[common_dates]
        
        log_s1 = np.log(s1)
        log_s2 = np.log(s2)
        
        # 优化点1：使用回归确定的对冲比率计算价差
        spread = log_s1 - self.hedge_ratio * log_s2
        
        zscore = pd.Series(index=spread.index, dtype=float)
        spread_mean_series = pd.Series(index=spread.index, dtype=float)
        spread_std_series = pd.Series(index=spread.index, dtype=float)
        
        for i in range(window, len(spread)):
            window_spread = spread.iloc[i-window:i]
            mean_spread = window_spread.mean()
            std_spread = window_spread.std()
            spread_mean_series.iloc[i] = mean_spread
            spread_std_series.iloc[i] = std_spread
            
            if std_spread > 0:
                zscore.iloc[i] = (spread.iloc[i] - mean_spread) / std_spread
            else:
                zscore.iloc[i] = 0
        
        zscore = zscore.fillna(0)
        spread_mean_series = spread_mean_series.fillna(method='bfill')
        spread_std_series = spread_std_series.fillna(method='bfill')
        
        return spread, zscore, spread_mean_series, spread_std_series

    def check_cointegration(self, stock1_data, stock2_data, date=None):
        """
        动态协整检验
        在给定数据窗口上检验两只股票是否仍存在协整关系
        """
        if len(stock1_data) < self.coint_check_window or len(stock2_data) < self.coint_check_window:
            return None, None
        
        # 对齐数据
        common_idx = stock1_data.index.intersection(stock2_data.index)
        if len(common_idx) < self.coint_check_window:
            return None, None
        
        s1_subset = stock1_data.loc[common_idx].iloc[-self.coint_check_window:]
        s2_subset = stock2_data.loc[common_idx].iloc[-self.coint_check_window:]
        
        log_s1 = np.log(s1_subset)
        log_s2 = np.log(s2_subset)
        
        # 进行协整检验
        coint_result = coint(log_s1, log_s2)
        coint_p = coint_result[1]
        
        # 价差平稳性检验
        spread = log_s1 - self.hedge_ratio * log_s2
        adf_result = adfuller(spread.dropna())
        adf_p = adf_result[1]
        
        if date:
            print(f"  [{date.strftime('%Y-%m-%d')}] 动态协整检验: p值={coint_p:.4f}, ADF p值={adf_p:.4f}, 窗口={self.coint_check_window}")
        
        return coint_p, adf_p

    def calculate_total_value(self, date, price1, price2):
        """计算投资组合总价值"""
        total_value = self.capital
        for stock_code, pos in self.positions.items():
            if stock_code == 'stock1_code':  # 注意：这里需要使用实际传入的股票代码变量名
                price = price1
            elif stock_code == 'stock2_code':
                price = price2
            else:
                price = pos['avg_price']
            total_value += price * pos['qty']
        return total_value

    def execute_strategy(self, stock1_code, stock2_code, stock1_data, stock2_data, 
                        window=40, entry_threshold=1.5, exit_threshold=0.3, 
                        stop_loss=2.0, max_holding_days=20, rebalance_threshold=0.8):
        """
        执行强弱轮动策略 (优化版，增加动态协整检验)
        注意：此函数中的 stock1_code, stock2_code 是具体的股票代码字符串
        """
        # 初始化配对状态
        self.pair_valid = True
        self.last_check_date = None
        self.coint_check_count = 0
        
        # 计算价差和Z-Score
        spread, zscore, spread_mean, spread_std = self.calculate_spread_zscore(
            stock1_data, stock2_data, window
        )
        if zscore is None:
            print(f"数据不足，无法计算z-score")
            return None
        
        holding_stock = None
        entry_date = None
        entry_zscore = 0
        holding_days = 0
        
        # 获取回测期间所有日期
        trading_dates = zscore.index[window:].tolist()
        
        for i, date in enumerate(trading_dates):
            if date not in stock1_data.index or date not in stock2_data.index:
                continue
            
            z = zscore.loc[date]
            price1 = stock1_data.loc[date]
            price2 = stock2_data.loc[date]
            
            # 优化点2：动态协整检验
            if self.dynamic_check and self.pair_valid:
                # 检查是否到了协整检验的日期
                need_check = False
                if self.last_check_date is None:
                    need_check = True
                else:
                    days_since_last = (date - self.last_check_date).days
                    if days_since_last >= self.coint_check_freq:
                        need_check = True
                
                if need_check:
                    # 获取截止到当前日期的历史数据
                    historical_data1 = stock1_data.loc[:date]
                    historical_data2 = stock2_data.loc[:date]
                    
                    coint_p, adf_p = self.check_cointegration(historical_data1, historical_data2, date)
                    
                    if coint_p is not None and adf_p is not None:
                        self.coint_check_count += 1
                        self.last_check_date = date
                        
                        # 如果协整关系破裂，强制平仓并标记配对失效
                        if coint_p > self.coint_threshold or adf_p > self.coint_threshold:
                            print(f"  ⚠️  [{date.strftime('%Y-%m-%d')}] 协整关系破裂！p值={coint_p:.4f}， 强制平仓。")
                            self.pair_valid = False
                            
                            # 强制平仓所有持仓
                            if holding_stock == 'stock1':
                                if stock1_code in self.positions:
                                    pos = self.positions[stock1_code]
                                    sell_volume = pos['qty']
                                    if sell_volume > 0:
                                        self.sell_stock(date, stock1_code, price1, sell_volume,
                                                      reason=f"强制平仓-协整关系破裂")
                            elif holding_stock == 'stock2':
                                if stock2_code in self.positions:
                                    pos = self.positions[stock2_code]
                                    sell_volume = pos['qty']
                                    if sell_volume > 0:
                                        self.sell_stock(date, stock2_code, price2, sell_volume,
                                                      reason=f"强制平仓-协整关系破裂")
                            holding_stock = None
                            entry_date = None
                            entry_zscore = 0
                            holding_days = 0
            
            # 如果配对已失效，跳过所有交易逻辑
            if not self.pair_valid:
                # 只记录净值，不交易
                total_value = self.calculate_total_value(date, price1, price2)
                daily_record = {
                    'date': date,
                    'zscore': z,
                    'price1': price1,
                    'price2': price2,
                    'capital': self.capital,
                    'total_value': total_value,
                    'holding_stock': None,
                    'holding_days': 0,
                    'pair_valid': False,
                    'hedge_ratio': self.hedge_ratio
                }
                self.daily_records.append(daily_record)
                self.equity_curve.append((date, total_value))
                continue
            
            has_stock1 = stock1_code in self.positions
            has_stock2 = stock2_code in self.positions
            total_value = self.calculate_total_value(date, price1, price2)
            
            daily_record = {
                'date': date,
                'zscore': z,
                'price1': price1,
                'price2': price2,
                'capital': self.capital,
                'total_value': total_value,
                'holding_stock': holding_stock,
                'holding_days': holding_days,
                'pair_valid': self.pair_valid,
                'hedge_ratio': self.hedge_ratio
            }
            self.daily_records.append(daily_record)
            
            # 交易逻辑（仅在配对有效时执行）
            # 情况1：无持仓
            if holding_stock is None:
                if z > entry_threshold:  # stock1相对高估，stock2低估
                    position_value = min(total_value * self.max_position_ratio, self.capital)
                    volume = int(position_value / price2 / 100) * 100
                    
                    if volume > 0:
                        success, _, _ = self.buy_stock(date, stock2_code, price2, volume, 
                                                      reason=f"开仓-买入低估股({stock2_code}), z={z:.2f}")
                        if success:
                            holding_stock = 'stock2'
                            entry_date = date
                            entry_zscore = z
                            holding_days = 0
                
                elif z < -entry_threshold:  # stock1相对低估，stock2高估
                    position_value = min(total_value * self.max_position_ratio, self.capital)
                    volume = int(position_value / price1 / 100) * 100
                    
                    if volume > 0:
                        success, _, _ = self.buy_stock(date, stock1_code, price1, volume, 
                                                      reason=f"开仓-买入低估股({stock1_code}), z={z:.2f}")
                        if success:
                            holding_stock = 'stock1'
                            entry_date = date
                            entry_zscore = z
                            holding_days = 0
            
            # 情况2：持有股票2
            elif holding_stock == 'stock2':
                holding_days += 1
                
                if z < -rebalance_threshold:  # stock1变得低估
                    if has_stock2:
                        pos = self.positions[stock2_code]
                        sell_volume = pos['qty']
                        if sell_volume > 0:
                            self.sell_stock(date, stock2_code, price2, sell_volume,
                                          reason=f"轮动-卖出高估股, z从{entry_zscore:.2f}到{z:.2f}")
                    
                    position_value = min(total_value * self.max_position_ratio, self.capital)
                    volume = int(position_value / price1 / 100) * 100
                    
                    if volume > 0:
                        success, _, _ = self.buy_stock(date, stock1_code, price1, volume,
                                                      reason=f"轮动-买入低估股, z={z:.2f}")
                        if success:
                            holding_stock = 'stock1'
                            entry_date = date
                            entry_zscore = z
                            holding_days = 0
                
                elif abs(z) < exit_threshold:  # 价差回归
                    if has_stock2:
                        pos = self.positions[stock2_code]
                        sell_volume = pos['qty']
                        if sell_volume > 0:
                            self.sell_stock(date, stock2_code, price2, sell_volume,
                                          reason=f"平仓-价差回归, z={z:.2f}")
                    holding_stock = None
                    entry_date = None
                    entry_zscore = 0
                    holding_days = 0
                
                elif abs(z) > stop_loss:  # 止损
                    if has_stock2:
                        pos = self.positions[stock2_code]
                        sell_volume = pos['qty']
                        if sell_volume > 0:
                            self.sell_stock(date, stock2_code, price2, sell_volume,
                                          reason=f"平仓-止损, z={z:.2f}")
                    holding_stock = None
                    entry_date = None
                    entry_zscore = 0
                    holding_days = 0
                
                elif holding_days >= max_holding_days:  # 时间止损
                    if has_stock2:
                        pos = self.positions[stock2_code]
                        sell_volume = pos['qty']
                        if sell_volume > 0:
                            self.sell_stock(date, stock2_code, price2, sell_volume,
                                          reason=f"平仓-时间止损, 持仓{holding_days}天")
                    holding_stock = None
                    entry_date = None
                    entry_zscore = 0
                    holding_days = 0
            
            # 情况3：持有股票1
            elif holding_stock == 'stock1':
                holding_days += 1
                
                if z > rebalance_threshold:  # stock2变得低估
                    if has_stock1:
                        pos = self.positions[stock1_code]
                        sell_volume = pos['qty']
                        if sell_volume > 0:
                            self.sell_stock(date, stock1_code, price1, sell_volume,
                                          reason=f"轮动-卖出高估股, z从{entry_zscore:.2f}到{z:.2f}")
                    
                    position_value = min(total_value * self.max_position_ratio, self.capital)
                    volume = int(position_value / price2 / 100) * 100
                    
                    if volume > 0:
                        success, _, _ = self.buy_stock(date, stock2_code, price2, volume,
                                                      reason=f"轮动-买入低估股, z={z:.2f}")
                        if success:
                            holding_stock = 'stock2'
                            entry_date = date
                            entry_zscore = z
                            holding_days = 0
                
                elif abs(z) < exit_threshold:  # 价差回归
                    if has_stock1:
                        pos = self.positions[stock1_code]
                        sell_volume = pos['qty']
                        if sell_volume > 0:
                            self.sell_stock(date, stock1_code, price1, sell_volume,
                                          reason=f"平仓-价差回归, z={z:.2f}")
                    holding_stock = None
                    entry_date = None
                    entry_zscore = 0
                    holding_days = 0
                
                elif abs(z) > stop_loss:  # 止损
                    if has_stock1:
                        pos = self.positions[stock1_code]
                        sell_volume = pos['qty']
                        if sell_volume > 0:
                            self.sell_stock(date, stock1_code, price1, sell_volume,
                                          reason=f"平仓-止损, z={z:.2f}")
                    holding_stock = None
                    entry_date = None
                    entry_zscore = 0
                    holding_days = 0
                
                elif holding_days >= max_holding_days:  # 时间止损
                    if has_stock1:
                        pos = self.positions[stock1_code]
                        sell_volume = pos['qty']
                        if sell_volume > 0:
                            self.sell_stock(date, stock1_code, price1, sell_volume,
                                          reason=f"平仓-时间止损, 持仓{holding_days}天")
                    holding_stock = None
                    entry_date = None
                    entry_zscore = 0
                    holding_days = 0
            
            self.equity_curve.append((date, total_value))
        
        return pd.DataFrame(self.daily_records).set_index('date')

# === 新增函数：在样本内计算配对参数 ===
def calculate_pair_parameters_in_sample(stock1_data, stock2_data, window_start, window_end):
    """
    在样本内数据上计算配对参数
    返回: hedge_ratio, coint_pvalue, adf_pvalue, 以及训练期内的价差统计
    """
    # 对齐数据
    common_dates = stock1_data.index.intersection(stock2_data.index)
    common_dates = common_dates[(common_dates >= window_start) & (common_dates <= window_end)]
    
    if len(common_dates) < 60:  # 至少需要60个交易日
        return None, None, None, None, None, None
    
    s1 = stock1_data.loc[common_dates]
    s2 = stock2_data.loc[common_dates]
    
    # 取对数价格
    log_s1 = np.log(s1)
    log_s2 = np.log(s2)
    
    # 优化点1：通过OLS回归计算对冲比率
    # 模型: log(S1) = hedge_ratio * log(S2) + intercept
    X = sm.add_constant(log_s2.values)  # 添加常数项
    model = sm.OLS(log_s1.values, X).fit()
    hedge_ratio = model.params[1]  # 获取对冲比率系数
    intercept = model.params[0]   # 获取截距项
    
    print(f"  对冲比率回归结果: hedge_ratio = {hedge_ratio:.4f}, R-squared = {model.rsquared:.4f}")
    
    # 计算训练期价差
    spread_in_sample = log_s1 - hedge_ratio * log_s2
    
    # 协整检验
    coint_result = coint(log_s1, log_s2)
    coint_pvalue = coint_result[1]
    
    # 价差平稳性检验
    adf_result = adfuller(spread_in_sample.dropna())
    adf_pvalue = adf_result[1]
    
    # 计算训练期价差的统计特征
    spread_mean = spread_in_sample.mean()
    spread_std = spread_in_sample.std()
    
    return hedge_ratio, intercept, coint_pvalue, adf_pvalue, spread_mean, spread_std

# === 第四步：主程序（优化版，包含样本内参数计算）===
if __name__ == "__main__":
    # 配置参数
    TDX_DATA_DIR = "C:/new_tdx/vipdoc"  # 修改为你的通达信数据目录
    
    # 时间段划分
    IN_SAMPLE_START = "2021-08-01"   # 样本内开始
    IN_SAMPLE_END = "2024-12-31"     # 样本内结束
    OUT_SAMPLE_START = "2025-01-01"  # 样本外开始
    OUT_SAMPLE_END = "2026-03-31"    # 样本外结束
    
    # 策略参数
    TARGET_INDUSTRY = "银行"  # 目标行业名称
    TOP_N_PAIRS = 3  # 只回测前N对！
    min_data_length = 100
    
    # 动态协整检验参数
    COINT_CHECK_WINDOW = 120  # 动态检验窗口长度
    COINT_CHECK_FREQ = 60     # 动态检验频率（交易日）
    COINT_THRESHOLD = 0.05    # 协整检验p值阈值
    
    print("="*60)
    print("A股配对交易策略（优化版：动态对冲比率 + 动态协整监控）")
    print(f"目标行业: {TARGET_INDUSTRY}")
    print(f"只回测排名前{TOP_N_PAIRS}的配对")
    print("="*60)
    
    # 从AKShare获取行业股票列表
    print(f"\n正在从东方财富获取 {TARGET_INDUSTRY} 行业股票列表...")
    local_csv_path = r"C:\Users\hz\Desktop\Strategy DVL\东方财富指定行业成分股.csv"
    industry_stocks = get_industry_stocks_from_akshare(TARGET_INDUSTRY, local_csv_path)
    
    if not industry_stocks:
        print(f"未能获取到 {TARGET_INDUSTRY} 行业的股票列表，程序退出")
        exit()
    
    print(f"获取到 {len(industry_stocks)} 只股票，开始策略回测...")
    
    print(f"\n第一步：样本内选股与参数计算 ({IN_SAMPLE_START} 到 {IN_SAMPLE_END})")
    
    # 加载样本内数据
    all_prices = []
    valid_stocks = []
    stock_data_dict = {}  # 存储股票数据
    
    for stock_code in industry_stocks:
        df = get_stock_data_from_tdx(stock_code, TDX_DATA_DIR)
        if df is not None:
            price_series = df['close'].loc[IN_SAMPLE_START:IN_SAMPLE_END]
            if len(price_series) >= min_data_length:
                all_prices.append(price_series)
                valid_stocks.append(stock_code)
                stock_data_dict[stock_code] = df['close']  # 存储完整数据
                print(f"  ✓ 加载: {stock_code}, {len(price_series)} 条数据")
            else:
                print(f"  ✗ 跳过: {stock_code}, 数据不足")
    
    print(f"\n成功加载 {len(valid_stocks)} 只股票的样本内数据")
    
    if len(valid_stocks) < 2:
        print("有效股票数量不足，无法进行配对分析")
        exit()
    
    # 计算SSD距离
    print(f"\n计算SSD距离矩阵...")
    n = len(valid_stocks)
    candidate_pairs = []
    
    for i in range(n):
        for j in range(i+1, n):
            data1 = all_prices[i]
            data2 = all_prices[j]
            
            common_dates = data1.index.intersection(data2.index)
            if len(common_dates) < min_data_length:
                continue
            
            data1_aligned = data1.loc[common_dates]
            data2_aligned = data2.loc[common_dates]
            
            # 归一化计算SSD
            norm_data1 = data1_aligned / data1_aligned.iloc[0]
            norm_data2 = data2_aligned / data2_aligned.iloc[0]
            ssd_dist = np.sum((norm_data1 - norm_data2) ** 2)
            
            # 计算相关性
            correlation = data1_aligned.corr(data2_aligned)
            
            candidate_pairs.append((valid_stocks[i], valid_stocks[j], ssd_dist, correlation))
    
    # 按SSD距离排序
    candidate_pairs.sort(key=lambda x: x[2])
    print(f"SSD筛选: 从 {len(candidate_pairs)} 对中选取了 {min(20, len(candidate_pairs))} 对进行协整检验")
    
    # 样本内参数计算与筛选
    selected_pairs_with_params = []  # 存储包含参数的结果
    for stock1, stock2, ssd_dist, correlation in candidate_pairs[:20]:
        # 获取完整数据
        if stock1 not in stock_data_dict or stock2 not in stock_data_dict:
            continue
        
        data1_full = stock_data_dict[stock1]
        data2_full = stock_data_dict[stock2]
        
        # 在样本内计算配对参数
        hedge_ratio, intercept, coint_p, adf_p, spread_mean, spread_std = calculate_pair_parameters_in_sample(
            data1_full, data2_full, IN_SAMPLE_START, IN_SAMPLE_END
        )
        
        if hedge_ratio is None:
            continue
        
        # 筛选条件
        if correlation > 0.90 and coint_p < 0.05 and adf_p < 0.05:
            selected_pairs_with_params.append({
                'stock1': stock1,
                'stock2': stock2,
                'ssd': ssd_dist,
                'corr': correlation,
                'hedge_ratio': hedge_ratio,
                'intercept': intercept,
                'coint_p': coint_p,
                'adf_p': adf_p,
                'spread_mean': spread_mean,
                'spread_std': spread_std
            })
            print(f"  ✓ 符合条件: {stock1} - {stock2}")
            print(f"    SSD距离: {ssd_dist:.1f}, 相关性: {correlation:.3f}")
            print(f"    对冲比率: {hedge_ratio:.4f}, 协整p值: {coint_p:.4f}, 价差标准差: {spread_std:.4f}")
    
    print(f"\n✅ 样本内选股与参数计算完成！共筛选出 {len(selected_pairs_with_params)} 对符合条件的股票")
    
    if not selected_pairs_with_params:
        print("没有符合条件的股票对，程序退出")
        exit()
    
    # 按SSD距离排序，只选取前TOP_N_PAIRS对
    selected_pairs_with_params.sort(key=lambda x: x['ssd'])
    top_pairs = selected_pairs_with_params[:TOP_N_PAIRS]
    
    print(f"\n只回测排名前{TOP_N_PAIRS}的配对:")
    for i, pair in enumerate(top_pairs):
        print(f"{i+1}. {pair['stock1']} - {pair['stock2']}: SSD={pair['ssd']:.1f}, "
              f"相关性={pair['corr']:.3f}, 对冲比率={pair['hedge_ratio']:.4f}")
    
    # 保存样本内筛选结果
    sample_in_df = pd.DataFrame([{
        'Stock1': p['stock1'],
        'Stock2': p['stock2'],
        'SSD距离': p['ssd'],
        '相关性': p['corr'],
        '对冲比率': p['hedge_ratio'],
        '截距项': p['intercept'],
        '协整p值': p['coint_p'],
        'ADF p值': p['adf_p'],
        '价差均值': p['spread_mean'],
        '价差标准差': p['spread_std']
    } for p in selected_pairs_with_params])
    sample_in_df.to_csv(f'A股配对_样本内筛选结果_优化版_前{TOP_N_PAIRS}对.csv', index=False, encoding='utf-8-sig')
    print(f"样本内筛选结果已保存到: A股配对_样本内筛选结果_优化版_前{TOP_N_PAIRS}对.csv")
    
    print(f"\n第二步：样本外回测 ({OUT_SAMPLE_START} 到 {OUT_SAMPLE_END})")
    print("="*60)
    
    # 对前TOP_N_PAIRS对股票进行样本外回测
    all_results = []
    
    for idx, pair in enumerate(top_pairs):
        stock1 = pair['stock1']
        stock2 = pair['stock2']
        hedge_ratio = pair['hedge_ratio']
        coint_p = pair['coint_p']
        
        print(f"\n回测第 {idx+1}/{len(top_pairs)} 对: {stock1} - {stock2}")
        print(f"对冲比率: {hedge_ratio:.4f}, 样本内协整p值: {coint_p:.4f}")
        
        # 创建策略实例，传入训练期计算出的参数
        strategy = ASharePairsTradingStrategyOptimized(
            initial_capital=1000000,
            commission_rate=0.0003,
            stamp_tax_rate=0.001,
            slippage=0.001,
            min_commission=5.0,
            max_position_ratio=0.5,
            min_position_ratio=0.1,
            hedge_ratio=hedge_ratio,  # 传入对冲比率
            coint_pvalue=coint_p,     # 传入协整p值
            coint_check_window=COINT_CHECK_WINDOW,
            coint_check_freq=COINT_CHECK_FREQ,
            coint_threshold=COINT_THRESHOLD,
            dynamic_check=True  # 启用动态协整检验
        )
        
        # 加载样本外数据
        df1 = get_stock_data_from_tdx(stock1, TDX_DATA_DIR)
        df2 = get_stock_data_from_tdx(stock2, TDX_DATA_DIR)
        
        if df1 is None or df2 is None:
            print(f"  ✗ 数据加载失败，跳过")
            continue
        
        # 提取样本外数据
        stock1_data = df1['close'].loc[OUT_SAMPLE_START:OUT_SAMPLE_END]
        stock2_data = df2['close'].loc[OUT_SAMPLE_START:OUT_SAMPLE_END]
        
        if len(stock1_data) < 60 or len(stock2_data) < 60:
            print(f"  ✗ 数据不足，跳过")
            continue
        
        # 执行策略
        result = strategy.execute_strategy(
            stock1_code=stock1,
            stock2_code=stock2,
            stock1_data=stock1_data,
            stock2_data=stock2_data,
            window=40,
            entry_threshold=1.5,
            exit_threshold=0.3,
            stop_loss=2.0,
            max_holding_days=20,
            rebalance_threshold=0.8
        )
        
        if result is not None:
            # 计算收益率
            initial_value = 1000000
            final_value = strategy.capital
            
            # 加上持仓市值
            for stock_code, pos in strategy.positions.items():
                if stock_code == stock1 and stock1_data.index[-1] in stock1_data.index:
                    price = stock1_data.loc[stock1_data.index[-1]]
                elif stock_code == stock2 and stock2_data.index[-1] in stock2_data.index:
                    price = stock2_data.loc[stock2_data.index[-1]]
                else:
                    price = pos['avg_price']
                
                final_value += price * pos['qty']
            
            total_return = (final_value - initial_value) / initial_value
            total_trades = len(strategy.trade_history)
            
            # 计算胜率
            win_trades = len([t for t in strategy.trade_history if t.get('pnl', 0) > 0])
            loss_trades = len([t for t in strategy.trade_history if t.get('pnl', 0) < 0])
            win_rate = win_trades / total_trades * 100 if total_trades > 0 else 0
            
            # 计算夏普比率
            if len(strategy.equity_curve) > 1:
                dates, values = zip(*strategy.equity_curve)
                returns = pd.Series(values).pct_change().dropna()
                if len(returns) > 1 and returns.std() > 0:
                    sharpe = np.sqrt(252) * returns.mean() / returns.std()
                else:
                    sharpe = 0
            else:
                sharpe = 0
            
            # 统计协整检验次数
            coint_checks = strategy.coint_check_count
            
            print(f"  ✓ 回测完成:")
            print(f"    初始资金: {initial_value:,.0f}")
            print(f"    最终资金: {final_value:,.0f}")
            print(f"    总收益率: {total_return:.2%}")
            print(f"    交易次数: {total_trades}")
            print(f"    胜率: {win_rate:.1f}%")
            print(f"    夏普比率: {sharpe:.2f}")
            print(f"    动态协整检验次数: {coint_checks}")
            print(f"    配对最终状态: {'有效' if strategy.pair_valid else '已失效'}")
            
            # 保存结果
            all_results.append({
                'pair': f"{stock1}-{stock2}",
                'ssd': pair['ssd'],
                'corr': pair['corr'],
                'hedge_ratio': hedge_ratio,
                'initial': initial_value,
                'final': final_value,
                'return': total_return,
                'trades': total_trades,
                'win_rate': win_rate,
                'sharpe': sharpe,
                'coint_checks': coint_checks,
                'pair_valid': strategy.pair_valid,
                'strategy': strategy
            })
            
            # 保存交易记录
            trades_df = pd.DataFrame(strategy.trade_history)
            trades_df.to_csv(f'A股配对_交易记录_{stock1}_{stock2}_优化版_前{TOP_N_PAIRS}对.csv', index=False, encoding='utf-8-sig')
            
            # 保存每日净值
            if strategy.equity_curve:
                equity_df = pd.DataFrame(strategy.equity_curve, columns=['date', 'value'])
                equity_df.set_index('date', inplace=True)
                equity_df.to_csv(f'A股配对_每日净值_{stock1}_{stock2}_优化版_前{TOP_N_PAIRS}对.csv', encoding='utf-8-sig')
    
    # 输出汇总结果
    if all_results:
        print(f"\n" + "="*60)
        print(f"样本外回测汇总结果（优化版，前{TOP_N_PAIRS}对）")
        print("="*60)
        
        # 计算总体表现
        total_initial = sum(r['initial'] for r in all_results)
        total_final = sum(r['final'] for r in all_results)
        overall_return = (total_final - total_initial) / total_initial
        
        # 统计配对有效性
        valid_pairs = sum(1 for r in all_results if r['pair_valid'])
        
        print(f"回测配对数量: {len(all_results)}")
        print(f"最终有效配对数量: {valid_pairs}")
        print(f"初始总资金: {total_initial:,.0f}")
        print(f"最终总资金: {total_final:,.0f}")
        print(f"总收益率: {overall_return:.2%}")
        
        # 按收益率排序
        sorted_results = sorted(all_results, key=lambda x: x['return'], reverse=True)
        
        print(f"\n配对表现排名:")
        for i, r in enumerate(sorted_results):
            status = "有效" if r['pair_valid'] else "失效"
            print(f"{i+1}. {r['pair']}: 收益率={r['return']:.2%}, 对冲比率={r['hedge_ratio']:.4f}, "
                  f"交易次数={r['trades']}, 胜率={r['win_rate']:.1f}%, 状态={status}")
        
        # 绘制净值曲线
        plt.rcParams["font.family"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "Microsoft YaHei"]
        plt.rcParams["axes.unicode_minus"] = False

        plt.figure(figsize=(16, 9))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, r in enumerate(sorted_results):
            if r['strategy'].daily_records:
                dates = [rec['date'] for rec in r['strategy'].daily_records]
                total_values = [rec['total_value'] for rec in r['strategy'].daily_records]
                
                net_values = [v / 1000000.0 for v in total_values]
                color_idx = i % len(colors)
                
                plt.plot(dates, net_values, label=f"{r['pair']} (收益率: {r['return']:.2%})", 
                         linewidth=2.5, color=colors[color_idx], alpha=0.9)
        
        # 初始净值参考线
        plt.axhline(y=1.0, color='r', linestyle='--', linewidth=1.5, label='初始净值(1.0)')
        
        # 计算等权重组合净值
        if len(all_results) > 1:
            combined_dates = None
            combined_values = None
            
            for r in all_results:
                if r['strategy'].daily_records:
                    dates_dict = {rec['date']: rec['total_value']/1000000.0 for rec in r['strategy'].daily_records}
                    
                    if combined_dates is None:
                        combined_dates = list(dates_dict.keys())
                        combined_values = [dates_dict[d] for d in combined_dates]
                    else:
                        # 对齐日期
                        common_dates = set(combined_dates) & set(dates_dict.keys())
                        if common_dates:
                            common_dates = sorted(list(common_dates))
                            combined_dates = common_dates
                            temp_vals1 = [dates_dict[d] for d in common_dates]
                            temp_vals2 = [combined_values[combined_dates.index(d)] for d in common_dates]
                            combined_values = [(v1+v2)/2 for v1, v2 in zip(temp_vals1, temp_vals2)]
            
            if combined_dates and combined_values:
                plt.plot(combined_dates, combined_values, label=f'等权重组合 (收益率: {overall_return:.2%})', 
                         linewidth=3, color='black', linestyle='-', alpha=0.8)
        
        plt.title(f'{TARGET_INDUSTRY}行业配对交易策略样本外回测（优化版，前{TOP_N_PAIRS}对）\n'
                  f'动态对冲比率 + 动态协整监控 | 总收益率: {overall_return:.2%}', 
                  fontsize=14, fontweight='bold')
        
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('净值（相对初始资金）', fontsize=12)
        
        plt.legend(loc='best', fontsize=10, frameon=True, shadow=True)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.xticks(rotation=45, ha='right', fontsize=10)       
        plt.tight_layout()
        plt.savefig(f'{TARGET_INDUSTRY}配对_样本外回测净值曲线_优化版_前{TOP_N_PAIRS}对.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 保存汇总结果
        summary_df = pd.DataFrame([{
            '股票对': r['pair'],
            'SSD距离': r['ssd'],
            '相关性': r['corr'],
            '对冲比率': r['hedge_ratio'],
            '初始资金': r['initial'],
            '最终资金': r['final'],
            '总收益率': r['return'],
            '年化收益率': (1 + r['return']) ** (252/len(r['strategy'].equity_curve)) - 1 if r['strategy'].equity_curve else 0,
            '交易次数': r['trades'],
            '胜率': r['win_rate'],
            '夏普比率': r['sharpe'],
            '协整检验次数': r['coint_checks'],
            '配对最终状态': '有效' if r['pair_valid'] else '失效'
        } for r in all_results])
        
        summary_df.to_csv(f'{TARGET_INDUSTRY}配对_样本外回测汇总_优化版_前{TOP_N_PAIRS}对.csv', index=False, encoding='utf-8-sig')
        print(f"\n详细结果已保存到: {TARGET_INDUSTRY}配对_样本外回测汇总_优化版_前{TOP_N_PAIRS}对.csv")
    else:
        print("\n没有成功的回测结果")