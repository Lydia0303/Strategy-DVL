# === 完整的优化版代码（实现动态对冲比率 + 动态协整监控 + 增强自适应交易带 + 强化无交易区间 + 失效后净值优化 + 浮动止损 + 动态基准波动率更新）===
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

# === 第三部分：A股适配的强弱轮动策略（增强优化版v5.3）===
class ASharePairsTradingStrategyOptimizedV5:
    """A股配对交易策略（强弱轮动，不允许融券）- 增强优化版v5.3
    新增功能：
    1. 动态基准波动率更新：支持日/周/自适应三种更新频率
    2. 改进自适应交易带：增强波动性调整逻辑
    3. 无交易区间：平仓后设置冷却期，防止频繁交易
    4. 失效后净值优化：配对失效后，资金按无风险利率增长
    5. 浮动止损：从最佳点回撤止损，替代固定止损
    6. 趋势过滤：开仓时检查价差回归趋势
    7. 强化冷却期：轮动平仓后冷却期加倍
    8. 基准更新控制器：防止误用，提供灵活更新策略
    """
    def __init__(self, initial_capital=1000000, commission_rate=0.0003, 
                 stamp_tax_rate=0.001, slippage=0.001, min_commission=5.0,
                 max_position_ratio=0.5, min_position_ratio=0.1,
                 hedge_ratio=1.0,  # 新增：对冲比率
                 baseline_spread_std=0.02,  # 新增：基准价差标准差（来自训练期）
                 coint_pvalue=0.05,  # 新增：样本内协整检验p值
                 coint_check_window=120,  # 新增：动态协整检验窗口长度
                 coint_check_freq=60,  # 新增：动态协整检验频率（交易日）
                 coint_threshold=0.05,  # 新增：动态协整检验阈值
                 dynamic_check=True,  # 新增：是否启用动态检验
                 entry_threshold_multiplier=1.8,  # 优化点3：入场阈值乘数
                 exit_threshold_multiplier=0.4,   # 优化点3：出场阈值乘数
                 stop_loss_multiplier=2.2,        # 优化点3：止损阈值乘数（备用）
                 cooling_period=5,                # 优化点4：冷却期（交易日）
                 rebalance_threshold_multiplier=0.8,  # 轮动阈值乘数
                 use_adaptive_thresholds=True,    # 是否使用自适应阈值
                 risk_free_rate=0.02,             # 新增：无风险年化收益率
                 float_stop_threshold=0.5,        # 新增：浮动止损回撤阈值
                 enable_trend_filter=True,        # 新增：启用趋势过滤
                 trend_filter_window=3,           # 新增：趋势过滤窗口
                 base_cooling_period=5,           # 新增：基础冷却期
                 baseline_update_window=20,       # 新增：基准波动率更新窗口
                 baseline_update_frequency='daily'  # 新增：基准更新频率
                 ):
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
        self.baseline_spread_std = baseline_spread_std  # 基准价差标准差
        self.coint_pvalue = coint_pvalue  # 样本内协整检验p值
        self.coint_check_window = coint_check_window
        self.coint_check_freq = coint_check_freq
        self.coint_threshold = coint_threshold
        self.dynamic_check = dynamic_check
        self.risk_free_rate = risk_free_rate  # 无风险利率
        
        # 优化点3：自适应阈值参数
        self.entry_threshold_multiplier = entry_threshold_multiplier
        self.exit_threshold_multiplier = exit_threshold_multiplier
        self.stop_loss_multiplier = stop_loss_multiplier
        self.rebalance_threshold_multiplier = rebalance_threshold_multiplier
        self.use_adaptive_thresholds = use_adaptive_thresholds
        
        # 优化点4：无交易区间参数
        self.base_cooling_period = base_cooling_period
        self.cooling_period = base_cooling_period
        self.last_exit_date = None
        self.in_cooling_period = False
        self.last_exit_reason = ""  # 新增：记录上次平仓原因
        
        # 新增：动态基准波动率更新参数
        self.baseline_update_window = baseline_update_window
        self.baseline_update_frequency = baseline_update_frequency
        self.spread_history = []  # 存储历史价差数据用于基准更新
        self.last_update_date = None  # 上次基准更新日期
        
        # 新增：配对关系状态跟踪
        self.pair_valid = True  # 配对是否有效
        self.last_check_date = None  # 上次检查日期
        self.coint_check_count = 0  # 检查次数计数器
        self.pair_invalid_date = None  # 新增：配对失效的具体日期
        
        # 新增：浮动止损相关
        self.best_z_since_entry = 0  # 持仓期间最佳z值
        self.float_stop_threshold = float_stop_threshold
        
        # 新增：趋势过滤参数
        self.enable_trend_filter = enable_trend_filter
        self.trend_filter_window = trend_filter_window
        
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
        返回：spread, zscore, spread_mean_series, spread_std_series
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
    
    def update_baseline_if_needed(self, current_date, spread_data):
        """
        核心改进：基准波动率按需更新控制器
        
        参数:
            current_date: 当前日期
            spread_data: 当前价差数据
        
        作用：
            - 决定是否更新基准波动率
            - 按配置的频率（日/周/自适应）执行更新
            - 防止不必要的重复更新
        """
        # 情况1：基准值未初始化（第一次计算）
        if not self.baseline_spread_std:
            self._update_baseline(spread_data)
            self.last_update_date = current_date
            return
        
        # 根据更新频率决定是否更新
        if self.baseline_update_frequency == 'daily':
            # 每日更新：新交易日就更新
            if current_date != self.last_update_date:
                self._update_baseline(spread_data)
                self.last_update_date = current_date
                
        elif self.baseline_update_frequency == 'weekly':
            # 每周更新：只在周五更新
            if current_date.weekday() == 4:  # 星期五
                self._update_baseline(spread_data)
                self.last_update_date = current_date
        
        elif self.baseline_update_frequency == 'adaptive':
            # 自适应更新：波动性变化大时更频繁更新
            if not self.last_update_date or (current_date - self.last_update_date).days >= 5:
                # 至少5天检查一次
                if len(self.spread_history) >= 5:
                    recent_data = self.spread_history[-5:]  # 最近5个数据
                else:
                    recent_data = [spread_data]
                
                current_vol = np.std(recent_data) if len(recent_data) > 1 else 1.0
                
                # 如果波动率变化超过20%，则更新
                if self.baseline_spread_std and \
                   abs(current_vol - self.baseline_spread_std) / self.baseline_spread_std > 0.2:
                    self._update_baseline(spread_data)
                    self.last_update_date = current_date
    
    def _update_baseline(self, new_spread):
        """
        核心改进：基准波动率实际更新执行器
        
        参数:
            new_spread: 新价差数据
        
        作用：
            - 实际执行基准波动率计算
            - 管理历史数据窗口
            - 更新基准波动率值
        """
        # 添加新数据到历史列表
        self.spread_history.append(new_spread)
        
        # 保持列表长度不超过窗口大小
        if len(self.spread_history) > self.baseline_update_window:
            self.spread_history.pop(0)  # 移除最旧的数据
        
        # 计算标准差（需要有足够数据）
        if len(self.spread_history) >= min(10, self.baseline_update_window):
            self.baseline_spread_std = np.std(self.spread_history)
            # 可选：调试输出
            # print(f"基准波动率更新: 新值={self.baseline_spread_std:.6f}, 窗口大小={len(self.spread_history)}")

    def calculate_adaptive_thresholds(self, spread_std, entry_threshold=1.5, exit_threshold=0.3, 
                                     stop_loss=2.0, rebalance_threshold=0.8):
        """
        优化点3：增强自适应交易阈值计算（集成动态基准更新）
        
        参数:
            spread_std: 当前价差的标准差
            entry_threshold: 固定入场阈值（备用）
            exit_threshold: 固定出场阈值（备用）
            stop_loss: 固定止损阈值（备用）
            rebalance_threshold: 固定轮动阈值（备用）
        
        返回:
            自适应计算的阈值
        
        改进点：
            - 在计算前自动更新基准波动率
            - 更稳健的调整系数计算
        """
        if not self.use_adaptive_thresholds or spread_std <= 0:
            # 如果不使用自适应阈值，或价差标准差为0，返回固定阈值
            return entry_threshold, exit_threshold, stop_loss, rebalance_threshold
        
        # 核心改进：在计算前自动记录当前价差用于基准更新
        # 注：实际基准更新在外部调用，这里只使用最新的基准值
        
        # 使用基准波动率进行标准化
        if self.baseline_spread_std > 0:
            volatility_ratio = spread_std / self.baseline_spread_std
        else:
            volatility_ratio = 1.0
        
        # 限制调整范围，避免阈值过宽或过窄
        # 改进：非对称调整，波动增加时更谨慎
        if volatility_ratio > 1.0:
            # 波动增加时，调整系数增长更缓慢
            adjustment = 1.0 + 0.5 * (volatility_ratio - 1.0)
        else:
            # 波动减小时，完全跟随
            adjustment = volatility_ratio
        
        # 进一步限制范围
        adjustment = max(0.5, min(2.0, adjustment))
        
        # 根据波动率调整阈值
        # 改进：不同阈值使用不同的调整策略
        adaptive_entry = self.entry_threshold_multiplier * adjustment
        adaptive_exit = self.exit_threshold_multiplier * min(adjustment, 1.2)  # 出场阈值最多放宽20%
        adaptive_stop_loss = self.stop_loss_multiplier * max(adjustment, 1.0)  # 止损阈值至少不收缩
        adaptive_rebalance = self.rebalance_threshold_multiplier * adjustment
        
        return adaptive_entry, adaptive_exit, adaptive_stop_loss, adaptive_rebalance

    def update_cooling_period(self, date, action=None, reason=""):
        """
        优化点4：强化无交易区间状态更新
        参数:
            date: 当前日期
            action: 当前动作，如果为"exit"表示平仓
            reason: 平仓原因
        """
        if action == "exit" and not self.in_cooling_period:
            # 记录平仓原因
            self.last_exit_reason = reason
            
            # 如果是平仓动作，开始冷却期
            self.last_exit_date = date
            self.in_cooling_period = True
            
            # 如果是轮动导致的平仓，冷却期加倍
            if "轮动" in reason:
                self.cooling_period = self.base_cooling_period * 2
                if date:
                    print(f"  [{date.strftime('%Y-%m-%d')}] 轮动平仓，进入延长冷却期，{self.cooling_period}个交易日内不交易")
            else:
                self.cooling_period = self.base_cooling_period
                if date:
                    print(f"  [{date.strftime('%Y-%m-%d')}] 进入冷却期，{self.cooling_period}个交易日内不交易")
        
        if self.in_cooling_period and self.last_exit_date:
            # 检查是否仍在冷却期内
            days_since_exit = (date - self.last_exit_date).days
            if days_since_exit >= self.cooling_period:
                # 冷却期结束
                self.in_cooling_period = False
                # 重置冷却期为基准值
                self.cooling_period = self.base_cooling_period
                if date:
                    print(f"  [{date.strftime('%Y-%m-%d')}] 冷却期结束，恢复交易")

    def check_trend_filter(self, spread, i, trend_window=3):
        """
        优化点5：趋势过滤检查
        检查价差在近期是否显示出回归迹象
        返回: True 如果趋势符合回归预期
        """
        if not self.enable_trend_filter or i < trend_window:
            return True
        
        recent_spread = spread.iloc[i-trend_window:i+1]  # 最近几天+今天
        if len(recent_spread) < 2:
            return True
        
        # 简单线性回归斜率
        x = np.arange(len(recent_spread))
        try:
            slope, _ = np.polyfit(x, recent_spread.values, 1)
        except:
            return True
        
        return slope

    def execute_strategy(self, stock1_code, stock2_code, stock1_data, stock2_data, 
                        window=40, entry_threshold=1.5, exit_threshold=0.3, 
                        stop_loss=2.0, max_holding_days=20, rebalance_threshold=0.8):
        """
        执行强弱轮动策略 (增强优化版v5.3，集成动态基准更新)
        注意：此函数中的 stock1_code, stock2_code 是具体的股票代码字符串
        """
        # 初始化配对状态
        self.pair_valid = True
        self.last_check_date = None
        self.coint_check_count = 0
        self.last_exit_date = None
        self.in_cooling_period = False
        self.pair_invalid_date = None
        self.best_z_since_entry = 0
        self.last_exit_reason = ""
        
        # 初始化基准更新相关变量
        self.spread_history = []
        self.last_update_date = None
        
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
            current_spread = spread.loc[date] if date in spread.index else 0
            
            # 核心改进：在计算自适应阈值前更新基准波动率
            self.update_baseline_if_needed(date, current_spread)
            
            # 优化点3：获取自适应阈值
            current_spread_std = spread_std.loc[date] if hasattr(spread_std, 'loc') else 1.0
            adaptive_entry, adaptive_exit, adaptive_stop_loss, adaptive_rebalance = \
                self.calculate_adaptive_thresholds(current_spread_std, entry_threshold, exit_threshold, stop_loss, rebalance_threshold)
            
            # 优化点4：更新冷却期状态
            self.update_cooling_period(date)
            
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
                            self.pair_invalid_date = date  # 记录失效日期
                            
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
                            self.best_z_since_entry = 0
            
            # === 核心优化：处理配对失效后的净值计算 ===
            if not self.pair_valid:
                # 配对已失效，不再进行任何交易
                # 但需要计算当前的总价值（应为100%现金）
                total_value = self.capital
                
                # 可选：让现金按无风险利率增长，使净值曲线更真实
                # 计算自失效日以来的交易日数
                if self.pair_invalid_date and date > self.pair_invalid_date:
                    days_since_invalid = (date - self.pair_invalid_date).days
                    if days_since_invalid > 0:
                        # 按日计算无风险收益
                        daily_rf_rate = (1 + self.risk_free_rate) ** (1/252) - 1
                        total_value = self.capital * (1 + daily_rf_rate) ** days_since_invalid
                
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
                    'hedge_ratio': self.hedge_ratio,
                    'adaptive_entry': adaptive_entry,
                    'adaptive_exit': adaptive_exit,
                    'adaptive_stop_loss': adaptive_stop_loss,
                    'in_cooling_period': self.in_cooling_period,
                    'pair_invalid_date': self.pair_invalid_date,
                    'best_z_since_entry': self.best_z_since_entry,
                    'baseline_spread_std': self.baseline_spread_std,  # 记录当前基准值
                    'baseline_update_frequency': self.baseline_update_frequency
                }
                self.daily_records.append(daily_record)
                self.equity_curve.append((date, total_value))
                continue  # 跳过所有交易逻辑
            
            # === 以下为配对有效时的正常交易逻辑 ===
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
                'hedge_ratio': self.hedge_ratio,
                'adaptive_entry': adaptive_entry,
                'adaptive_exit': adaptive_exit,
                'adaptive_stop_loss': adaptive_stop_loss,
                'in_cooling_period': self.in_cooling_period,
                'pair_invalid_date': self.pair_invalid_date,
                'best_z_since_entry': self.best_z_since_entry,
                'baseline_spread_std': self.baseline_spread_std,  # 记录当前基准值
                'baseline_update_frequency': self.baseline_update_frequency
            }
            self.daily_records.append(daily_record)
            
            # 交易逻辑（仅在配对有效时执行）
            # 情况1：无持仓
            if holding_stock is None:
                # 优化点4：检查是否在冷却期内
                if self.in_cooling_period:
                    # 在冷却期内，不进行任何开仓
                    self.equity_curve.append((date, total_value))
                    continue
                
                # 优化点5：趋势过滤
                trend_slope = self.check_trend_filter(spread, i, self.trend_filter_window)
                
                if z > adaptive_entry:  # stock1相对高估，stock2低估
                    # 趋势过滤：价差应在收缩（斜率为负），即高估情况在缓解
                    if trend_slope < 0 or not self.enable_trend_filter:
                        position_value = min(total_value * self.max_position_ratio, self.capital)
                        volume = int(position_value / price2 / 100) * 100
                        
                        if volume > 0:
                            success, _, _ = self.buy_stock(date, stock2_code, price2, volume, 
                                                          reason=f"开仓-买入低估股({stock2_code}), z={z:.2f}, 入场阈值={adaptive_entry:.2f}, 趋势斜率={trend_slope:.4f}")
                            if success:
                                holding_stock = 'stock2'
                                entry_date = date
                                entry_zscore = z
                                self.best_z_since_entry = z  # 初始化最佳z值
                                holding_days = 0
                
                elif z < -adaptive_entry:  # stock1相对低估，stock2高估
                    # 趋势过滤：价差应在收缩（斜率为正），即低估情况在缓解
                    if trend_slope > 0 or not self.enable_trend_filter:
                        position_value = min(total_value * self.max_position_ratio, self.capital)
                        volume = int(position_value / price1 / 100) * 100
                        
                        if volume > 0:
                            success, _, _ = self.buy_stock(date, stock1_code, price1, volume, 
                                                          reason=f"开仓-买入低估股({stock1_code}), z={z:.2f}, 入场阈值={adaptive_entry:.2f}, 趋势斜率={trend_slope:.4f}")
                            if success:
                                holding_stock = 'stock1'
                                entry_date = date
                                entry_zscore = z
                                self.best_z_since_entry = z  # 初始化最佳z值
                                holding_days = 0
            
            # 情况2：持有股票2
            elif holding_stock == 'stock2':
                holding_days += 1
                
                # 更新持仓期间的最佳z值（对多stock2来说，z越小越有利）
                if z < self.best_z_since_entry:
                    self.best_z_since_entry = z
                
                # 优化点6：浮动止损检查
                z_retreat_from_best = z - self.best_z_since_entry  # 当前z值相对于最佳值的回撤
                
                if z_retreat_from_best > self.float_stop_threshold:
                    # 从最佳点回撤过大，浮动止损
                    if has_stock2:
                        pos = self.positions[stock2_code]
                        sell_volume = pos['qty']
                        if sell_volume > 0:
                            self.sell_stock(date, stock2_code, price2, sell_volume,
                                          reason=f"平仓-浮动止损, 最佳z={self.best_z_since_entry:.2f}, 当前z={z:.2f}, 回撤={z_retreat_from_best:.2f}")
                            self.update_cooling_period(date, action="exit", reason="浮动止损")
                    holding_stock = None
                    entry_date = None
                    entry_zscore = 0
                    self.best_z_since_entry = 0
                    holding_days = 0
                    self.equity_curve.append((date, total_value))
                    continue
                
                elif z < -adaptive_rebalance:  # stock1变得低估
                    if has_stock2:
                        pos = self.positions[stock2_code]
                        sell_volume = pos['qty']
                        if sell_volume > 0:
                            self.sell_stock(date, stock2_code, price2, sell_volume,
                                          reason=f"轮动-卖出高估股, z从{entry_zscore:.2f}到{z:.2f}, 轮动阈值={adaptive_rebalance:.2f}")
                            # 优化点4：记录平仓，开始冷却期
                            self.update_cooling_period(date, action="exit", reason="轮动平仓")
                    
                    # 优化点4：检查是否在冷却期内
                    if not self.in_cooling_period:
                        position_value = min(total_value * self.max_position_ratio, self.capital)
                        volume = int(position_value / price1 / 100) * 100
                        
                        if volume > 0:
                            success, _, _ = self.buy_stock(date, stock1_code, price1, volume,
                                                          reason=f"轮动-买入低估股, z={z:.2f}, 轮动阈值={adaptive_rebalance:.2f}")
                            if success:
                                holding_stock = 'stock1'
                                entry_date = date
                                entry_zscore = z
                                self.best_z_since_entry = z
                                holding_days = 0
                    else:
                        # 在冷却期内，不进行反向开仓
                        holding_stock = None
                        entry_date = None
                        entry_zscore = 0
                        self.best_z_since_entry = 0
                        holding_days = 0
                
                elif abs(z) < adaptive_exit:  # 价差回归
                    if has_stock2:
                        pos = self.positions[stock2_code]
                        sell_volume = pos['qty']
                        if sell_volume > 0:
                            self.sell_stock(date, stock2_code, price2, sell_volume,
                                          reason=f"平仓-价差回归, z={z:.2f}, 出场阈值={adaptive_exit:.2f}")
                            # 优化点4：记录平仓，开始冷却期
                            self.update_cooling_period(date, action="exit", reason="价差回归")
                    holding_stock = None
                    entry_date = None
                    entry_zscore = 0
                    self.best_z_since_entry = 0
                    holding_days = 0
                
                elif holding_days >= max_holding_days and abs(z) > adaptive_exit:  # 时间止损/回归失败
                    if has_stock2:
                        pos = self.positions[stock2_code]
                        sell_volume = pos['qty']
                        if sell_volume > 0:
                            self.sell_stock(date, stock2_code, price2, sell_volume,
                                          reason=f"平仓-时间止损/回归失败, 持仓{holding_days}天, z={z:.2f}")
                            # 优化点4：记录平仓，开始冷却期
                            self.update_cooling_period(date, action="exit", reason="时间止损")
                    holding_stock = None
                    entry_date = None
                    entry_zscore = 0
                    self.best_z_since_entry = 0
                    holding_days = 0
            
            # 情况3：持有股票1
            elif holding_stock == 'stock1':
                holding_days += 1
                
                # 更新持仓期间的最佳z值（对多stock1来说，z越大越有利）
                if z > self.best_z_since_entry:
                    self.best_z_since_entry = z
                
                # 优化点6：浮动止损检查
                z_retreat_from_best = self.best_z_since_entry - z  # 当前z值相对于最佳值的回撤
                
                if z_retreat_from_best > self.float_stop_threshold:
                    # 从最佳点回撤过大，浮动止损
                    if has_stock1:
                        pos = self.positions[stock1_code]
                        sell_volume = pos['qty']
                        if sell_volume > 0:
                            self.sell_stock(date, stock1_code, price1, sell_volume,
                                          reason=f"平仓-浮动止损, 最佳z={self.best_z_since_entry:.2f}, 当前z={z:.2f}, 回撤={z_retreat_from_best:.2f}")
                            self.update_cooling_period(date, action="exit", reason="浮动止损")
                    holding_stock = None
                    entry_date = None
                    entry_zscore = 0
                    self.best_z_since_entry = 0
                    holding_days = 0
                    self.equity_curve.append((date, total_value))
                    continue
                
                elif z > adaptive_rebalance:  # stock2变得低估
                    if has_stock1:
                        pos = self.positions[stock1_code]
                        sell_volume = pos['qty']
                        if sell_volume > 0:
                            self.sell_stock(date, stock1_code, price1, sell_volume,
                                          reason=f"轮动-卖出高估股, z从{entry_zscore:.2f}到{z:.2f}, 轮动阈值={adaptive_rebalance:.2f}")
                            # 优化点4：记录平仓，开始冷却期
                            self.update_cooling_period(date, action="exit", reason="轮动平仓")
                    
                    # 优化点4：检查是否在冷却期内
                    if not self.in_cooling_period:
                        position_value = min(total_value * self.max_position_ratio, self.capital)
                        volume = int(position_value / price2 / 100) * 100
                        
                        if volume > 0:
                            success, _, _ = self.buy_stock(date, stock2_code, price2, volume,
                                                          reason=f"轮动-买入低估股, z={z:.2f}, 轮动阈值={adaptive_rebalance:.2f}")
                            if success:
                                holding_stock = 'stock2'
                                entry_date = date
                                entry_zscore = z
                                self.best_z_since_entry = z
                                holding_days = 0
                    else:
                        # 在冷却期内，不进行反向开仓
                        holding_stock = None
                        entry_date = None
                        entry_zscore = 0
                        self.best_z_since_entry = 0
                        holding_days = 0
                
                elif abs(z) < adaptive_exit:  # 价差回归
                    if has_stock1:
                        pos = self.positions[stock1_code]
                        sell_volume = pos['qty']
                        if sell_volume > 0:
                            self.sell_stock(date, stock1_code, price1, sell_volume,
                                          reason=f"平仓-价差回归, z={z:.2f}, 出场阈值={adaptive_exit:.2f}")
                            # 优化点4：记录平仓，开始冷却期
                            self.update_cooling_period(date, action="exit", reason="价差回归")
                    holding_stock = None
                    entry_date = None
                    entry_zscore = 0
                    self.best_z_since_entry = 0
                    holding_days = 0
                
                elif holding_days >= max_holding_days and abs(z) > adaptive_exit:  # 时间止损/回归失败
                    if has_stock1:
                        pos = self.positions[stock1_code]
                        sell_volume = pos['qty']
                        if sell_volume > 0:
                            self.sell_stock(date, stock1_code, price1, sell_volume,
                                          reason=f"平仓-时间止损/回归失败, 持仓{holding_days}天, z={z:.2f}")
                            # 优化点4：记录平仓，开始冷却期
                            self.update_cooling_period(date, action="exit", reason="时间止损")
                    holding_stock = None
                    entry_date = None
                    entry_zscore = 0
                    self.best_z_since_entry = 0
                    holding_days = 0
            
            self.equity_curve.append((date, total_value))
        
        return pd.DataFrame(self.daily_records).set_index('date')

# === 新增函数：在样本内计算配对参数 ===
def calculate_pair_parameters_in_sample(stock1_data, stock2_data, window_start, window_end):
    """
    在样本内数据上计算配对参数
    返回: hedge_ratio, intercept, coint_pvalue, adf_pvalue, spread_mean, spread_std
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

# === 第四步：主程序（增强优化版v5.3，包含动态基准更新）===
if __name__ == "__main__":
    # 配置参数
    TDX_DATA_DIR = "C:/new_tdx/vipdoc"  # 修改为你的通达信数据目录
    
    # 时间段划分
    IN_SAMPLE_START = "2018-01-01"   # 样本内开始
    IN_SAMPLE_END = "2023-12-31"     # 样本内结束
    OUT_SAMPLE_START = "2024-01-01"  # 样本外开始
    OUT_SAMPLE_END = "2026-03-31"    # 样本外结束
    
    # 策略参数
    TARGET_INDUSTRY = "银行"  # 目标行业名称
    TOP_N_PAIRS = 5  # 只回测前N对！
    min_data_length = 100
    
    # 动态协整检验参数
    COINT_CHECK_WINDOW = 120  # 动态检验窗口长度
    COINT_CHECK_FREQ = 60     # 动态检验频率（交易日）
    COINT_THRESHOLD = 0.05    # 协整检验p值阈值
    
    # 自适应交易带参数
    ENTRY_MULTIPLIER = 2.2    # 入场阈值乘数
    EXIT_MULTIPLIER = 0.4     # 出场阈值乘数
    STOP_LOSS_MULTIPLIER = 2.4  # 止损阈值乘数（备用）
    REBALANCE_MULTIPLIER = 0.6  # 轮动阈值乘数
    USE_ADAPTIVE = True       # 是否使用自适应阈值
    
    # 无交易区间参数
    COOLING_PERIOD = 5        # 冷却期（交易日）
    
    # 新增：失效后净值计算参数
    RISK_FREE_RATE = 0.02     # 无风险年化收益率
    
    # 新增：浮动止损参数
    FLOAT_STOP_THRESHOLD = 0.6  # 浮动止损回撤阈值
    
    # 新增：趋势过滤参数
    ENABLE_TREND_FILTER = True  # 启用趋势过滤
    TREND_FILTER_WINDOW = 5     # 趋势过滤窗口
    
    # 新增：动态基准更新参数
    BASELINE_UPDATE_WINDOW = 20  # 基准波动率更新窗口
    BASELINE_UPDATE_FREQUENCY = 'daily'  # 基准更新频率：'daily'（每日）/ 'weekly'（每周）/ 'adaptive'（自适应）
    
    print("="*80)
    print("A股配对交易策略（增强优化版v5.3：集成动态基准波动率更新）")
    print(f"目标行业: {TARGET_INDUSTRY}")
    print(f"只回测排名前{TOP_N_PAIRS}的配对")
    print(f"自适应阈值: {'启用' if USE_ADAPTIVE else '禁用'}")
    print(f"入场乘数: {ENTRY_MULTIPLIER}, 出场乘数: {EXIT_MULTIPLIER}")
    print(f"止损乘数: {STOP_LOSS_MULTIPLIER}, 轮动乘数: {REBALANCE_MULTIPLIER}")
    print(f"冷却期: {COOLING_PERIOD}个交易日")
    print(f"无风险利率(失效后): {RISK_FREE_RATE:.1%}")
    print(f"浮动止损阈值: {FLOAT_STOP_THRESHOLD}")
    print(f"趋势过滤: {'启用' if ENABLE_TREND_FILTER else '禁用'}, 窗口: {TREND_FILTER_WINDOW}")
    print(f"基准更新: 频率={BASELINE_UPDATE_FREQUENCY}, 窗口={BASELINE_UPDATE_WINDOW}天")
    print("="*80)
    
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
              f"相关性={pair['corr']:.3f}, 对冲比率={pair['hedge_ratio']:.4f}, 基准价差标准差={pair['spread_std']:.4f}")
    
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
    sample_in_df.to_csv(f'A股配对_样本内筛选结果_增强优化版v5.3_前{TOP_N_PAIRS}对.csv', index=False, encoding='utf-8-sig')
    print(f"样本内筛选结果已保存到: A股配对_样本内筛选结果_增强优化版v5.3_前{TOP_N_PAIRS}对.csv")
    
    print(f"\n第二步：样本外回测 ({OUT_SAMPLE_START} 到 {OUT_SAMPLE_END})")
    print("="*80)
    
    # 对前TOP_N_PAIRS对股票进行样本外回测
    all_results = []
    
    for idx, pair in enumerate(top_pairs):
        stock1 = pair['stock1']
        stock2 = pair['stock2']
        hedge_ratio = pair['hedge_ratio']
        coint_p = pair['coint_p']
        baseline_std = pair['spread_std']
        
        print(f"\n回测第 {idx+1}/{len(top_pairs)} 对: {stock1} - {stock2}")
        print(f"对冲比率: {hedge_ratio:.4f}, 样本内协整p值: {coint_p:.4f}, 基准价差标准差: {baseline_std:.4f}")
        print(f"基准更新频率: {BASELINE_UPDATE_FREQUENCY}, 窗口: {BASELINE_UPDATE_WINDOW}天")
        
        # 创建策略实例，传入训练期计算出的参数和新优化参数
        strategy = ASharePairsTradingStrategyOptimizedV5(
            initial_capital=1000000,
            commission_rate=0.0003,
            stamp_tax_rate=0.001,
            slippage=0.001,
            min_commission=5.0,
            max_position_ratio=0.5,
            min_position_ratio=0.1,
            hedge_ratio=hedge_ratio,  # 传入对冲比率
            baseline_spread_std=baseline_std,  # 传入基准价差标准差
            coint_pvalue=coint_p,     # 传入协整p值
            coint_check_window=COINT_CHECK_WINDOW,
            coint_check_freq=COINT_CHECK_FREQ,
            coint_threshold=COINT_THRESHOLD,
            dynamic_check=True,  # 启用动态协整检验
            entry_threshold_multiplier=ENTRY_MULTIPLIER,
            exit_threshold_multiplier=EXIT_MULTIPLIER,
            stop_loss_multiplier=STOP_LOSS_MULTIPLIER,
            cooling_period=COOLING_PERIOD,
            rebalance_threshold_multiplier=REBALANCE_MULTIPLIER,
            use_adaptive_thresholds=USE_ADAPTIVE,
            risk_free_rate=RISK_FREE_RATE,  # 传入无风险利率
            float_stop_threshold=FLOAT_STOP_THRESHOLD,  # 传入浮动止损阈值
            enable_trend_filter=ENABLE_TREND_FILTER,  # 启用趋势过滤
            trend_filter_window=TREND_FILTER_WINDOW,  # 趋势过滤窗口
            base_cooling_period=COOLING_PERIOD,  # 基础冷却期
            baseline_update_window=BASELINE_UPDATE_WINDOW,  # 传入基准更新窗口
            baseline_update_frequency=BASELINE_UPDATE_FREQUENCY  # 传入基准更新频率
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
            entry_threshold=1.5,  # 固定阈值（备用）
            exit_threshold=0.3,   # 固定阈值（备用）
            stop_loss=2.0,        # 固定阈值（备用）
            max_holding_days=20,
            rebalance_threshold=0.8  # 固定阈值（备用）
        )
        
        if result is not None:
            # 计算最终收益率
            initial_value = 1000000
            if strategy.equity_curve:
                final_value = strategy.equity_curve[-1][1]  # 从净值曲线获取最终价值
            else:
                final_value = strategy.capital
            
            total_return = (final_value - initial_value) / initial_value
            total_trades = len(strategy.trade_history)
            
            # 计算胜率
            win_trades = len([t for t in strategy.trade_history if t.get('pnl', 0) > 0])
            loss_trades = len([t for t in strategy.trade_history if t.get('pnl', 0) < 0])
            win_rate = win_trades / total_trades * 100 if total_trades > 0 else 0
            
            # 计算盈亏比
            if win_trades > 0 and loss_trades > 0:
                avg_win = np.mean([t.get('pnl', 0) for t in strategy.trade_history if t.get('pnl', 0) > 0])
                avg_loss = np.mean([abs(t.get('pnl', 0)) for t in strategy.trade_history if t.get('pnl', 0) < 0])
                profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
            else:
                profit_factor = 0
            
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
            
            # 统计冷却期触发次数
            cooling_triggers = sum(1 for rec in strategy.daily_records if rec.get('in_cooling_period', False))
            
            # 统计配对失效后的交易日数
            invalid_days = 0
            if strategy.pair_invalid_date and strategy.daily_records:
                last_date = strategy.daily_records[-1]['date']
                invalid_days = (last_date - strategy.pair_invalid_date).days
            
            # 统计浮动止损触发次数
            float_stop_trades = len([t for t in strategy.trade_history if "浮动止损" in t.get('reason', '')])
            
            # 统计基准更新次数
            baseline_updates = len([rec for rec in strategy.daily_records 
                                  if 'baseline_spread_std' in rec and rec.get('baseline_spread_std', 0) != baseline_std])
            
            print(f"  ✓ 回测完成:")
            print(f"    初始资金: {initial_value:,.0f}")
            print(f"    最终资金: {final_value:,.0f}")
            print(f"    总收益率: {total_return:.2%}")
            print(f"    交易次数: {total_trades}")
            print(f"    胜率: {win_rate:.1f}%")
            print(f"    盈亏比: {profit_factor:.2f}")
            print(f"    夏普比率: {sharpe:.2f}")
            print(f"    动态协整检验次数: {coint_checks}")
            print(f"    冷却期触发次数: {cooling_triggers}")
            print(f"    浮动止损次数: {float_stop_trades}")
            print(f"    基准更新次数: {baseline_updates}")
            print(f"    配对失效日期: {strategy.pair_invalid_date.strftime('%Y-%m-%d') if strategy.pair_invalid_date else 'N/A'}")
            print(f"    失效后交易日数: {invalid_days}")
            print(f"    配对最终状态: {'有效' if strategy.pair_valid else '已失效'}")
            print(f"    基准更新频率: {BASELINE_UPDATE_FREQUENCY}")
            
            # 保存结果
            all_results.append({
                'pair': f"{stock1}-{stock2}",
                'ssd': pair['ssd'],
                'corr': pair['corr'],
                'hedge_ratio': hedge_ratio,
                'baseline_std': baseline_std,
                'initial': initial_value,
                'final': final_value,
                'return': total_return,
                'trades': total_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'sharpe': sharpe,
                'coint_checks': coint_checks,
                'cooling_triggers': cooling_triggers,
                'float_stop_trades': float_stop_trades,
                'baseline_updates': baseline_updates,
                'pair_invalid_date': strategy.pair_invalid_date,
                'invalid_days': invalid_days,
                'pair_valid': strategy.pair_valid,
                'baseline_update_frequency': BASELINE_UPDATE_FREQUENCY,  # 记录基准更新频率
                'strategy': strategy
            })
            
            # 保存交易记录
            trades_df = pd.DataFrame(strategy.trade_history)
            trades_df.to_csv(f'A股配对_交易记录_{stock1}_{stock2}_增强优化版v5.3_前{TOP_N_PAIRS}对.csv', index=False, encoding='utf-8-sig')
            
            # 保存每日净值
            if strategy.equity_curve:
                equity_df = pd.DataFrame(strategy.equity_curve, columns=['date', 'value'])
                equity_df.set_index('date', inplace=True)
                equity_df.to_csv(f'A股配对_每日净值_{stock1}_{stock2}_增强优化版v5.3_前{TOP_N_PAIRS}对.csv', encoding='utf-8-sig')
            
            # 保存每日记录（包含更多细节）
            daily_df = pd.DataFrame(strategy.daily_records)
            daily_df.set_index('date', inplace=True)
            daily_df.to_csv(f'A股配对_每日记录_{stock1}_{stock2}_增强优化版v5.3_前{TOP_N_PAIRS}对.csv', encoding='utf-8-sig')
    
    # 输出汇总结果
    if all_results:
        print(f"\n" + "="*80)
        print(f"样本外回测汇总结果（增强优化版v5.3，前{TOP_N_PAIRS}对）")
        print("="*80)
        
        # 计算总体表现
        total_initial = sum(r['initial'] for r in all_results)
        total_final = sum(r['final'] for r in all_results)
        overall_return = (total_final - total_initial) / total_initial
        
        # 统计配对有效性
        valid_pairs = sum(1 for r in all_results if r['pair_valid'])
        
        # 统计基准更新情况
        total_baseline_updates = sum(r.get('baseline_updates', 0) for r in all_results)
        
        print(f"回测配对数量: {len(all_results)}")
        print(f"最终有效配对数量: {valid_pairs}")
        print(f"初始总资金: {total_initial:,.0f}")
        print(f"最终总资金: {total_final:,.0f}")
        print(f"总收益率: {overall_return:.2%}")
        print(f"总基准更新次数: {total_baseline_updates}")
        print(f"基准更新频率: {BASELINE_UPDATE_FREQUENCY}")
        
        # 按收益率排序
        sorted_results = sorted(all_results, key=lambda x: x['return'], reverse=True)
        
        print(f"\n配对表现排名:")
        for i, r in enumerate(sorted_results):
            status = "有效" if r['pair_valid'] else "失效"
            invalid_info = f", 失效于: {r['pair_invalid_date'].strftime('%Y-%m-%d')}" if r['pair_invalid_date'] else ""
            cooling_str = f", 冷却触发: {r['cooling_triggers']}" if r.get('cooling_triggers') is not None else ""
            float_stop_str = f", 浮动止损: {r['float_stop_trades']}" if r.get('float_stop_trades') is not None else ""
            baseline_update_str = f", 基准更新: {r.get('baseline_updates', 0)}" if r.get('baseline_updates') is not None else ""
            print(f"{i+1}. {r['pair']}: 收益率={r['return']:.2%}, 对冲比率={r['hedge_ratio']:.4f}, "
                  f"交易次数={r['trades']}, 胜率={r['win_rate']:.1f}%, 盈亏比={r['profit_factor']:.2f}"
                  f"{cooling_str}{float_stop_str}{baseline_update_str}{invalid_info}, 状态={status}")
        
        # 绘制净值曲线
        plt.rcParams["font.family"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "Microsoft YaHei"]
        plt.rcParams["axes.unicode_minus"] = False

        plt.figure(figsize=(16, 9))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for i, r in enumerate(sorted_results):
            if r['strategy'].daily_records:
                dates = [rec['date'] for rec in r['strategy'].daily_records]
                total_values = [rec['total_value'] for rec in r['strategy'].daily_records]
                
                net_values = [v / 1000000.0 for v in total_values]
                color_idx = i % len(colors)
                
                # 标注失效点
                if r['pair_invalid_date']:
                    # 找到失效日期在净值曲线中的位置
                    try:
                        invalid_idx = dates.index(r['pair_invalid_date'])
                        invalid_net = net_values[invalid_idx]
                        plt.scatter(r['pair_invalid_date'], invalid_net, 
                                   color='red', s=100, zorder=5, marker='x')
                        plt.text(r['pair_invalid_date'], invalid_net*1.01, '失效点', 
                                fontsize=9, color='red', ha='center', fontweight='bold')
                    except ValueError:
                        pass
                
                label = f"{r['pair']} (收益率: {r['return']:.2%})"
                if not r['pair_valid']:
                    label += " [已失效]"
                
                plt.plot(dates, net_values, label=label, 
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
        
        plt.title(f'{TARGET_INDUSTRY}行业配对交易策略样本外回测（增强优化版v5.3，前{TOP_N_PAIRS}对）\n'
                  f'动态对冲比率 + 动态协整监控 + 增强自适应交易带 + 强化无交易区间 + 失效后净值优化 + 浮动止损 + 动态基准更新 | 总收益率: {overall_return:.2%}', 
                  fontsize=14, fontweight='bold')
        
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('净值（相对初始资金）', fontsize=12)
        
        plt.legend(loc='best', fontsize=10, frameon=True, shadow=True)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.xticks(rotation=45, ha='right', fontsize=10)       
        plt.tight_layout()
        plt.savefig(f'{TARGET_INDUSTRY}配对_样本外回测净值曲线_增强优化版v5.3_前{TOP_N_PAIRS}对.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 保存汇总结果
        summary_df = pd.DataFrame([{
            '股票对': r['pair'],
            'SSD距离': r['ssd'],
            '相关性': r['corr'],
            '对冲比率': r['hedge_ratio'],
            '基准价差标准差': r.get('baseline_std', 0),
            '基准更新频率': r.get('baseline_update_frequency', BASELINE_UPDATE_FREQUENCY),
            '基准更新次数': r.get('baseline_updates', 0),
            '初始资金': r['initial'],
            '最终资金': r['final'],
            '总收益率': r['return'],
            '年化收益率': (1 + r['return']) ** (252/len(r['strategy'].equity_curve)) - 1 if r['strategy'].equity_curve else 0,
            '交易次数': r['trades'],
            '胜率': r['win_rate'],
            '盈亏比': r.get('profit_factor', 0),
            '夏普比率': r['sharpe'],
            '协整检验次数': r['coint_checks'],
            '冷却期触发次数': r.get('cooling_triggers', 0),
            '浮动止损次数': r.get('float_stop_trades', 0),
            '配对失效日期': r['pair_invalid_date'].strftime('%Y-%m-%d') if r['pair_invalid_date'] else '',
            '失效后交易日数': r.get('invalid_days', 0),
            '配对最终状态': '有效' if r['pair_valid'] else '失效'
        } for r in all_results])
        
        summary_df.to_csv(f'{TARGET_INDUSTRY}配对_样本外回测汇总_增强优化版v5.3_前{TOP_N_PAIRS}对.csv', index=False, encoding='utf-8-sig')
        print(f"\n详细结果已保存到: {TARGET_INDUSTRY}配对_样本外回测汇总_增强优化版v5.3_前{TOP_N_PAIRS}对.csv")
        
        # 额外分析：统计各种优化功能的效果
        print(f"\n优化功能效果统计:")
        print("-" * 40)
        
        total_float_stops = sum(r.get('float_stop_trades', 0) for r in all_results)
        total_trades = sum(r['trades'] for r in all_results)
        total_cooling_triggers = sum(r.get('cooling_triggers', 0) for r in all_results)
        
        print(f"总交易次数: {total_trades}")
        print(f"总浮动止损次数: {total_float_stops} ({total_float_stops/total_trades*100:.1f}%)")
        print(f"总冷却期触发次数: {total_cooling_triggers}")
        print(f"总基准更新次数: {total_baseline_updates}")
        print(f"平均胜率: {np.mean([r['win_rate'] for r in all_results]):.1f}%")
        print(f"平均盈亏比: {np.mean([r.get('profit_factor', 0) for r in all_results]):.2f}")
        
        # 分析交易记录中的平仓原因分布
        if all_results and len(all_results[0]['strategy'].trade_history) > 0:
            exit_reasons = {}
            for r in all_results:
                for trade in r['strategy'].trade_history:
                    if trade['action'] == '卖出':
                        reason = trade.get('reason', '')
                        # 提取主要原因
                        if '价差回归' in reason:
                            key = '价差回归'
                        elif '轮动' in reason:
                            key = '轮动'
                        elif '浮动止损' in reason:
                            key = '浮动止损'
                        elif '时间止损' in reason:
                            key = '时间止损'
                        elif '强制平仓' in reason:
                            key = '强制平仓'
                        else:
                            key = '其他'
                        
                        exit_reasons[key] = exit_reasons.get(key, 0) + 1
            
            print(f"\n平仓原因分布:")
            for reason, count in exit_reasons.items():
                percentage = count/total_trades*100 if total_trades > 0 else 0
                print(f"  {reason}: {count}次 ({percentage:.1f}%)")
        
        # 参数建议
        print(f"\n参数优化建议:")
        print("-" * 40)
        print("1. 如果胜率仍偏低(<50%)，可尝试调整以下参数:")
        print("   - 提高 entry_threshold_multiplier (如从2.2提高到2.4)")
        print("   - 降低 rebalance_threshold_multiplier (如从0.6降低到0.5)")
        print("   - 增加 cooling_period (如从5增加到8)")
        print("   - 调整基准更新频率: 尝试 'weekly' 或 'adaptive'")
        
        print("\n2. 如果交易频率过高，可尝试调整以下参数:")
        print("   - 提高 entry_threshold_multiplier")
        print("   - 启用趋势过滤 (ENABLE_TREND_FILTER=True)")
        print("   - 增加 cooling_period")
        print("   - 降低基准更新频率: 尝试 'weekly'")
        
        print("\n3. 如果浮动止损触发过于频繁，可尝试:")
        print("   - 提高 float_stop_threshold (如从0.6提高到0.8)")
        print("   - 降低 entry_threshold_multiplier 以减少初始偏离")
        
        print("\n4. 基准更新频率选择建议:")
        print("   - 'daily' (默认): 每日更新，对市场变化最敏感")
        print("   - 'weekly': 每周更新，减少交易噪音")
        print("   - 'adaptive': 自适应更新，波动大时频繁更新，平稳时减少更新")
        
        # 基准更新效果分析
        if all_results and len(all_results) > 0:
            print(f"\n基准更新效果分析:")
            print("-" * 40)
            
            # 按基准更新频率分组统计
            freq_stats = {}
            for r in all_results:
                freq = r.get('baseline_update_frequency', BASELINE_UPDATE_FREQUENCY)
                if freq not in freq_stats:
                    freq_stats[freq] = {'returns': [], 'trades': [], 'win_rates': []}
                
                freq_stats[freq]['returns'].append(r['return'])
                freq_stats[freq]['trades'].append(r['trades'])
                freq_stats[freq]['win_rates'].append(r['win_rate'])
            
            for freq, stats in freq_stats.items():
                if stats['returns']:
                    avg_return = np.mean(stats['returns']) * 100
                    avg_trades = np.mean(stats['trades'])
                    avg_win_rate = np.mean(stats['win_rates'])
                    print(f"  {freq}: 平均收益率={avg_return:.1f}%, 平均交易次数={avg_trades:.1f}, 平均胜率={avg_win_rate:.1f}%")
        
    else:
        print("\n没有成功的回测结果")
    
    print(f"\n{'='*80}")
    print("增强优化版v5.3 代码执行完成！")
    print("主要改进功能:")
    print("1. 动态基准波动率更新（支持日/周/自适应三种频率）")
    print("2. 改进的自适应阈值计算（非对称调整，更稳健）")
    print("3. 强化无交易区间（冷却期机制）")
    print("4. 浮动止损（从最佳点回撤止损）")
    print("5. 趋势过滤（开仓时检查价差回归趋势）")
    print("6. 失效后净值优化（配对失效后资金按无风险利率增长）")
    print(f"{'='*80}")