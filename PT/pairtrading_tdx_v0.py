# === 第一部分：导入必要的库 ===
import struct
import pandas as pd
import numpy as np
import os
import warnings
from statsmodels.tsa.stattools import coint, adfuller
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# === 第二部分：定义读取通达信数据的函数 ===
def read_tdx_day_file(file_path):
    """读取通达信日线数据文件(.day)"""
    with open(file_path, 'rb') as f:
        data = f.read()
    
    records = []
    for i in range(0, len(data), 32):
        if i + 32 > len(data):
            break
            
        # 解析二进制数据
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

def compute_ssd_distance_matrix(price_data_list, normalize=True):
    """计算股票价格序列之间的SSD距离矩阵"""
    n = len(price_data_list)
    ssd_matrix = np.zeros((n, n))
    
    if normalize:
        normalized_data = []
        for price_series in price_data_list:
            if len(price_series) > 0:
                norm_series = price_series / price_series.iloc[0]
                normalized_data.append(norm_series)
            else:
                normalized_data.append(pd.Series())
    else:
        normalized_data = price_data_list
    
    for i in range(n):
        for j in range(i+1, n):
            if len(normalized_data[i]) > 0 and len(normalized_data[j]) > 0:
                common_idx = normalized_data[i].index.intersection(normalized_data[j].index)
                if len(common_idx) > 0:
                    dist = np.sum((normalized_data[i].loc[common_idx] - normalized_data[j].loc[common_idx])**2)
                    ssd_matrix[i, j] = dist
                    ssd_matrix[j, i] = dist
                else:
                    ssd_matrix[i, j] = np.inf
                    ssd_matrix[j, i] = np.inf
            else:
                ssd_matrix[i, j] = np.inf
                ssd_matrix[j, i] = np.inf
    
    return ssd_matrix

# === 第三部分：A股适配的强弱轮动策略 ===
class ASharePairsTradingStrategy:
    """A股配对交易策略（强弱轮动，不允许融券）"""
    
    def __init__(self, initial_capital=1000000, commission_rate=0.0003, 
                 stamp_tax_rate=0.001, slippage=0.001, min_commission=5.0,
                 max_position_ratio=0.5, min_position_ratio=0.1):
        """
        参数:
        - initial_capital: 初始资金
        - commission_rate: 佣金费率（双向）
        - stamp_tax_rate: 印花税（卖出时）
        - slippage: 滑点
        - min_commission: 最低佣金
        - max_position_ratio: 单只股票最大仓位比例
        - min_position_ratio: 单只股票最小仓位比例
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.slippage = slippage
        self.min_commission = min_commission
        self.max_position_ratio = max_position_ratio
        self.min_position_ratio = min_position_ratio
        
        # 持仓记录
        self.positions = {}  # 股票代码 -> {'qty': 数量, 'avg_price': 平均成本, 'value': 市值}
        self.trade_history = []  # 交易记录
        self.equity_curve = []  # 每日净值
        self.daily_records = []  # 每日记录
        
    def calculate_trading_cost(self, price, volume, is_buy=True):
        """计算交易成本（A股规则）"""
        # 1. 应用滑点
        if is_buy:
            exec_price = price * (1 + self.slippage)
        else:
            exec_price = price * (1 - self.slippage)
        
        # 2. 计算交易金额
        trade_amount = exec_price * volume
        
        # 3. 计算佣金
        commission = max(trade_amount * self.commission_rate, self.min_commission)
        
        # 4. 计算印花税（卖出时收取）
        stamp_tax = 0
        if not is_buy:
            stamp_tax = trade_amount * self.stamp_tax_rate
        
        # 5. 总成本
        total_cost = commission + stamp_tax
        
        return exec_price, total_cost
    
    def buy_stock(self, date, stock_code, price, volume, reason=""):
        """买入股票（A股只能做多）"""
        if volume <= 0:
            return False, 0, 0
        
        # 计算实际成交价和成本
        exec_price, cost = self.calculate_trading_cost(price, volume, is_buy=True)
        
        # 计算所需资金
        required_capital = exec_price * volume + cost
        
        if required_capital > self.capital:
            # 资金不足，按最大可买数量
            max_volume = int(self.capital / (exec_price * 1.01))  # 预留1%缓冲
            if max_volume <= 0:
                return False, 0, 0
            volume = max_volume
            required_capital = exec_price * volume + cost
        
        # 更新资金
        self.capital -= required_capital
        
        # 更新持仓
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
        
        # 记录交易
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
        
        # 不能卖出超过持仓的数量
        volume = min(volume, pos['qty'])
        
        # 计算实际成交价和成本
        exec_price, cost = self.calculate_trading_cost(price, volume, is_buy=False)
        
        # 计算卖出金额
        sell_amount = exec_price * volume
        
        # 计算盈亏
        pnl = (exec_price - pos['avg_price']) * volume - cost
        
        # 更新资金
        self.capital += sell_amount - cost
        
        # 更新持仓
        pos['qty'] -= volume
        if pos['qty'] <= 0:
            del self.positions[stock_code]
        
        # 记录交易
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
    
    def rolling_regression(self, stock1_data, stock2_data, window=60):
        """滚动回归计算残差"""
        n = len(stock1_data)
        alphas, betas, residuals = [], [], []
        
        for i in range(window, n):
            x = stock2_data.iloc[i-window:i]
            y = stock1_data.iloc[i-window:i]
            
            X = np.vstack([np.ones(window), x.values]).T
            beta, alpha = np.linalg.lstsq(X, y.values, rcond=None)[0]
            
            resid = y.iloc[-1] - (alpha + beta * x.iloc[-1])
            
            alphas.append(alpha)
            betas.append(beta)
            residuals.append(resid)
        
        dates = stock1_data.index[window:]
        df = pd.DataFrame({
            'date': dates,
            'alpha': alphas,
            'beta': betas,
            'resid': residuals
        }).set_index('date')
        
        return df
    
    def calculate_zscore(self, residuals, window=60):
        """计算残差的z-score"""
        mean_resid = residuals.rolling(window).mean()
        std_resid = residuals.rolling(window).std()
        zscore = (residuals - mean_resid) / std_resid
        return zscore.fillna(0)
    
    def calculate_portfolio_value(self, date, stock1_data, stock2_data):
        """计算投资组合总价值"""
        total_value = self.capital
        
        for stock_code, pos in self.positions.items():
            if stock_code == 'CASH':
                continue
            
            if stock_code in stock1_data.index and date in stock1_data.index:
                price = stock1_data.loc[date]
            elif stock_code in stock2_data.index and date in stock2_data.index:
                price = stock2_data.loc[date]
            else:
                # 获取最新价格
                price = pos['avg_price']  # 默认用成本价
            
            total_value += price * pos['qty']
        
        return total_value
    
    def execute_strategy(self, stock1_code, stock2_code, stock1_data, stock2_data, 
                        window=60, entry_threshold=1.2, exit_threshold=0.3, 
                        stop_loss=2.5, max_holding_days=30, rebalance_days=5):
        """
        执行强弱轮动策略 逻辑：买入低估的，等待价差回归
        """
        # 对齐数据
        common_dates = stock1_data.index.intersection(stock2_data.index)
        stock1_data = stock1_data.loc[common_dates]
        stock2_data = stock2_data.loc[common_dates]
        
        if len(stock1_data) < window * 2:
            print(f"数据不足: 只有{len(stock1_data)}条，至少需要{window*2}条")
            return None
        
        # 计算滚动回归和z-score
        reg_result = self.rolling_regression(stock1_data, stock2_data, window)
        zscore = self.calculate_zscore(reg_result['resid'], window)
        
        # 初始化状态
        in_position = False
        entry_date = None
        entry_zscore = 0
        holding_days = 0
        last_rebalance_date = None
        current_position_stock = None  # 记录当前持有哪只股票
        
        # 逐日执行策略
        for date in zscore.index[window:]:
            # 跳过初始窗口
            if date not in stock1_data.index or date not in stock2_data.index:
                continue
            
            z = zscore.loc[date] if date in zscore.index else 0
            price1 = stock1_data.loc[date]
            price2 = stock2_data.loc[date]
            
            # 每日计算持仓市值
            # 1. 重新计算现金（之前的交易可能改变）
            cash = self.capital
            # 2. 计算股票市值
            stock_value = 0
            for stock_code, pos in self.positions.items():
                if stock_code == stock1_code:
                    stock_value += pos['qty'] * price1
                elif stock_code == stock2_code:
                    stock_value += pos['qty'] * price2
                # 如果还持有其他股票（理论上不应该有）
                else:
                    # 默认用成本价
                    stock_value += pos['qty'] * pos.get('avg_price', 0)
            # 3. 总市值 = 现金 + 股票市值
            total_value = cash + stock_value
            
            # 记录每日状态
            daily_record = {
                'date': date,
                'zscore': z,
                'price1': price1,
                'price2': price2,
                'capital': cash, # 现金
                'stock_value': stock_value, # 股票市值
                'total_value': total_value, # 总市值
                'holding_days': holding_days,
                'in_position': in_position,
                'position_stock': current_position_stock
            }
            self.daily_records.append(daily_record)  # 将每日记录添加到列表中

            # 获取当前持仓
            has_stock1 = stock1_code in self.positions
            has_stock2 = stock2_code in self.positions
            
            if not in_position:  # 无配对仓位
                if abs(z) > entry_threshold:
                    # 计算仓位大小
                    position_value = min(total_value * self.max_position_ratio, self.capital)
                    
                    if z > entry_threshold:  # stock1高估，stock2低估，买入stock2
                        # 检查是否有股票1持仓（如果有，先卖出）
                        if has_stock1:
                           pos = self.positions[stock1_code]
                           self.sell_stock(date, stock1_code, price1, pos['qty'], 
                                           reason=f"调仓-卖出高估股, z={z:.2f}")
                        
                        # 买入低估股stock2
                        volume = int(position_value / price2 / 100) * 100
                        if volume > 0:
                            success, _, _ = self.buy_stock(date, stock2_code, price2, volume, 
                                                          reason=f"开仓-买入低估股, z={z:.2f}")
                            if success:
                                in_position = True
                                current_position_stock = stock2_code
                                entry_date = date
                                entry_zscore = z
                                holding_days = 0
                                print(f"📈 {date}: 开仓买入{stock2_code}(低估), z={z:.2f}")
                    
                    elif z < -entry_threshold:  # stock1低估，stock2高估，买入stock1
                        # 检查是否有stock2持仓（如果有，先卖出）
                        if has_stock2:
                           pos = self.positions[stock2_code]
                           self.sell_stock(date, stock2_code, price2, pos['qty'], 
                                           reason=f"调仓-卖出高估股, z={z:.2f}")
                        # 买入低估股stock1
                        volume = int(position_value / price1 / 100) * 100
                        if volume > 0:
                            success, _, _ = self.buy_stock(date, stock1_code, price1, volume, 
                                                          reason=f"开仓-买入低估股, z={z:.2f}")
                            if success:
                                in_position = True
                                current_position_stock = stock1_code
                                entry_date = date
                                entry_zscore = z
                                holding_days = 0
                                print(f"📈 {date}: 开仓买入{stock1_code}(低估), z={z:.2f}")
            
            else:  # 已有配对仓位
                holding_days += 1
                
                # 判断是否需要再平衡
                need_rebalance = False
                rebalance_reason=""
                if last_rebalance_date is None or (date - last_rebalance_date).days >= rebalance_days:
                    #检查是否需要轮动（当前持仓变成高估的）
                    if z > 0.8 and current_position_stock == stock1_code:  # 当前持有stock1，但stock1变得高估
                        need_rebalance = True
                        rebalance_reason = "再平衡-卖出高估股"
                    elif z <-0.8 and current_position_stock == stock2_code:   #当前持有stock2，但stock2变得高估
                        need_rebalance = True
                        rebalance_reason = "再平衡-卖出高估股"
                
                # 平仓条件判断
                should_close = False
                close_reason = ""
                
                if abs(z) < exit_threshold:  # 价差回归
                    should_close = True
                    close_reason = "回归平仓"
                elif abs(z) > stop_loss:  # 止损
                    should_close = True
                    close_reason = "止损平仓"
                elif holding_days >= max_holding_days:  # 时间止损
                    should_close = True
                    close_reason = "时间平仓"
                
                if should_close:
                    # 平仓：卖出当前持仓
                    if current_position_stock == stock1_code and has_stock1:
                        pos = self.positions[stock1_code]
                        self.sell_stock(date, stock1_code, price1, pos['qty'], 
                                       reason=f"平仓-{close_reason}")
                    
                    elif current_position_stock == stock2_code and has_stock2:
                        pos = self.positions[stock2_code]
                        self.sell_stock(date, stock2_code, price2, pos['qty'], 
                                       reason=f"平仓-{close_reason}")
                    
                    in_position = False
                    current_position_stock = None
                    entry_date = None
                    entry_zscore = 0
                    holding_days = 0
                    
                    print(f"📉 {date}: 平仓-{close_reason}, z={z:.2f}, 持仓{holding_days}天")
                
                elif need_rebalance:  # 再平衡
                    last_rebalance_date = date
                    
                    if z > 0.8 and current_position_stock == stock1_code:  # stock1变得高估
                        if has_stock1:  # 卖出高估的stock1
                            pos = self.positions[stock1_code]
                            sell_qty = pos['qty']
                            if sell_qty > 0:
                                self.sell_stock(date, stock1_code, price1, sell_qty, 
                                               reason=f"{rebalance_reason}")
                        
                        # 买入低估的stock2
                        rebalance_value = min(total_value * 0.3, self.capital)  # 30%仓位
                        buy_qty = int(rebalance_value / price2 / 100) * 100
                        if buy_qty > 0:
                            self.buy_stock(date, stock2_code, price2, buy_qty, 
                                          reason="再平衡-买入低估股")
                        # 更新持仓状态
                        current_position_stock = stock2_code
                        entry_date = date
                        entry_zscore = z
                        holding_days = 0

                        print(f"🔄 {date}: 轮动-卖出{stock1_code}(高估), 买入{stock2_code}(低估), z={z:.2f}")
                    
                    elif z < -0.8 and current_position_stock == stock2_code:  # stock2变得高估
                        if has_stock2:  # 卖出高估的stock2
                            pos = self.positions[stock2_code]
                            sell_qty = pos['qty']
                            if sell_qty > 0:
                                self.sell_stock(date, stock2_code, price2, sell_qty, 
                                               reason=f"{rebalance_reason}")
                        
                        # 买入低估的stock1
                        rebalance_value = min(total_value * 0.3, self.capital)  # 30%仓位
                        buy_qty = int(rebalance_value / price1 / 100) * 100
                        if buy_qty > 0:
                            self.buy_stock(date, stock1_code, price1, buy_qty, 
                                          reason="再平衡-买入低估股")
                        
                        # 更新持仓状态
                        current_position_stock = stock1_code
                        entry_date = date
                        entry_zscore = z
                        holding_days = 0

                        print(f"🔄 {date}: 轮动-卖出{stock2_code}(高估), 买入{stock1_code}(低估), z={z:.2f}")
            
            # 记录每日净值
            self.equity_curve.append((date, total_value))
            self.daily_records.append(daily_record)
        
        return pd.DataFrame(self.daily_records).set_index('date')

# === 第四步：主程序（样本内选股 + 样本外回测） ===
if __name__ == "__main__":
    # 配置参数
    TDX_DATA_DIR = "C:/new_tdx/vipdoc"  # 修改为你的通达信数据目录
    
    # 时间段划分
    IN_SAMPLE_START = "2021-08-01"   # 样本内开始
    IN_SAMPLE_END = "2024-12-31"     # 样本内结束
    OUT_SAMPLE_START = "2025-01-01"  # 样本外开始
    OUT_SAMPLE_END = "2026-03-23"    # 样本外结束
    
    # 股票列表（银行股示例）
    industry_stocks = ['601288', '601398', '601077', '001227', '600036', '601988', '601128', '002936', 
                      '601939', '600926', '600000', '002142', '601963', '601818', '601860', '601229',
                      '603323', '600016', '600015', '601838', '600908', '601009', '601187', '601166',
                      '601528', '601169', '601997', '002958', '002807', '601658', '600919', '601665',
                      '601577', '601825', '600928', '601916', '002966', '601998', '002839', '000001', '002948']
    
    # SSD参数
    ssd_top_n = 20
    min_data_length = 100
    
    print("="*60)
    print("A股配对交易策略（强弱轮动，不允许融券）")
    print("="*60)
    
    print(f"\n第一步：样本内选股 ({IN_SAMPLE_START} 到 {IN_SAMPLE_END})")
    
    # 加载样本内数据
    all_prices = []
    valid_stocks = []
    
    for stock_code in industry_stocks:
        df = get_stock_data_from_tdx(stock_code, TDX_DATA_DIR)
        if df is not None:
            price_series = df['close'].loc[IN_SAMPLE_START:IN_SAMPLE_END]
            if len(price_series) >= min_data_length:
                all_prices.append(price_series)
                valid_stocks.append(stock_code)
                print(f"  ✓ 加载: {stock_code}, {len(price_series)} 条数据")
            else:
                print(f"  ✗ 跳过: {stock_code}, 数据不足")
    
    print(f"\n成功加载 {len(valid_stocks)} 只股票的样本内数据")
    
    if len(valid_stocks) < 2:
        print("有效股票数量不足，无法进行配对分析")
        exit()
    
    # 计算SSD距离
    print(f"\n计算SSD距离矩阵...")
    ssd_matrix = compute_ssd_distance_matrix(all_prices, normalize=True)
    
    # 筛选SSD最小的前N对
    candidate_pairs = []
    n = len(valid_stocks)
    for i in range(n):
        for j in range(i+1, n):
            dist = ssd_matrix[i, j]
            if dist < np.inf:
                candidate_pairs.append((i, j, dist))
    
    candidate_pairs.sort(key=lambda x: x[2])
    top_pairs = candidate_pairs[:min(ssd_top_n, len(candidate_pairs))]
    
    print(f"SSD筛选: 从 {len(candidate_pairs)} 对中选取了 {len(top_pairs)} 对进行协整检验")
    
    # 协整检验
    selected_pairs = []
    for i, j, ssd_dist in top_pairs:
        stock1 = valid_stocks[i]
        stock2 = valid_stocks[j]
        
        data1 = all_prices[i]
        data2 = all_prices[j]
        
        common_dates = data1.index.intersection(data2.index)
        if len(common_dates) < min_data_length:
            continue
        
        data1_aligned = data1.loc[common_dates]
        data2_aligned = data2.loc[common_dates]
        
        # 协整检验
        log_data1 = np.log(data1_aligned)
        log_data2 = np.log(data2_aligned)
        
        coint_result = coint(log_data1, log_data2)
        coint_p = coint_result[1]
        
        spread = log_data1 - log_data2
        adf_result = adfuller(spread)
        adf_p = adf_result[1]
        
        correlation = log_data1.corr(log_data2)
        
        if correlation > 0.85 and coint_p < 0.05 and adf_p < 0.05:
            selected_pairs.append((stock1, stock2, ssd_dist, correlation, coint_p, adf_p))
            print(f"  ✓ 符合条件: {stock1} - {stock2}")
            print(f"    SSD距离: {ssd_dist:.1f}, 相关性: {correlation:.3f}, 协整p值: {coint_p:.4f}")
    
    print(f"\n✅ 样本内选股完成！共筛选出 {len(selected_pairs)} 对符合条件的股票")
    
    if not selected_pairs:
        print("没有符合条件的股票对，程序退出")
        exit()
    
    # 保存样本内筛选结果
    sample_in_df = pd.DataFrame(selected_pairs, 
                                columns=['Stock1', 'Stock2', 'SSD距离', '相关性', '协整p值', 'ADF p值'])
    sample_in_df.to_csv('A股配对_样本内筛选结果.csv', index=False, encoding='utf-8-sig')
    print("样本内筛选结果已保存到: A股配对_样本内筛选结果.csv")
    
    print(f"\n第二步：样本外回测 ({OUT_SAMPLE_START} 到 {OUT_SAMPLE_END})")
    print("="*60)
    
    # 对每对股票进行样本外回测
    all_results = []
    
    for idx, (stock1, stock2, ssd_dist, corr, coint_p, adf_p) in enumerate(selected_pairs):
        print(f"\n回测第 {idx+1}/{len(selected_pairs)} 对: {stock1} - {stock2}")
        print(f"SSD距离: {ssd_dist:.1f}, 相关性: {corr:.3f}")
        
        # 创建策略实例
        strategy = ASharePairsTradingStrategy(
            initial_capital=1000000,
            commission_rate=0.0003,  # 0.03%
            stamp_tax_rate=0.001,    # 0.1%
            slippage=0.001,          # 0.1%
            min_commission=5.0,
            max_position_ratio=0.5,  # 单只股票最大50%
            min_position_ratio=0.1   # 单只股票最小10%
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
            rebalance_days=3
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
            
            print(f"  ✓ 回测完成:")
            print(f"    初始资金: {initial_value:,.0f}")
            print(f"    最终资金: {final_value:,.0f}")
            print(f"    总收益率: {total_return:.2%}")
            print(f"    交易次数: {total_trades}")
            print(f"    胜率: {win_rate:.1f}%")
            print(f"    夏普比率: {sharpe:.2f}")
            
            # 保存结果
            all_results.append({
                'pair': f"{stock1}-{stock2}",
                'ssd': ssd_dist,
                'corr': corr,
                'initial': initial_value,
                'final': final_value,
                'return': total_return,
                'trades': total_trades,
                'win_rate': win_rate,
                'sharpe': sharpe,
                'strategy': strategy
            })
            
            # 保存交易记录
            trades_df = pd.DataFrame(strategy.trade_history)
            trades_df.to_csv(f'A股配对_交易记录_{stock1}_{stock2}.csv', index=False, encoding='utf-8-sig')
            
            # 保存每日净值
            if strategy.equity_curve:
                equity_df = pd.DataFrame(strategy.equity_curve, columns=['date', 'value'])
                equity_df.set_index('date', inplace=True)
                equity_df.to_csv(f'A股配对_每日净值_{stock1}_{stock2}.csv', encoding='utf-8-sig')
    
    # 输出汇总结果
    if all_results:
        print(f"\n" + "="*60)
        print("样本外回测汇总结果")
        print("="*60)
        
        # 计算总体表现
        total_initial = sum(r['initial'] for r in all_results)
        total_final = sum(r['final'] for r in all_results)
        overall_return = (total_final - total_initial) / total_initial
        
        print(f"总配对数量: {len(all_results)}")
        print(f"初始总资金: {total_initial:,.0f}")
        print(f"最终总资金: {total_final:,.0f}")
        print(f"总收益率: {overall_return:.2%}")
        
        # 按收益率排序
        sorted_results = sorted(all_results, key=lambda x: x['return'], reverse=True)
        
        print(f"\n各配对表现排名:")
        for i, r in enumerate(sorted_results):
            print(f"{i+1}. {r['pair']}: 收益率={r['return']:.2%}, 交易次数={r['trades']}, 胜率={r['win_rate']:.1f}%")
        

        # 绘制净值曲线（每日市值变化）
        plt.rcParams["font.family"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "Microsoft YaHei"]
        plt.rcParams["axes.unicode_minus"] = False

        plt.figure(figsize=(16, 9))
        #高对比度颜色列表
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        #只展示前5个配对
        for i, r in enumerate(all_results[:5]):
            if r['strategy'].daily_records:
                dates = [rec['date'] for rec in r['strategy'].daily_records]
                total_values = [rec['total_value'] for rec in r['strategy'].daily_records]

                # 计算净值
                net_values = [v / 1000000.0 for v in total_values]

                color_idx = i % len(colors)

                # 绘制曲线，标签包含配对名称和收益率（百分比形式）
                plt.plot(dates, net_values, label=f"{r['pair']} (收益率: {r['return']:.2%})", 
                         linewidth=2.5, color=colors[color_idx], alpha=0.9)
        
        #初始净值参考线
        plt.axhline(y=1.0, color='r', linestyle='--', linewidth=1.5, label='初始净值(1.0)')

        plt.title('A股配对交易强弱轮动策略样本外回测（2025-01-01至2026-03-23）\n'
          '总收益率: {:.2%} | 最佳收益: {:.2%} | 最差收益: {:.2%} | 平均收益: {:.2%}'.format(
              overall_return, 
              max(r['return'] for r in all_results), 
              min(r['return'] for r in all_results), 
              sum(r['return'] for r in all_results) / len(all_results)
          ), fontsize=14, fontweight='bold')
        
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('净值（相对初始资金）', fontsize=12)

        plt.legend(loc='best', fontsize=10, frameon=True, shadow=True)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        # 日期刻度旋转（避免重叠）
        plt.xticks(rotation=45, ha='right', fontsize=10)       
        plt.tight_layout()
        plt.savefig('A股配对_样本外回测净值曲线.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 保存汇总结果
        summary_df = pd.DataFrame([{
            '股票对': r['pair'],
            'SSD距离': r['ssd'],
            '相关性': r['corr'],
            '初始资金': r['initial'],
            '最终资金': r['final'],
            '总收益率': r['return'],
            '年化收益率': (1 + r['return']) ** (252/len(r['strategy'].equity_curve)) - 1 if r['strategy'].equity_curve else 0,
            '交易次数': r['trades'],
            '胜率': r['win_rate'],
            '夏普比率': r['sharpe']
        } for r in all_results])
        
        summary_df.to_csv('A股配对_样本外回测汇总.csv', index=False, encoding='utf-8-sig')
        print(f"\n详细结果已保存到: A股配对_样本外回测汇总.csv")
    else:
        print("\n没有成功的回测结果")