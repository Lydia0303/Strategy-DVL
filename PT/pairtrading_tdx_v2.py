# === 完整的优化版代码（只回测前2对） ===
# === 第一部分：导入必要的库 ===
import struct
import pandas as pd
import numpy as np
import os
import warnings
from statsmodels.tsa.stattools import coint, adfuller
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

# === 第三部分：A股适配的强弱轮动策略 ===
class ASharePairsTradingStrategy:
    """A股配对交易策略（强弱轮动，不允许融券）"""
    
    def __init__(self, initial_capital=1000000, commission_rate=0.0003, 
                 stamp_tax_rate=0.001, slippage=0.001, min_commission=5.0,
                 max_position_ratio=0.5, min_position_ratio=0.1):
        
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.slippage = slippage
        self.min_commission = min_commission
        self.max_position_ratio = max_position_ratio
        self.min_position_ratio = min_position_ratio
        
        # 持仓记录
        self.positions = {}  # 股票代码 -> {'qty': 数量, 'avg_price': 平均成本, 'entry_date': 入场日期}
        self.trade_history = []  # 交易记录
        self.equity_curve = []  # 每日净值
        self.daily_records = []  # 每日记录
        
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
        
        # 确保买入数量是100的整数倍
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
        """计算价差的z-score序列"""
        common_dates = stock1_data.index.intersection(stock2_data.index)
        if len(common_dates) < window * 2:
            return None, None
        
        s1 = stock1_data.loc[common_dates]
        s2 = stock2_data.loc[common_dates]
        
        log_s1 = np.log(s1)
        log_s2 = np.log(s2)
        spread = log_s1 - log_s2
        
        zscore = pd.Series(index=spread.index, dtype=float)
        
        for i in range(window, len(spread)):
            window_spread = spread.iloc[i-window:i]
            mean_spread = window_spread.mean()
            std_spread = window_spread.std()
            
            if std_spread > 0:
                zscore.iloc[i] = (spread.iloc[i] - mean_spread) / std_spread
            else:
                zscore.iloc[i] = 0
        
        zscore = zscore.fillna(0)
        
        return spread, zscore
    
    def calculate_total_value(self, date, price1, price2):
        """计算投资组合总价值"""
        total_value = self.capital
        
        for stock_code, pos in self.positions.items():
            if stock_code == 'stock1':
                price = price1
            elif stock_code == 'stock2':
                price = price2
            else:
                price = pos['avg_price']
            
            total_value += price * pos['qty']
        
        return total_value
    
    def execute_strategy(self, stock1_code, stock2_code, stock1_data, stock2_data, 
                        window=40, entry_threshold=1.5, exit_threshold=0.3, 
                        stop_loss=2.0, max_holding_days=20, rebalance_threshold=0.8):
        """执行强弱轮动策略"""
        spread, zscore = self.calculate_spread_zscore(stock1_data, stock2_data, window)
        if zscore is None:
            print(f"数据不足，无法计算z-score")
            return None
        
        holding_stock = None
        entry_date = None
        entry_zscore = 0
        holding_days = 0
        
        for i, date in enumerate(zscore.index[window:]):
            if date not in stock1_data.index or date not in stock2_data.index:
                continue
            
            z = zscore.loc[date]
            price1 = stock1_data.loc[date]
            price2 = stock2_data.loc[date]
            
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
                'holding_days': holding_days
            }
            self.daily_records.append(daily_record)
            
            # 情况1：无持仓
            if holding_stock is None:
                if z > entry_threshold:  # stock1相对高估，stock2低估
                    position_value = min(total_value * self.max_position_ratio, self.capital)
                    volume = int(position_value / price2 / 100) * 100
                    
                    if volume > 0:
                        success, _, _ = self.buy_stock(date, stock2_code, price2, volume, 
                                                      reason=f"开仓-买入低估股(stock2), z={z:.2f}")
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
                                                      reason=f"开仓-买入低估股(stock1), z={z:.2f}")
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

# === 第四步：主程序（样本内选股 + 只回测前2对） ===
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
    
    # 策略参数
    TOP_N_PAIRS = 2  # <<< 修改这里：只回测前2对！
    min_data_length = 100
    
    print("="*60)
    print("A股配对交易策略（强弱轮动，不允许融券）")
    print(f"只回测排名前{TOP_N_PAIRS}的配对")
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
    
    # 协整检验
    selected_pairs = []
    for stock1, stock2, ssd_dist, correlation in candidate_pairs[:20]:  # 只检查前20对
        # 找到对应的数据
        idx1 = valid_stocks.index(stock1)
        idx2 = valid_stocks.index(stock2)
        data1 = all_prices[idx1]
        data2 = all_prices[idx2]
        
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
        
        # 价差平稳性检验
        spread = log_data1 - log_data2
        adf_result = adfuller(spread.dropna())
        adf_p = adf_result[1]
        
        if correlation > 0.85 and coint_p < 0.05 and adf_p < 0.05:
            selected_pairs.append((stock1, stock2, ssd_dist, correlation, coint_p, adf_p))
            print(f"  ✓ 符合条件: {stock1} - {stock2}")
            print(f"    SSD距离: {ssd_dist:.1f}, 相关性: {correlation:.3f}, 协整p值: {coint_p:.4f}")
    
    print(f"\n✅ 样本内选股完成！共筛选出 {len(selected_pairs)} 对符合条件的股票")
    
    if not selected_pairs:
        print("没有符合条件的股票对，程序退出")
        exit()
    
    # 按SSD距离排序，只选取前TOP_N_PAIRS对
    selected_pairs.sort(key=lambda x: x[2])  # 按SSD距离排序
    top_pairs = selected_pairs[:TOP_N_PAIRS]
    
    print(f"\n只回测排名前{TOP_N_PAIRS}的配对:")
    for i, (stock1, stock2, ssd_dist, corr, coint_p, adf_p) in enumerate(top_pairs):
        print(f"{i+1}. {stock1} - {stock2}: SSD={ssd_dist:.1f}, 相关性={corr:.3f}")
    
    # 保存样本内筛选结果
    sample_in_df = pd.DataFrame(selected_pairs, 
                                columns=['Stock1', 'Stock2', 'SSD距离', '相关性', '协整p值', 'ADF p值'])
    sample_in_df.to_csv('A股配对_样本内筛选结果_前2对.csv', index=False, encoding='utf-8-sig')
    print("样本内筛选结果已保存到: A股配对_样本内筛选结果_前2对.csv")
    
    print(f"\n第二步：样本外回测 ({OUT_SAMPLE_START} 到 {OUT_SAMPLE_END})")
    print("="*60)
    
    # 对前TOP_N_PAIRS对股票进行样本外回测
    all_results = []
    
    for idx, (stock1, stock2, ssd_dist, corr, coint_p, adf_p) in enumerate(top_pairs):
        print(f"\n回测第 {idx+1}/{len(top_pairs)} 对: {stock1} - {stock2}")
        print(f"SSD距离: {ssd_dist:.1f}, 相关性: {corr:.3f}")
        
        # 创建策略实例
        strategy = ASharePairsTradingStrategy(
            initial_capital=1000000,
            commission_rate=0.0003,
            stamp_tax_rate=0.001,
            slippage=0.001,
            min_commission=5.0,
            max_position_ratio=0.5,
            min_position_ratio=0.1
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
            trades_df.to_csv(f'A股配对_交易记录_{stock1}_{stock2}_前2对.csv', index=False, encoding='utf-8-sig')
            
            # 保存每日净值
            if strategy.equity_curve:
                equity_df = pd.DataFrame(strategy.equity_curve, columns=['date', 'value'])
                equity_df.set_index('date', inplace=True)
                equity_df.to_csv(f'A股配对_每日净值_{stock1}_{stock2}_前2对.csv', encoding='utf-8-sig')
    
    # 输出汇总结果
    if all_results:
        print(f"\n" + "="*60)
        print(f"样本外回测汇总结果（前{TOP_N_PAIRS}对）")
        print("="*60)
        
        # 计算总体表现
        total_initial = sum(r['initial'] for r in all_results)
        total_final = sum(r['final'] for r in all_results)
        overall_return = (total_final - total_initial) / total_initial
        
        print(f"回测配对数量: {len(all_results)}")
        print(f"初始总资金: {total_initial:,.0f}")
        print(f"最终总资金: {total_final:,.0f}")
        print(f"总收益率: {overall_return:.2%}")
        
        # 按收益率排序
        sorted_results = sorted(all_results, key=lambda x: x['return'], reverse=True)
        
        print(f"\n配对表现排名:")
        for i, r in enumerate(sorted_results):
            print(f"{i+1}. {r['pair']}: 收益率={r['return']:.2%}, 交易次数={r['trades']}, 胜率={r['win_rate']:.1f}%")
        
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
            # 创建等权重组合
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
        
        plt.title(f'A股配对交易强弱轮动策略样本外回测（前{TOP_N_PAIRS}对）\n'
                  f'总收益率: {overall_return:.2%} | 最佳收益: {max(r["return"] for r in all_results):.2%} | '
                  f'平均收益: {sum(r["return"] for r in all_results) / len(all_results):.2%}', 
                  fontsize=14, fontweight='bold')
        
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('净值（相对初始资金）', fontsize=12)
        
        plt.legend(loc='best', fontsize=10, frameon=True, shadow=True)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.xticks(rotation=45, ha='right', fontsize=10)       
        plt.tight_layout()
        plt.savefig(f'A股配对_样本外回测净值曲线_前{TOP_N_PAIRS}对.png', dpi=300, bbox_inches='tight')
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
        
        summary_df.to_csv(f'A股配对_样本外回测汇总_前{TOP_N_PAIRS}对.csv', index=False, encoding='utf-8-sig')
        print(f"\n详细结果已保存到: A股配对_样本外回测汇总_前{TOP_N_PAIRS}对.csv")
    else:
        print("\n没有成功的回测结果")