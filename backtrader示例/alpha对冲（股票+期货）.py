import backtrader as bt
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta

#定义获取沪深300指数成分股的全局函数
def get_hs300_constituents(date):
    stock300 = ak.index_stock_cons(symbol="000300") 
    return stock300['品种代码'].tolist()

# 定义获取沪深300成分股股票历史行情数据的全局函数
def get_stock_data(symbol, start_date, end_date):
    stock_data = ak.stock_zh_a_hist(symbol=symbol, start_date=start_date, end_date=end_date, adjust="qfq")
    if not stock_data.empty:
        stock_data.index = pd.to_datetime(stock_data['日期'])
        stock_data = stock_data[['开盘', '最高', '最低', '收盘', '成交量']]
        stock_data.columns = ['open', 'high', 'low', 'close', 'volume']
    return stock_data
# 定义获取沪深300期货主力合约历史行情数据的全局函数
def get_futures_data(symbol, start_date, end_date):
    # 注意：这里使用 futures_main_sina 可能无法正确获取历史主力合约数据
    # 建议使用更合适的接口（如 ak.futures_zh_main_contracts）
    try:
        futures_data = ak.futures_zh_daily_sina(symbol=symbol, start_date=start_date, end_date=end_date)
        if not futures_data.empty:
            futures_data.index = pd.to_datetime(futures_data['日期'])
            futures_data = futures_data[['开盘价', '最高价', '最低价', '收盘价', '成交量']]
            futures_data.columns = ['open', 'high', 'low', 'close', 'volume']
            return futures_data  # 返回DataFrame，而非backtrader数据源
        else:
            print(f"期货数据为空: {symbol} ({start_date} - {end_date})")
            return None
    except Exception as e:
        print(f"获取期货数据失败: {e}")
        return None

def get_previous_trading_date(exchange='SHSE', date=None):
    """
    获取指定日期的前一交易日（默认使用上交所交易日历）
    :param exchange: 交易所代码，默认'SHSE'（上交所）
    :param date: 输入日期（格式：YYYY-MM-DD或datetime对象）
    :return: 前一交易日日期（datetime.date对象）
    """
    if date is None:
        date = datetime.now()
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y%m%d")
    elif isinstance(date, datetime):
        date = date
    
    # 获取交易日历
    trade_dates = ak.tool_trade_date_hist_sina()['trade_date'].astype(str)
    trade_dates = [datetime.strptime(d, "%Y-%m-%d") for d in trade_dates]
    trade_dates = sorted([d.date() for d in trade_dates])  # 转换为date对象列表
    
    # 查找前一交易日
    try:
        idx = trade_dates.index(date.date())
        if idx == 0:
            raise ValueError("输入日期为最早交易日，无前一交易日")
        prev_date = trade_dates[idx-1]
        return prev_date.strftime("%Y%m%d")  # 输出为YYYYMMDD格式
    except ValueError:
        raise ValueError(f"日期{date.strftime('%Y%m%d')}不在交易日历中")
    
class AlphaHedgeStrategy(bt.Strategy):
    params = (
        ('stock_percentage', 0.4),
        ('futures_percentage', 0.4),
        ('futures_symbol', None)
    )

    def __init__(self):
        self.stock_data = {}
        self.futures_data = None
        self.last_trade_date = None
        self.futures_data_index = -1  # 存储期货数据在datas中的索引
        self.futures_symbol = None
    
    def start(self):
        # 在回测开始时获取主力合约
        if not self.p.futures_symbol:
            self.futures_symbol = self.get_futures_main_contract(date=self.data0.datetime.date())
        else:
            self.futures_symbol = self.p.futures_symbol
            
        if self.futures_symbol:
            # 尝试从已添加的数据中查找期货数据
            for data in self.datas:
                if data._name == self.futures_symbol:
                    self.futures_data = data
                    break
                    
            if not self.futures_data:
                print(f"策略中未找到期货数据: {self.futures_symbol}")
        else:
            print("无法获取主力合约，回测将不包含期货对冲")

    def next(self):
        current_date = self.datas[0].datetime.date(0)
        current_time = self.datas[0].datetime.time(0)

        # 检查是否为每月第一个交易日且时间为09:35:00
        if (current_date.day == 1 and current_time.hour == 9 and current_time.minute == 35 and current_date != self.last_trade_date):
            self.last_trade_date = current_date
            self.execute_algo()
    
    def get_futures_main_contract(self, date):
        # 获取指定日期的沪深300股指期货主力合约（AKshare目前已有接口无法获取到历史主力合约，这里用当前实时的主力合约替代，可能与历史日期不完全匹配）
        try:
            date_str = date.strftime("%Y%m%d") if isinstance(date, datetime) else date
            futures_data=ak.futures_main_sina(symbol="IF")
            if not futures_data.empty:
                main_contract = futures_data.iloc[0]['合约代码']
                print(f"获取到主力合约: {main_contract}，日期: {date_str}")
                return main_contract
    
            else:
                raise ValueError("No futures main contract found for the date")
        except Exception as e:
            print(f"获取主力合约失败: {e}")
            return None
    def execute_algo(self):
        # 执行策略逻辑
        print(f"Executing strategy on {self.datas[0].datetime.date(0)} at {self.datas[0].datetime.time(0)}")
        current_date = self.last_trade_date  # 当前调仓日（每月第一个交易日）
        # 获取前一交易日（用于获取成分股和期货合约）
        last_trade_day = get_previous_trading_date(date=current_date)
        # 获取沪深300成分股
        stock300 = get_hs300_constituents(last_trade_day)  # 修正：使用last_trade_day而非current_date
        print(f"HS300 constituents: {stock300}")
        # 获取有交易的股票
        not_suspended_symbols = self.get_not_suspended_symbols(stock300)
        # 获取EV/EBITDA大于0且排名最小的30只股票
        selected_stocks = self.get_selected_stocks(not_suspended_symbols)
        # 平掉不在标的池中的股票持仓
        self.close_positions(selected_stocks)
        # 等权购买标的池中的股票
        self.buy_selected_stocks(selected_stocks)

        # 获取主力合约（新增步骤）
        last_day = self.last_trade_date  # 调仓日的前一交易日
        futures_symbol = self.get_futures_main_contract(last_day)  # 调用类方法获取合约代码
        if not futures_symbol:
            print("无法获取主力合约，跳过对冲操作")
            return
    
        # 获取期货数据
        if self.futures_data is None:
            # 从全局获取期货数据（修正：使用正确的参数）
            futures_df = get_futures_data(
                symbol=futures_symbol,
                start_date=self.datas[0].datetime.date(0).strftime("%Y%m%d"),
                end_date=self.datas[0].datetime.date(-1).strftime("%Y%m%d")  # 使用当前数据的最后日期
            )
            
            if futures_df is not None:
                # 转换为backtrader数据源
                self.futures_data = bt.feeds.PandasData(dataname=futures_df, name=futures_symbol)
                # 查找期货数据在datas中的索引
                for i, data in enumerate(self.datas):
                    if data._name == futures_symbol:
                        self.futures_data_index = i
                        break

        # 获取股指期货的保证金比率
        margin_ratio = self.get_futures_margin_ratio()
        print(f"Margin ratio for futures: {margin_ratio}")
        # 更新股指期货的权重并进行对冲
        self.hedge_futures(margin_ratio, futures_symbol)  # 传递合约代码至对冲函数
    
    # 新增：获取期货历史数据的类方法
    def get_futures_data(self, symbol, start_date, end_date):
        futures_data = ak.futures_main_sina(symbol=symbol, start_date=start_date, end_date=end_date)
        if not futures_data.empty:
            futures_data.index = pd.to_datetime(futures_data['日期'])
            futures_data = futures_data[['开盘价', '最高价', '最低价', '收盘价', '成交量']]
            futures_data.columns = ['open', 'high', 'low', 'close', 'volume']
            return bt.feeds.PandasData(dataname=futures_data)
        else:
            return None
        
    def get_not_suspended_symbols(self, stock300):
        not_suspended_info = ak.stock_zh_a_spot_em()
        not_suspended_symbols = not_suspended_info[not_suspended_info['stock_code'].isin(stock300)]['stock_code'].tolist()
        return not_suspended_symbols
    
    def get_financial_data(stock_code, current_date):
        # 获取资产负债表
        balance_sheet = ak.stock_financial_report_sina(stock=stock_code, symbol="资产负债表")
        # 获取利润表
        income_statement = ak.stock_financial_report_sina(stock=stock_code, symbol="利润表")
        # 获取财务分析指标
        financial_analysis_indicator = ak.stock_financial_analysis_indicator(symbol=stock_code, start_year=current_date)
    
        return balance_sheet, income_statement, financial_analysis_indicator
    def get_selected_stocks(self, not_suspended_symbols):
        # 初始化存放EV/EBIT数据的DataFrame
        ev_ebit_df = pd.DataFrame(columns=['stock_code', 'EV/EBIT'])
    
        current_date = self.last_trade_date.strftime('%Y%m%d')  # 转换为YYYYMMDD格式，适配akshare接口
    
        # 遍历每只未停牌股票，获取财务数据并计算EV/EBIT
        for stock_code in not_suspended_symbols:
            try:
                # 获取单只股票的财务数据（需确保get_financial_data函数与当前策略兼容）
                balance_sheet, income_statement, _ = self.get_financial_data(stock_code, current_date)
            
                # 过滤当前报告期数据
                filtered_balance = balance_sheet[balance_sheet['报告日'] == current_date]
                filtered_income = income_statement[income_statement['报告日'] == current_date]
            
                # 检查数据是否存在
                if filtered_balance.empty or filtered_income.empty:
                   continue  # 跳过无数据的股票
            
                # 从资产负债表获取数据（注意字段可能需根据实际返回调整）
                market_value = filtered_balance['负债和所有者权益(或股东权益)总计'].values[0]
                debt = filtered_balance['负债合计'].values[0]
                cash = filtered_balance['货币资金'].values[0]
            
                # 从利润表获取数据
                net_profit = filtered_income['净利润'].values[0]
                income_tax = filtered_income['所得税费用'].values[0]
                financial_expenses = filtered_income['财务费用'].values[0]
            
                # 计算EV和EBIT（EBITDA = EBIT + 折旧 + 摊销，需补充折旧摊销数据，假设从财务分析指标获取）
                # 注意：当前单只股票逻辑计算的是EV/EBIT，若需EBITDA，需额外获取折旧和摊销数据
                ev = market_value + debt - cash
                ebit = net_profit + income_tax + financial_expenses
                if ebit == 0:
                    ev_ebit_ratio = float('inf')  # 避免除零错误
                else:
                    ev_ebit_ratio = ev / ebit  # 此处暂用EV/EBIT替代，如需EBITDA需补充计算
                
                # 将结果存入DataFrame
                ev_ebit_df = ev_ebit_df.append({
                'stock_code': stock_code,
                'EV/EBIT': ev_ebit_ratio
                }, ignore_index=True)
            
            except Exception as e:
                print(f"Error processing {stock_code}: {e}")
                continue  # 跳过异常股票
    
        # 筛选EV/EBIT>0的股票，并按升序排序取前30
        if not ev_ebit_df.empty:
            selected_stocks = ev_ebit_df[ev_ebit_df['EV/EBIT'] > 0].sort_values(
                by='EV/EBIT', ascending=True
            ).head(30)['stock_code'].tolist()
        else:
            selected_stocks = []  # 无有效数据时返回空列表
    
        return selected_stocks
    
    def close_positions(self, selected_stocks):
        for data in self.datas:
            if data._name not in selected_stocks:
                self.close(data)

    def buy_selected_stocks(self, selected_stocks):
        stock_percentage = self.p.stock_percentage / len(selected_stocks)
        for stock in selected_stocks:
            self.buy(data=self.stock_data[stock], percent=stock_percentage)

    def get_futures_margin_ratio(self):
        futures_rule_df = ak.futures_rule(date=self.last_trade_date.strftime('%Y%m%d'))
        margin_ratio = futures_rule_df.loc[futures_rule_df['品种'] == '沪深300股指期货', '交易保证金比例'].values[0]
        return float(margin_ratio.strip('%')) / 100

    def hedge_futures(self, margin_ratio, futures_symbol):
        futures_percentage = self.p.futures_percentage * margin_ratio
        # 使用主力合约代码下单
        self.sell(data=self.datas[self.futures_data_index], percent=futures_percentage)  # 假设期货数据在datas中的索引已知
        
        # 或通过symbol查找数据（更灵活）
        for data in self.datas:
            if data._name == futures_symbol:
                self.sell(data=data, percent=futures_percentage)
                break


if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.addstrategy(AlphaHedgeStrategy)

    start_date="20240701"
    end_date="20241001"

    # 获取前一交易日，用于确定初始主力合约
    last_day = get_previous_trading_date(date=start_date)

    # 获取沪深300成分股的历史价格数据
    stock300 = get_hs300_constituents(date=last_day)[:5]  # 使用前一交易日的成分股
    for stock in stock300:
        try:
            stock_data = get_stock_data(symbol=stock, start_date=start_date, end_date=end_date)
            if not stock_data.empty:
                cerebro.adddata(bt.feeds.PandasData(dataname=stock_data, name=stock))
            else:
                print(f"No data for {stock}")
        except Exception as e:
            print(f"Error processing data for {stock}: {e}")
   
    
    # 设置初始资金
    cerebro.broker.setcash(10000000)
    # 设置佣金比例
    cerebro.broker.setcommission(commission=0.0001)
    # 设置滑点比例
    cerebro.broker.set_slippage_perc(perc=0.0001)

        # ========== 添加分析器 ==========
    # 累计收益率
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    # 年化收益率
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annual_return')
    # 夏普比率（默认无风险利率=0，时间周期=252天）
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0, timeframe=bt.TimeFrame.Days)
    # 最大回撤
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    # 交易统计（胜率、盈亏比等）
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    # 时间加权收益率
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')

    # 启动回测
    results = cerebro.run()
    # 运行回测后处理分析器结果
def print_backtest_results(results):
    strat = results[0]  # 获取策略实例
    
    # 1. 累计收益率
    ret_analyzer = strat.analyzers.returns.get_analysis()
    cum_return = ret_analyzer['rtot']  # 总收益率
    print(f"累计收益率: {cum_return:.2%}")
    
    # 2. 年化收益率
    annual_ret = strat.analyzers.annual_return.get_analysis()
    print(f"年化收益率: {list(annual_ret.values())[0]:.2%}")  # 取第一年的年化
    
    # 3. 夏普比率
    sharpe = strat.analyzers.sharpe.get_analysis()
    print(f"夏普比率: {sharpe['sharperatio']:.2f}")
    
    # 4. 最大回撤
    drawdown = strat.analyzers.drawdown.get_analysis()
    print(f"最大回撤: {drawdown['max']['drawdown']:.2%}")
    
    # 5. 交易统计（胜率、盈亏比）
    trade_stats = strat.analyzers.trades.get_analysis()
    win_ratio = trade_stats.won.total / trade_stats.total.closed * 100
    profit_factor = trade_stats.won.pnl.total / abs(trade_stats.lost.pnl.total)
    print(f"胜率: {win_ratio:.2f}%")
    print(f"盈亏比: {profit_factor:.2f}")

     # ===== 输出结果 =====
    print("\n========== 回测结果 ==========")
    print(f"最终资金: {cerebro.broker.getvalue():,.2f}")
    print_backtest_results(results)  # 调用上面定义的函数
   
    
    # 绘制回测结果
    cerebro.plot()
