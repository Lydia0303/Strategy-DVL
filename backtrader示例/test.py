import akshare as ak
import pandas as pd

def get_financial_data(stock_code, date):
    # 获取资产负债表
    balance_sheet = ak.stock_financial_report_sina(stock=stock_code, symbol="资产负债表")
    # 获取利润表
    income_statement = ak.stock_financial_report_sina(stock=stock_code, symbol="利润表")
    # 获取财务分析指标
    financial_analysis_indicator = ak.stock_financial_analysis_indicator(symbol=stock_code, start_year=date)
    
    return balance_sheet, income_statement, financial_analysis_indicator

def calculate_ev_ebit(balance_sheet, income_statement, report_date):
    report_date_str = str(report_date)  # 确保报告日是字符串格式
    filtered_df_balance_sheet = balance_sheet[balance_sheet['报告日'] == report_date_str]
    filtered_df_income_statement = income_statement[income_statement['报告日'] == report_date_str]

    
    # 从资产负债表中提取数据
    market_value = filtered_df_balance_sheet['负债和所有者权益(或股东权益)总计'].values[0] #获取市值（负债和所有者权益总计）
    debt = filtered_df_balance_sheet['负债合计'].values[0]
    cash = filtered_df_balance_sheet['货币资金'].values[0]
    
    # 从利润表中提取数据
    net_profit = filtered_df_income_statement['净利润'].values[0]
    income_tax = filtered_df_income_statement['所得税费用'].values[0]
    financial_expenses = filtered_df_income_statement['财务费用'].values[0]
    
    
    # 计算EV和EBIT
    EV = market_value + debt - cash
    EBIT = net_profit + income_tax + financial_expenses
    
    # 计算EV/EBIT
    EV_EBIT = EV / EBIT if EBIT != 0 else float('inf')
    
    return EV_EBIT

# 示例：获取特定股票在特定日期的财务数据并计算EV/EBITDA
stock_code = "600004"  # 示例股票代码
report_date = "20241231"  # 示例日期

balance_sheet, income_statement, financial_analysis_indicator = get_financial_data(stock_code, report_date)

# 打印表头
print("资产负债表表头：")
print(balance_sheet.columns)
print("\n利润表表头：")
print(income_statement.columns)
print("\n财务分析指标表头：")
print(financial_analysis_indicator.columns)

# 计算EV/EBITDA
ev_ebitda = calculate_ev_ebit(balance_sheet, income_statement, report_date)
print(f"EV/EBITDA for {stock_code} on {report_date}: {ev_ebitda}")