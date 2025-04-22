import akshare as ak
import pandas as pd

def check_and_print_date_info(stock1, stock2, benchmark, start_date, end_date):
    # 获取股票和沪深300指数的历史数据
    stock1_data = ak.stock_zh_a_daily(symbol=stock1, start_date=start_date, end_date=end_date, adjust="qfq")
    stock2_data = ak.stock_zh_a_daily(symbol=stock2, start_date=start_date, end_date=end_date, adjust="qfq")
    benchmark_data = ak.stock_zh_index_daily_em(symbol=benchmark, start_date=start_date, end_date=end_date)

    # 打印数据的前几行
    print("Stock1 Data Head:")
    print(stock1_data.head())
    print("\nStock2 Data Head:")
    print(stock2_data.head())
    print("\nBenchmark Data Head:")
    print(benchmark_data.head())

    # 检查日期列是否存在
    if 'date' not in stock1_data.columns:
        print(f"Error: 'date' column not found in stock1_data. Available columns: {stock1_data.columns}")
    else:
        print(f"Stock1 Data 'date' column exists. First date: {stock1_data['date'].iloc[0]}")

    if 'date' not in stock2_data.columns:
        print(f"Error: 'date' column not found in stock2_data. Available columns: {stock2_data.columns}")
    else:
        print(f"Stock2 Data 'date' column exists. First date: {stock2_data['date'].iloc[0]}")

    if 'date' not in benchmark_data.columns:
        print(f"Error: 'date' column not found in benchmark_data. Available columns: {benchmark_data.columns}")
    else:
        print(f"Benchmark Data 'date' column exists. First date: {benchmark_data['date'].iloc[0]}")

# 调用函数
if __name__ == "__main__":
    stock1 = "601328"  # 交通银行
    stock2 = "601998"  # 中信证券
    benchmark = "000300"  # 沪深300指数
    start_date = "2024-01-01"
    end_date = "2024-04-30"

    check_and_print_date_info(stock1, stock2, benchmark, start_date, end_date)