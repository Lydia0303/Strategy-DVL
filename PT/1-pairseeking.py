import akshare as ak
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint, adfuller

# 获取行业成分股
sw_index_third_cons_df = ak.sw_index_third_cons(symbol="851811.SI")
print(sw_index_third_cons_df)
sw_index_third_cons_df_stocks = sw_index_third_cons_df['股票代码'].tolist()

# 获取股票的历史数据
def get_stock_data(stock_code, start_date, end_date):
    try:
        data = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        data['日期'] = pd.to_datetime(data['日期'])
        data.set_index('日期', inplace=True)
        return data['收盘']
    except Exception as e:
        print(f"Error fetching data for {stock_code}: {e}")
        return None

# 设置日期范围
start_date = "20230101"
end_date = "20241231"

# 存储股票对及其相关性、协整性和平稳性结果
results = []

# 遍历股票对
for i in range(len(sw_index_third_cons_df_stocks)):
    for j in range(i + 1, len(sw_index_third_cons_df_stocks)):
        stock1 = sw_index_third_cons_df_stocks[i]
        stock2 = sw_index_third_cons_df_stocks[j]

        # 将股票代码从 "600313.SH" 转换为 "600313"
        stock1_code = stock1.split('.')[0]  # 去掉 ".SH" 或 ".SZ"
        stock2_code = stock2.split('.')[0]  # 去掉 ".SH" 或 ".SZ"

        # 获取股票数据
        data1 = get_stock_data(stock1_code, start_date, end_date)
        data2 = get_stock_data(stock2_code, start_date, end_date)

        if data1 is not None and data2 is not None:
            # 确保数据长度一致
            common_dates = data1.index.intersection(data2.index)
            data1 = data1.loc[common_dates]
            data2 = data2.loc[common_dates]

            # 对股票价格取对数
            log_data1 = np.log(data1)
            log_data2 = np.log(data2)

            # 计算相关性（对数价格的相关性）
            correlation = log_data1.corr(log_data2)

            # 协整性检验（对数价格的协整性）
            coint_result = coint(log_data1, log_data2)
            coint_t = coint_result[0]
            coint_p = coint_result[1]

            # 检查价差的平稳性（对数价差的平稳性）
            spread = log_data1 - log_data2
            adf_result = adfuller(spread)
            adf_stat = adf_result[0]
            adf_p = adf_result[1]

            # 筛选条件：相关性高于0.9，协整性显著（p<0.05），价差平稳（p<0.05）
            if correlation > 0.9 and coint_p < 0.05 and adf_p < 0.05:
                results.append({
                    'Stock1': stock1,
                    'Stock2': stock2,
                    'Correlation': correlation,
                    'Cointegration_t': coint_t,
                    'Cointegration_p': coint_p,
                    'ADF_stat': adf_stat,
                    'ADF_p': adf_p
                })

# 将结果存储到DataFrame中
results_df = pd.DataFrame(results)
print(results_df)