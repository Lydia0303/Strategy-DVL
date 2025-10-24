
import akshare as ak
import pandas as pd
# sw_index_third_info_df = ak.sw_index_third_info()
# sw_index_third_info_df.to_excel("sw_index_third_info.xlsx", index=False, engine="openpyxl")

stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol="002978", period="daily", start_date="20220101", end_date="20241231", adjust="qfq")
stock_zh_a_hist_df.to_excel("002978.xlsx", index=False)
print("数据已保存到EXCEL文件中。")



"""
import akshare as ak
import pandas as pd

# 定义股票代码列表
stock_codes = ["002329", "001318"]  # 示例股票代码

# 定义起始日期和结束日期
start_date = "20220101"
end_date = "20241231"

# 创建一个空的DataFrame，用于存储所有股票的数据
all_stock_data = pd.DataFrame()

# 遍历股票代码列表，下载每只股票的数据
for code in stock_codes:
    try:
        # 下载股票数据
        stock_data = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        
        # 添加股票代码列
        stock_data['code'] = code
        
        # 将数据添加到总DataFrame中
        all_stock_data = pd.concat([all_stock_data, stock_data], ignore_index=True)
        
        print(f"成功下载股票 {code} 的数据")
    except Exception as e:
        print(f"下载股票 {code} 的数据时出错: {e}")

# 打印所有股票的数据
# print(all_stock_data)

# 保存到CSV文件
all_stock_data.to_csv("all_stock_data.csv", index=False)
"""