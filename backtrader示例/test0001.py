import akshare as ak

stock_board_industry_summary_ths_df = ak.stock_board_industry_summary_ths()
stock_board_industry_summary_ths_df.to_excel("同花顺一级行业.xlsx", index=False)
