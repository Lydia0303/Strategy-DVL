import akshare as ak

# stock_board_industry_name_em_df = ak.stock_board_industry_name_em()
# dfcf_industry_l1=stock_board_industry_name_em_df.to_csv('东方财富一级行业板块.csv', index=False, encoding='utf-8-sig')

stock_board_industry_cons_em_df = ak.stock_board_industry_cons_em(symbol="粮食种植")
print(stock_board_industry_cons_em_df)
dfcf_industry=stock_board_industry_cons_em_df.to_csv('东方财富粮食种植行业成分股.csv', index=False, encoding='utf-8-sig')

