import akshare as ak
df = ak.bond_cb_jsl('20231229')
print(df['转股溢价率'].dtype)   # 大概率是 float64
print(df['转股溢价率'].head())