# pip install akshare pandas tqdm
import akshare as ak
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# 1. 读取本地持仓文件
csv_path = r"C:\Users\hz\Desktop\日常\光照项目\持仓原始数据\hold_103157988_20250905_1506.csv"
hold_df = pd.read_csv(csv_path, dtype={'证券代码': 'str'})
hold_df['证券代码'] = hold_df['证券代码'].str.zfill(6)          # 补 0

# 2. 抓取全市场可转债比价表
print('正在抓取可转债比价表...')
cb_comp = ak.bond_cov_comparison()
# 只保留我们需要的列
cb_comp = cb_comp[['转债代码','转债名称','转债最新价','转股溢价率','纯债溢价率',
                   '正股代码','正股名称']].copy()
cb_comp['转债代码'] = cb_comp['转债代码'].astype(str).str.zfill(6)
cb_comp['正股代码'] = cb_comp['正股代码'].astype(str).str.zfill(6)

# 3. 抓取正股所属行业（申万一级）
print('正在抓取正股行业板块...')
# 用 stock_individual_info_em 接口，字段里含“行业”
def get_industry(stock_code: str) -> str:
    try:
        info = ak.stock_individual_info_em(symbol=stock_code)
        # 返回 DataFrame，指标名在行索引，value 在列
        return info.loc[info['item']=='行业','value'].values[0]
    except Exception as e:
        return ''

uniq_stocks = cb_comp['正股代码'].drop_duplicates().tolist()
industry_map = {code: get_industry(code) for code in tqdm(uniq_stocks, desc='行业抓取')}
cb_comp['所属行业'] = cb_comp['正股代码'].map(industry_map)

# 4. 与本地持仓匹配
result = hold_df.merge(cb_comp,
                       left_on='证券代码',
                       right_on='转债代码',
                       how='left')

# 5. 保存结果
save_path = Path(csv_path).with_name(f'可转债比价及行业_{pd.Timestamp.today().strftime("%Y%m%d")}.csv')
result.to_csv(save_path, index=False, encoding='utf_8_sig')
print(f'结果已保存至：{save_path}')
result