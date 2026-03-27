# === 第一部分：导入必要的库 ===
import struct
import pandas as pd
import numpy as np
import os
from statsmodels.tsa.stattools import coint, adfuller
from scipy.spatial.distance import pdist, squareform
from itertools import combinations

# === 第二部分：定义读取通达信数据的函数 ===
def read_tdx_day_file(file_path):
    """读取通达信日线数据文件(.day)"""
    with open(file_path, 'rb') as f:
        data = f.read()
    
    records = []
    for i in range(0, len(data), 32):
        if i + 32 > len(data):
            break
            
        # 解析二进制数据
        buffer = data[i:i+32]
        # 格式：I(4字节日期) f(开盘) f(最高) f(最低) f(收盘) f(成交额) I(成交量) I(保留)
        date, open_price, high, low, close, amount, volume, _ = struct.unpack('IfffffII', buffer)
        
        # 通达信日期格式：YYYYMMDD
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
    """
    从通达信数据目录获取股票数据
    stock_code: 如 '600000' 或 '000001'
    tdx_data_dir: 通达信数据目录，如 'D:/new_tdx/vipdoc'
    """
    # 判断市场
    if stock_code.startswith('6'):
        market = 'sh'
    elif stock_code.startswith('0') or stock_code.startswith('3'):
        market = 'sz'
    elif stock_code.startswith('688'):
        market = 'sh'
    else:
        raise ValueError(f"未知股票代码: {stock_code}")
    
    # 构建文件路径
    file_name = f'{market}{stock_code}.day'
    # 注意路径：沪市在sh/lday，深市在sz/lday
    file_path = os.path.join(tdx_data_dir, f'{market}/lday', file_name)
    
    if os.path.exists(file_path):
        return read_tdx_day_file(file_path)
    else:
        print(f"文件不存在: {file_path}")
        return None
def compute_ssd_distance_matrix(price_data_list, normalize=True):
    """
    计算股票价格序列之间的SSD距离矩阵
    
    参数:
    price_data_list: 包含所有股票收盘价Series的列表
    normalize: 是否对价格进行标准化（消除绝对价格差异）
    
    返回:
    ssd_matrix: SSD距离矩阵
    """
    n = len(price_data_list)
    ssd_matrix = np.zeros((n, n))
    
    if normalize:
        # 标准化：将每只股票的价格除以其初始值
        # 这样可以消除价格的绝对差异，专注于相对走势
        normalized_data = []
        for price_series in price_data_list:
            if len(price_series) > 0:
                norm_series = price_series / price_series.iloc[0]
                normalized_data.append(norm_series)
            else:
                normalized_data.append(pd.Series())
    else:
        normalized_data = price_data_list
    
    # 计算两两之间的SSD距离
    for i in range(n):
        for j in range(i+1, n):
            if len(normalized_data[i]) > 0 and len(normalized_data[j]) > 0:
                # 确保时间序列对齐
                common_idx = normalized_data[i].index.intersection(normalized_data[j].index)
                if len(common_idx) > 0:
                    # 计算平方和距离
                    dist = np.sum((normalized_data[i].loc[common_idx] - normalized_data[j].loc[common_idx])**2)
                    ssd_matrix[i, j] = dist
                    ssd_matrix[j, i] = dist
                else:
                    ssd_matrix[i, j] = np.inf
                    ssd_matrix[j, i] = np.inf
            else:
                ssd_matrix[i, j] = np.inf
                ssd_matrix[j, i] = np.inf
    
    return ssd_matrix

# === 第三部分：配对交易筛选逻辑 ===
if __name__ == "__main__":
    # 配置参数
    TDX_DATA_DIR = "C:/new_tdx/vipdoc"  # 修改为你的通达信数据目录"C:\new_tdx\vipdoc"
    
    # 你的行业股票列表（这里用示例，替换为你的实际行业股票）
    industry_stocks = ['601288', '601398', '601077', '001227', '600036', '601988', '601128', '002936', '601939', '600926', '600000', '002142', '601963', '601818', '601860', '601229',
                       '603323', '600016', '600015', '601838', '600908', '601009', '601187', '601166', '601528', '601169', '601997', '002958', '002807', '601658', '600919', '601665',
                       '601577', '601825', '600928', '601916', '002966', '601998', '002839', '000001', '002948']  # 替换为你的行业股票代码列表
    
    # 设置日期范围
    start_date = "2021-08-01"
    end_date = "2024-12-31"

    # SSD参数
    ssd_top_n = 30  # 取SSD距离最小的前N对股票进行协整检验
    min_data_length = 100  # 最小数据长度要求

    print(f"开始处理 {len(industry_stocks)} 只股票...")

    # 第一步：加载所有股票数据
    all_prices = []
    valid_stocks = []  # 记录有效股票
    for stock_code in industry_stocks:
        df = get_stock_data_from_tdx(stock_code, TDX_DATA_DIR)
        if df is not None:
            # 提取收盘价并筛选日期范围
            price_series = df['close'].loc[start_date:end_date]
            if len(price_series) >= min_data_length:
                all_prices.append(price_series)
                valid_stocks.append(stock_code)
            else:
                print(f"股票 {stock_code} 数据不足: {len(price_series)} 条")
        else:
            print(f"无法读取股票 {stock_code} 数据")
    
    print(f"成功加载 {len(valid_stocks)} 只有效股票数据")
    
    if len(valid_stocks) < 2:
        print("有效股票数量不足，无法进行配对分析")
        exit()
    
    # 第二步：计算SSD距离矩阵
    print("计算SSD距离矩阵...")
    ssd_matrix = compute_ssd_distance_matrix(all_prices, normalize=True)
    
    # 第三步：获取SSD距离最小的前N对股票
    candidate_pairs = []
    n = len(valid_stocks)
    
    for i in range(n):
        for j in range(i+1, n):
            dist = ssd_matrix[i, j]
            if dist < np.inf:  # 排除无效对
                candidate_pairs.append((i, j, dist))
    
    # 按SSD距离从小到大排序
    candidate_pairs.sort(key=lambda x: x[2])
    
    # 取前N对
    top_pairs = candidate_pairs[:min(ssd_top_n, len(candidate_pairs))]
    
    print(f"SSD筛选结果: 从 {len(candidate_pairs)} 对中选取了 {len(top_pairs)} 对进行协整检验")
    print("=" * 60)
    
    # 第四步：对SSD筛选出的股票对进行协整检验
    results = []
    
    for idx, (i, j, ssd_dist) in enumerate(top_pairs, 1):
        stock1 = valid_stocks[i]
        stock2 = valid_stocks[j]
        
        data1 = all_prices[i]
        data2 = all_prices[j]
        
        # 确保时间序列对齐
        common_dates = data1.index.intersection(data2.index)
        if len(common_dates) < min_data_length:
            print(f"股票对 {stock1}-{stock2} 共同交易日不足: {len(common_dates)} 天")
            continue
        
        data1_aligned = data1.loc[common_dates]
        data2_aligned = data2.loc[common_dates]
        
        print(f"处理第 {idx}/{len(top_pairs)} 对: {stock1} 和 {stock2}, SSD距离: {ssd_dist:.2f}")
        
        # 对股票价格取对数
        log_data1 = np.log(data1_aligned)
        log_data2 = np.log(data2_aligned)
        
        # 计算相关性
        correlation = log_data1.corr(log_data2)
        
        # 协整性检验
        coint_result = coint(log_data1, log_data2)
        coint_t = coint_result[0]
        coint_p = coint_result[1]
        
        # 价差平稳性检验
        spread = log_data1 - log_data2
        adf_result = adfuller(spread)
        adf_stat = adf_result[0]
        adf_p = adf_result[1]
        
        # 筛选条件
        if correlation > 0.9 and coint_p < 0.05 and adf_p < 0.05:
            results.append({
                'Stock1': stock1,
                'Stock2': stock2,
                'SSD_Distance': ssd_dist,
                'Correlation': correlation,
                'Cointegration_t': coint_t,
                'Cointegration_p': coint_p,
                'ADF_stat': adf_stat,
                'ADF_p': adf_p
            })
            print(f"  √ 发现符合条件的配对!")
        else:
            print(f"  × 不满足条件: corr={correlation:.4f}, coint_p={coint_p:.4f}, adf_p={adf_p:.4f}")
    
    # 输出结果
    if results:
        results_df = pd.DataFrame(results)
        
        # 按SSD距离排序
        results_df = results_df.sort_values('SSD_Distance')
        
        print(f"\n{'='*60}")
        print(f"找到 {len(results_df)} 对符合条件的股票：")
        print('-'*60)
        
        for idx, row in results_df.iterrows():
            print(f"配对 {idx+1}: {row['Stock1']} - {row['Stock2']}")
            print(f"  SSD距离: {row['SSD_Distance']:.2f}")
            print(f"  相关性: {row['Correlation']:.4f}")
            print(f"  协整p值: {row['Cointegration_p']:.4f}")
            print(f"  ADF p值: {row['ADF_p']:.4f}")
            print('-'*30)
        
        # 保存结果到CSV
        results_df.to_csv('配对交易_SSD筛选结果.csv', index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到: 配对交易_SSD筛选结果.csv")
        
        # 输出SSD距离最小的前5对
        print(f"\nSSD距离最小的5对股票：")
        top_5_ssd = results_df.nsmallest(5, 'SSD_Distance')
        for _, row in top_5_ssd.iterrows():
            print(f"{row['Stock1']} - {row['Stock2']}: SSD={row['SSD_Distance']:.2f}, 相关性={row['Correlation']:.4f}")
            
    else:
        print("\n未找到符合条件的股票对")
        
    # 输出SSD距离矩阵信息
    print(f"\nSSD距离统计:")
    valid_distances = [dist for _, _, dist in candidate_pairs if dist < np.inf]
    if valid_distances:
        print(f"最小距离: {min(valid_distances):.2f}")
        print(f"最大距离: {max(valid_distances):.2f}")
        print(f"平均距离: {np.mean(valid_distances):.2f}")
        print(f"中位数距离: {np.median(valid_distances):.2f}")