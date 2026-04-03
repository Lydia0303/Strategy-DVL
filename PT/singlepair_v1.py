import pandas as pd
import numpy as np
import struct
import os
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
import matplotlib.pyplot as plt

# 定义股票代码
stock1 = "601997" 
stock2 = "601860"  

# 通达信本地数据根目录
TDX_ROOT_DIR = "C:/new_tdx/vipdoc"

# 定义获取数据的时间范围
start_date = "20210101"  # 起始日期
end_date = "20260303"    # 结束日期

def get_tdx_file_path(stock_code):
    """
    根据股票代码获取通达信日线文件完整路径
    文件路径格式: C:/new_tdx/vipdoc/{market}/lday/{stock_code}.day
    """
    code = str(stock_code).upper().strip()
    
    # 判断市场
    if code.startswith('SH') or code.endswith('.SH'):
        market = 'sh'
    elif code.startswith('SZ') or code.endswith('.SZ'):
        market = 'sz'
    else:
        # 纯数字代码，6开头为沪市，0/3开头为深市
        pure_code = code.zfill(6)
        if pure_code.startswith('6'):
            market = 'sh'
        else:
            market = 'sz'
    
    # 提取纯数字代码
    if code.startswith('SH'):
        pure_code = code[2:]
    elif code.startswith('SZ'):
        pure_code = code[2:]
    elif code.endswith('.SH'):
        pure_code = code[:-3]
    elif code.endswith('.SZ'):
        pure_code = code[:-3]
    else:
        pure_code = code
    
    # 构建文件路径
    file_path = os.path.join(TDX_ROOT_DIR, market, "lday", f"{market}{pure_code}.day")
    return market, pure_code, file_path

def read_tdx_day_file(file_path):
    """
    读取通达信 .day 二进制文件，返回 DataFrame
    """
    data = []
    try:
        with open(file_path, 'rb') as f:
            while True:
                buffer = f.read(32)  # 每条记录32字节
                if len(buffer) < 32:
                    break
                # 解包：日期(4), 开盘(4), 最高(4), 最低(4), 收盘(4), 成交量(4), 成交额(4), 保留(4)
                fields = struct.unpack('IIIIIIII', buffer)
                date = fields[0]
                open_price = fields[1] / 100.0
                high = fields[2] / 100.0
                low = fields[3] / 100.0
                close = fields[4] / 100.0
                volume = fields[5]
                amount = fields[6] / 100.0
                # 保留字段忽略
                data.append([date, open_price, high, low, close, volume, amount])
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        return None
    except Exception as e:
        print(f"读取文件 {file_path} 出错: {e}")
        return None

    if not data:
        print(f"文件 {file_path} 中没有数据")
        return None

    df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'amount'])
    df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    return df

# 获取股票1的历史数据
print(f"获取股票 {stock1} 的数据...")
market1, pure_code1, file_path1 = get_tdx_file_path(stock1)
print(f"  市场: {market1}, 代码: {pure_code1}")
print(f"  文件路径: {file_path1}")

stock1_data = read_tdx_day_file(file_path1)
if stock1_data is None or stock1_data.empty:
    print(f"无法读取股票 {stock1} 的数据")
    exit(1)
print(f"  读取成功: {len(stock1_data)} 条记录")

# 获取股票2的历史数据
print(f"\n获取股票 {stock2} 的数据...")
market2, pure_code2, file_path2 = get_tdx_file_path(stock2)
print(f"  市场: {market2}, 代码: {pure_code2}")
print(f"  文件路径: {file_path2}")

stock2_data = read_tdx_day_file(file_path2)
if stock2_data is None or stock2_data.empty:
    print(f"无法读取股票 {stock2} 的数据")
    exit(1)
print(f"  读取成功: {len(stock2_data)} 条记录")

# 根据时间范围筛选数据
print(f"\n筛选日期范围: {start_date} 到 {end_date}")
start_dt = pd.Timestamp(start_date)
end_dt = pd.Timestamp(end_date)

stock1_data = stock1_data[(stock1_data.index >= start_dt) & (stock1_data.index <= end_dt)]
stock2_data = stock2_data[(stock2_data.index >= start_dt) & (stock2_data.index <= end_dt)]

print(f"筛选后 {stock1} 数据量: {len(stock1_data)}")
print(f"筛选后 {stock2} 数据量: {len(stock2_data)}")

# 检查数据是否为空
if stock1_data.empty or stock2_data.empty:
    print("错误: 筛选后的数据为空，请检查日期范围")
    exit(1)

# 合并数据
merged_data = pd.DataFrame({
    'close_1': stock1_data['close'],
    'close_2': stock2_data['close']
}).dropna()

if merged_data.empty:
    print("错误: 合并后的数据为空")
    exit(1)

# 输出数据概览
print(f"\n{'='*50}")
print("数据概览:")
print('='*50)
print(f"\n股票 {stock1} 的数据:")
print(f"日期范围: {stock1_data.index[0]} 到 {stock1_data.index[-1]}")
print(f"前5行数据:")
print(stock1_data[['open', 'high', 'low', 'close', 'volume']].head())

print(f"\n股票 {stock2} 的数据:")
print(f"日期范围: {stock2_data.index[0]} 到 {stock2_data.index[-1]}")
print(f"前5行数据:")
print(stock2_data[['open', 'high', 'low', 'close', 'volume']].head())

print(f"\n合并后的配对数据 (前5行):")
print(merged_data.head())

# 对数价格序列
merged_data['log_close_1'] = np.log(merged_data['close_1'])
merged_data['log_close_2'] = np.log(merged_data['close_2'])

# 计算相关性
correlation = merged_data['log_close_1'].corr(merged_data['log_close_2'])
print(f"\n对数价格相关性: {correlation:.4f}")

# ADF检验
adf_result_1 = adfuller(merged_data['log_close_1'].dropna())
adf_result_2 = adfuller(merged_data['log_close_2'].dropna())

print(f"\nADF检验结果({stock1}): 统计量={adf_result_1[0]:.4f}, p-值={adf_result_1[1]:.4f}")
print(f"ADF检验结果({stock2}): 统计量={adf_result_2[0]:.4f}, p-值={adf_result_2[1]:.4f}")

if adf_result_1[1] < 0.05 and adf_result_2[1] < 0.05:
    print("两个资产的对数收盘价都是平稳的")
else:
    print("至少有一个资产的对数收盘价不是平稳的")

# 协整性检验
coint_result = coint(merged_data['log_close_1'], merged_data['log_close_2'])
coint_t, p_value, _ = coint_result

print(f"\n协整检验 t-统计量: {coint_t:.4f}")
print(f"协整检验 p-值: {p_value:.4f}")

if p_value < 0.05:
    print("两个资产存在协整关系")
else:
    print("两个资产不存在协整关系")

# 可视化
plt.rcParams["font.family"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.figure(figsize=(14, 10))

# 对数价格序列可视化
plt.subplot(3, 1, 1)
merged_data['log_close_1'].plot(label=f'{stock1} 对数价格')
merged_data['log_close_2'].plot(label=f'{stock2} 对数价格')
plt.title('Log Price Series (Formation Period)')
plt.xlabel('日期')
plt.ylabel('log price')
plt.legend()
plt.grid(True, alpha=0.3)

# 差分对数价格序列可视化
plt.subplot(3, 1, 2)
merged_data['log_close_1'].diff().plot(label=f'{stock1} 对数价格差分')
merged_data['log_close_2'].diff().plot(label=f'{stock2} 对数价格差分')
plt.title('Differenced Log Price Series (Formation Period)')
plt.xlabel('date')
plt.ylabel('difference')
plt.legend()
plt.grid(True, alpha=0.3)

# 价差序列可视化
plt.subplot(3, 1, 3)
spread = merged_data['log_close_1'] - merged_data['log_close_2']
spread.plot(label='spread', linewidth=1.5)
mu = spread.mean()
sd = spread.std()
plt.axhline(y=mu, color='black', linestyle='--', label='均值')
plt.axhline(y=mu + 2 * sd, color='red', linestyle='--', label='±2标准差')
plt.axhline(y=mu - 2 * sd, color='red', linestyle='--')
plt.axhline(y=mu + 1.5 * sd, color='orange', linestyle='--', label='±1.5标准差')
plt.axhline(y=mu - 1.5 * sd, color='orange', linestyle='--')
plt.axhline(y=mu + 1 * sd, color='green', linestyle='--', label='±1标准差')
plt.axhline(y=mu - 1 * sd, color='green', linestyle='--')
plt.title('Log Spread(Formation Period)')
plt.xlabel('date')
plt.ylabel('spread')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()