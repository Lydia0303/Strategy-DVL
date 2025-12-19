import pandas as pd
import matplotlib.pyplot as plt

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 设置中文字体（改用Windows系统自带字体，兼容性更强）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

def analyze_position(file_path):
    # 读取文件（指定引擎，兼容xls和xlsx）
    try:
        # 优先尝试openpyxl引擎（处理xlsx）
        df = pd.read_excel(file_path, engine='openpyxl')
    except:
        # 失败则用xlrd引擎（处理xls）
        df = pd.read_excel(file_path, engine='xlrd')

    # 将成交日期列转换为字符串类型，并按照正确格式解析为日期时间类型
    df['成交日期'] = pd.to_datetime(df['成交日期'].astype(str), format='%Y%m%d')

    # 根据证券名称判断是否为 ETF
    df['类型'] = df['证券名称'].apply(lambda x: 'ETF' if 'ETF' in str(x) else '股票')

    # 计算持仓变动，买入为正，卖出为负
    df['持仓变动'] = df.apply(
        lambda row: row['成交数量'] if row['买卖标志'] == '买入' else -row['成交数量'], 
        axis=1
    )

    # 按成交日期和类型分组，计算每日每种类型的持仓总和
    daily_position = df.groupby(['成交日期', '类型'])['持仓变动'].sum().reset_index()

    # 计算每日总持仓
    daily_total = daily_position.groupby('成交日期')['持仓变动'].sum().reset_index()
    daily_total.columns = ['成交日期', '总持仓']

    # 合并数据，计算每种类型的持仓占比
    result = pd.merge(daily_position, daily_total, on='成交日期', how='left')
    result['占比'] = result['持仓变动'] / result['总持仓'] * 100

    # 重塑表格以便展示
    pivot_result = result.pivot(index='成交日期', columns='类型', values='占比').fillna(0)

    # 转换日期格式并输出结果
    pivot_result.index = pivot_result.index.strftime('%Y-%m-%d')
    print("每日股票和 ETF 持仓占比（%）：")
    print(pivot_result)

    # 绘制柱状图
    ax = pivot_result.plot(kind='bar', stacked=True, figsize=(10, 6))

    # 设置图表标题和标签
    plt.title('每日股票和 ETF 持仓占比')
    plt.xlabel('日期')
    plt.ylabel('占比（%）')

    # 添加数据标签
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        ax.annotate(f'{height:.2f}%', (x + width/2, y + height/2), ha='center', va='center')

    plt.xticks(rotation=0)
    plt.show()

    return pivot_result

# 文件路径（Windows路径处理：前面加r表示原始字符串，避免转义错误）
file_path = r"C:\Users\hz\Documents\20251022_历史成交查询.xlsx"

# 调用函数
analyze_position(file_path)