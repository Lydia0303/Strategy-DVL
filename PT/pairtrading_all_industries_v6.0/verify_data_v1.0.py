# === 数据路径验证与行业统计脚本 ===
# 在正式回测前运行，检查数据完整性

import struct
import pandas as pd
import os
import glob

def parse_industry_file(txt_path):
    """解析行业板块文件并统计"""
    industry_map = {}
    st_stocks = set()
    all_stocks = set()

    print(f"正在解析文件: {txt_path}")
    print("-" * 60)

    if not os.path.exists(txt_path):
        print(f"✗ 文件不存在: {txt_path}")
        return None, None, None

    try:
        with open(txt_path, 'r', encoding='gbk', errors='ignore') as f:
            lines = f.readlines()

        print(f"总行数: {len(lines)}")

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) >= 4:
                market_code = parts[0]
                industry_name = parts[1]
                stock_code = parts[2]
                stock_name = parts[3]

                all_stocks.add(stock_code)

                # 识别ST股票
                if 'ST' in stock_name.upper() or '*ST' in stock_name.upper():
                    st_stocks.add(stock_code)
                    continue

                if industry_name not in industry_map:
                    industry_map[industry_name] = []

                if stock_code not in industry_map[industry_name]:
                    industry_map[industry_name].append({
                        'code': stock_code,
                        'name': stock_name,
                        'market': market_code
                    })

        return industry_map, st_stocks, all_stocks

    except Exception as e:
        print(f"✗ 解析失败: {e}")
        return None, None, None

def check_tdx_data_path(data_dir):
    """检查通达信数据路径"""
    print(f"\n检查数据路径: {data_dir}")
    print("-" * 60)

    # 检查常见路径
    paths_to_check = [
        data_dir,
        os.path.join(data_dir, "sh", "lday"),
        os.path.join(data_dir, "sz", "lday"),
        os.path.join("C:/new_tdx/vipdoc", "sh", "lday"),
        os.path.join("D:/new_tdx/vipdoc", "sh", "lday"),
    ]

    for path in paths_to_check:
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"{exists} {path}")

    # 尝试读取一个示例文件
    sample_files = [
        os.path.join(data_dir, "sh", "lday", "sh600000.day"),
        os.path.join("C:/new_tdx/vipdoc", "sh", "lday", "sh600000.day"),
        os.path.join("D:/new_tdx/vipdoc", "sh", "lday", "sh600000.day"),
    ]

    print(f"\n尝试读取示例数据文件(浦发银行sh600000):")
    for path in sample_files:
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    chunk = f.read(32)
                    if len(chunk) == 32:
                        date, open_p, high, low, close, vol, amt, _ = struct.unpack('IIIIIIII', chunk)
                        date_str = str(date)
                        year = int(date_str[:4])
                        month = int(date_str[4:6])
                        day = int(date_str[6:8])
                        print(f"✓ 成功读取: {path}")
                        print(f"  最早数据: {year}-{month:02d}-{day:02d}")
                        print(f"  收盘价: {close/100:.2f}")
                        return True
            except Exception as e:
                print(f"✗ 读取失败 {path}: {e}")

    return False

def print_industry_summary(industry_map, st_stocks, all_stocks):
    """打印行业统计摘要"""
    print(f"\n{'='*80}")
    print("行业板块统计摘要")
    print(f"{'='*80}")

    print(f"\n【总体统计】")
    print(f"行业总数: {len(industry_map)}")
    print(f"股票总数(含ST): {len(all_stocks)}")
    print(f"ST股票数: {len(st_stocks)} ({len(st_stocks)/len(all_stocks)*100:.1f}%)")
    print(f"有效股票数: {len(all_stocks) - len(st_stocks)}")

    print(f"\n【行业列表】")

    # 按股票数量排序
    sorted_industries = sorted(industry_map.items(), key=lambda x: len(x[1]), reverse=True)

    for i, (name, stocks) in enumerate(sorted_industries, 1):
        st_count = sum(1 for s in stocks if s['code'] in st_stocks)
        clean_count = len(stocks) - st_count
        print(f"{i:2d}. {name:20s} | 成分股: {len(stocks):3d} | 有效: {clean_count:3d}")

    print(f"\n【股票数量Top 10行业】")
    for i, (name, stocks) in enumerate(sorted_industries[:10], 1):
        sample_stocks = [s['code'] for s in stocks[:3]]
        print(f"{i}. {name}: {len(stocks)}只 (示例: {', '.join(sample_stocks)}...)")

    print(f"\n【ST股票示例】")
    st_list = list(st_stocks)[:10]
    print(f"前10只: {', '.join(st_list)}")
    if len(st_stocks) > 10:
        print(f"... 等共{len(st_stocks)}只")
    return sorted_industries  # 返回排序后的列表

def main():
    # 配置
    LOCAL_INDUSTRY_TXT_PATH = "C:/new_tdx/T0002/export/行业板块.txt"
    TDX_DATA_DIR = "C:/new_tdx/vipdoc"

    print("="*80)
    print("A股全行业配对策略 - 数据验证工具 v1.0")
    print("="*80)


    # 1. 解析行业文件
    print("\n[步骤1] 解析行业板块文件...")
    industry_map, st_stocks, all_stocks = parse_industry_file(LOCAL_INDUSTRY_TXT_PATH)

    if industry_map is None:
        print("\n✗ 行业文件解析失败，请检查文件路径")
        return

    # 2. 打印统计
    sorted_industries=print_industry_summary(industry_map, st_stocks, all_stocks)

    # 3. 检查数据路径
    print(f"\n[步骤2] 检查通达信数据路径...")
    data_ok = check_tdx_data_path(TDX_DATA_DIR)

    if not data_ok:
        print("\n⚠️ 未能读取示例数据文件，请检查：")
        print("1. 通达信软件是否安装")
        print("2. 数据路径是否正确")
        print("3. 日线数据是否已下载")

    # 4. 保存行业列表供参考
    output_file = "行业列表统计.csv"
    with open(output_file, 'w', encoding='utf-8-sig') as f:
        f.write("行业名称,成分股数量,有效股票数量\n")
        for name, stocks in sorted_industries:
            st_count = sum(1 for s in stocks if s['code'] in st_stocks)
            f.write(f"{name},{len(stocks)},{len(stocks)-st_count}\n")

    print(f"\n✓ 行业列表已保存: {output_file}")

    print(f"\n{'='*80}")
    print("验证完成！")
    print(f"{'='*80}")

    if data_ok and industry_map:
        print("\n✓ 数据验证通过，可以运行正式回测程序")
    else:
        print("\n⚠️ 请解决上述问题后再运行回测")

if __name__ == "__main__":
    main()
