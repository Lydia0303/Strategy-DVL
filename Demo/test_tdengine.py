from taosrest import connect
import pandas as pd
from typing import List, Dict, Any

def analyze_tdengine_structure():
    """分析TDengine数据结构"""
    print("🔍 分析TDengine数据结构")
    
    try:
        # 连接
        conn = connect(
            url="http://172.16.66.188:6041",
            user="root",
            password="taosdata"
        )
        print("✅ 连接成功")
        
        # 1. 重新运行之前的查询看看SQL是什么
        print("\n📊 重新运行查询:")
        result = list(conn.query("SHOW stock.TABLES"))
        print(f"查询结果类型: {type(result)}")
        print(f"结果数量: {len(result)}")
        
        if result:
            print("\n前10个表:")
            for i, table in enumerate(result[:10], 1):
                print(f"{i:3d}. {table[0]}")
            
            # 2. 检查第一个表的结构
            first_table = result[0][0]
            print(f"\n🔍 分析表: {first_table}")
            
            # 获取表结构
            try:
                desc = list(conn.query(f"DESCRIBE stock.{first_table}"))
                print(f"表结构 ({len(desc)}列):")
                for col in desc:
                    col_name = col[0]
                    col_type = col[1]
                    col_length = col[2] if len(col) > 2 and col[2] else ""
                    
                    if col_length:
                        print(f"  - {col_name}: {col_type}({col_length})")
                    else:
                        print(f"  - {col_name}: {col_type}")
                        
            except Exception as e:
                print(f"查询表结构失败: {e}")
            
            # 3. 查询数据样例
            try:
                sample = list(conn.query(f"SELECT * FROM stock.{first_table} LIMIT 5"))
                if sample:
                    print(f"\n📈 数据样例 (前{len(sample)}行):")
                    
                    # 获取列名
                    desc = list(conn.query(f"DESCRIBE stock.{first_table}"))
                    columns = [col[0] for col in desc]
                    
                    for i, row in enumerate(sample, 1):
                        print(f"\n行 {i}:")
                        for col_name, value in zip(columns, row):
                            if isinstance(value, (int, float)):
                                print(f"  {col_name}: {value:.4f}" if isinstance(value, float) else f"  {col_name}: {value}")
                            else:
                                print(f"  {col_name}: {value}")
                                
            except Exception as e:
                print(f"查询数据失败: {e}")
        
        # 4. 查看是否有超级表
        print("\n🔍 查看超级表:")
        try:
            stables = list(conn.query("SHOW stock.STABLES"))
            if stables:
                print(f"发现 {len(stables)} 个超级表:")
                for stable in stables:
                    print(f"  - {stable[0]}")
                    
                    # 查看超级表结构
                    try:
                        desc = list(conn.query(f"DESCRIBE stock.{stable[0]}"))
                        print(f"    超级表结构 ({len(desc)}列):")
                        for col in desc[:5]:  # 只显示前5列
                            print(f"      - {col[0]}: {col[1]}")
                        if len(desc) > 5:
                            print(f"      ... 等{len(desc)}列")
                    except:
                        pass
            else:
                print("无超级表")
        except Exception as e:
            print(f"查询超级表失败: {e}")
        
        conn.close()
        print("\n✅ 分析完成")
        
    except Exception as e:
        print(f"❌ 连接失败: {e}")

def get_stock_data_sample():
    """获取股票数据样例"""
    print("📈 获取股票数据样例")
    
    try:
        conn = connect(
            url="http://172.16.66.188:6041",
            user="root",
            password="taosdata"
        )
        
        # 1. 获取一个股票表
        tables = list(conn.query("SHOW stock.TABLES"))
        if not tables:
            print("⚠ 无表")
            return
        
        # 选择一个表
        target_table = tables[0][0]
        print(f"选择表: {target_table}")
        
        # 2. 查询数据统计
        print("\n📊 数据统计:")
        try:
            # 检查是否有时间字段
            desc = list(conn.query(f"DESCRIBE stock.{target_table}"))
            time_fields = []
            for col in desc:
                if 'timestamp' in col[1].lower() or 'time' in col[0].lower():
                    time_fields.append(col[0])
            
            if time_fields:
                time_field = time_fields[0]
                stats_sql = f"""
                SELECT 
                    MIN({time_field}) as 最早时间,
                    MAX({time_field}) as 最近时间,
                    COUNT(*) as 总行数
                FROM stock.{target_table}
                """
                
                stats = list(conn.query(stats_sql))
                if stats and stats[0]:
                    print(f"  时间范围: {stats[0][0]} 到 {stats[0][1]}")
                    print(f"  总行数: {stats[0][2]:,}")
                    
                    # 查询最新数据
                    latest_sql = f"""
                    SELECT * FROM stock.{target_table} 
                    ORDER BY {time_field} DESC 
                    LIMIT 3
                    """
                    
                    latest = list(conn.query(latest_sql))
                    if latest:
                        print(f"\n📈 最新3条数据:")
                        columns = [col[0] for col in desc]
                        
                        for i, row in enumerate(latest, 1):
                            print(f"\n  行 {i}:")
                            for col_name, value in zip(columns, row):
                                if col_name == time_field:
                                    print(f"    {col_name}: {value}")
                                elif isinstance(value, (int, float)):
                                    if 'price' in col_name.lower() or 'close' in col_name.lower():
                                        print(f"    {col_name}: {value:.4f}")
                                    elif 'volume' in col_name.lower():
                                        if value >= 1000000:
                                            print(f"    {col_name}: {value/1000000:.2f}M")
                                        elif value >= 1000:
                                            print(f"    {col_name}: {value/1000:.2f}K")
                                        else:
                                            print(f"    {col_name}: {value}")
                                    else:
                                        print(f"    {col_name}: {value}")
                                else:
                                    print(f"    {col_name}: {value}")
            else:
                print("  未找到时间字段")
                
        except Exception as e:
            print(f"  统计查询失败: {e}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ 失败: {e}")

def find_kline_tables():
    """查找K线数据表"""
    print("🔍 查找K线数据表")
    
    try:
        conn = connect(
            url="http://172.16.66.188:6041",
            user="root",
            password="taosdata"
        )
        
        # 获取所有表
        tables = list(conn.query("SHOW stock.TABLES"))
        print(f"总表数: {len(tables)}")
        
        # 分析表结构
        kline_tables = []
        
        for i, table_info in enumerate(tables[:20], 1):  # 只检查前20个
            table_name = table_info[0]
            
            try:
                # 获取表结构
                desc = list(conn.query(f"DESCRIBE stock.{table_name}"))
                
                # 检查是否是K线表
                has_ts = any('timestamp' in col[1].lower() for col in desc)
                has_price = any(price in col[0].lower() for price in ['open', 'high', 'low', 'close'])
                has_volume = any(vol in col[0].lower() for vol in ['volume', 'amount'])
                
                if has_ts and (has_price or has_volume):
                    kline_tables.append(table_name)
                    print(f"\n✅ 发现K线表 {i}: {table_name}")
                    print(f"   字段: {[col[0] for col in desc]}")
                    
                    # 查询数据样例
                    try:
                        sample = list(conn.query(f"SELECT * FROM stock.{table_name} LIMIT 1"))
                        if sample:
                            print(f"   样例: {sample[0]}")
                    except:
                        pass
                        
            except Exception as e:
                continue
        
        print(f"\n🎯 发现 {len(kline_tables)} 个K线表")
        
        if kline_tables:
            # 详细分析第一个K线表
            analyze_kline_table(conn, kline_tables[0])
        
        conn.close()
        
    except Exception as e:
        print(f"❌ 失败: {e}")

def analyze_kline_table(conn, table_name):
    """分析K线表"""
    print(f"\n📊 详细分析K线表: {table_name}")
    
    try:
        # 获取表结构
        desc = list(conn.query(f"DESCRIBE stock.{table_name}"))
        columns = [col[0] for col in desc]
        
        print(f"表结构 ({len(columns)}列):")
        for i, col in enumerate(columns, 1):
            print(f"{i:2d}. {col}")
        
        # 查找时间字段
        time_field = None
        for col in desc:
            if 'timestamp' in col[1].lower():
                time_field = col[0]
                break
        
        if time_field:
            print(f"\n⏰ 时间字段: {time_field}")
            
            # 查询时间范围
            time_sql = f"""
            SELECT 
                MIN({time_field}) as 最早,
                MAX({time_field}) as 最晚,
                COUNT(*) as 总数
            FROM stock.{table_name}
            """
            
            time_stats = list(conn.query(time_sql))
            if time_stats and time_stats[0]:
                print(f"📅 时间范围: {time_stats[0][0]} 到 {time_stats[0][1]}")
                print(f"📈 数据总量: {time_stats[0][2]:,} 行")
            
            # 查询不同时间频率
            freq_sql = f"""
            SELECT 
                COUNT(*) as 数量,
                DATE_FORMAT({time_field}, '%Y-%m-%d') as 日期
            FROM stock.{table_name}
            GROUP BY DATE_FORMAT({time_field}, '%Y-%m-%d')
            ORDER BY 日期 DESC
            LIMIT 5
            """
            
            try:
                freqs = list(conn.query(freq_sql))
                if freqs:
                    print(f"\n📅 最近5天数据量:")
                    for freq in freqs:
                        print(f"  {freq[1]}: {freq[0]:,} 行")
            except:
                pass
        
        # 查询最新数据
        latest_sql = f"SELECT * FROM stock.{table_name} ORDER BY {time_field if time_field else columns[0]} DESC LIMIT 5"
        latest = list(conn.query(latest_sql))
        
        if latest:
            print(f"\n📈 最新5条数据:")
            
            # 创建简单的表格显示
            header = " | ".join([f"{col:12}" for col in columns[:5]])
            print(header)
            print("-" * len(header) * 2)
            
            for row in latest:
                row_display = []
                for i, value in enumerate(row[:5]):  # 只显示前5列
                    if isinstance(value, float):
                        row_display.append(f"{value:12.4f}")
                    elif isinstance(value, int):
                        row_display.append(f"{value:12,d}")
                    else:
                        display = str(value)[:12]
                        row_display.append(f"{display:12}")
                
                print(" | ".join(row_display))
        
    except Exception as e:
        print(f"分析失败: {e}")

# 量化数据获取类
class StockDataFetcher:
    """股票数据获取器"""
    
    def __init__(self, host="172.16.66.188", port=6041, 
                 user="root", password="taosdata"):
        self.conn = None
        self.url = f"http://{host}:{port}"
        self.user = user
        self.password = password
    
    def connect(self):
        """连接数据库"""
        if not self.conn:
            self.conn = connect(
                url=self.url,
                user=self.user,
                password=self.password
            )
    
    def get_table_list(self, database="stock"):
        """获取表列表"""
        self.connect()
        tables = list(self.conn.query(f"SHOW {database}.TABLES"))
        return [table[0] for table in tables] if tables else []
    
    def get_stock_data(self, table_name, start_time=None, end_time=None, limit=1000):
        """获取股票数据"""
        self.connect()
        
        # 获取表结构
        desc = list(self.conn.query(f"DESCRIBE stock.{table_name}"))
        columns = [col[0] for col in desc]
        
        # 构建查询
        query = f"SELECT * FROM stock.{table_name}"
        
        # 查找时间字段
        time_field = None
        for col in desc:
            if 'timestamp' in col[1].lower():
                time_field = col[0]
                break
        
        conditions = []
        if time_field and start_time:
            conditions.append(f"{time_field} >= '{start_time}'")
        if time_field and end_time:
            conditions.append(f"{time_field} <= '{end_time}'")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += f" ORDER BY {time_field if time_field else columns[0]}"
        query += f" LIMIT {limit}"
        
        # 执行查询
        data = list(self.conn.query(query))
        
        if data:
            # 转换为DataFrame
            df = pd.DataFrame(data, columns=columns)
            
            # 设置时间索引
            if time_field and time_field in df.columns:
                df[time_field] = pd.to_datetime(df[time_field])
                df.set_index(time_field, inplace=True)
            
            return df
        else:
            return pd.DataFrame()
    
    def get_multiple_stocks(self, table_names, **kwargs):
        """获取多个股票数据"""
        data_dict = {}
        for table in table_names:
            df = self.get_stock_data(table, **kwargs)
            if not df.empty:
                data_dict[table] = df
        return data_dict
    
    def close(self):
        """关闭连接"""
        if self.conn:
            self.conn.close()

# 主程序
if __name__ == "__main__":
    print("="*60)
    print("TDengine 股票数据分析工具")
    print("="*60)
    
    # 1. 分析数据结构
    analyze_tdengine_structure()
    
    # 2. 获取数据样例
    get_stock_data_sample()
    
    # 3. 查找K线表
    find_kline_tables()
    
    # 4. 测试数据获取器
    print("\n" + "="*60)
    print("测试数据获取器")
    print("="*60)
    
    fetcher = StockDataFetcher()
    
    try:
        # 获取表列表
        tables = fetcher.get_table_list()
        if tables:
            print(f"发现 {len(tables)} 个表")
            
            # 获取前3个表的数据
            for i, table in enumerate(tables[:3], 1):
                print(f"\n📈 获取表 {i}: {table}")
                
                df = fetcher.get_stock_data(table, limit=5)
                if not df.empty:
                    print(f"数据形状: {df.shape}")
                    print(f"列名: {list(df.columns)}")
                    print(f"\n前3行数据:")
                    print(df.head(3))
                else:
                    print("无数据")
                    
    finally:
        fetcher.close()
        print("\n✅ 完成")