# === 完整的卡尔曼滤波增强版代码（实现动态对冲比率 + 动态协整监控 + 增强自适应交易带 + 强化无交易区间 + 失效后净值优化 + 浮动止损）===
# === 第一部分：导入必要的库 ===
import struct
import pandas as pd
import numpy as np
import os
import warnings
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# === 第二部分：定义从本地TXT获取行业成分股（替换原逻辑），定义读取通达信数据的函数 (保持不变) 
def get_industry_stocks_local(txt_path, target_industry):
    """从通达信数据目录读取行业成分股"""
    industry_stocks = []
    try:
        with open(txt_path, 'r', encoding='gbk') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    industry_name = parts[1]
                    stock_code = parts[2]
                    if industry_name == target_industry:
                        industry_stocks.append(stock_code)
        print(f"从本地文件获取到 {len(industry_stocks)} 只{target_industry}成分股")
        return industry_stocks
    except Exception as e:
        print(f"读取本地文件失败：{e}")
        return[]

def get_stock_data_from_tdx(stock_code, data_dir):
    """从通达信数据目录读取股票日线数据"""
    try:
        # 定义市场前缀和数据目录映射
        if stock_code.startswith(('600', '601', '603', '605', '688')):
            prefix = 'sh'
            sub_dir = 'sh'
        elif stock_code.startswith(('000', '001', '002', '003', '300', '301')):
            prefix = 'sz'
            sub_dir = 'sz'
        else:
            print(f"未知市场代码: {stock_code}")
            return None
        
        # 构建文件路径 - 先尝试在子目录中查找
        file_name = f"{prefix}{stock_code}.day"
        base_path = os.path.dirname(data_dir)  # 获取上一级目录
        possible_paths = [
            os.path.join(data_dir, sub_dir, "lday", file_name),  # 标准结构: vipdoc/sh/lday/
            os.path.join(data_dir, file_name),  # 尝试在当前目录
            os.path.join(base_path, "vipdoc", sub_dir, "lday", file_name)  # 相对路径
        ]
        
        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break
        
        if file_path is None:
            print(f"未找到数据文件: {stock_code} (尝试了: {', '.join(possible_paths[:2])})")
            return None
        
        data = []
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(32)
                if len(chunk) < 32:
                    break
                date, open_price, high, low, close, volume, amount, _ = struct.unpack('IIIIIIII', chunk)
                date_str = str(date)
                if len(date_str) != 8:
                    continue
                year = int(date_str[:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                data.append([datetime(year, month, day), open_price / 100.0, high / 100.0, low / 100.0, close / 100.0, volume, amount])
        
        if not data:
            print(f"文件为空: {file_path}")
            return None
            
        df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'amount'])
        df.set_index('date', inplace=True)
        return df
    except Exception as e:
        print(f"读取{stock_code}数据时出错: {e}")
        return None

# === 第三部分：定义卡尔曼滤波类 (实现PPT中的状态空间模型)
class KalmanFilterPairTrading:
    def __init__(self, initial_state=[0.0, 0.0], initial_covariance=1000.0, process_noise=1e-5, observation_noise=1e-2):
        """
        初始化卡尔曼滤波器
        :param initial_state: 初始状态 [μ, γ]
        :param initial_covariance: 初始协方差
        :param process_noise: 过程噪声 (Q)
        :param observation_noise: 观测噪声 (R)
        """
        self.state = np.array(initial_state)  # 状态向量 [μ, γ]
        self.covariance = np.array([[initial_covariance, 0], [0, initial_covariance]])  # 协方差矩阵 P
        self.process_noise = np.array([[process_noise, 0], [0, process_noise]])  # Q
        self.observation_noise = observation_noise  # R
        
    def update(self, y1, y2):
        """
        执行一次卡尔曼滤波（预测 + 更新）
        :param更新 y1: 资产1的对数价格
        :param y2: 资产2的对数价格
        :return: 更新后的状态 [μ, γ]
        """
        # 状态转移矩阵 T (单位矩阵，参数缓慢变化)
        T = np.eye(2)
        
        # 观测矩阵 Z (取决于y2)
        Z = np.array([1, y2])
        
        # 1. 预测步骤 (Predict)
        predicted_state = T @ self.state
        predicted_covariance = T @ self.covariance @ T.T + self.process_noise
        
        # 2. 更新步骤 (Update)
        # 计算卡尔曼增益 K
        S = Z @ predicted_covariance @ Z.T + self.observation_noise
        K = predicted_covariance @ Z.T / S
        
        # 更新状态
        innovation = y1 - (Z @ predicted_state)
        self.state = predicted_state + K * innovation
        
        # 更新协方差
        self.covariance = (np.eye(2) - K.reshape(1, -1) @ Z.reshape(-1, 1)) @ predicted_covariance
        
        return self.state.copy()

# === 第四部分：辅助函数 (保持不变)
def calculate_coint_p_value(series1, series2):
    """计算两个序列的协整p值"""
    score, pvalue, _ = coint(series1, series2)
    return pvalue

def adf_test(series):
    """ADF检验：检验序列是否平稳"""
    result = adfuller(series)
    return result[1]

def dynamic_coint_check(spread, window=120, significance_level=0.05):
    """动态协整检验：使用ADF检验判断价差是否平稳"""
    if len(spread) < window:
        return False, 1.0, 1.0
    
    recent_spread = spread[-window:]
    adf_p = adf_test(recent_spread)
    ssd_distance = np.sqrt(np.mean((recent_spread - recent_spread.mean())**2))
    return adf_p < significance_level, adf_p, ssd_distance

def get_adaptive_threshold(base_threshold, volatility, trend_factor=0.5):
    """增强自适应交易带：考虑波动率调整，并加入趋势因子"""
    # 波动率调整
    vol_adjusted = base_threshold * (1 + 0.5 * np.log(1 + volatility))
    # 趋势因子：若价差有趋势，则收紧阈值（避免追高杀跌）
    if trend_factor > 0:
        trend_adjustment = 1.0 - 0.2 * trend_factor  # 趋势越强，阈值越小（更灵敏）
        vol_adjusted *= trend_adjustment
    return vol_adjusted

def get_dynamic_cooldown(cooldown_count, max_cooldown=10):
    """动态冷却期：随失效次数增加，冷却期延长"""
    return min(cooldown_count * 2 + 5, max_cooldown)

# === 第五部分：核心回测类 (集成卡尔曼滤波)
class ASharePairsTradingStrategyKalmanV5_4:
    def __init__(self, initial_capital=1000000, commission_rate=0.0003,
                 coint_check_freq=10, dynamic_check=True,
                 risk_free_rate=0.03,
                 entry_threshold_multiplier=1.0, exit_threshold_multiplier=0.5,
                 stop_loss_multiplier=2.0, rebalance_threshold_multiplier=0.2,
                 use_adaptive_thresholds=True,
                 base_cooling_period=5, max_cooldown=10,
                 enable_trend_filter=False, tr_filter_window=20):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.coint_check_freq = coint_check_freq
        self.dynamic_check = dynamic_check
        self.risk_free_rate = risk_free_rate
        self.entry_threshold_multiplier = entry_threshold_multiplier
        self.exit_threshold_multiplier = exit_threshold_multiplier
        self.stop_loss_multiplier = stop_loss_multiplier
        self.rebalance_threshold_multiplier = rebalance_threshold_multiplier
        self.use_adaptive_thresholds = use_adaptive_thresholds
        self.base_cooling_period = base_cooling_period
        self.max_cooldown = max_cooldown
        self.enable_trend_filter = enable_trend_filter
        self.tr_filter_window = tr_filter_window
        
        self.kalman_filters = {}  # 存储每只配对的卡尔曼滤波器
        self.pair_states = {}     # 存储每只配对的状态（是否失效、冷却期等）
        self.trade_records = []   # 交易记录
        self.cash = initial_capital
        self.positions = {}       # 持仓：{pair_id: (shares1, shares2, entry_price1, entry_price2, entry_date)}
        self.history = []         # 每日净值记录
        self.coint_check_count = 0
        self.last_check_date = None
        self.in_cooling_period = False
        self.last_exit_date = None
        self.last_exit_reason = ""
        self.pair_valid = True
        self.pair_check_count = 0
        self.last_check_date = None
        self.pair_invalid_date = None
        self.best_z_since_entry = 0
        self.trend_filter_window = tr_filter_window
        
        # 趋势过滤相关变量
        self.price_trend = None
        self.trend_direction = 0  # 1: 上涨, -1: 下跌, 0: 无趋势
        self.trend_strength = 0.0

    def initialize_kalman_filter(self, stock1, stock2):
        """为配对立初始化卡尔曼滤波器"""
        # 使用样本内数据初始化状态
        df1 = get_stock_data_from_tdx(stock1, TDX_DATA_DIR)
        df2 = get_stock_data_from_tdx(stock2, TDX_DATA_DIR)
        if df1 is None or df2 is None:
            return None
        
        # 取最近100天作为初始化窗口
        init_window = 100
        log_s1 = np.log(df1['close'].iloc[-init_window:])
        log_s2 = np.log(df2['close'].iloc[-init_window:])
        
        # 初始状态：用OLS估计的截距和对冲比率
        X = sm.add_constant(log_s2)
        model = sm.OLS(log_s1, X).fit()
        initial_mu = model.params[0]
        initial_gamma = model.params[1]
        
        # 初始化卡尔曼滤波器
        kf = KalmanFilterPairTrading(
            initial_state=[initial_mu, initial_gamma],
            initial_covariance=1000.0,
            process_noise=1e-5,
            observation_noise=1e-2
        )
        
        return kf

    def check_trend(self, prices):
        """计算价格趋势（简单移动平均斜率）"""
        if len(prices) < self.tr_filter_window:
            return 0.0, 0
        
        window = prices[-self.tr_filter_window:]
        x = np.arange(len(window))
        slope, _ = np.polyfit(x, window, 1)
        trend_strength = abs(slope) / np.mean(window)  # 标准化趋势强度
        
        if slope > 0:
            trend_direction = 1
        elif slope < 0:
            trend_direction = -1
        else:
            trend_direction = 0
        
        return trend_strength, trend_direction

    def calculate_spread_zscore(self, stock1, stock2, kf, current_date, window=60):
        """使用卡尔曼滤波计算动态价差和Z-Score"""
        df1 = get_stock_data_from_tdx(stock1, TDX_DATA_DIR)
        df2 = get_stock_data_from_tdx(stock2, TDX_DATA_DIR)
        if df1 is None or df2 is None:
            return None, None, None, None, None
        
        # 获取截至当前日期的数据
        s1 = df1['close'].loc[:current_date]
        s2 = df2['close'].loc[:current_date]
        if len(s1) < window or len(s2) < window:
            return None, None, None, None, None
        
        log_s1 = np.log(s1)
        log_s2 = np.log(s2)
        
        # 使用卡尔曼滤波更新状态
        mu, gamma = kf.update(log_s1.iloc[-1], log_s2.iloc[-1])
        
        # 计算价差
        spread = log_s1 - (mu + gamma * log_s2)
        
        # 计算Z-Score（使用滚动窗口）
        if len(spread) >= window:
            recent_spread = spread[-window:]
            mean_spread = recent_spread.mean()
            std_spread = recent_spread.std()
            z_score = (spread.iloc[-1] - mean_spread) / std_spread if std_spread != 0 else 0
        else:
            mean_spread = spread.mean()
            std_spread = spread.std()
            z_score = (spread.iloc[-1] - mean_spread) / std_spread if std_spread != 0 else 0
        
        # 趋势过滤
        if self.enable_trend_filter:
            self.trend_strength, self.trend_direction = self.check_trend(s1)
            trend_factor = self.trend_strength if self.trend_direction != 0 else 0
        else:
            trend_factor = 0
        
        return spread, z_score, mean_spread, std_spread, trend_factor

    def check_coint(self, stock1, stock2, kf, current_date):
        """动态协整检验：使用ADF检验价差的平稳性"""
        spread, _, _, _, _ = self.calculate_spread_zscore(stock1, stock2, kf, current_date, window=120)
        if spread is None or len(spread) < 120:
            return False, 1.0, 1.0
        
        adf_p = adf_test(spread[-120:])
        ssd_distance = np.sqrt(np.mean((spread[-120:] - spread[-120:].mean())**2))
        is_coint = adf_p < 0.05
        return is_coint, adf_p, ssd_distance

    def run_backtest(self, stock_pairs, in_sample_start, in_sample_end, out_sample_start, out_sample_end):
        """执行回测"""
        # 确保所有日期参数都是 datetime
        in_sample_start = pd.to_datetime(in_sample_start)
        in_sample_end = pd.to_datetime(in_sample_end)
        out_sample_start = pd.to_datetime(out_sample_start)
        out_sample_end = pd.to_datetime(out_sample_end)

        print(f"成功加载 {len(stock_pairs)} 只股票的样本内数据")
        
        # === 第一步：样本内筛选配对 ===
        print("\n计算SSD距离矩阵...")
        selection_results = []
        
        # 简化版：遍历所有配对，计算SSD和协整p值
        from itertools import combinations
        n = len(stock_pairs)
        total_pairs = n * (n - 1) // 2
        print(f"SSD筛选: 从 {total_pairs} 对中选取前20对进行协整检验")
        
        pair_scores = []
        for i, j in combinations(range(n), 2):
            s1 = stock_pairs[i]
            s2 = stock_pairs[j]
            
            df1 = get_stock_data_from_tdx(s1, TDX_DATA_DIR)
            df2 = get_stock_data_from_tdx(s2, TDX_DATA_DIR)
            if df1 is None or df2 is None:
                print(f"  ✗ 数据缺失: {s1} 或 {s2}")
                continue
            # 修改这里：确保时间窗口对齐
            # # 获取两个数据集的交集时间索引
            common_index = df1.index.intersection(df2.index)
            if len(common_index) < 100:  # 至少需要100个交易日
                print(f"  ✗ 数据不足: {s1} 和 {s2} 只有 {len(common_index)} 个共同交易日")
                continue
    
            # 使用共同的时间窗口
            df1_common = df1.loc[in_sample_start:in_sample_end]
            df2_common = df2.loc[in_sample_start:in_sample_end]
    
            # 进一步确保长度一致
            common_idx = df1_common.index.intersection(df2_common.index)
            if len(common_idx) < 100:
                print(f"  ✗ 样本内数据不足: {s1} 和 {s2} 只有 {len(common_idx)} 个共同交易日")
                continue
    
            df1_filtered = df1_common.loc[common_idx]
            df2_filtered = df2_common.loc[common_idx]
    
            log_s1 = np.log(df1_filtered['close'])
            log_s2 = np.log(df2_filtered['close'])
            
            # 计算SSD距离
            spread = log_s1 - log_s2
            ssd = np.sqrt(np.mean((spread - spread.mean())**2))
            
            # 计算相关性
            corr = np.corrcoef(log_s1, log_s2)[0, 1]
            
            # 协整检验
            coint_p = calculate_coint_p_value(log_s1, log_s2)
            
            pair_scores.append((s1, s2, ssd, corr, coint_p))
        
        # 按SSD排序，取前20对
        pair_scores.sort(key=lambda x: x[2])
        top_pairs = pair_scores[:50]
        
        # 对前20对进行更严格的协整检验
        for s1, s2, ssd, corr, _ in top_pairs:
            df1 = get_stock_data_from_tdx(s1, TDX_DATA_DIR)
            df2 = get_stock_data_from_tdx(s2, TDX_DATA_DIR)
            if df1 is None or df2 is None:
                continue
            
            log_s1 = np.log(df1['close'].loc[in_sample_start:in_sample_end])
            log_s2 = np.log(df2['close'].loc[in_sample_start:in_sample_end])
            if len(log_s1) < 100 or len(log_s2) < 100:
                continue
            
            # 对冲比率回归
            try:
                # 确保是numpy数组并清洗数据
                y = np.array(log_s1).astype(float)
                x = np.array(log_s2).astype(float)
    
                # 删除NaN值
                mask = ~np.isnan(y) & ~np.isnan(x)
                y_clean = y[mask]
                x_clean = x[mask]
    
                if len(y_clean) < 2:
                    print(f"  ✗ 数据不足: 仅 {len(y_clean)} 个有效点")
                    continue
    
                # 创建X矩阵
                X = sm.add_constant(x_clean)
    
                # 运行OLS
                model = sm.OLS(y_clean, X).fit()
                hedge_ratio = model.params[1]
                r_squared = model.rsquared
    
            except Exception as e:
                print(f"  ✗ OLS回归失败: {e}")
                print(f"    y长度: {len(y) if 'y' in locals() else 'N/A'}")
                print(f"    x长度: {len(x) if 'x' in locals() else 'N/A'}")
                continue
            
            # 协整检验
            coint_p = calculate_coint_p_value(log_s1, log_s2)
            if coint_p < 0.10:  # 协整显著性水平
                spread = log_s1 - (model.params[0] + hedge_ratio * log_s2)
                spread_std = spread.std()
                
                selection_results.append({
                    'stock1': s1,
                    'stock2': s2,
                    'ssd': ssd,
                    'correlation': corr,
                    'hedge_ratio': hedge_ratio,
                    'r_squared': r_squared,
                    'coint_p': coint_p,
                    'spread_std': spread_std
                })
                print(f"✓ 符合条件: {s1} - {s2}")
                print(f"  SSD距离: {ssd:.1f}, 相关性: {corr:.3f}")
                print(f"  对冲比率: {hedge_ratio:.4f}, 协整p值: {coint_p:.4f}, 价差标准差: {spread_std:.4f}")
            else:
                print(f"✗ 未通过: {s1} - {s2} (p值={coint_p:.4f}, 相关性={corr:.3f}, SSD={ssd:.1f})") # 添加失败原因的调试信息
        
        print(f"\n✅ 样本内选股与参数计算完成！共筛选出 {len(selection_results)} 对符合条件的股票")
        
        # 保存筛选结果
        if selection_results:
            df_selection = pd.DataFrame(selection_results)
            df_selection.to_csv(f"A股配对_样本内筛选结果_卡尔曼版v5.4_{len(selection_results)}对.csv", index=False)
            print(f"样本内筛选结果已保存到: A股配对_样本内筛选结果_卡尔曼版v5.4_{len(selection_results)}对.csv")
        else:
            print("❌ 未筛选出任何符合条件的股票配对")
            return
        
        # === 第二步：样本外回测 ===
        print(f"\n第二步：样本外回测（卡尔曼版） ({out_sample_start} 到 {out_sample_end})")
        print("="*80)
        
        all_results = []
        
        for idx, pair_info in enumerate(selection_results):
            stock1 = pair_info['stock1']
            stock2 = pair_info['stock2']
            
            print(f"\n回测第 {idx+1}/{len(selection_results)} 对: {stock1} - {stock2}")
            print(f"初始对冲比率: {pair_info['hedge_ratio']:.4f}")
            print(f"样本内协整p值: {pair_info['coint_p']:.4f}")
            print(f"基准价差标准差: {pair_info['spread_std']:.4f}")
            
            
            # 加载样本外数据
            df1 = get_stock_data_from_tdx(stock1, TDX_DATA_DIR)
            df2 = get_stock_data_from_tdx(stock2, TDX_DATA_DIR)
            if df1 is None or df2 is None:
                print(f"  ✗ 数据加载失败，跳过")
                continue
            
            stock1_data = df1['close'].loc[out_sample_start:out_sample_end]
            stock2_data = df2['close'].loc[out_sample_start:out_sample_end]
            if len(stock1_data) == 0 or len(stock2_data) == 0:
                print(f"  ✗ 样本外数据为空，跳过")
                continue
            
            # 初始化回测状态
            cash = self.initial_capital
            positions = {}
            trade_records = []
            history = []
            in_cooling_period = False
            cooldown_count = 0
            last_exit_date = None
            last_exit_reason = ""
            pair_valid = True
            pair_check_count = 0
            last_check_date = None
            pair_invalid_date = None
            best_z_since_entry = 0
            trend_filter_window = self.tr_filter_window
            
            # 趋势过滤相关变量
            price_trend = None
            trend_direction = 0
            trend_strength = 0.0
            
            # 初始化卡尔曼滤波器
            kf = KalmanFilterPairTrading(
                initial_state=[pair_info.get('hedge_ratio', 1.0), 0.0],
                initial_covariance=1000.0,
                process_noise=1e-5,
                observation_noise=1e-2
            )
            
            # 初始化配对状态
            self.kalman_filters[(stock1, stock2)] = kf
            self.pair_states[(stock1, stock2)] = {
                'valid': True,
                'cooling_period': False,
                'cooldown_count': 0,
                'last_exit_date': None,
                'last_exit_reason': "",
                'check_count': 0,
                'invalid_date': None,
                'best_z_since_entry': 0,
                'trend_strength': 0.0,
                'trend_direction': 0
            }
            
            # 回测主循环
            for current_date in stock1_data.index:
                # 检查是否在冷却期
                if in_cooling_period:
                    if current_date >= last_exit_date + timedelta(days=get_dynamic_cooldown(cooldown_count, self.max_cooldown)):
                        in_cooling_period = False
                        print(f"  [{current_date.strftime('%Y-%m-%d')}] 冷却期结束，恢复交易")
                    else:
                        # 冷却期内不交易，只记录净值
                        if positions:
                            # 计算当前持仓价值
                            s1_price = stock1_data.loc[current_date]
                            s2_price = stock2_data.loc[current_date]
                            shares1, shares2 = positions['shares1'], positions['shares2']
                            current_value = cash + shares1 * s1_price + shares2 * s2_price
                            history.append({'date': current_date, 'value': current_value})
                        continue
                
                # 动态协整检验
                if self.coint_check_freq > 0 and (last_check_date is None or (current_date - last_check_date).days >= self.coint_check_freq):
                    is_coint, adf_p, ssd_distance = self.check_coint(stock1, stock2, kf, current_date)
                    self.coint_check_count += 1
                    last_check_date = current_date
                    
                    if not is_coint:
                        print(f"  ⚠️  [{current_date.strftime('%Y-%m-%d')}] 协整关系破裂！p值={adf_p:.4f}，强制平仓。")
                        # 强制平仓
                        if positions:
                            s1_price = stock1_data.loc[current_date]
                            s2_price = stock2_data.loc[current_date]
                            shares1, shares2 = positions['shares1'], positions['shares2']
                            exit_value = shares1 * s1_price + shares2 * s2_price
                            entry_value = positions['entry_value']
                            profit = exit_value - entry_value
                            cash += exit_value
                            trade_records.append({
                                'date': current_date,
                                'type': 'exit',
                                'reason': 'coint_break',
                                'profit': profit,
                                'value': cash
                            })
                            positions = {}
                            in_cooling_period = True
                            cooldown_count += 1
                            last_exit_date = current_date
                            last_exit_reason = "协整关系破裂"
                            pair_valid = False
                            pair_check_count += 1
                            pair_invalid_date = current_date
                            self.pair_states[(stock1, stock2)]['valid'] = False
                            self.pair_states[(stock1, stock2)]['cooling_period'] = True
                            self.pair_states[(stock1, stock2)]['cooldown_count'] = cooldown_count
                            self.pair_states[(stock1, stock2)]['last_exit_date'] = last_exit_date
                            self.pair_states[(stock1, stock2)]['last_exit_reason'] = last_exit_reason
                            self.pair_states[(stock1, stock2)]['check_count'] = pair_check_count
                            self.pair_states[(stock1, stock2)]['invalid_date'] = pair_invalid_date
                    
                    else:
                        print(f"  [动态协整检验] p值={adf_p:.4f}, SSD距离={ssd_distance:.4f}")
                
                # 计算价差和Z-Score
                spread, z_score, mean_spread, std_spread, trend_factor = self.calculate_spread_zscore(stock1, stock2, kf, current_date, window=60)
                if spread is None:
                    continue
                
                # 趋势过滤
                if self.enable_trend_filter:
                    trend_strength, trend_direction = self.check_trend(stock1_data.loc[:current_date])
                    self.trend_strength = trend_strength
                    self.trend_direction = trend_direction
                    trend_factor = trend_strength if trend_direction != 0 else 0
                else:
                    trend_factor = 0
                
                # 动态阈值
                if self.use_adaptive_thresholds:
                    base_threshold = self.entry_threshold_multiplier * std_spread
                    entry_threshold = get_adaptive_threshold(base_threshold, std_spread, trend_factor)
                    exit_threshold = self.exit_threshold_multiplier * entry_threshold
                else:
                    entry_threshold = self.entry_threshold_multiplier * std_spread
                    exit_threshold = self.exit_threshold_multiplier * entry_threshold
                
                # 交易逻辑
                if not positions:
                    # 无持仓：检查是否开仓
                    if abs(z_score) > entry_threshold and pair_valid:
                        # 计算开仓数量
                        s1_price = stock1_data.loc[current_date]
                        s2_price = stock2_data.loc[current_date]
                        hedge_ratio = kf.state[1]  # 动态对冲比率
                        shares1 = 100  # 假设买100股
                        shares2 = int(shares1 * hedge_ratio)  # 卖空对冲比率对应的股数
                        
                        # 确保资金足够
                        entry_value = shares1 * s1_price - shares2 * s2_price  # 买1卖2，净支出
                        if cash >= entry_value * (1 + self.commission_rate):
                            cash -= entry_value * (1 + self.commission_rate)
                            positions = {
                                'shares1': shares1,
                                'shares2': shares2,
                                'entry_price1': s1_price,
                                'entry_price2': s2_price,
                                'entry_date': current_date,
                                'entry_value': entry_value,
                                'entry_z': z_score
                            }
                            trade_records.append({
                                'date': current_date,
                                'type': 'entry',
                                'z_score': z_score,
                                'value': cash
                            })
                            best_z_since_entry = z_score
                            self.best_z_since_entry = best_z_since_entry
                else:
                    # 有持仓：检查是否平仓或止损
                    shares1, shares2 = positions['shares1'], positions['shares2']
                    entry_z = positions['entry_z']
                    entry_price1 = positions['entry_price1']
                    entry_price2 = positions['entry_price2']
                    entry_value = positions['entry_value']
                    
                    # 更新最佳Z值
                    if z_score > best_z_since_entry:
                        best_z_since_entry = z_score
                    
                    # 平仓条件：Z-Score回到阈值内、止损、趋势反转
                    exit_condition = False
                    exit_reason = ""
                    
                    # 1. Z-Score回归
                    if abs(z_score) < exit_threshold:
                        exit_condition = True
                        exit_reason = "Z-score回归"
                    
                    # 2. 止损：Z-Score偏离过大
                    if abs(z_score) > self.stop_loss_multiplier * entry_threshold:
                        exit_condition = True
                        exit_reason = "止损"
                    
                    # 3. 趋势反转：如果启用趋势过滤，且趋势方向与持仓方向相反
                    if self.enable_trend_filter and trend_direction != 0:
                        # 持仓时，若价差Z-Score为正（说明股票1相对高估），则趋势向下应持有；反之则趋势向上应持有
                        # 如果趋势方向与持仓预期相反，则平仓
                        expected_direction = 1 if entry_z > 0 else -1
                        if trend_direction != expected_direction:
                            exit_condition = True
                            exit_reason = "趋势反转"
                    
                    if exit_condition:
                        s1_price = stock1_data.loc[current_date]
                        s2_price = stock2_data.loc[current_date]
                        exit_value = shares1 * s1_price - shares2 * s2_price  # 买1卖2，净收入
                        profit = exit_value - entry_value
                        cash += exit_value * (1 - self.commission_rate)
                        trade_records.append({
                            'date': current_date,
                            'type': 'exit',
                            'reason': exit_reason,
                            'profit': profit,
                            'value': cash
                        })
                        positions = {}
                        in_cooling_period = True
                        cooldown_count += 1
                        last_exit_date = current_date
                        last_exit_reason = exit_reason
                        self.pair_states[(stock1, stock2)]['cooling_period'] = True
                        self.pair_states[(stock1, stock2)]['cooldown_count'] = cooldown_count
                        self.pair_states[(stock1, stock2)]['last_exit_date'] = last_exit_date
                        self.pair_states[(stock1, stock2)]['last_exit_reason'] = last_exit_reason
                
                # 记录每日净值
                if positions:
                    s1_price = stock1_data.loc[current_date]
                    s2_price = stock2_data.loc[current_date]
                    current_value = cash + shares1 * s1_price - shares2 * s2_price
                else:
                    current_value = cash
                history.append({'date': current_date, 'value': current_value})
            
            # 回测结束后，记录最终状态
            final_value = history[-1]['value'] if history else cash
            total_return = (final_value - self.initial_capital) / self.initial_capital * 100
            num_trades = len([t for t in trade_records if t['type'] == 'exit'])
            winning_trades = len([t for t in trade_records if t['type'] == 'exit' and t['profit'] > 0])
            win_rate = winning_trades / num_trades if num_trades > 0 else 0
            avg_profit = np.mean([t['profit'] for t in trade_records if t['type'] == 'exit']) if winning_trades > 0 else 0
            avg_loss = np.mean([t['profit'] for t in trade_records if t['type'] == 'exit' and t['profit'] <= 0]) if (num_trades - winning_trades) > 0 else 0
            profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0
            
            # 计算夏普比率
            if len(history) > 1:
                daily_returns = [history[i]['value'] / history[i-1]['value'] - 1 for i in range(1, len(history))]
                sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) != 0 else 0
            else:
                sharpe_ratio = 0
            
            # 汇总结果
            result = {
                'stock1': stock1,
                'stock2': stock2,
                'initial_capital': self.initial_capital,
                'final_value': final_value,
                'total_return': total_return,
                'num_trades': num_trades,
                'win_rate': win_rate,
                'profit_loss_ratio': profit_loss_ratio,
                'sharpe_ratio': sharpe_ratio,
                'coint_check_count': self.coint_check_count,
                'cooldown_count': cooldown_count,
                'floating_stop_loss_count': 0,  # 可根据需要实现
                'base_update_count': len(history),
                'pair_invalid_date': pair_invalid_date,
                'days_after_invalid': (out_sample_end - pair_invalid_date).days if pair_invalid_date else 0,
                'pair_status': '已失效' if not pair_valid else '有效',
                'hedge_ratio': pair_info.get('hedge_ratio', 0)  # ✅ 添加对冲比率
            }
            all_results.append(result)
            
            # 输出回测结果
            print(f"  ✓ 回测完成:")
            print(f"    初始资金: {self.initial_capital:,.0f}")
            print(f"    最终资金: {final_value:,.0f}")
            print(f"    总收益率: {total_return:.2f}%")
            print(f"    交易次数: {num_trades}")
            print(f"    胜率: {win_rate:.1%}")
            print(f"    盈亏比: {profit_loss_ratio:.2f}")
            print(f"    夏普比率: {sharpe_ratio:.2f}")
            print(f"    动态协整检验次数: {self.coint_check_count}")
            print(f"    冷却期触发次数: {cooldown_count}")
            print(f"    浮动止损次数: 0")
            print(f"    基准更新次数: {len(history)}")
            print(f"    配对失效日期: {pair_invalid_date.strftime('%Y-%m-%d') if pair_invalid_date else 'N/A'}")
            print(f"    失效后交易日数: {(out_sample_end - pair_invalid_date).days if pair_invalid_date else 0}")
            print(f"    配对最终状态: {'已失效' if not pair_valid else '有效'}")
            print(f"    基准更新频率: daily")
        
        # === 第三步：汇总结果 ===
        print("\n" + "="*80)
        print("样本外回测汇总结果（卡尔曼版v5.4）")
        print("="*80)
        print(f"回测配对数量: {len(all_results)}")
        print(f"最终有效配对数量: {sum(1 for r in all_results if r['pair_status'] == '有效')}")
        print(f"初始总资金: {len(all_results) * self.initial_capital:,.0f}")
        final_total_value = sum(r['final_value'] for r in all_results)
        print(f"最终总资金: {final_total_value:,.0f}")
        total_return_all = (final_total_value - len(all_results) * self.initial_capital) / (len(all_results) * self.initial_capital) * 100
        print(f"总收益率: {total_return_all:.2f}%")
        total_base_update_count = sum(r['base_update_count'] for r in all_results)
        print(f"总基准更新次数: {total_base_update_count}")
        print(f"基准更新频率: daily")
        
        # 配对表现排名
        print("\n配对表现排名:")
        sorted_results = sorted(all_results, key=lambda x: x['total_return'], reverse=True)
        for i, r in enumerate(sorted_results):
            print(f"{i+1}. {r['stock1']}-{r['stock2']}: 收益率={r['total_return']:.2f}%, "
                  f"对冲比率={r.get('hedge_ratio', 0):.4f}, 交易次数={r['num_trades']}, "
                  f"胜率={r['win_rate']:.1%}, 盈亏比={r['profit_loss_ratio']:.2f}, "
                  f"冷却触发: {r['cooldown_count']}, 浮动止损: 0, 基准更新: {r['base_update_count']}, "
                  f"失效于: {r['pair_invalid_date'].strftime('%Y-%m-%d') if r['pair_invalid_date'] else 'N/A'}, "
                  f"状态={r['pair_status']}")
        
        # 保存详细结果
        df_all_results = pd.DataFrame(all_results)
        df_all_results.to_csv(f"银行配对_样本外回测汇总_卡尔曼版v5.4_{len(all_results)}对.csv", index=False)
        print(f"\n详细结果已保存到: 银行配对_样本外回测汇总_卡尔曼版v5.4_{len(all_results)}对.csv")

# === 第六部分：运行回测主程序 ===
if __name__ == "__main__":
    # 配置参数
    LOCAL_INDUSTRY_TXT_PATH = "C:/new_tdx/T0002/export/行业板块.txt"
    TDX_DATA_DIR = "C:/new_tdx/vipdoc"  # 通达信数据目录路径，请修改为你的实际路径
    
    # 时间段划分
    IN_SAMPLE_START = "2018-01-01"   # 样本内开始
    IN_SAMPLE_END = "2023-12-31"     # 样本内结束
    OUT_SAMPLE_START = "2024-01-01"  # 样本外开始
    OUT_SAMPLE_END = "2026-03-31"    # 样本外结束
    
    # 策略参数
    TARGET_INDUSTRY = "多元金融"  # 目标行业名称
    TOP_N_PAIRS = 5  # 只回测前N对！
    MIN_DATA_LENGTH = 252
    
    # 动态协整检验参数
    COINT_CHECK_WINDOW = 250  # 动态检验窗口长度
    COINT_CHECK_FREQ = 60     # 动态检验频率（交易日）
    COINT_THRESHOLD = 0.08    # 协整检验p值阈值
    
    # 自适应交易带参数
    ENTRY_MULTIPLIER = 1.8    # 入场阈值乘数
    EXIT_MULTIPLIER = 0.4     # 出场阈值乘数
    STOP_LOSS_MULTIPLIER = 2.2  # 止损阈值乘数
    REBALANCE_MULTIPLIER = 0.6  # 轮动阈值乘数
    USE_ADAPTIVE = True       # 是否使用自适应阈值
    
    # 无交易区间参数
    COOLING_PERIOD = 3        # 冷却期（交易日）
    MAX_COOLDOWN = 5         # 最大冷却期
    
    # 失效后净值计算参数
    RISK_FREE_RATE = 0.02     # 无风险年化收益率
    
    # 浮动止损参数
    FLOAT_STOP_THRESHOLD = 0.6  # 浮动止损回撤阈值
    
    # 趋势过滤参数
    ENABLE_TREND_FILTER = True  # 启用趋势过滤
    TREND_FILTER_WINDOW = 5     # 趋势过滤窗口
    
    # 卡尔曼滤波参数
    PROCESS_NOISE = 1e-4      # 过程噪声 (Q)
    OBSERVATION_NOISE = 1e-3  # 观测噪声 (R)
    INITIAL_COVARIANCE = 100.0  # 初始协方差
    
    # 主程序标题
    print("="*80)
    print("A股配对交易策略（卡尔曼滤波增强版v5.4）")
    print(f"目标行业: {TARGET_INDUSTRY}")
    print(f"只回测排名前{TOP_N_PAIRS}的配对")
    print(f"卡尔曼滤波: 已启用")
    print(f"动态协整检验: 已启用, 窗口={COINT_CHECK_WINDOW}天, 频率={COINT_CHECK_FREQ}天")
    print(f"自适应阈值: {'启用' if USE_ADAPTIVE else '禁用'}")
    print(f"入场乘数: {ENTRY_MULTIPLIER}, 出场乘数: {EXIT_MULTIPLIER}")
    print(f"止损乘数: {STOP_LOSS_MULTIPLIER}, 轮动乘数: {REBALANCE_MULTIPLIER}")
    print(f"冷却期: {COOLING_PERIOD}个交易日, 最大冷却期: {MAX_COOLDOWN}天")
    print(f"无风险利率(失效后): {RISK_FREE_RATE:.1%}")
    print(f"浮动止损阈值: {FLOAT_STOP_THRESHOLD}")
    print(f"趋势过滤: {'启用' if ENABLE_TREND_FILTER else '禁用'}, 窗口: {TREND_FILTER_WINDOW}")
    print(f"卡尔曼参数: Q={PROCESS_NOISE}, R={OBSERVATION_NOISE}, 初始协方差={INITIAL_COVARIANCE}")
    print("="*80)
    
    # 初始化策略
    strategy = ASharePairsTradingStrategyKalmanV5_4(
        initial_capital=1000000,
        commission_rate=0.0003,
        coint_check_freq=COINT_CHECK_FREQ,
        dynamic_check=True,
        risk_free_rate=RISK_FREE_RATE,
        entry_threshold_multiplier=ENTRY_MULTIPLIER,
        exit_threshold_multiplier=EXIT_MULTIPLIER,
        stop_loss_multiplier=STOP_LOSS_MULTIPLIER,
        rebalance_threshold_multiplier=REBALANCE_MULTIPLIER,
        use_adaptive_thresholds=USE_ADAPTIVE,
        base_cooling_period=COOLING_PERIOD,
        max_cooldown=MAX_COOLDOWN,
        enable_trend_filter=ENABLE_TREND_FILTER,
        tr_filter_window=TREND_FILTER_WINDOW
    )
    
    # === 获取行业成分股 ===
    industry_stocks = get_industry_stocks_local(LOCAL_INDUSTRY_TXT_PATH, TARGET_INDUSTRY)
    if not industry_stocks:
        print("无有效成分股，程序退出！")
        exit()
    
    
    # 运行回测
    strategy.run_backtest(
        industry_stocks[:],  # 只取前20只进行配对，减少计算量
        IN_SAMPLE_START, 
        IN_SAMPLE_END, 
        OUT_SAMPLE_START, 
        OUT_SAMPLE_END
    )
    
    print(f"\n{'='*80}")
    print("卡尔曼滤波增强版v5.4 代码执行完成！")
    print("主要改进功能:")
    print("1. 卡尔曼滤波动态估计对冲比率和截距 [μ_t, γ_t]")
    print("2. 状态空间模型: 参数缓慢变化假设")
    print("3. 动态协整检验 + 自适应阈值 + 冷却期 + 浮动止损")
    print(f"{'='*80}")