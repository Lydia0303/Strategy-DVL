"""
训练 & 预测
"""
from xgboost import XGBRegressor
# import numpy as np
import optuna
from sklearn.metrics import root_mean_squared_error

def train_model(X, y, tune: bool = False):
    if tune:
        def objective(trial):
            params = dict(
                max_depth=trial.suggest_int("max_depth", 3, 8),
                n_estimators=300,
                learning_rate=0.05,
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample", 0.6, 1.0),
                reg_alpha=trial.suggest_float("alpha", 0, 1),
                n_jobs=-1,
                random_state=42,
            )
            model = XGBRegressor(**params).fit(X, y)
            return root_mean_squared_error(y, model.predict(X), squared=False)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=30)
        best = study.best_params
    else:
        best = {}
    final_params = dict(
        n_estimators=300,
        learning_rate=0.10,
        max_depth=best.get("max_depth", 8),
        subsample=best.get("subsample", 0.8),
        colsample_bytree=best.get("colsample", 0.8),
        min_child_weight=1,       # 叶子最少 1 样本
        reg_alpha=best.get("alpha", 0),
        reg_lambda=0,             # 无 L2
        n_jobs=-1,
        random_state=42,
    )
    return XGBRegressor(**final_params).fit(X, y)

"""
def train_model(train_df, features, target="label"):
    print(f"[调试] 训练集形状 before clean: {train_df.shape}")
    print("[调试] NaN 统计:\n", train_df[features + [target]].isnull().sum())
    print("[调试] inf 统计:\n", np.isinf(train_df[features]).sum())
    # 1. 去掉 NaN,特征列或目标列中存在NaN的行（因为模型无法处理缺失值）
    train_df = train_df.dropna(subset=[target] + features)
    # 2. 去掉 inf / -inf
    train_df = train_df[~np.isinf(train_df[features]).any(axis=1)]
    # 3. 去掉负溢价（可选，先跑通）
    train_df = train_df[train_df["premium"] >= 0]
    if train_df.empty:  # 如果清洗后的数据为空,抛出错误，避免后续代码崩溃
        raise ValueError("训练集为空，请检查 NaN 或筛选条件！")
    print(f"[调试] 训练集 NaN 统计:")
    print(train_df[features + [target]].isnull().sum())  # 再次打印清洗后的缺失值统计，确认清洗生效
    # 训练模型
    X, y = train_df[features], train_df[target]   # 提取特征矩阵X（特征列）和目标向量y（预测目标）
    # 初始化XGBoost回归模型，设置超参数
    model = XGBRegressor(n_estimators=300,  # 树的数量（300棵树），越多可能越准但计算越慢
                         max_depth=4,   # 每棵树的最大深度（控制过拟合，4层较浅，避免过度复杂）
                         learning_rate=0.05,   # 学习率（步长），较小的值需要更多树配合
                         subsample=0.8,   # 每棵树的样本采样比例（80%），增加随机性，减少过拟合
                         random_state=42,   # 随机种子（固定值确保结果可复现）
                         n_jobs=-1)   # 并行计算的线程数（-1表示使用所有可用线程）
    model.fit(X, y)   # 用特征X和目标y训练模型
    return model   # 返回训练好的模型

def predict_model(model, test_df, features):
    return model.predict(test_df[features])

"""