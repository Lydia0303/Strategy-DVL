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
