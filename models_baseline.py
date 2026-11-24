# -*- coding: utf-8 -*-
"""基线模型：季节朴素 + LightGBM"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from config import SEED


def seasonal_naive(series: pd.Series, horizon: int) -> pd.Series:
    """季节朴素：用 horizon 步之前的值作为预测。"""
    return series.shift(horizon)


def fit_lightgbm(X_train, y_train, X_val, y_val):
    """LightGBM：验证集早停，防止过拟合。"""
    model = lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model
