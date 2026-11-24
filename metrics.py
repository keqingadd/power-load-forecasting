# -*- coding: utf-8 -*-
"""评估指标：MAE / RMSE / MAPE / sMAPE"""
import numpy as np
import pandas as pd


def _clean(y_true, y_pred):
    y_true = np.asarray(y_true, dtype="float64").ravel()
    y_pred = np.asarray(y_pred, dtype="float64").ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def evaluate(y_true, y_pred, name: str) -> dict:
    yt, yp = _clean(y_true, y_pred)
    if len(yt) == 0:
        return dict(model=name, n=0, MAE=np.nan, RMSE=np.nan, MAPE=np.nan, sMAPE=np.nan)
    err = yt - yp
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = np.abs(yt)
    mape = float(np.mean(np.abs(err) / np.where(denom < 1e-6, np.nan, denom)) * 100)
    smape = float(np.mean(2 * np.abs(err) / np.where(
        np.abs(yt) + np.abs(yp) < 1e-6, np.nan, np.abs(yt) + np.abs(yp))) * 100)
    return dict(model=name, n=len(yt), MAE=mae, RMSE=rmse, MAPE=mape, sMAPE=smape)


def to_table(rows) -> pd.DataFrame:
    return pd.DataFrame(rows)[["model", "n", "MAE", "RMSE", "MAPE", "sMAPE"]]


def show(df: pd.DataFrame) -> str:
    out = df.copy()
    for c in ["MAE", "RMSE"]:
        out[c] = out[c].map(lambda v: f"{v:,.1f}")
    for c in ["MAPE", "sMAPE"]:
        out[c] = out[c].map(lambda v: f"{v:.2f}%")
    return out.to_string(index=False)
