# -*- coding: utf-8 -*-
"""数据读取、清洗、特征构造、时间切分"""
import numpy as np
import pandas as pd

from config import (LOAD_CSV, WEATHER_CSV, FREQ, TRAIN_END, TEST_START,
                    LAGS, LOOKBACK)


def load_load_series(freq: str = FREQ) -> pd.Series:
    """读取 15 分钟负荷，重建完整时间索引，线性插值补齐 388 个缺失时点。"""
    df = pd.read_csv(LOAD_CSV, encoding="utf-8-sig")
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"])
    s = df.set_index("ds")["y"].sort_index().astype("float64")
    idx = pd.date_range(s.index.min(), s.index.max(), freq=freq)
    return s.reindex(idx).interpolate(limit_direction="both").rename("load")


def load_weather() -> pd.DataFrame:
    """逐日气象：去重、清洗温度单位，返回 index=日期 的 tmax / tmin。"""
    df = pd.read_csv(WEATHER_CSV, encoding="utf-8-sig")
    df = df.drop_duplicates(subset=["日期"], keep="first").copy()
    raw = df["日期"].astype(str)
    clean = (raw.str.replace("年", "-", regex=False)
                .str.replace("月", "-", regex=False)
                .str.replace("日", "", regex=False))
    out = pd.DataFrame({"date": pd.to_datetime(clean, errors="coerce")})
    for src, dst in [("最高温度", "tmax"), ("最低温度", "tmin")]:
        out[dst] = pd.to_numeric(
            df[src].astype(str).str.replace("℃", "", regex=False).str.strip(),
            errors="coerce")
    out = out.dropna(subset=["date"]).set_index("date").sort_index()
    return out


def holiday_frame(index: pd.DatetimeIndex):
    """节假日 / 调休特征（chinese-calendar，未安装则返回 None）。"""
    try:
        import chinese_calendar as cc
    except Exception:
        return None
    dates = pd.Series(index.normalize().date).drop_duplicates()
    mapping_h = {d: int(cc.is_holiday(d)) for d in dates}
    mapping_w = {d: int(cc.is_workday(d)) for d in dates}
    day = index.normalize().date
    return pd.DataFrame({
        "is_holiday": [mapping_h[d] for d in day],
        "is_workday": [mapping_w[d] for d in day],
    }, index=index)


def build_features(series: pd.Series, include_weather: bool = True) -> pd.DataFrame:
    """构造单步预测的特征表（只用历史信息，杜绝未来泄漏）。"""
    df = pd.DataFrame({"y": series})
    df["lag0"] = series
    for lag in LAGS:
        df[f"lag{lag}"] = series.shift(lag)
    past = series.shift(1)
    df["roll96_mean"] = past.rolling(96).mean()
    df["roll96_std"] = past.rolling(96).std()
    df["roll96_max"] = past.rolling(96).max()
    df["roll672_mean"] = past.rolling(672).mean()

    idx = series.index
    df["tod"] = idx.hour * 4 + idx.minute // 15      # 每天 0~95
    df["dow"] = idx.dayofweek
    df["month"] = idx.month
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    hf = holiday_frame(idx)
    if hf is not None:
        df["is_holiday"] = hf["is_holiday"].values
        df["is_workday"] = hf["is_workday"].values

    if include_weather:
        w = load_weather()
        day = idx.normalize()
        df["tmax"] = w["tmax"].reindex(day).values
        df["tmin"] = w["tmin"].reindex(day).values
        df["tmax"] = df["tmax"].interpolate(limit_direction="both")
        df["tmin"] = df["tmin"].interpolate(limit_direction="both")
    return df


CAL_COLS = ["tod", "dow", "month", "is_weekend", "is_holiday", "is_workday",
            "tmax", "tmin"]


def features_for_horizon(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """把日历/气象特征对齐到"目标时刻 t+h"（预测目标时刻的星期、时段是确定已知的）。"""
    out = df.copy()
    cols = [c for c in CAL_COLS if c in out.columns]
    out[cols] = df[cols].shift(-horizon)
    return out


EXOG_COLS = ["tod_sin", "tod_cos", "dow_sin", "dow_cos", "is_weekend", "is_holiday",
             "tmax", "tmin"]
FUTURE_COV_COLS = ["tod_sin", "tod_cos", "dow_sin", "dow_cos", "is_weekend", "is_holiday"]


def build_exog_frame(series: pd.Series, include_weather: bool = True) -> pd.DataFrame:
    """逐时刻外生特征（时间 t 时刻已知）：时段、星期、周末、节假日、气温。"""
    idx = series.index
    f = pd.DataFrame(index=idx)
    tod = idx.hour * 4 + idx.minute // 15
    f["tod_sin"] = np.sin(2 * np.pi * tod / 96)
    f["tod_cos"] = np.cos(2 * np.pi * tod / 96)
    dow = idx.dayofweek
    f["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    f["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    f["is_weekend"] = (dow >= 5).astype("float32")
    hf = holiday_frame(idx)
    f["is_holiday"] = hf["is_holiday"].values if hf is not None else 0.0
    if include_weather:
        w = load_weather()
        day = idx.normalize()
        f["tmax"] = w["tmax"].reindex(day).values
        f["tmin"] = w["tmin"].reindex(day).values
        f["tmax"] = f["tmax"].interpolate(limit_direction="both")
        f["tmin"] = f["tmin"].interpolate(limit_direction="both")
    return f.astype("float32")


def make_multichannel_windows(series: pd.Series, exog: pd.DataFrame,
                              lookback: int = LOOKBACK, horizon: int = 1):
    """多通道滑窗（view，不复制）：
    X  : (n, lookback, 1+len(EXOG_COLS))  过去负荷 + 过去外生特征
    FX : (n, horizon, len(FUTURE_COV_COLS)) 目标时刻的外生特征（未来已知）
    Y  : (n, horizon)                      目标负荷
    """
    y = series.to_numpy(dtype="float32")
    past = np.concatenate([y.reshape(-1, 1), exog[EXOG_COLS].to_numpy("float32")], axis=1)
    fut = exog[FUTURE_COV_COLS].to_numpy("float32")
    n = len(y) - lookback - horizon + 1
    X = np.lib.stride_tricks.sliding_window_view(
        past, (lookback, past.shape[1]))[:, 0][:n]
    FX = np.lib.stride_tricks.sliding_window_view(
        fut, (horizon, fut.shape[1]))[:, 0][:n]
    Y = np.lib.stride_tricks.sliding_window_view(y[lookback:], horizon)[:n]
    return X, FX, Y


def split_masks(index: pd.DatetimeIndex):
    """返回 (训练, 验证, 测试) 三个布尔掩码。"""
    t_train_end = pd.Timestamp(TRAIN_END)
    t_test = pd.Timestamp(TEST_START)
    train = index < t_train_end
    val = (index >= t_train_end) & (index < t_test)
    test = index >= t_test
    return train, val, test


def make_windows(values: np.ndarray, lookback: int = LOOKBACK, horizon: int = 1):
    """滑窗构造 (X, y)：X 为回看序列，y 为未来 horizon 步。"""
    n = len(values) - lookback - horizon + 1
    X = np.lib.stride_tricks.sliding_window_view(values, lookback)[:n]
    Y = np.lib.stride_tricks.sliding_window_view(values[lookback:], horizon)[:n]
    return np.ascontiguousarray(X, dtype="float32"), np.ascontiguousarray(Y, dtype="float32")
