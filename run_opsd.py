# -*- coding: utf-8 -*-
"""E6：跨数据泛化实验 —— 在 OPSD 欧洲小时级负荷上复跑同一套方法

数据：D:\电力负荷预测毕设数据\raw\02_opsd_europe_hourly\time_series_60min_singleindex.csv
用法（PyCharm 里改 Parameters 或命令行）：
    python run_opsd.py                  # 默认德国 DE
    python run_opsd.py --country FR     # 法国
    python run_opsd.py --country GB     # 英国
输出：results/opsd_metrics.csv、results/figs/opsd_prediction.png
"""
import argparse
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA_ROOT, FIGS_DIR, RESULTS_DIR
from metrics import evaluate, show, to_table
from models_baseline import fit_lightgbm

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OPSD_CSV = DATA_ROOT / "raw" / "02_opsd_europe_hourly" / "time_series_60min_singleindex.csv"


def load_country(country: str) -> pd.Series:
    """读取某国负荷列（优先 actual，其次 forecast），返回小时级 Series。"""
    head = pd.read_csv(OPSD_CSV, nrows=0)
    cols = list(head.columns)
    cand = [c for c in cols if c.startswith(f"{country}_load_actual")
            or c.startswith(f"{country}_load_forecast")]
    if not cand:
        raise SystemExit(f"找不到 {country} 的负荷列，可用列示例：{cols[:8]}")
    col = cand[0]
    df = pd.read_csv(OPSD_CSV, usecols=["utc_timestamp", col])
    df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True)
    s = df.set_index("utc_timestamp")[col].astype("float64").sort_index()
    s = s[~s.index.duplicated()]
    s = s.interpolate(limit_direction="both")
    print(f"[数据] {country} 列={col} | {len(s):,} 小时 | {s.index.min()} ~ {s.index.max()}")
    return s.rename("load")


def build_features(s: pd.Series):
    idx = s.index
    df = pd.DataFrame({"y": s})
    df["lag1"] = s.shift(1)
    df["lag2"] = s.shift(2)
    df["lag3"] = s.shift(3)
    df["lag24"] = s.shift(24)
    df["lag25"] = s.shift(25)
    df["lag48"] = s.shift(48)
    df["lag72"] = s.shift(72)
    df["lag168"] = s.shift(168)
    past = s.shift(1)
    df["roll24_mean"] = past.rolling(24).mean()
    df["roll24_std"] = past.rolling(24).std()
    df["roll168_mean"] = past.rolling(168).mean()
    df["hour"] = idx.hour
    df["dow"] = idx.dayofweek
    df["month"] = idx.month
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    return df


def main():
    ap = argparse.ArgumentParser(description="OPSD 跨数据泛化实验")
    ap.add_argument("--country", type=str, default="DE")
    ap.add_argument("--test-start", type=str, default="2020-04-01")
    ap.add_argument("--train-end", type=str, default="2020-01-01")
    args = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    s = load_country(args.country)
    df = build_features(s)
    target = s.shift(-1)
    ok = target.notna().to_numpy()
    t_tr_end = pd.Timestamp(args.train_end, tz="UTC")
    t_te = pd.Timestamp(args.test_start, tz="UTC")
    idx = df.index
    train = (idx < t_tr_end) & ok
    val = (idx >= t_tr_end) & (idx < t_te) & ok
    test = (idx >= t_te) & ok
    print(f"[切分] 训练 {train.sum():,} | 验证 {val.sum():,} | 测试 {test.sum():,}")

    cols = [c for c in df.columns if c != "y"]
    X = df[cols].to_numpy(dtype="float64")
    y = target.to_numpy(dtype="float64")

    rows = []
    for S, name in [(24, "朴素-前一日同时刻"), (168, "朴素-前一周同时刻")]:
        naive = s.shift(S - 1).to_numpy()[test]
        rows.append(evaluate(y[test], naive, name))

    model = fit_lightgbm(X[train], y[train], X[val], y[val])
    pred = model.predict(X[test])
    rows.append(evaluate(y[test], pred, "LightGBM"))
    print(f"[LightGBM] 最佳迭代 {model.best_iteration_}")

    table = to_table(rows).sort_values("MAE")
    print(f"\n=== OPSD {args.country} 单步预测（测试集 {args.test_start} 之后）===")
    print(show(table))
    table.to_csv(RESULTS_DIR / "opsd_metrics.csv", index=False, encoding="utf-8-sig")

    idx_test = idx[test]
    n = min(24 * 14, len(idx_test))
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(idx_test[-n:], y[test][-n:], color="black", lw=1.0, label="真实值")
    ax.plot(idx_test[-n:], pred[-n:], color="#d62728", lw=1.0, label="LightGBM")
    ax.set_title(f"OPSD {args.country}：最后两周预测对比")
    ax.set_ylabel("负荷 (MW)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "opsd_prediction.png", dpi=150)
    plt.close(fig)
    print(f"[输出] {RESULTS_DIR / 'opsd_metrics.csv'}")
    print(f"[图片] {FIGS_DIR / 'opsd_prediction.png'}")


if __name__ == "__main__":
    main()
