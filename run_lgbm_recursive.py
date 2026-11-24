# -*- coding: utf-8 -*-
"""E2c：LightGBM 递归多步（用 h=1 的 LightGBM 逐点滚动预测未来 96 步）

与 LSTM 递归多步在同一批起点、同一口径下对比。
运行：python run_lgbm_recursive.py
输出：results/lgbm_recursive_metrics.csv、results/figs/lgbm_recursive_per_step.png
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIGS_DIR, RESULTS_DIR, TEST_START, TRAIN_END
from data_utils import (build_features, features_for_horizon, load_load_series,
                        split_masks)
from metrics import evaluate, show, to_table
from models_baseline import fit_lightgbm

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

H = 96
LAGS = (1, 2, 4, 8, 96, 97, 192, 288, 672)


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="LightGBM 递归多步")
    p.add_argument("--residual", action="store_true",
                   help="残差模式：预测 y(t+1)-y(t-95)，最终预测 = 前一日同时刻 + 残差")
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    series = load_load_series()
    values = series.to_numpy(dtype="float64")
    idx = series.index
    base_df = build_features(series)
    df = features_for_horizon(base_df, 1)
    if args.residual:
        target = (series.shift(-1) - series.shift(95)).reindex(idx)   # y(t+1) - y(t-95)
    else:
        target = series.shift(-1).reindex(idx)
    ok = target.notna().to_numpy()
    train, val, test = split_masks(idx)
    train, val, test = train & ok, val & ok, test & ok
    cols = [c for c in df.columns if c != "y"]
    X = df[cols].to_numpy(dtype="float64")
    y = target.to_numpy(dtype="float64")

    model = fit_lightgbm(X[train], y[train], X[val], y[val])
    print(f"[LightGBM] 训练完成，最佳迭代 {model.best_iteration_}")

    # 预取日历/气象列（预测目标时刻 t+1 时使用 row=t+1 的值）
    cal = {c: base_df[c].to_numpy() for c in ["tod", "dow", "month", "is_weekend",
                                              "is_holiday", "is_workday", "tmax", "tmin"]}
    origin_start = pd.Timestamp(TEST_START)
    origins = [i for i in range(len(idx)) if idx[i] >= origin_start and idx[i].hour == 0
               and idx[i].minute == 0 and i + H < len(idx) and i >= 672]
    print(f"[起点] {len(origins)} 个（{idx[origins[0]]} ~ {idx[origins[-1]]}）")

    pred_all = np.zeros((len(origins), H))
    for oi, i in enumerate(origins):
        hist = list(values[i - 672: i + 1])       # 已知历史（最近 673 点，保证 lag672 可算）
        for k in range(1, H + 1):
            t = i + k - 1                           # 当前"已知到"的时刻下标
            cur = np.asarray(hist, dtype="float64")   # 含最新预测值，长度随 k 增长
            past = cur[:-1]
            j = t + 1                               # 目标时刻
            feat = {
                "lag0": cur[-1],
                "roll96_mean": past[-96:].mean(),
                "roll96_std": past[-96:].std(),
                "roll96_max": past[-96:].max(),
                "roll672_mean": past[-672:].mean(),
                "tod": cal["tod"][j],
                "dow": cal["dow"][j],
                "month": cal["month"][j],
                "is_weekend": cal["is_weekend"][j],
                "is_holiday": cal["is_holiday"][j],
                "is_workday": cal["is_workday"][j],
                "tmax": cal["tmax"][j],
                "tmin": cal["tmin"][j],
            }
            for L in LAGS:
                feat[f"lag{L}"] = cur[-1 - L]
            vec = np.asarray([feat[c] for c in cols], dtype="float64").reshape(1, -1)
            pred = float(model.predict(vec)[0])
            if args.residual:
                pred = values[t - 95] + pred      # 前一日同时刻 + 残差
            pred_all[oi, k - 1] = pred
            hist.append(pred)

    truth = np.stack([values[i + 1: i + 1 + H] for i in origins])
    naive = np.stack([values[i + 1 - 96: i + 1 - 96 + H] for i in origins])

    rows = [evaluate(truth.ravel(), pred_all.ravel(), "LightGBM 递归多步"),
            evaluate(truth.ravel(), naive.ravel(), "朴素(前一日同时刻)")]
    table = to_table(rows)
    print()
    print("=== 未来 1 天：LightGBM 递归多步 vs 朴素 ===")
    print(show(table))
    table.to_csv(RESULTS_DIR / "lgbm_recursive_metrics.csv", index=False, encoding="utf-8-sig")

    per_step = np.mean(np.abs(pred_all - truth), axis=0)
    per_naive = np.mean(np.abs(naive - truth), axis=0)
    print("\n逐步 MAE:")
    for k in (1, 4, 8, 16, 32, 48, 64, 80, 96):
        print(f"  step {k:3d}: LightGBM递归 {per_step[k-1]:8.1f} kW | 朴素 {per_naive[k-1]:8.1f} kW")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    steps = np.arange(1, H + 1)
    ax.plot(steps, per_step, label="LightGBM 递归多步", color="#d62728", lw=1.6)
    ax.plot(steps, per_naive, label="朴素(前一日同时刻)", color="#1f77b4", lw=1.2, alpha=0.8)
    lstm_csv = RESULTS_DIR / "recursive_metrics.csv"
    if lstm_csv.exists():
        try:
            lstm = pd.read_csv(lstm_csv, encoding="utf-8-sig")
            print("\n[已有对比] LSTM 递归多步:", lstm.iloc[0].to_dict())
        except Exception:
            pass
    ax.set_title("未来 1 天：LightGBM 递归多步误差随步长变化")
    ax.set_xlabel("预测步（1 步 = 15 分钟）")
    ax.set_ylabel("MAE (kW)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "lgbm_recursive_per_step.png", dpi=150)
    plt.close(fig)
    print(f"[输出] {RESULTS_DIR / 'lgbm_recursive_metrics.csv'}")
    print(f"[图片] {FIGS_DIR / 'lgbm_recursive_per_step.png'}")


if __name__ == "__main__":
    main()
