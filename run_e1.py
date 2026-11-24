# -*- coding: utf-8 -*-
"""E1：基线对比实验（季节朴素 / LightGBM）

口径：已知 t 时刻及以前的信息，预测 t+1 时刻的负荷（提前 15 分钟）。
运行：在 PyCharm 中右键本文件 -> Run，或命令行 python run_e1.py
输出：results/e1_metrics.csv、results/figs/*.png，并打印对比表。
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import RESULTS_DIR, FIGS_DIR, TEST_START, TRAIN_END
from data_utils import (load_load_series, build_features, features_for_horizon,
                        split_masks)
from metrics import evaluate, to_table, show
from models_baseline import fit_lightgbm

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    series = load_load_series()
    print(f"[数据] 负荷 {len(series):,} 点 | {series.index.min()} ~ {series.index.max()}")
    df = features_for_horizon(build_features(series), 1)
    idx = df.index
    target = series.shift(-1).reindex(idx)      # 预测目标：t+1
    ok = target.notna().to_numpy()

    train, val, test = split_masks(idx)
    train, val, test = train & ok, val & ok, test & ok
    print(f"[切分] 训练 {train.sum():,} | 验证 {val.sum():,} | 测试 {test.sum():,}  "
          f"(训练 < {TRAIN_END} ≤ 验证 < {TEST_START} ≤ 测试)")

    y = target.to_numpy(dtype="float64")
    feat_cols = [c for c in df.columns if c != "y"]
    X = df[feat_cols].to_numpy(dtype="float64")
    print(f"[特征] {len(feat_cols)} 个（含 lag0 当前值、滞后、滚动统计、日历、气象、节假日）")

    rows = []
    # 季节朴素基线：预测 t+1 时，用 t+1-S 时刻的值（S=96 为一天，672 为一周）
    for S, name in [(1, "朴素-前一时刻"), (96, "朴素-前一日同时刻"), (672, "朴素-前一周同时刻")]:
        naive = series.shift(S - 1).reindex(idx).to_numpy()[test]
        rows.append(evaluate(y[test], naive, name))

    # LightGBM
    t1 = time.time()
    model = fit_lightgbm(X[train], y[train], X[val], y[val])
    pred = model.predict(X[test])
    rows.append(evaluate(y[test], pred, "LightGBM"))
    print(f"[LightGBM] 训练+预测 {time.time() - t1:.1f}s, 最佳迭代 {model.best_iteration_}")

    table = to_table(rows).sort_values("MAE")
    print()
    print("=== E1 基线对比（测试集 2021-07-01 ~ 2021-08-31，单步提前 15 分钟）===")
    print(show(table))
    table.to_csv(RESULTS_DIR / "e1_metrics.csv", index=False, encoding="utf-8-sig")

    # ---- 图 1：测试集最后 7 天预测对比 ----
    n = min(672, int(test.sum()))
    idx_test = idx[test][-n:]
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(idx_test, y[test][-n:], color="black", lw=1.0, label="真实值")
    ax.plot(idx_test, pred[-n:], color="#d62728", lw=1.0, label="LightGBM")
    naive_daily = series.shift(95).reindex(idx).to_numpy()[test]
    ax.plot(idx_test, naive_daily[-n:], color="#1f77b4", lw=0.9, alpha=0.7,
            label="朴素基线(前一日)")
    ax.set_title("测试集最后 7 天：预测 vs 真实")
    ax.set_ylabel("负荷 (kW)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "e1_prediction_vs_truth.png", dpi=150)
    plt.close(fig)

    # ---- 图 2：特征重要性 ----
    imp = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    imp[::-1].plot(kind="barh", ax=ax, color="#2ca02c")
    ax.set_title("LightGBM 特征重要性 Top15")
    ax.set_xlabel("split importance")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "e1_feature_importance.png", dpi=150)
    plt.close(fig)

    # ---- 图 3：残差分布 ----
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(y[test] - pred, bins=60, color="#9467bd", alpha=0.85)
    ax.set_title("LightGBM 残差分布（测试集）")
    ax.set_xlabel("真实值 - 预测值 (kW)")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "e1_residual_hist.png", dpi=150)
    plt.close(fig)

    print(f"\n[完成] 总耗时 {time.time() - t0:.1f}s")
    print(f"[输出] {RESULTS_DIR / 'e1_metrics.csv'}")
    print(f"[图片] {FIGS_DIR}")


if __name__ == "__main__":
    main()
