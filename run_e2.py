# -*- coding: utf-8 -*-
"""E2：多步预测对比（h = 1 / 4 / 96 步，即 15 分钟 / 1 小时 / 1 天）

统一口径：已知 t 时刻及以前的信息，预测 t+h 时刻的负荷。
LightGBM 采用"直接多步"（每个 h 单独训练一个模型），与两种季节朴素基线对比。
运行：python run_e2.py
输出：results/e2_metrics.csv、results/figs/e2_mae_vs_horizon.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIGS_DIR, RESULTS_DIR, TEST_START, TRAIN_END
from data_utils import (load_load_series, build_features, features_for_horizon,
                        split_masks)
from metrics import evaluate, to_table, show
from models_baseline import fit_lightgbm

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HORIZONS = (1, 4, 96)
NAME = {1: "15 分钟", 4: "1 小时", 96: "1 天"}


def main():
    ap = argparse.ArgumentParser(description="LightGBM 直接多步")
    ap.add_argument("--residual", action="store_true",
                    help="残差模式：预测 y(t+h)-y(t+h-96)，最终预测 = 前一日同时刻 + 残差")
    args = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    series = load_load_series()
    base_df = build_features(series)
    idx = base_df.index

    rows = []
    for h in HORIZONS:
        df = features_for_horizon(base_df, h)
        feat_cols = [c for c in df.columns if c != "y"]
        X = df[feat_cols].to_numpy(dtype="float64")
        target = series.shift(-h).reindex(idx)
        base_arr = series.shift(96 - h).reindex(idx).to_numpy()   # 前一日同时刻
        ok = target.notna().to_numpy()
        if args.residual:
            ok &= np.isfinite(base_arr)
        train, val, test = split_masks(idx)
        train, val, test = train & ok, val & ok, test & ok
        y = target.to_numpy(dtype="float64")

        # 朴素基线：目标时刻的同日同时刻（S=96）与同周同时刻（S=672）
        for S, tag in [(96, "日"), (672, "周")]:
            naive = series.shift(S - h).reindex(idx).to_numpy()[test]
            rows.append(evaluate(y[test], naive, f"朴素-{tag}(h={h})"))

        if args.residual:
            y_fit = y - base_arr
            model = fit_lightgbm(X[train], y_fit[train], X[val], y_fit[val])
            pred = base_arr[test] + model.predict(X[test])
            tag = "LightGBM-残差"
        else:
            model = fit_lightgbm(X[train], y[train], X[val], y[val])
            pred = model.predict(X[test])
            tag = "LightGBM"
        rows.append(evaluate(y[test], pred, f"{tag}(h={h})"))
        print(f"[h={h}] {tag} 最佳迭代={model.best_iteration_}")

    table = to_table(rows)
    print()
    print("=== E2 多步预测对比（测试集 2021-07-01 ~ 2021-08-31）===")
    print(show(table))
    table.to_csv(RESULTS_DIR / "e2_metrics.csv", index=False, encoding="utf-8-sig")

    # 图：MAE 随预测提前量变化
    lgb_tag = "LightGBM-残差" if args.residual else "LightGBM"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for prefix, label, color in [("朴素-日", "朴素(前一日同时刻)", "#1f77b4"),
                                 ("朴素-周", "朴素(前一周同时刻)", "#2ca02c"),
                                 (lgb_tag, lgb_tag, "#d62728")]:
        ys = []
        for h in HORIZONS:
            row = table[table["model"] == f"{prefix}(h={h})"]
            ys.append(float(row["MAE"].iloc[0]))
        ax.plot(np.arange(len(HORIZONS)), ys, marker="o", label=label, color=color)
    ax.set_title("多步预测：误差随预测提前量变化")
    ax.set_xlabel("预测提前量（1 步 = 15 分钟）")
    ax.set_ylabel("MAE (kW)")
    ax.set_xticks(np.arange(len(HORIZONS)))
    ax.set_xticklabels([f"{h} 步（{NAME[h]}）" for h in HORIZONS])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "e2_mae_vs_horizon.png", dpi=150)
    plt.close(fig)
    print(f"[输出] {RESULTS_DIR / 'e2_metrics.csv'}")
    print(f"[图片] {FIGS_DIR / 'e2_mae_vs_horizon.png'}")


if __name__ == "__main__":
    main()
