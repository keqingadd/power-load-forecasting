# -*- coding: utf-8 -*-
"""E2b：递归多步预测（用已训练的 h=1 LSTM+Attention 逐点滚动预测未来 96 步）

支持两种输入：
    --features exog  用多通道外生特征模型（默认，推荐；模型需先用 run_lstm.py --features exog --horizon 1 训练）
    --features load  用单变量负荷模型

运行：python run_recursive.py
输出：results/recursive_metrics.csv、results/figs/recursive_vs_direct.png
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config import FIGS_DIR, LOOKBACK, RESULTS_DIR, TEST_START
from data_utils import (FUTURE_COV_COLS, build_exog_frame, load_load_series,
                        split_masks)
from metrics import evaluate, show, to_table
from models_lstm_attn import LSTMAttention

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

H = 96        # 一次预测未来 1 天（96 个 15 分钟点）


def parse_args():
    p = argparse.ArgumentParser(description="LSTM 递归多步预测")
    p.add_argument("--features", type=str, default="exog", choices=["exog", "load"])
    p.add_argument("--model", type=str, default="", help="模型文件名（默认按 features 自动选）")
    p.add_argument("--horizon", type=int, default=H)
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    series = load_load_series()
    values = series.to_numpy(dtype="float32")
    idx = series.index
    tr, _, _ = split_masks(idx)
    mu, sd = float(values[tr].mean()), float(values[tr].std())
    z = (values - mu) / sd

    exog = None
    if args.features == "exog":
        exog = build_exog_frame(series)
        for col in ["tmax", "tmin"]:
            m = float(np.nanmean(exog[col].to_numpy()[tr]))
            s_ = float(np.nanstd(exog[col].to_numpy()[tr])) or 1.0
            exog[col] = (exog[col] - m) / s_
        exog = exog.fillna(0.0).astype("float32")
        exog_vals = exog.to_numpy()
    else:
        exog_vals = None

    model_name = args.model or f"lstm_attn_h1_{args.features}.pt"
    model_path = RESULTS_DIR / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型 {model_path}，请先运行: "
                                f"python run_lstm.py --features {args.features} --horizon 1 --epochs 20")

    start = pd.Timestamp(TEST_START)
    origins = [i for i in range(len(idx)) if idx[i] >= start and idx[i].hour == 0
               and idx[i].minute == 0 and i + args.horizon < len(idx) and i >= LOOKBACK]
    print(f"[起点] {len(origins)} 个（{idx[origins[0]]} ~ {idx[origins[-1]]}），每次滚动 {args.horizon} 步")

    blob = torch.load(model_path, map_location="cpu")
    if isinstance(blob, dict) and "state_dict" in blob:
        state, meta = blob["state_dict"], blob.get("meta", {})
    else:
        state, meta = blob, {}
    residual = bool(meta.get("residual", False))
    mu_r = float(meta.get("mu_r", 0.0))
    sd_r = float(meta.get("sd_r", 1.0))
    n_feat = int(meta.get("n_features", 1 if exog is None else 1 + exog.shape[1]))
    n_fut = int(meta.get("n_future_cov", 0 if exog is None else len(FUTURE_COV_COLS)))

    net = LSTMAttention(n_features=n_feat, hidden=64, layers=2, horizon=1,
                        n_future_cov=n_fut)
    net.load_state_dict(state)
    net.eval()
    tag = "residual" if residual else args.features
    print(f"[模型] {model_name} | 残差模式={residual} | 输入通道={n_feat}")

    # 初始窗口：过去 lookback 个时刻的 [负荷, 外生特征]
    if exog is None:
        windows = np.stack([z[i - LOOKBACK + 1: i + 1] for i in origins])[:, :, None]
    else:
        windows = np.stack([np.concatenate(
            [z[i - LOOKBACK + 1: i + 1, None], exog_vals[i - LOOKBACK + 1: i + 1]], axis=1)
            for i in origins])

    rec_pred = np.zeros((len(origins), args.horizon), dtype="float32")
    with torch.no_grad():
        for k in range(1, args.horizon + 1):
            if exog is None:
                fx = None
            else:
                fx = torch.tensor(np.stack([exog_vals[i + k, :len(FUTURE_COV_COLS)] for i in origins])[:, None, :])
            out, _ = net(torch.tensor(windows), fx)
            step = out.numpy().reshape(-1)
            if residual:
                base_v = values[np.array(origins) + k - 96]          # 前一日同时刻（已知）
                load = base_v + step * sd_r + mu_r
            else:
                load = step * sd + mu
            rec_pred[:, k - 1] = load
            norm_load = ((load - mu) / sd).reshape(-1, 1, 1)
            if exog is None:
                windows = np.concatenate([windows[:, 1:, :], norm_load], axis=1)
            else:
                new_rows = np.concatenate([norm_load[:, :, 0],
                                           np.stack([exog_vals[i + k] for i in origins])], axis=1)
                windows = np.concatenate([windows[:, 1:, :], new_rows[:, None, :]], axis=1)

    truth = np.stack([values[i + 1: i + 1 + args.horizon] for i in origins])
    naive_pred = np.stack([values[i + 1 - 96: i + 1 - 96 + args.horizon] for i in origins])

    # 直接多步模型（若存在）同起点对比
    direct_pred = None
    direct_path = RESULTS_DIR / f"lstm_attn_h{args.horizon}_{args.features}.pt"
    if direct_path.exists():
        if exog is None:
            Xo = np.stack([z[i - LOOKBACK + 1: i + 1] for i in origins])[:, :, None]
            fxo = None
        else:
            Xo = np.stack([np.concatenate(
                [z[i - LOOKBACK + 1: i + 1, None], exog_vals[i - LOOKBACK + 1: i + 1]], axis=1)
                for i in origins])
            fxo = torch.tensor(np.stack([
                exog_vals[i + 1: i + 1 + args.horizon, :len(FUTURE_COV_COLS)] for i in origins]))
        netd = LSTMAttention(n_features=1 if exog is None else 1 + exog.shape[1],
                             hidden=64, layers=2, horizon=args.horizon,
                             n_future_cov=0 if exog is None else len(FUTURE_COV_COLS))
        netd.load_state_dict(torch.load(direct_path, map_location="cpu"))
        netd.eval()
        with torch.no_grad():
            pd_, _ = netd(torch.tensor(Xo), fxo)
        direct_pred = pd_.numpy() * sd + mu

    per_rec = np.mean(np.abs(rec_pred - truth), axis=0)
    per_naive = np.mean(np.abs(naive_pred - truth), axis=0)

    rows = [evaluate(truth.ravel(), rec_pred.ravel(), f"LSTM+Attn 递归多步({tag})"),
            evaluate(truth.ravel(), naive_pred.ravel(), "朴素(前一日同时刻)")]
    if direct_pred is not None:
        rows.append(evaluate(truth.ravel(), direct_pred.ravel(), f"LSTM+Attn 直接多步({args.features})"))
    table = to_table(rows)
    print()
    print("=== 未来 1 天预测对比（每日 0 点起点，每次滚动 96 步）===")
    print(show(table))
    table.to_csv(RESULTS_DIR / "recursive_metrics.csv", index=False, encoding="utf-8-sig")

    print("\n逐步 MAE:")
    for k in (1, 4, 8, 16, 32, 48, 64, 80, 96):
        print(f"  step {k:3d} ({k*15//60:>2d}h{k*15%60:02d}m): 递归 {per_rec[k-1]:8.1f} kW | 朴素 {per_naive[k-1]:8.1f} kW")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    steps = np.arange(1, args.horizon + 1)
    ax.plot(steps, per_rec, label=f"LSTM+Attn 递归多步({tag})", color="#d62728", lw=1.6)
    ax.plot(steps, per_naive, label="朴素(前一日同时刻)", color="#1f77b4", lw=1.2, alpha=0.8)
    if direct_pred is not None:
        per_direct = np.mean(np.abs(direct_pred - truth), axis=0)
        ax.plot(steps, per_direct, label=f"LSTM+Attn 直接多步({args.features})", color="#2ca02c", lw=1.2, alpha=0.8)
        print(f"\n直接多步 vs 递归：step1 {per_direct[0]:.0f} vs {per_rec[0]:.0f} | "
              f"step96 {per_direct[-1]:.0f} vs {per_rec[-1]:.0f}")
    ax.set_title("未来 1 天预测：误差随预测步长变化")
    ax.set_xlabel("预测步（1 步 = 15 分钟）")
    ax.set_ylabel("MAE (kW)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "recursive_vs_direct.png", dpi=150)
    plt.close(fig)
    print(f"[输出] {RESULTS_DIR / 'recursive_metrics.csv'}")
    print(f"[图片] {FIGS_DIR / 'recursive_vs_direct.png'}")


if __name__ == "__main__":
    main()
