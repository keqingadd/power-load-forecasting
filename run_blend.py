# -*- coding: utf-8 -*-
"""E7：组合预测（朴素基线 + 残差 LSTM 递归多步 加权）

在验证集（2021 上半年每日 0 点起点）上网格搜索权重 w，
再在测试集（2021 年 7-8 月）上评估：
    pred = w * 朴素(前一日同时刻) + (1-w) * LSTM+Attn 递归多步
运行：python run_blend.py
输出：results/blend_metrics.csv、results/figs/blend_recursive.png
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config import FIGS_DIR, LOOKBACK, RESULTS_DIR, TEST_START, TRAIN_END
from data_utils import FUTURE_COV_COLS, build_exog_frame, load_load_series, split_masks
from metrics import evaluate, show, to_table
from models_lstm_attn import LSTMAttention

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

H = 96
MODEL = "lstm_attn_h1_residual.pt"


def daily_origins(idx, start, end, need):
    out = []
    for i in range(len(idx)):
        if idx[i] >= start and idx[i] < end and idx[i].hour == 0 and idx[i].minute == 0:
            if i >= need - 1 and i + H < len(idx):
                out.append(i)
    return out


def main():
    series = load_load_series()
    values = series.to_numpy("float64")
    idx = series.index
    tr, _, _ = split_masks(idx)
    mu, sd = float(values[tr].mean()), float(values[tr].std())

    exog = build_exog_frame(series)
    for c in ["tmax", "tmin"]:
        exog[c] = (exog[c] - np.nanmean(exog[c].to_numpy()[tr])) / (np.nanstd(exog[c].to_numpy()[tr]) or 1)
    exog = exog.fillna(0.0).astype("float32")
    exog_vals = exog.to_numpy()

    blob = torch.load(RESULTS_DIR / MODEL, map_location="cpu")
    state, meta = blob["state_dict"], blob.get("meta", {})
    residual = bool(meta.get("residual", False))
    mu_r, sd_r = float(meta.get("mu_r", 0.0)), float(meta.get("sd_r", 1.0))
    net = LSTMAttention(n_features=int(meta.get("n_features", 9)), hidden=64, layers=2,
                        horizon=1, n_future_cov=int(meta.get("n_future_cov", 6)))
    net.load_state_dict(state)
    net.eval()
    print(f"[模型] {MODEL} 残差={residual}")

    def rollout(origins):
        windows = np.stack([np.concatenate(
            [((values[i - LOOKBACK + 1: i + 1] - mu) / sd)[:, None],
             exog_vals[i - LOOKBACK + 1: i + 1]], axis=1) for i in origins]).astype("float32")
        pred = np.zeros((len(origins), H))
        with torch.no_grad():
            for k in range(1, H + 1):
                fx = torch.tensor(np.stack(
                    [exog_vals[i + k, :len(FUTURE_COV_COLS)] for i in origins])[:, None, :])
                out, _ = net(torch.tensor(windows), fx)
                step = out.numpy().reshape(-1)
                if residual:
                    load = values[np.array(origins) + k - 96] + step * sd_r + mu_r
                else:
                    load = step * sd + mu
                pred[:, k - 1] = load
                norm = ((load - mu) / sd).astype("float32").reshape(-1, 1, 1)
                new_rows = np.concatenate([norm[:, :, 0],
                                           np.stack([exog_vals[i + k] for i in origins])],
                                          axis=1).astype("float32")
                windows = np.concatenate([windows[:, 1:, :], new_rows[:, None, :]], axis=1)
        truth = np.stack([values[i + 1: i + 1 + H] for i in origins])
        naive = np.stack([values[i + 1 - 96: i + 1 - 96 + H] for i in origins])
        return pred, naive, truth

    # 验证集：2021-01-01 ~ 2021-06-30，且目标不落入测试集
    t_train_end, t_test = pd.Timestamp(TRAIN_END), pd.Timestamp(TEST_START)
    val_origins = daily_origins(idx, t_train_end, t_test, LOOKBACK)
    val_origins = [i for i in val_origins if i + H < np.where(idx >= t_test)[0][0]]
    pred_v, naive_v, truth_v = rollout(val_origins)
    print(f"[验证] 起点 {len(val_origins)} 个（{idx[val_origins[0]]} ~ {idx[val_origins[-1]]}）")

    ws = np.arange(0, 1.0001, 0.05)
    scores = [np.mean(np.abs(w * naive_v + (1 - w) * pred_v - truth_v)) for w in ws]
    w_best = float(ws[int(np.argmin(scores))])
    print(f"[权重] 验证集最优 w(朴素权重) = {w_best:.2f}，验证 MAE = {min(scores):,.1f} kW")

    test_origins = daily_origins(idx, t_test, idx[-1] + pd.Timedelta(minutes=1), LOOKBACK)
    pred_t, naive_t, truth_t = rollout(test_origins)
    print(f"[测试] 起点 {len(test_origins)} 个（{idx[test_origins[0]]} ~ {idx[test_origins[-1]]}）")

    blend = w_best * naive_t + (1 - w_best) * pred_t

    # 逐步长加权：每个步长 k 单独在验证集上选权重 w_k
    w_step = []
    for k in range(H):
        errs = [np.mean(np.abs(w * naive_v[:, k] + (1 - w) * pred_v[:, k] - truth_v[:, k]))
                for w in ws]
        w_step.append(float(ws[int(np.argmin(errs))]))
    w_step = np.array(w_step)
    blend_step = np.stack([w_step[k] * naive_t[:, k] + (1 - w_step[k]) * pred_t[:, k]
                           for k in range(H)], axis=1)

    rows = [evaluate(truth_t.ravel(), blend_step.ravel(), "组合(逐步长权重)"),
            evaluate(truth_t.ravel(), blend.ravel(), f"组合(全局w朴素={w_best:.2f})"),
            evaluate(truth_t.ravel(), pred_t.ravel(), "LSTM+Attn 递归多步(残差)"),
            evaluate(truth_t.ravel(), naive_t.ravel(), "朴素(前一日同时刻)")]
    table = to_table(rows)
    print()
    print("=== 未来 1 天：组合预测 vs 单模型（测试集） ===")
    print(show(table))
    table.to_csv(RESULTS_DIR / "blend_metrics.csv", index=False, encoding="utf-8-sig")

    per_bs = np.mean(np.abs(blend_step - truth_t), axis=0)
    per_b = np.mean(np.abs(blend - truth_t), axis=0)
    per_r = np.mean(np.abs(pred_t - truth_t), axis=0)
    per_n = np.mean(np.abs(naive_t - truth_t), axis=0)
    print("\n逐步 MAE（部分）:")
    print("  逐步长权重:", np.round(w_step[::12], 2), "...")
    for k in (1, 16, 32, 48, 64, 80, 96):
        print(f"  step {k:3d}: 逐步组合 {per_bs[k-1]:8.1f} | 全局组合 {per_b[k-1]:8.1f} | "
              f"残差LSTM {per_r[k-1]:8.1f} | 朴素 {per_n[k-1]:8.1f}")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    steps = np.arange(1, H + 1)
    ax.plot(steps, per_bs, label="组合(逐步长权重)", color="#9467bd", lw=2.2)
    ax.plot(steps, per_b, label=f"组合(全局 w={w_best:.2f})", color="#ff7f0e", lw=1.2, ls="--")
    ax.plot(steps, per_r, label="LSTM+Attn 递归多步(残差)", color="#d62728", lw=1.3)
    ax.plot(steps, per_n, label="朴素(前一日同时刻)", color="#1f77b4", lw=1.2, alpha=0.8)
    ax.set_title("未来 1 天：组合预测误差随步长变化")
    ax.set_xlabel("预测步（1 步 = 15 分钟）")
    ax.set_ylabel("MAE (kW)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "blend_recursive.png", dpi=150)
    plt.close(fig)
    print(f"[输出] {RESULTS_DIR / 'blend_metrics.csv'}")
    print(f"[图片] {FIGS_DIR / 'blend_recursive.png'}")


if __name__ == "__main__":
    main()
