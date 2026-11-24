# -*- coding: utf-8 -*-
"""训练 LSTM + 注意力模型（支持多通道外生特征、残差预测）

用法（PyCharm 里改 Run Configuration 的 Parameters，或命令行）：
    python run_lstm.py                                      # 默认：外生特征，单步，MAE，20 epoch
    python run_lstm.py --horizon 96 --epochs 40             # 直接预测未来 1 天
    python run_lstm.py --horizon 96 --epochs 40 --residual  # 残差预测：Δ = y(t+h) - y(t+h-96)
    python run_lstm.py --features load                      # 单变量（仅历史负荷）
    python run_lstm.py --loss quantile                      # 分位数损失（区间预测）
    python run_lstm.py --epochs 3 --max-train 20000         # 快速试跑
"""
import argparse
import os
import time

# Windows 下 conda 的 MKL 与 torch 的 OpenMP 可能冲突，先设置再导入 torch
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from config import (FIGS_DIR, HORIZON, LOOKBACK, RESULTS_DIR, SEED, TEST_START,
                    TRAIN_END)
from data_utils import (build_exog_frame, load_load_series, make_multichannel_windows,
                        make_windows, split_masks)
from metrics import evaluate, show, to_table
from models_lstm_attn import LSTMAttention, build_loss

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def parse_args():
    p = argparse.ArgumentParser(description="LSTM + Attention 负荷预测")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--lookback", type=int, default=LOOKBACK)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--features", type=str, default="exog", choices=["exog", "load"])
    p.add_argument("--residual", action="store_true",
                   help="残差模式：预测 y(t+h)-y(t+h-96)，最终预测 = 前一日同时刻 + 残差")
    p.add_argument("--loss", type=str, default="mae",
                   choices=["mse", "mae", "huber", "quantile"])
    p.add_argument("--quantiles", type=str, default="0.1,0.5,0.9",
                   help="分位数损失的分位点，逗号分隔，默认 0.1,0.5,0.9（80% 区间）")
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    series = load_load_series()
    values = series.to_numpy(dtype="float64")
    print(f"[数据] {len(values):,} 点 | {series.index.min()} ~ {series.index.max()}")

    train_mask_raw, _, _ = split_masks(series.index)
    mu = float(values[train_mask_raw].mean())
    sd = float(values[train_mask_raw].std())
    z = (values - mu) / sd
    z_series = pd.Series(z, index=series.index, name="load")

    # ---------- 输入通道 ----------
    if args.features == "exog":
        exog = build_exog_frame(series)
        for col in ["tmax", "tmin"]:
            mu_c = float(np.nanmean(exog[col].to_numpy()[train_mask_raw]))
            sd_c = float(np.nanstd(exog[col].to_numpy()[train_mask_raw])) or 1.0
            exog[col] = (exog[col] - mu_c) / sd_c
        exog = exog.fillna(0.0).astype("float32")
        X, FX, _ = make_multichannel_windows(z_series, exog, args.lookback, args.horizon)
        n_features, n_future_cov = X.shape[2], FX.shape[2]
        print(f"[输入] 多通道 {n_features} 个，未来协变量 {n_future_cov} 个")
    else:
        X, _ = make_windows(z, args.lookback, args.horizon)
        X = X[:, :, None]
        FX = None
        n_features, n_future_cov = 1, 0
        print("[输入] 单变量（仅历史负荷）")

    n = len(X)
    target_time = pd.DatetimeIndex(series.index[args.lookback + args.horizon - 1:])
    train, val, test = split_masks(target_time)
    print(f"[窗口] 共 {n:,} | 训练 {train.sum():,} | 验证 {val.sum():,} | 测试 {test.sum():,}")

    # ---------- 目标值：原始负荷 或 残差 Δ = y(t+h) - y(t+h-96) ----------
    t_idx = np.arange(n) + args.lookback - 1
    Yraw = np.stack([values[t_idx + k] for k in range(1, args.horizon + 1)],
                    axis=1).astype("float64")
    base = np.stack([values[t_idx + k - 96] for k in range(1, args.horizon + 1)],
                    axis=1).astype("float64")   # 前一日同时刻（已知基线）
    if args.residual:
        R = Yraw - base
        mu_r = float(R[train].mean())
        sd_r = float(R[train].std()) or 1.0
        Yn = (R - mu_r) / sd_r
        print(f"[目标] 残差模式：Δ = y(t+h) - y(t+h-96)，训练集 Δ 均值 {mu_r:,.1f} kW，标准差 {sd_r:,.1f} kW")
    else:
        mu_r, sd_r = 0.0, 1.0
        Yn = (Yraw - mu) / sd
        print("[目标] 原始负荷（标准化）")

    # ---------- 训练 ----------
    Xtr, Ytr = X[train], Yn[train]
    FXtr = FX[train] if FX is not None else None
    if args.max_train and len(Xtr) > args.max_train:
        keep = np.linspace(0, len(Xtr) - 1, args.max_train).astype(int)
        Xtr, Ytr = Xtr[keep], Ytr[keep]
        FXtr = FXtr[keep] if FXtr is not None else None

    qs = [float(x) for x in args.quantiles.split(",")]
    n_quantiles = len(qs) if args.loss == "quantile" else 1
    if args.residual and args.loss == "quantile":
        raise SystemExit("暂不支持 --residual 与 --loss quantile 组合，请分开做实验")
    net = LSTMAttention(n_features=n_features, hidden=args.hidden, layers=args.layers,
                        horizon=args.horizon, n_future_cov=n_future_cov,
                        n_quantiles=n_quantiles)
    lossf = build_loss(args.loss, quantiles=qs)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)

    Xv = torch.tensor(np.ascontiguousarray(X[val]))
    Yv = torch.tensor(np.ascontiguousarray(Yn[val]))
    FXv = torch.tensor(np.ascontiguousarray(FX[val])) if FX is not None else None

    best_val, best_state, bad = float("inf"), None, 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        net.train()
        perm = np.random.permutation(len(Xtr))
        total = 0.0
        for i in range(0, len(perm), args.batch_size):
            b = perm[i:i + args.batch_size]
            xb = torch.tensor(np.ascontiguousarray(Xtr[b]))
            yb = torch.tensor(np.ascontiguousarray(Ytr[b]))
            fxb = torch.tensor(np.ascontiguousarray(FXtr[b])) if FXtr is not None else None
            opt.zero_grad()
            pred, _ = net(xb, fxb)
            loss = lossf(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            total += loss.item() * len(b)

        net.eval()
        with torch.no_grad():
            pv, _ = net(Xv, FXv)
            vm = float(torch.mean(torch.abs(pv - Yv)))
        sched.step(vm)
        improved = vm < best_val
        if improved:
            best_val, best_state, bad = vm, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
        print(f"  epoch {ep:2d}/{args.epochs} | train_loss={total / len(Xtr):.4f} "
              f"| val_MAE={vm:.4f}{' *' if improved else ''} | {time.time() - t0:.0f}s")
        if bad >= args.patience:
            print(f"  验证集 {args.patience} 轮未提升，早停")
            break

    if best_state is not None:
        net.load_state_dict(best_state)

    # ---------- 测试集评估 ----------
    net.eval()
    Xte = torch.tensor(np.ascontiguousarray(X[test]))
    FXte = torch.tensor(np.ascontiguousarray(FX[test])) if FX is not None else None
    with torch.no_grad():
        pt, _ = net(Xte, FXte)
    truth = Yraw[test]
    if args.residual:
        tag = "residual"
    elif args.loss == "quantile":
        tag = f"{args.features}_quantile"
    else:
        tag = args.features
    interval_info = None
    lo = hi = None
    if args.loss == "quantile":
        q_pred = pt.numpy().reshape(len(X[test]), args.horizon, n_quantiles) * sd + mu
        mid = qs.index(0.5) if 0.5 in qs else n_quantiles // 2
        pred = q_pred[:, :, mid]
        lo, hi = q_pred[:, :, 0], q_pred[:, :, -1]
        nominal = (qs[-1] - qs[0]) * 100
        picp = float(np.mean((truth >= lo) & (truth <= hi)) * 100)
        pinaw = float(np.mean(hi - lo) / (truth.max() - truth.min()))
        interval_info = dict(PICP=picp, PINAW=pinaw, nominal=nominal,
                             width=float(np.mean(hi - lo)))
        print(f"[区间] 名义覆盖率 {nominal:.0f}% | 实测 PICP={picp:.2f}% | PINAW={pinaw:.4f} | 平均宽度 {np.mean(hi-lo):,.1f} kW")
    else:
        out = pt.numpy().reshape(len(X[test]), args.horizon)
        pred = base[test] + out * sd_r + mu_r if args.residual else out * sd + mu

    row = evaluate(truth.ravel(), pred.ravel(),
                   f"LSTM+Attn({tag},{args.loss},h={args.horizon})")
    row_naive = evaluate(truth.ravel(), base[test].ravel(), "朴素(前一日同时刻)")
    table = to_table([row, row_naive])
    print("\n=== 测试集结果 ===")
    print(show(table))
    table.to_csv(RESULTS_DIR / f"lstm_h{args.horizon}_{tag}_metrics.csv",
                 index=False, encoding="utf-8-sig")
    if interval_info is not None:
        pd.DataFrame([dict(model=f"LSTM+Attn(quantile,h={args.horizon})",
                           nominal_coverage=interval_info["nominal"],
                           PICP=interval_info["PICP"], PINAW=interval_info["PINAW"],
                           mean_width_kw=interval_info["width"])]).to_csv(
            RESULTS_DIR / "interval_metrics.csv", index=False, encoding="utf-8-sig")
        print(f"[输出] {RESULTS_DIR / 'interval_metrics.csv'}")

    if args.horizon > 1:
        per_step = np.mean(np.abs(truth - pred), axis=0)
        per_naive = np.mean(np.abs(truth - base[test]), axis=0)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(np.arange(1, len(per_step) + 1), per_step, marker="o", ms=3,
                color="#d62728", label=f"LSTM+Attn ({tag})")
        ax.plot(np.arange(1, len(per_naive) + 1), per_naive, color="#1f77b4", lw=1.2,
                alpha=0.8, label="朴素(前一日同时刻)")
        ax.set_title(f"直接多步：误差随预测步长变化（h={args.horizon}，{tag}）")
        ax.set_xlabel("预测步（1 步 = 15 分钟）")
        ax.set_ylabel("MAE (kW)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(FIGS_DIR / f"lstm_h{args.horizon}_{tag}_per_step_mae.png", dpi=150)
        plt.close(fig)
        print(f"[图片] {FIGS_DIR / ('lstm_h%d_%s_per_step_mae.png' % (args.horizon, tag))}")

    if args.horizon == 1:
        idx = target_time[test][-672:]
        fig, ax = plt.subplots(figsize=(13, 4.5))
        ax.plot(idx, truth.ravel()[-672:], color="black", lw=1.0, label="真实值")
        ax.plot(idx, pred.ravel()[-672:], color="#d62728", lw=1.0, label="LSTM+Attention")
        if lo is not None:
            ax.fill_between(idx, lo.ravel()[-672:], hi.ravel()[-672:],
                            color="#d62728", alpha=0.20,
                            label=f"{int((qs[-1]-qs[0])*100)}% 预测区间")
        ax.set_title(f"LSTM + Attention：最后 7 天预测（{tag}）")
        ax.set_ylabel("负荷 (kW)")
        ax.legend(loc="upper left", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIGS_DIR / "lstm_prediction_vs_truth.png", dpi=150)
        plt.close(fig)
        if lo is not None:
            fig, ax = plt.subplots(figsize=(13, 4.5))
            ax.fill_between(idx, lo.ravel()[-672:], hi.ravel()[-672:],
                            color="#d62728", alpha=0.20,
                            label=f"P{int(qs[0]*100)}-P{int(qs[-1]*100)} 区间")
            ax.plot(idx, pred.ravel()[-672:], color="#d62728", lw=1.0, label="P50 预测")
            ax.plot(idx, truth.ravel()[-672:], color="black", lw=0.9, label="真实值")
            ax.set_title(f"分位数区间预测（PICP={interval_info['PICP']:.1f}%，"
                         f"PINAW={interval_info['PINAW']:.3f}）")
            ax.set_ylabel("负荷 (kW)")
            ax.legend(loc="upper left", fontsize=9)
            fig.tight_layout()
            fig.savefig(FIGS_DIR / "quantile_interval.png", dpi=150)
            plt.close(fig)

    with torch.no_grad():
        _, alpha = net(Xte[:512], FXte[:512] if FXte is not None else None)
    alpha_mean = alpha.numpy().mean(axis=0)
    fig, ax = plt.subplots(figsize=(11, 2.6))
    ax.bar(np.arange(len(alpha_mean)), alpha_mean, color="#2ca02c", width=1.0)
    ax.set_title(f"平均注意力权重（{tag}）：模型关注哪些历史时点")
    ax.set_xlabel("回看步：0 = 最早（1 天前） … 95 = 最近一个 15 分钟点")
    ax.set_ylabel("权重")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / f"lstm_attention_heatmap_h{args.horizon}_{tag}.png", dpi=150)
    plt.close(fig)

    # 保存 checkpoint（含元信息，供递归脚本读取）
    model_path = RESULTS_DIR / f"lstm_attn_h{args.horizon}_{tag}.pt"
    torch.save({
        "state_dict": net.state_dict(),
        "meta": {
            "features": args.features,
            "residual": bool(args.residual),
            "horizon": args.horizon,
            "lookback": args.lookback,
            "n_features": n_features,
            "n_future_cov": n_future_cov,
            "mu": mu, "sd": sd, "mu_r": mu_r, "sd_r": sd_r,
        },
    }, model_path)
    print(f"[完成] 耗时 {time.time() - t0:.0f}s，模型已保存到 {model_path}")


if __name__ == "__main__":
    main()
