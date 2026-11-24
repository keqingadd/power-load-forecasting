# -*- coding: utf-8 -*-
"""毕设原型系统：电力负荷预测 Web 演示（Streamlit）

启动：
    streamlit run app.py
或在 PyCharm 里把 app.py 的脚本参数设置为：
    run app.py --server.port 8501
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RESULTS_DIR, TEST_START, TRAIN_END, LOOKBACK
from data_utils import FUTURE_COV_COLS, build_exog_frame, load_load_series, split_masks
from metrics import evaluate
from models_lstm_attn import LSTMAttention

st.set_page_config(page_title="电力负荷预测原型系统", layout="wide")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

H = 96
MODELS = {
    "LSTM+Attention（残差，推荐）": "lstm_attn_h1_residual.pt",
    "LSTM+Attention（外生特征）": "lstm_attn_h1_exog.pt",
    "分位数区间 LSTM（P10/P50/P90）": "lstm_attn_h1_exog_quantile.pt",
}


@st.cache_data
def load_data():
    series = load_load_series()
    values = series.to_numpy("float64")
    tr, _, _ = split_masks(series.index)
    mu, sd = float(values[tr].mean()), float(values[tr].std())
    exog = build_exog_frame(series)
    for c in ["tmax", "tmin"]:
        exog[c] = (exog[c] - np.nanmean(exog[c].to_numpy()[tr])) / (np.nanstd(exog[c].to_numpy()[tr]) or 1)
    exog = exog.fillna(0.0).astype("float32")
    return series, values, mu, sd, exog


@st.cache_resource
def load_model(name):
    blob = torch.load(RESULTS_DIR / name, map_location="cpu")
    if isinstance(blob, dict) and "state_dict" in blob:
        state, meta = blob["state_dict"], blob.get("meta", {})
    else:
        state, meta = blob, {}
    net = LSTMAttention(n_features=int(meta.get("n_features", 9)), hidden=64, layers=2,
                        horizon=1, n_future_cov=int(meta.get("n_future_cov", 6)),
                        n_quantiles=3 if "quantile" in name else 1)
    net.load_state_dict(state)
    net.eval()
    return net, meta


def rollout(net, meta, origins, series, values, mu, sd, exog, quantile=False):
    exog_vals = exog.to_numpy()
    residual = bool(meta.get("residual", False))
    mu_r = float(meta.get("mu_r", 0.0))
    sd_r = float(meta.get("sd_r", 1.0))
    windows = np.stack([np.concatenate(
        [((values[i - LOOKBACK + 1: i + 1] - mu) / sd)[:, None],
         exog_vals[i - LOOKBACK + 1: i + 1]], axis=1) for i in origins]).astype("float32")
    pred = np.zeros((len(origins), H))
    bands = np.zeros((len(origins), H, 3)) if quantile else None
    with torch.no_grad():
        for k in range(1, H + 1):
            fx = torch.tensor(np.stack(
                [exog_vals[i + k, :len(FUTURE_COV_COLS)] for i in origins])[:, None, :])
            out, _ = net(torch.tensor(windows), fx)
            raw = out.numpy().reshape(len(origins), -1)
            if quantile:
                q = raw * sd + mu                      # (n, 3)
                bands[:, k - 1, :] = q
                load = q[:, 1]
            elif residual:
                load = values[np.array(origins) + k - 96] + raw[:, 0] * sd_r + mu_r
            else:
                load = raw[:, 0] * sd + mu
            pred[:, k - 1] = load
            norm = ((load - mu) / sd).astype("float32").reshape(-1, 1, 1)
            new_rows = np.concatenate([norm[:, :, 0],
                                       np.stack([exog_vals[i + k] for i in origins])],
                                      axis=1).astype("float32")
            windows = np.concatenate([windows[:, 1:, :], new_rows[:, None, :]], axis=1)
    truth = np.stack([values[i + 1: i + 1 + H] for i in origins])
    naive = np.stack([values[i + 1 - 96: i + 1 - 96 + H] for i in origins])
    return pred, truth, naive, bands


def main():
    series, values, mu, sd, exog = load_data()
    idx = series.index

    st.title("电力负荷预测原型系统")
    st.caption("数据：某地区电网 2018-01 ~ 2021-08，15 分钟负荷 · 模型：LSTM + 注意力")

    with st.sidebar:
        st.header("预测设置")
        model_label = st.selectbox("选择模型", list(MODELS.keys()))
        default_date = pd.Timestamp("2021-08-01").date()
        day = st.date_input("预测起点日期（每天 0 点开始，预测未来 1 天）", value=default_date)
        st.caption("可选范围：2021-07-01 ~ 2021-08-30")

    origins = [i for i in range(len(idx))
               if idx[i].date() == day and idx[i].hour == 0 and idx[i].minute == 0
               and i >= LOOKBACK and i + H < len(idx)]
    if not origins:
        st.error("该日期不在可预测范围内（需要 2021-07-01 ~ 2021-08-30 之间的日期）")
        return
    i = origins[0]
    model_name = MODELS[model_label]
    net, meta = load_model(model_name)
    quantile = "quantile" in model_name
    pred, truth, naive, bands = rollout(net, meta, [i], series, values, mu, sd, exog, quantile)

    p, t, n = pred[0], truth[0], naive[0]
    row = evaluate(t, p, model_label)
    row_naive = evaluate(t, n, "朴素(前一日同时刻)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE (kW)", f"{row['MAE']:,.0f}", f"比朴素 {row['MAE']-row_naive['MAE']:+,.0f}")
    c2.metric("RMSE (kW)", f"{row['RMSE']:,.0f}")
    c3.metric("MAPE", f"{row['MAPE']:.2f}%")
    c4.metric("峰值误差 (kW)", f"{np.max(np.abs(t-p)):,.0f}")

    hours = pd.date_range(idx[i] + pd.Timedelta(minutes=15), periods=H, freq="15min")
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(hours, t, color="black", lw=1.1, label="真实值")
    ax.plot(hours, p, color="#d62728", lw=1.1, label=model_label)
    ax.plot(hours, n, color="#1f77b4", lw=0.9, alpha=0.7, label="朴素(前一日同时刻)")
    if quantile and bands is not None:
        ax.fill_between(hours, bands[0][:, 0], bands[0][:, 2], color="#d62728", alpha=0.18,
                        label="P10-P90 区间")
    ax.set_title(f"{idx[i].date()} 未来 1 天预测对比")
    ax.set_ylabel("负荷 (kW)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("预测明细（前 8 个点）")
    detail = pd.DataFrame({
        "时间": hours[:8].strftime("%Y-%m-%d %H:%M"),
        "真实值(kW)": np.round(t[:8], 1),
        "预测值(kW)": np.round(p[:8], 1),
        "误差(kW)": np.round(t[:8] - p[:8], 1),
    })
    st.dataframe(detail, use_container_width=True)

    st.caption("说明：原型仅用于演示，模型训练/评估代码见 run_lstm.py 与 run_recursive.py。")


if __name__ == "__main__":
    main()
