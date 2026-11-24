# -*- coding: utf-8 -*-
"""全局配置：路径、切分时间、随机种子、预测口径"""
from pathlib import Path

# ---------- 数据（原始数据只读，位于另一个独立目录） ----------
DATA_ROOT = Path("D:/电力负荷预测毕设数据")
RAW_DIR = DATA_ROOT / "raw"
LOAD_CSV = RAW_DIR / "01_china_region_15min" / "附件1-区域15分钟负荷数据.csv"
WEATHER_CSV = RAW_DIR / "01_china_region_15min" / "附件3-气象数据.csv"
INDUSTRY_CSV = RAW_DIR / "01_china_region_15min" / "附件2-行业日负荷数据.csv"

# ---------- 工程输出 ----------
PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
FIGS_DIR = RESULTS_DIR / "figs"

# ---------- 时间切分（严格按时间顺序，禁止打乱） ----------
TRAIN_END = "2021-01-01"   # 训练集：< 2021-01-01
TEST_START = "2021-07-01"  # 验证集：[TRAIN_END, TEST_START)，测试集：>= TEST_START
FREQ = "15min"

# ---------- 建模口径 ----------
SEED = 42
LOOKBACK = 96               # 回看 1 天（96 个 15 分钟点）
HORIZON = 1                 # 单步预测：提前 15 分钟
LAGS = (1, 2, 4, 8, 96, 97, 192, 288, 672)
