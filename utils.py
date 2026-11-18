import random

import numpy as np
import torch
from sklearn.linear_model import Ridge


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def regression_metrics(pred, gt, eps=1e-6):
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    diff = pred - gt
    return {
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "mae": float(np.mean(np.abs(diff))),
        "mape": float(100.0 * np.mean(np.abs(diff) / np.maximum(np.abs(gt), eps))),
    }


def mean_std(values):
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def print_simple_baselines(train_df, test_df, horizon):
    """Original diagnostic baselines; these are NOT paper Table 2 baselines."""
    xp_train = np.stack(train_df["roi_past"]).astype(np.float64)
    yf_train = np.stack(train_df["roi_future"]).astype(np.float64)
    xp_test = np.stack(test_df["roi_past"]).astype(np.float64)
    yf_test = np.stack(test_df["roi_future"]).astype(np.float64)
    last = np.repeat(xp_test[:, -1:], horizon, axis=1)
    last_h = xp_test[:, -horizon:] if xp_test.shape[1] >= horizon else last
    mean = np.repeat(xp_test.mean(axis=1, keepdims=True), horizon, axis=1)
    t = np.arange(xp_test.shape[1], dtype=np.float64)
    centered_t = t - t.mean()
    slope = ((xp_test - xp_test.mean(axis=1, keepdims=True)) * centered_t).sum(axis=1) / ((centered_t ** 2).sum() + 1e-12)
    future_t = np.arange(len(t), len(t) + horizon, dtype=np.float64)
    linear = xp_test.mean(axis=1, keepdims=True) + slope[:, None] * (future_t[None, :] - t.mean())
    ridge = np.column_stack([
        Ridge(alpha=1.0).fit(xp_train, yf_train[:, h]).predict(xp_test)
        for h in range(horizon)
    ])
    for name, pred in (("last_repeat", last), ("lastH_copy", last_h),
                       ("mean_repeat", mean), ("linear_extrap", linear),
                       ("ridge", ridge)):
        m = regression_metrics(pred, yf_test)
        print(f"[BASELINE {name}] RMSE={m['rmse']:.4f} MAE={m['mae']:.4f} MAPE={m['mape']:.2f}%")
