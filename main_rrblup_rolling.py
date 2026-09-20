import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error


def rmse_metric_np(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def rrmse_metric_np(y_true, y_pred):
    top_k = max(1, int(len(y_true) * 0.2))
    top_idx = np.argpartition(y_true, -top_k)[-top_k:]
    return rmse_metric_np(y_true, y_pred) / max(float(np.mean(y_true[top_idx])), 1e-6)


def success_rate_at_threshold(pred, true, threshold):
    selected = pred > threshold
    count = int(selected.sum())
    if count == 0:
        return float("nan"), 0
    return float(np.mean(true[selected] > threshold)), count


def precision_recall_at_threshold(pred, true, threshold):
    pred_pos = pred > threshold
    actual_pos = true > threshold
    pred_n = int(pred_pos.sum())
    actual_n = int(actual_pos.sum())
    tp = int(np.logical_and(pred_pos, actual_pos).sum())
    precision = tp / pred_n if pred_n > 0 else float("nan")
    recall = tp / actual_n if actual_n > 0 else float("nan")
    return float(precision), float(recall), pred_n, actual_n


def fit_rrblup(X_fixed, Z, y, alpha):
    y_mean = float(y.mean())
    y_center = y - y_mean
    X_aug = np.concatenate([X_fixed, Z], axis=1)
    penalty = np.concatenate(
        [
            np.zeros(X_fixed.shape[1], dtype=np.float32),
            np.full(Z.shape[1], alpha, dtype=np.float32),
        ]
    )
    aug_X = np.vstack([X_aug, np.diag(np.sqrt(penalty))])
    aug_y = np.concatenate([y_center, np.zeros(len(penalty), dtype=np.float32)])
    coef, *_ = np.linalg.lstsq(aug_X, aug_y, rcond=None)
    return y_mean, coef


def predict_rrblup(X_fixed, Z, intercept, coef):
    n_fixed = X_fixed.shape[1]
    return intercept + X_fixed @ coef[:n_fixed] + Z @ coef[n_fixed:]


seed = 20260314
np.random.seed(seed % (2**32 - 1))
torch.manual_seed(seed)

pheno = pd.read_csv("Data/Clean/phenotype_all.csv")
G_raw = pd.read_csv("Data/Clean/geno1.csv")

G_mat = torch.tensor(G_raw.iloc[:, 10:].to_numpy(), dtype=torch.float32).t() * 2
ID = G_raw.columns[10:]
id_to_idx = {sid: i for i, sid in enumerate(ID)}
pheno = pheno[pheno["SID"].isin(ID)].reset_index(drop=True)
N = len(pheno)

G = torch.zeros(N, G_mat.shape[1], dtype=torch.float32)
for i in range(N):
    G[i, :] = G_mat[id_to_idx[pheno["SID"].iloc[i]], :]

HIR = torch.tensor(pheno["HIR"].to_numpy(), dtype=torch.float32) / 100
y_all = HIR.cpu().numpy()
year_np = pheno["Year"].to_numpy()
years = sorted(int(y) for y in pheno.loc[~pheno["HIR"].isna(), "Year"].unique())

X_fixed_all = pd.get_dummies(pheno["Year"].astype(str), prefix="year", drop_first=True).astype(float).to_numpy(dtype=np.float32)
Z_all = G.cpu().numpy()

run_dir = os.path.join("Result", f"rrblup_rolling_{seed}")
os.makedirs(run_dir, exist_ok=True)
fold_results = []
precision_recall_thresholds = [0.10, 0.15, 0.20]
alphas = [0.1, 1.0, 5.0, 10.0]

for fold_id, test_year in enumerate(years[1:], start=1):
    test_idx = np.where((year_np == test_year) & ~np.isnan(y_all))[0]
    train_pool = np.where((year_np < test_year) & ~np.isnan(y_all))[0]
    if len(test_idx) == 0 or len(train_pool) <= 24:
        continue

    np.random.seed((seed + fold_id) % (2**32 - 1))
    perm = np.random.permutation(len(train_pool))
    val_size = max(24, int(0.1 * len(train_pool)))
    val_idx = train_pool[perm[:val_size]]
    tra_idx = train_pool[perm[val_size:]]

    X_fixed_train = X_fixed_all[tra_idx]
    X_fixed_val = X_fixed_all[val_idx]
    X_fixed_test = X_fixed_all[test_idx]
    Z_train = Z_all[tra_idx]
    Z_val = Z_all[val_idx]
    Z_test = Z_all[test_idx]
    y_train = y_all[tra_idx]
    y_val = y_all[val_idx]
    y_test = y_all[test_idx]

    z_mean = Z_train.mean(axis=0, keepdims=True)
    z_std = Z_train.std(axis=0, keepdims=True)
    z_std[z_std < 1e-6] = 1.0
    Z_train = (Z_train - z_mean) / z_std
    Z_val = (Z_val - z_mean) / z_std
    Z_test = (Z_test - z_mean) / z_std

    best_alpha = None
    best_val_rrmse = float("inf")
    best_intercept = None
    best_coef = None
    for alpha in alphas:
        intercept, coef = fit_rrblup(X_fixed_train, Z_train, y_train, alpha)
        val_pred = predict_rrblup(X_fixed_val, Z_val, intercept, coef)
        val_rrmse = rrmse_metric_np(y_val, val_pred)
        if val_rrmse < best_val_rrmse:
            best_val_rrmse = val_rrmse
            best_alpha = alpha
            best_intercept = intercept
            best_coef = coef.copy()

    test_pred = predict_rrblup(X_fixed_test, Z_test, best_intercept, best_coef)
    sr015, n015 = success_rate_at_threshold(test_pred, y_test, 0.15)
    sr020, n020 = success_rate_at_threshold(test_pred, y_test, 0.20)
    fold_row = {"test_year": test_year, "n_train": int(len(tra_idx)), "n_val": int(len(val_idx)), "n_test": int(len(test_idx)), "best_alpha": best_alpha, "best_val_hir_rrmse": best_val_rrmse, "final_test_hir_rrmse": rrmse_metric_np(y_test, test_pred), "final_test_hir_rmse": rmse_metric_np(y_test, test_pred), "test_success_rate_0.15": sr015, "test_success_rate_0.15_n": n015, "test_success_rate_0.20": sr020, "test_success_rate_0.20_n": n020}
    for threshold in precision_recall_thresholds:
        threshold_suffix = f"{threshold:.2f}"
        precision, recall, pred_n, actual_n = precision_recall_at_threshold(test_pred, y_test, threshold)
        fold_row[f"test_precision_{threshold_suffix}"] = precision
        fold_row[f"test_recall_{threshold_suffix}"] = recall
        fold_row[f"test_predicted_positive_n_{threshold_suffix}"] = pred_n
        fold_row[f"test_actual_positive_n_{threshold_suffix}"] = actual_n
    fold_results.append(fold_row)

fold_df = pd.DataFrame(fold_results).sort_values("test_year")
fold_df.to_csv(os.path.join(run_dir, "rolling_metrics.csv"), index=False)
with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "run_dir": run_dir,
            "seed": seed,
            "model": "RR-BLUP_rolling",
            "years": fold_df["test_year"].tolist(),
            "precision_recall_thresholds": precision_recall_thresholds,
            "mean_test_hir_rrmse": float(fold_df["final_test_hir_rrmse"].mean()),
            "std_test_hir_rrmse": float(fold_df["final_test_hir_rrmse"].std(ddof=0)),
            "mean_test_hir_rmse": float(fold_df["final_test_hir_rmse"].mean()),
            "std_test_hir_rmse": float(fold_df["final_test_hir_rmse"].std(ddof=0)),
            "mean_test_precision_0.10": float(fold_df["test_precision_0.10"].mean()),
            "mean_test_recall_0.10": float(fold_df["test_recall_0.10"].mean()),
            "mean_test_precision_0.15": float(fold_df["test_precision_0.15"].mean()),
            "mean_test_recall_0.15": float(fold_df["test_recall_0.15"].mean()),
            "mean_test_precision_0.20": float(fold_df["test_precision_0.20"].mean()),
            "mean_test_recall_0.20": float(fold_df["test_recall_0.20"].mean()),
        },
        f,
        indent=2,
        ensure_ascii=False,
    )
