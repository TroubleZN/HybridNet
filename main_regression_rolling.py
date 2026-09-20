import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
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


seed = int(datetime.now().strftime("%Y%m%d")) % (2**31 - 1)
np.random.seed(seed % (2**32 - 1))
torch.manual_seed(seed)
print(f"随机种子: {seed}")

data_path = "Data/"
pheno_path = data_path + "01_Data_Pheno/"

PD = pd.read_csv(pheno_path + "Planting Dates/PD_2018-2024.csv")
env_ames = pd.read_excel(data_path + "Clean/ames.xlsx")
pheno = pd.read_csv(data_path + "Clean/phenotype_all.csv")
G_raw = pd.read_csv(data_path + "Clean/geno1.csv")

G_mat = torch.tensor(G_raw.iloc[:, 10:].to_numpy(), dtype=torch.float32).t() * 2
ID = G_raw.columns[10:]
id_to_idx = {sid: i for i, sid in enumerate(ID)}

pheno = pheno[pheno["SID"].isin(ID)].reset_index(drop=True)
N = len(pheno)

G = torch.zeros(N, G_mat.shape[1], dtype=torch.float32)
for i in range(N):
    G[i, :] = G_mat[id_to_idx[pheno["SID"].iloc[i]], :]

depths_log = np.log([12, 24, 50])
log4 = np.log(4)
moistures = env_ames[["SoilMoist2", "SoilMoist3", "SoilMoist4"]].values


def interp_row(row):
    return np.interp(log4, depths_log[::-1], row[::-1])


theta4_base = np.apply_along_axis(interp_row, axis=1, arr=moistures)
rain_mm = env_ames["SoilMoist1"] * 25.4
delta_theta = np.where(
    rain_mm < 1,
    0,
    np.where(
        rain_mm <= 20,
        0.5 * (1 - np.exp(-rain_mm / 3)) * (env_ames["SoilMoist2"] - theta4_base),
        env_ames["SoilMoist2"] - theta4_base,
    ),
)
env_ames["SoilMoist1"] = theta4_base + delta_theta

env_all = {}
for i in range(len(PD)):
    year = PD.iloc[i, 0]
    for rep in [1, 2]:
        dt = pd.Timestamp(PD.iloc[i, rep] + " 09:00:00")
        ht = dt + pd.Timedelta("130 Days")
        env_all[f"{year}_{rep}"] = env_ames[(env_ames.Time >= dt) & (env_ames.Time < ht)]

env = torch.zeros(N, 130 * 24, 13, dtype=torch.float32)
for i in range(N):
    key = f"{pheno.Year[i]}_{pheno.REP[i]}"
    env_i = env_all[key].iloc[:, [2, 9, 10, 11, 12, 5, 13, 14, 15, 3, 4, 6, 8]]
    env[i, :, :] = torch.tensor(env_i.to_numpy(), dtype=torch.float32)

mg = torch.zeros(N, 2, dtype=torch.float32)
mg[:, 1] = 31363 / 44560
for i in range(N):
    key = f"{pheno.Year[i]}_{pheno.REP[i]}"
    plant = env_all[key].iloc[0, 1]
    harvest = pd.Timestamp(str(pheno.Year[i]) + "-08-24 09:00:00")
    mg[i, 0] = (harvest - plant).days

HIR = torch.tensor(pheno["HIR"].to_numpy(), dtype=torch.float32) / 100
y_all = HIR.cpu().numpy()
year_np = pheno["Year"].to_numpy()
years = sorted(int(y) for y in pheno.loc[~pheno["HIR"].isna(), "Year"].unique())

window_size = 240
num_windows = env.shape[1] // window_size
env_w = env[:, :num_windows * window_size, :].reshape(env.shape[0], num_windows, window_size, env.shape[2])
env_features = torch.cat(
    [
        env_w.mean(dim=2),
        env_w.max(dim=2)[0],
        env_w.min(dim=2)[0],
    ],
    dim=2,
).reshape(env.shape[0], -1)

X_all = torch.cat([mg, G, env_features], dim=1).cpu().numpy()
run_dir = os.path.join("Result", f"regression_rolling_{seed}")
os.makedirs(run_dir, exist_ok=True)
print(f"结果输出目录: {run_dir}")

alphas = [0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
fold_results = []
precision_recall_thresholds = [0.10, 0.15, 0.20]

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

    X_train = X_all[tra_idx]
    y_train = y_all[tra_idx]
    X_val = X_all[val_idx]
    y_val = y_all[val_idx]
    X_test = X_all[test_idx]
    y_test = y_all[test_idx]

    feature_mean = X_train.mean(axis=0, keepdims=True)
    feature_std = X_train.std(axis=0, keepdims=True)
    feature_std[feature_std < 1e-6] = 1.0

    X_train = (X_train - feature_mean) / feature_std
    X_val = (X_val - feature_mean) / feature_std
    X_test = (X_test - feature_mean) / feature_std

    best_alpha = None
    best_val_rrmse = float("inf")
    best_model = None
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_rrmse = rrmse_metric_np(y_val, val_pred)
        if val_rrmse < best_val_rrmse:
            best_val_rrmse = val_rrmse
            best_alpha = alpha
            best_model = model

    test_pred = best_model.predict(X_test)
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
            "model": "RidgeRegression_rolling",
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
