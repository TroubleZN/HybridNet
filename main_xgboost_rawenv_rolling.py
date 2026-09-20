import json
import os

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import mean_squared_error


def rmse_metric_np(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def rrmse_metric_np(y_true, y_pred):
    top_k = max(1, int(len(y_true) * 0.2))
    top_idx = np.argpartition(y_true, -top_k)[-top_k:]
    top_true_mean = max(float(np.mean(y_true[top_idx])), 1e-6)
    return rmse_metric_np(y_true, y_pred) / top_true_mean


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


seed = 20260314
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
theta4_base = np.apply_along_axis(lambda row: np.interp(log4, depths_log[::-1], row[::-1]), axis=1, arr=moistures)
rain_mm = env_ames["SoilMoist1"] * 25.4
delta_theta = np.where(rain_mm < 1, 0, np.where(rain_mm <= 20, 0.5 * (1 - np.exp(-rain_mm / 3)) * (env_ames["SoilMoist2"] - theta4_base), env_ames["SoilMoist2"] - theta4_base))
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
years = sorted(int(y) for y in pheno.loc[~pheno["HIR"].isna(), "Year"].unique())
year_np = pheno["Year"].to_numpy()

env_features = env.reshape(env.shape[0], -1)
X_all = torch.cat([mg, G, env_features], dim=1).cpu().numpy()
y_all = HIR.cpu().numpy()

params = {
    "n_estimators": 600,
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": seed,
    "n_jobs": 1,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "verbosity": 0,
}

run_dir = os.path.join("Result", f"xgboost_rawenv_rolling_{seed}")
os.makedirs(run_dir, exist_ok=True)
print(f"结果输出目录: {run_dir}")

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

    X_train = np.ascontiguousarray(X_all[tra_idx], dtype=np.float32)
    y_train = np.ascontiguousarray(y_all[tra_idx], dtype=np.float32)
    X_val = np.ascontiguousarray(X_all[val_idx], dtype=np.float32)
    y_val = np.ascontiguousarray(y_all[val_idx], dtype=np.float32)
    X_test = np.ascontiguousarray(X_all[test_idx], dtype=np.float32)
    y_test = np.ascontiguousarray(y_all[test_idx], dtype=np.float32)

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val), (X_test, y_test)], verbose=False)
    y_pred = model.predict(X_test)

    sr015, n015 = success_rate_at_threshold(y_pred, y_test, 0.15)
    sr020, n020 = success_rate_at_threshold(y_pred, y_test, 0.20)
    val_rmse_best = min(model.evals_result()["validation_1"]["rmse"])
    val_top_true_mean = max(float(np.mean(np.sort(y_val)[-max(1, int(len(y_val) * 0.2)):])), 1e-6)

    fold_row = {
        "test_year": test_year,
        "n_train": int(len(tra_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "best_val_hir_rrmse": float(val_rmse_best / val_top_true_mean),
        "final_test_hir_rrmse": rrmse_metric_np(y_test, y_pred),
        "final_test_hir_rmse": rmse_metric_np(y_test, y_pred),
        "test_success_rate_0.15": sr015,
        "test_success_rate_0.15_n": n015,
        "test_success_rate_0.20": sr020,
        "test_success_rate_0.20_n": n020,
    }
    for threshold in precision_recall_thresholds:
        threshold_suffix = f"{threshold:.2f}"
        precision, recall, pred_n, actual_n = precision_recall_at_threshold(y_pred, y_test, threshold)
        fold_row[f"test_precision_{threshold_suffix}"] = precision
        fold_row[f"test_recall_{threshold_suffix}"] = recall
        fold_row[f"test_predicted_positive_n_{threshold_suffix}"] = pred_n
        fold_row[f"test_actual_positive_n_{threshold_suffix}"] = actual_n
    fold_results.append(fold_row)

fold_df = pd.DataFrame(fold_results).sort_values("test_year")
rolling_metrics_path = os.path.join(run_dir, "rolling_metrics.csv")
summary_path = os.path.join(run_dir, "summary.json")
config_path = os.path.join(run_dir, "config.json")

fold_df.to_csv(rolling_metrics_path, index=False)
summary = {
    "run_dir": run_dir,
    "seed": seed,
    "model": "XGBoost_rawenv_rolling",
    "env_feature_mode": "raw_flatten",
    "env_feature_dim": int(env_features.shape[1]),
    "total_feature_dim": int(X_all.shape[1]),
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
}
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

config = {
    "run_dir": run_dir,
    "seed": seed,
    "model": "XGBoost_rawenv_rolling",
    "env_feature_mode": "raw_flatten",
    "env_feature_dim": int(env_features.shape[1]),
    "total_feature_dim": int(X_all.shape[1]),
    "params": params,
    "precision_recall_thresholds": precision_recall_thresholds,
    "status": "finished",
}
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
