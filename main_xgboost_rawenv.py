import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from Code.res_plot import plot_success


def pearson_corr_np(y_true, y_pred):
    if len(y_true) < 2:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def rmse_metric_np(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def rrmse_metric_np(y_true, y_pred):
    top_ratio = 0.2
    top_k = max(1, int(len(y_true) * top_ratio))
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
    rep1 = PD.iloc[i, 1]
    dt1 = pd.Timestamp(rep1 + " 09:00:00")
    ht1 = dt1 + pd.Timedelta("130 Days")
    env_all[f"{year}_1"] = env_ames[(env_ames.Time >= dt1) & (env_ames.Time < ht1)]

    rep2 = PD.iloc[i, 2]
    dt2 = pd.Timestamp(rep2 + " 09:00:00")
    ht2 = dt2 + pd.Timedelta("130 Days")
    env_all[f"{year}_2"] = env_ames[(env_ames.Time >= dt2) & (env_ames.Time < ht2)]

env = torch.zeros(N, 130 * 24, 13, dtype=torch.float32)
for i in range(N):
    key = f"{pheno.Year[i]}_{pheno.REP[i]}"
    env_i = env_all[key].iloc[:, [2, 9, 10, 11, 12, 5, 13, 14, 15, 3, 4, 6, 8]]
    env[i, :, :] = torch.tensor(env_i.to_numpy(), dtype=torch.float32)

mg = torch.zeros(N, 2, dtype=torch.float32)
mg[:, 1] = 31363 / 44560
for i in range(N):
    year = pheno.Year[i]
    key = f"{pheno.Year[i]}_{pheno.REP[i]}"
    plant = env_all[key].iloc[0, 1]
    harvest = pd.Timestamp(str(year) + "-08-24 09:00:00")
    mg[i, 0] = (harvest - plant).days

HIR = torch.tensor(pheno["HIR"].to_numpy(), dtype=torch.float32) / 100
label_mask = ~torch.isnan(HIR)

perm = torch.randperm(N)
test_size = int(N * 0.3)
test_idx = perm[:test_size]
train_idx = perm[test_size:]

perm2 = torch.randperm(len(train_idx))
validate_size = int(0.1 * N)
val_idx = train_idx[perm2[:validate_size]]
tra_idx = train_idx[perm2[validate_size:]]

env_features = env.reshape(env.shape[0], -1)
X_all = torch.cat([mg, G, env_features], dim=1).cpu().numpy()
y_all = HIR.cpu().numpy()

train_valid_idx = tra_idx[label_mask[tra_idx]].cpu().numpy()
val_valid_idx = val_idx[label_mask[val_idx]].cpu().numpy()
test_valid_idx = test_idx[label_mask[test_idx]].cpu().numpy()

X_train = np.ascontiguousarray(X_all[train_valid_idx], dtype=np.float32)
y_train = np.ascontiguousarray(y_all[train_valid_idx], dtype=np.float32)
X_val = np.ascontiguousarray(X_all[val_valid_idx], dtype=np.float32)
y_val = np.ascontiguousarray(y_all[val_valid_idx], dtype=np.float32)
X_test = np.ascontiguousarray(X_all[test_valid_idx], dtype=np.float32)
y_test = np.ascontiguousarray(y_all[test_valid_idx], dtype=np.float32)

result_root = "Result"
os.makedirs(result_root, exist_ok=True)
run_name = f"xgboost_rawenv_{seed}"
run_dir = os.path.join(result_root, run_name)
os.makedirs(run_dir, exist_ok=True)
print(f"结果输出目录: {run_dir}")

metrics_path = os.path.join(run_dir, "metrics.txt")
config_path = os.path.join(run_dir, "config.json")
history_path = os.path.join(run_dir, "history.csv")
curve_path = os.path.join(run_dir, "training_curves.png")
scatter_path = os.path.join(run_dir, "HIR.png")
success_path = os.path.join(run_dir, "success_rate.png")
model_path = os.path.join(run_dir, "model.json")
checkpoint_path = os.path.join(run_dir, "checkpoint.npz")
precision_recall_thresholds = [0.10, 0.15, 0.20]

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
    "verbosity": 1,
}

model = xgb.XGBRegressor(**params)
model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_val, y_val), (X_test, y_test)],
    verbose=False,
)

evals_result = model.evals_result()
train_rmse_hist = evals_result["validation_0"]["rmse"]
val_rmse_hist = evals_result["validation_1"]["rmse"]
test_rmse_hist = evals_result["validation_2"]["rmse"]

y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

train_rrmse = rrmse_metric_np(y_train, y_train_pred)
val_rrmse = rrmse_metric_np(y_val, y_val_pred)
test_rrmse = rrmse_metric_np(y_test, y_test_pred)
train_rmse = rmse_metric_np(y_train, y_train_pred)
val_rmse = rmse_metric_np(y_val, y_val_pred)
test_rmse = rmse_metric_np(y_test, y_test_pred)

train_sr_015, train_n_015 = success_rate_at_threshold(y_train_pred, y_train, 0.15)
val_sr_015, val_n_015 = success_rate_at_threshold(y_val_pred, y_val, 0.15)
test_sr_015, test_n_015 = success_rate_at_threshold(y_test_pred, y_test, 0.15)
train_sr_020, train_n_020 = success_rate_at_threshold(y_train_pred, y_train, 0.20)
val_sr_020, val_n_020 = success_rate_at_threshold(y_val_pred, y_val, 0.20)
test_sr_020, test_n_020 = success_rate_at_threshold(y_test_pred, y_test, 0.20)

precision_recall_results = {"train": {}, "val": {}, "test": {}}
for threshold in precision_recall_thresholds:
    threshold_key = f"{int(round(threshold * 100)):02d}"
    precision_recall_results["train"][threshold_key] = precision_recall_at_threshold(y_train_pred, y_train, threshold)
    precision_recall_results["val"][threshold_key] = precision_recall_at_threshold(y_val_pred, y_val, threshold)
    precision_recall_results["test"][threshold_key] = precision_recall_at_threshold(y_test_pred, y_test, threshold)

history_df = pd.DataFrame(
    {
        "epoch": np.arange(1, len(train_rmse_hist) + 1),
        "train_hir_rmse": train_rmse_hist,
        "val_hir_rmse": val_rmse_hist,
        "test_hir_rmse": test_rmse_hist,
    }
)
history_df["train_hir_rrmse"] = history_df["train_hir_rmse"] / max(float(np.mean(np.sort(y_train)[-max(1, int(len(y_train) * 0.2)):])), 1e-6)
history_df["val_hir_rrmse"] = history_df["val_hir_rmse"] / max(float(np.mean(np.sort(y_val)[-max(1, int(len(y_val) * 0.2)):])), 1e-6)
history_df["test_hir_rrmse"] = history_df["test_hir_rmse"] / max(float(np.mean(np.sort(y_test)[-max(1, int(len(y_test) * 0.2)):])), 1e-6)
history_df.to_csv(history_path, index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=200)
axes[0].plot(history_df["epoch"], history_df["train_hir_rmse"], label="train")
axes[0].plot(history_df["epoch"], history_df["val_hir_rmse"], label="val")
axes[0].plot(history_df["epoch"], history_df["test_hir_rmse"], label="test")
axes[0].set_title("HIR RMSE")
axes[0].set_xlabel("Boosting Round")
axes[0].legend()
axes[0].grid(alpha=0.2)

axes[1].plot(history_df["epoch"], history_df["train_hir_rrmse"], label="train")
axes[1].plot(history_df["epoch"], history_df["val_hir_rrmse"], label="val")
axes[1].plot(history_df["epoch"], history_df["test_hir_rrmse"], label="test")
axes[1].set_title("HIR RRMSE")
axes[1].set_xlabel("Boosting Round")
axes[1].legend()
axes[1].grid(alpha=0.2)
fig.tight_layout()
fig.savefig(curve_path)
plt.close(fig)

fig, ax = plt.subplots()
plt.scatter(y_train_pred, y_train, label="train")
plt.scatter(y_test_pred, y_test, label="test")
plt.plot([0, 0.4], [0, 0.4], "red", linestyle="--")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(
    "Test RMSE: "
    + str(round(test_rmse, 3))
    + "  Test RRMSE: "
    + str(round(test_rrmse, 3))
    + "  Test Corr: "
    + str(round(pearson_corr_np(y_test, y_test_pred), 3))
)
plt.legend()
plt.savefig(scatter_path)
plt.close(fig)

plot_success(
    torch.cat([torch.tensor(y_train_pred), torch.tensor(y_test_pred)]),
    torch.cat([torch.tensor(y_train), torch.tensor(y_test)]),
    range(len(y_train_pred)),
    range(len(y_train_pred), len(y_train_pred) + len(y_test_pred)),
    fig_name=success_path,
)

model.get_booster().save_model(model_path)
np.savez(
    checkpoint_path,
    train_idx=train_idx.cpu().numpy(),
    val_idx=val_idx.cpu().numpy(),
    test_idx=test_idx.cpu().numpy(),
    train_valid_idx=train_valid_idx,
    val_valid_idx=val_valid_idx,
    test_valid_idx=test_valid_idx,
)

config = {
    "run_dir": run_dir,
    "timestamp": run_name,
    "seed": seed,
    "model": "XGBoost_rawenv",
    "env_feature_mode": "raw_flatten",
    "env_feature_dim": int(env_features.shape[1]),
    "total_feature_dim": int(X_all.shape[1]),
    "params": params,
    "precision_recall_thresholds": precision_recall_thresholds,
    "status": "finished",
}

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

with open(metrics_path, "w", encoding="utf-8") as f:
    f.write(f"run_dir: {run_dir}\n")
    f.write("device: cpu\n")
    f.write(f"seed: {seed}\n")
    f.write("env_feature_mode: raw_flatten\n")
    f.write(f"env_feature_dim: {int(env_features.shape[1])}\n")
    f.write(f"total_feature_dim: {int(X_all.shape[1])}\n")
    f.write(f"best_val_loss_during_training: {min(history_df['val_hir_rrmse']):.6f}\n")
    f.write(f"final_train_HIR_RRMSE: {train_rrmse:.6f}\n")
    f.write(f"final_train_HIR_RMSE: {train_rmse:.6f}\n")
    f.write(f"final_val_HIR_RRMSE: {val_rrmse:.6f}\n")
    f.write(f"final_val_HIR_RMSE: {val_rmse:.6f}\n")
    f.write(f"final_test_HIR_RRMSE: {test_rrmse:.6f}\n")
    f.write(f"final_test_HIR_RMSE: {test_rmse:.6f}\n")
    f.write(f"train_success_rate_0.15: {train_sr_015:.6f} (n={train_n_015})\n")
    f.write(f"val_success_rate_0.15: {val_sr_015:.6f} (n={val_n_015})\n")
    f.write(f"test_success_rate_0.15: {test_sr_015:.6f} (n={test_n_015})\n")
    f.write(f"train_success_rate_0.20: {train_sr_020:.6f} (n={train_n_020})\n")
    f.write(f"val_success_rate_0.20: {val_sr_020:.6f} (n={val_n_020})\n")
    f.write(f"test_success_rate_0.20: {test_sr_020:.6f} (n={test_n_020})\n")
    for split in ["train", "val", "test"]:
        for threshold in precision_recall_thresholds:
            threshold_key = f"{int(round(threshold * 100)):02d}"
            precision, recall, pred_n, actual_n = precision_recall_results[split][threshold_key]
            f.write(f"{split}_precision_{threshold:.2f}: {precision:.6f} (pred_n={pred_n}, actual_n={actual_n})\n")
            f.write(f"{split}_recall_{threshold:.2f}: {recall:.6f} (pred_n={pred_n}, actual_n={actual_n})\n")
