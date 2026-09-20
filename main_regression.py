import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from Code.res_plot import plot_success


def rmse_metric_np(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def rrmse_metric_np(y_true, y_pred):
    top_k = max(1, int(len(y_true) * 0.2))
    top_idx = np.argpartition(y_true, -top_k)[-top_k:]
    return rmse_metric_np(y_true, y_pred) / max(float(np.mean(y_true[top_idx])), 1e-6)


def pearson_corr_np(y_true, y_pred):
    if len(y_true) < 2:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


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
y_all = HIR.cpu().numpy()

perm = torch.randperm(N)
test_size = int(N * 0.3)
test_idx = perm[:test_size].cpu().numpy()
train_idx = perm[test_size:].cpu().numpy()

perm2 = torch.randperm(len(train_idx))
validate_size = int(0.1 * N)
val_idx = train_idx[perm2[:validate_size].cpu().numpy()]
tra_idx = train_idx[perm2[validate_size:].cpu().numpy()]

train_valid_idx = tra_idx[~np.isnan(y_all[tra_idx])]
val_valid_idx = val_idx[~np.isnan(y_all[val_idx])]
test_valid_idx = test_idx[~np.isnan(y_all[test_idx])]

X_train = X_all[train_valid_idx]
y_train = y_all[train_valid_idx]
X_val = X_all[val_valid_idx]
y_val = y_all[val_valid_idx]
X_test = X_all[test_valid_idx]
y_test = y_all[test_valid_idx]

feature_mean = X_train.mean(axis=0, keepdims=True)
feature_std = X_train.std(axis=0, keepdims=True)
feature_std[feature_std < 1e-6] = 1.0

X_train = (X_train - feature_mean) / feature_std
X_val = (X_val - feature_mean) / feature_std
X_test = (X_test - feature_mean) / feature_std

alphas = [0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
precision_recall_thresholds = [0.10, 0.15, 0.20]
best_alpha = None
best_val_rrmse = float("inf")
history = []
best_model = None

for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    row = {
        "alpha": alpha,
        "train_hir_rmse": rmse_metric_np(y_train, train_pred),
        "val_hir_rmse": rmse_metric_np(y_val, val_pred),
        "test_hir_rmse": rmse_metric_np(y_test, test_pred),
        "train_hir_rrmse": rrmse_metric_np(y_train, train_pred),
        "val_hir_rrmse": rrmse_metric_np(y_val, val_pred),
        "test_hir_rrmse": rrmse_metric_np(y_test, test_pred),
    }
    history.append(row)
    if row["val_hir_rrmse"] < best_val_rrmse:
        best_val_rrmse = row["val_hir_rrmse"]
        best_alpha = alpha
        best_model = model

run_dir = os.path.join("Result", f"regression_{seed}")
os.makedirs(run_dir, exist_ok=True)
print(f"结果输出目录: {run_dir}")

train_pred = best_model.predict(X_train)
val_pred = best_model.predict(X_val)
test_pred = best_model.predict(X_test)

train_rmse = rmse_metric_np(y_train, train_pred)
val_rmse = rmse_metric_np(y_val, val_pred)
test_rmse = rmse_metric_np(y_test, test_pred)
train_rrmse = rrmse_metric_np(y_train, train_pred)
val_rrmse = rrmse_metric_np(y_val, val_pred)
test_rrmse = rrmse_metric_np(y_test, test_pred)

train_sr_015, train_n_015 = success_rate_at_threshold(train_pred, y_train, 0.15)
val_sr_015, val_n_015 = success_rate_at_threshold(val_pred, y_val, 0.15)
test_sr_015, test_n_015 = success_rate_at_threshold(test_pred, y_test, 0.15)
train_sr_020, train_n_020 = success_rate_at_threshold(train_pred, y_train, 0.20)
val_sr_020, val_n_020 = success_rate_at_threshold(val_pred, y_val, 0.20)
test_sr_020, test_n_020 = success_rate_at_threshold(test_pred, y_test, 0.20)
precision_recall_results = {"train": {}, "val": {}, "test": {}}
for threshold in precision_recall_thresholds:
    threshold_key = f"{int(round(threshold * 100)):02d}"
    precision_recall_results["train"][threshold_key] = precision_recall_at_threshold(train_pred, y_train, threshold)
    precision_recall_results["val"][threshold_key] = precision_recall_at_threshold(val_pred, y_val, threshold)
    precision_recall_results["test"][threshold_key] = precision_recall_at_threshold(test_pred, y_test, threshold)

history_df = pd.DataFrame(history)
history_path = os.path.join(run_dir, "history.csv")
history_df.to_csv(history_path, index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=200)
axes[0].plot(history_df["alpha"], history_df["train_hir_rmse"], marker="o", label="train")
axes[0].plot(history_df["alpha"], history_df["val_hir_rmse"], marker="o", label="val")
axes[0].plot(history_df["alpha"], history_df["test_hir_rmse"], marker="o", label="test")
axes[0].set_xscale("log")
axes[0].set_title("RMSE vs Alpha")
axes[0].set_xlabel("alpha")
axes[0].legend()
axes[0].grid(alpha=0.2)

axes[1].plot(history_df["alpha"], history_df["train_hir_rrmse"], marker="o", label="train")
axes[1].plot(history_df["alpha"], history_df["val_hir_rrmse"], marker="o", label="val")
axes[1].plot(history_df["alpha"], history_df["test_hir_rrmse"], marker="o", label="test")
axes[1].set_xscale("log")
axes[1].set_title("RRMSE vs Alpha")
axes[1].set_xlabel("alpha")
axes[1].legend()
axes[1].grid(alpha=0.2)
fig.tight_layout()
curve_path = os.path.join(run_dir, "training_curves.png")
fig.savefig(curve_path)
plt.close(fig)

fig, ax = plt.subplots()
plt.scatter(train_pred, y_train, label="train")
plt.scatter(test_pred, y_test, label="test")
plt.plot([0, 0.4], [0, 0.4], "red", linestyle="--")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(
    "Test RMSE: "
    + str(round(test_rmse, 3))
    + "  Test RRMSE: "
    + str(round(test_rrmse, 3))
    + "  Test Corr: "
    + str(round(pearson_corr_np(y_test, test_pred), 3))
)
plt.legend()
scatter_path = os.path.join(run_dir, "HIR.png")
plt.savefig(scatter_path)
plt.close(fig)

success_path = os.path.join(run_dir, "success_rate.png")
plot_success(
    torch.cat([torch.tensor(train_pred), torch.tensor(test_pred)]),
    torch.cat([torch.tensor(y_train), torch.tensor(y_test)]),
    range(len(train_pred)),
    range(len(train_pred), len(train_pred) + len(test_pred)),
    fig_name=success_path,
)

coef_path = os.path.join(run_dir, "coef.npy")
np.save(coef_path, best_model.coef_)
intercept_path = os.path.join(run_dir, "intercept.npy")
np.save(intercept_path, np.array([best_model.intercept_]))

checkpoint_path = os.path.join(run_dir, "checkpoint.npz")
np.savez(
    checkpoint_path,
    feature_mean=feature_mean,
    feature_std=feature_std,
    train_idx=train_idx,
    val_idx=val_idx,
    test_idx=test_idx,
    train_valid_idx=train_valid_idx,
    val_valid_idx=val_valid_idx,
    test_valid_idx=test_valid_idx,
)

config = {
    "run_dir": run_dir,
    "seed": seed,
    "model": "RidgeRegression",
    "best_alpha": best_alpha,
    "n_features": int(X_train.shape[1]),
    "precision_recall_thresholds": precision_recall_thresholds,
    "status": "finished",
}
config_path = os.path.join(run_dir, "config.json")
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

metrics_path = os.path.join(run_dir, "metrics.txt")
with open(metrics_path, "w", encoding="utf-8") as f:
    f.write(f"run_dir: {run_dir}\n")
    f.write("device: cpu\n")
    f.write(f"seed: {seed}\n")
    f.write(f"best_val_loss_during_training: {best_val_rrmse:.6f}\n")
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
