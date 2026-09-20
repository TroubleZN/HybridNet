import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
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
print(f"随机种子: {seed}")

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

# Fixed effects: intercept is implicit; year dummies are explicit fixed effects.
X_fixed_all = pd.get_dummies(pheno["Year"].astype(str), prefix="year", drop_first=True).astype(float).to_numpy(dtype=np.float32)
Z_all = G.cpu().numpy()

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

X_fixed_train = X_fixed_all[train_valid_idx]
X_fixed_val = X_fixed_all[val_valid_idx]
X_fixed_test = X_fixed_all[test_valid_idx]
Z_train = Z_all[train_valid_idx]
Z_val = Z_all[val_valid_idx]
Z_test = Z_all[test_valid_idx]
y_train = y_all[train_valid_idx]
y_val = y_all[val_valid_idx]
y_test = y_all[test_valid_idx]

z_mean = Z_train.mean(axis=0, keepdims=True)
z_std = Z_train.std(axis=0, keepdims=True)
z_std[z_std < 1e-6] = 1.0
Z_train = (Z_train - z_mean) / z_std
Z_val = (Z_val - z_mean) / z_std
Z_test = (Z_test - z_mean) / z_std

alphas = [0.1, 1.0, 5.0, 10.0]
precision_recall_thresholds = [0.10, 0.15, 0.20]
history = []
best_coef = None
best_intercept = None
best_alpha = None
best_val_rrmse = float("inf")

for alpha in alphas:
    intercept, coef = fit_rrblup(X_fixed_train, Z_train, y_train, alpha)
    train_pred = predict_rrblup(X_fixed_train, Z_train, intercept, coef)
    val_pred = predict_rrblup(X_fixed_val, Z_val, intercept, coef)
    test_pred = predict_rrblup(X_fixed_test, Z_test, intercept, coef)
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
        best_intercept = intercept
        best_coef = coef.copy()
        best_alpha = alpha

run_dir = os.path.join("Result", f"rrblup_{seed}")
os.makedirs(run_dir, exist_ok=True)
print(f"结果输出目录: {run_dir}")

train_pred = predict_rrblup(X_fixed_train, Z_train, best_intercept, best_coef)
val_pred = predict_rrblup(X_fixed_val, Z_val, best_intercept, best_coef)
test_pred = predict_rrblup(X_fixed_test, Z_test, best_intercept, best_coef)

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
history_df.to_csv(os.path.join(run_dir, "history.csv"), index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=200)
axes[0].plot(history_df["alpha"], history_df["train_hir_rmse"], marker="o", label="train")
axes[0].plot(history_df["alpha"], history_df["val_hir_rmse"], marker="o", label="val")
axes[0].plot(history_df["alpha"], history_df["test_hir_rmse"], marker="o", label="test")
axes[0].set_xscale("log")
axes[0].set_title("RMSE vs Alpha")
axes[0].legend()
axes[0].grid(alpha=0.2)

axes[1].plot(history_df["alpha"], history_df["train_hir_rrmse"], marker="o", label="train")
axes[1].plot(history_df["alpha"], history_df["val_hir_rrmse"], marker="o", label="val")
axes[1].plot(history_df["alpha"], history_df["test_hir_rrmse"], marker="o", label="test")
axes[1].set_xscale("log")
axes[1].set_title("RRMSE vs Alpha")
axes[1].legend()
axes[1].grid(alpha=0.2)
fig.tight_layout()
fig.savefig(os.path.join(run_dir, "training_curves.png"))
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
plt.savefig(os.path.join(run_dir, "HIR.png"))
plt.close(fig)

plot_success(
    torch.cat([torch.tensor(train_pred), torch.tensor(test_pred)]),
    torch.cat([torch.tensor(y_train), torch.tensor(y_test)]),
    range(len(train_pred)),
    range(len(train_pred), len(train_pred) + len(test_pred)),
    fig_name=os.path.join(run_dir, "success_rate.png"),
)

np.save(os.path.join(run_dir, "coef.npy"), best_coef)
np.save(os.path.join(run_dir, "intercept.npy"), np.array([best_intercept]))
np.savez(
    os.path.join(run_dir, "checkpoint.npz"),
    fixed_columns=pd.get_dummies(pheno["Year"].astype(str), prefix="year", drop_first=True).columns.to_numpy(),
    z_mean=z_mean,
    z_std=z_std,
    train_idx=train_idx,
    val_idx=val_idx,
    test_idx=test_idx,
)

with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "run_dir": run_dir,
            "seed": seed,
            "model": "RR-BLUP",
            "best_alpha": best_alpha,
            "n_fixed_effects": int(X_fixed_train.shape[1]),
            "n_markers": int(Z_train.shape[1]),
            "precision_recall_thresholds": precision_recall_thresholds,
            "status": "finished",
        },
        f,
        indent=2,
        ensure_ascii=False,
    )

with open(os.path.join(run_dir, "metrics.txt"), "w", encoding="utf-8") as f:
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
