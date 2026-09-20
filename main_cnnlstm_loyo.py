import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from Code.cnnlstm_shared import StableHIRCNNLSTM


def pearson_corr(y_true, y_pred):
    y_true = y_true.float()
    y_pred = y_pred.float()
    vx = y_true - y_true.mean()
    vy = y_pred - y_pred.mean()
    denom = torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum())
    if denom <= 0:
        return torch.tensor(0.0, device=y_true.device)
    return (vx * vy).sum() / denom


def rmse_metric(y_true, y_pred):
    return torch.sqrt(((y_pred - y_true) ** 2).mean())


def rrmse_metric(y_true, y_pred):
    top_k = max(1, int(len(y_true) * 0.2))
    top_idx = torch.topk(y_true, top_k).indices
    return rmse_metric(y_true, y_pred) / y_true[top_idx].mean().clamp(min=1e-6)


def success_rate_at_threshold(pred, true, threshold):
    selected = pred > threshold
    count = int(selected.sum().item())
    if count == 0:
        return float("nan"), 0
    return float((true[selected] > threshold).float().mean().item()), count


def precision_recall_at_threshold(pred, true, threshold):
    pred_pos = pred > threshold
    actual_pos = true > threshold
    pred_n = int(pred_pos.sum().item())
    actual_n = int(actual_pos.sum().item())
    if pred_n == 0:
        precision = float("nan")
    else:
        precision = float(actual_pos[pred_pos].float().mean().item())
    if actual_n == 0:
        recall = float("nan")
    else:
        recall = float(pred_pos[actual_pos].float().mean().item())
    return precision, recall, pred_n, actual_n


def evaluate_model(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb_env, xb_g, xb_mg, yb in loader:
            pred = model(xb_env.to(device), xb_g.to(device), xb_mg.to(device))
            preds.append(pred)
            trues.append(yb.to(device))
    pred = torch.cat(preds)
    true = torch.cat(trues)
    return {
        "pred": pred,
        "true": true,
        "rmse": rmse_metric(true, pred),
        "rrmse": rrmse_metric(true, pred),
        "corr": pearson_corr(true, pred),
    }


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"当前设备: {device}")

seed = 20260314
torch.manual_seed(seed)
np.random.seed(seed % (2**32 - 1))
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
    year = pheno.Year[i]
    key = f"{pheno.Year[i]}_{pheno.REP[i]}"
    plant = env_all[key].iloc[0, 1]
    harvest = pd.Timestamp(str(year) + "-08-24 09:00:00")
    mg[i, 0] = (harvest - plant).days

HIR = torch.tensor(pheno["HIR"].to_numpy(), dtype=torch.float32) / 100
years = sorted(int(y) for y in pheno.loc[~pheno["HIR"].isna(), "Year"].unique())
year_tensor = torch.tensor(pheno["Year"].to_numpy(), dtype=torch.int64)

result_root = "Result"
os.makedirs(result_root, exist_ok=True)
run_name = f"cnnlstm_loyo_{seed}"
run_dir = os.path.join(result_root, run_name)
os.makedirs(run_dir, exist_ok=True)

batch_size = 32
lr_init = 1e-4
weight_decay = 1e-4
max_epochs = 300
early_stop_patience = 40
lr_patience = 12
min_delta = 5e-4
min_lr = 1e-6
min_val_samples = 24
precision_recall_thresholds = [0.10, 0.15, 0.20]

fold_results = []
for fold_id, test_year in enumerate(years, start=1):
    test_idx = torch.where(year_tensor == test_year)[0]
    test_idx = test_idx[~torch.isnan(HIR[test_idx])]
    train_pool = torch.where(year_tensor != test_year)[0]
    train_pool = train_pool[~torch.isnan(HIR[train_pool])]
    if len(test_idx) == 0 or len(train_pool) <= min_val_samples:
        continue

    fold_seed = seed + fold_id
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed % (2**32 - 1))
    perm = torch.randperm(len(train_pool))
    val_size = max(min_val_samples, int(0.1 * len(train_pool)))
    val_idx = train_pool[perm[:val_size]]
    tra_idx = train_pool[perm[val_size:]]

    env_mean = env[tra_idx].mean(dim=(0, 1), keepdim=True)
    env_std = env[tra_idx].std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
    G_mean = G[tra_idx].mean(dim=0, keepdim=True)
    G_std = G[tra_idx].std(dim=0, keepdim=True).clamp(min=1e-6)
    mg_mean = mg[tra_idx].mean(dim=0, keepdim=True)
    mg_std = mg[tra_idx].std(dim=0, keepdim=True).clamp(min=1e-6)

    env_norm = (env - env_mean) / env_std
    G_norm = (G - G_mean) / G_std
    mg_norm = (mg - mg_mean) / mg_std

    train_ds = TensorDataset(env_norm[tra_idx], G_norm[tra_idx], mg_norm[tra_idx], HIR[tra_idx])
    val_ds = TensorDataset(env_norm[val_idx], G_norm[val_idx], mg_norm[val_idx], HIR[val_idx])
    test_ds = TensorDataset(env_norm[test_idx], G_norm[test_idx], mg_norm[test_idx], HIR[test_idx])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = StableHIRCNNLSTM(geno_dim=G.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    lr = lr_init
    fail = 0
    best_val_rrmse = 1e9
    best_state = None

    for epoch in range(max_epochs):
        model.train()
        for xb_env, xb_g, xb_mg, yb in train_dl:
            optimizer.zero_grad()
            pred = model(xb_env.to(device), xb_g.to(device), xb_mg.to(device))
            loss = loss_fn(pred, yb.to(device))
            loss.backward()
            optimizer.step()

        val_eval = evaluate_model(model, val_dl, device)
        improved = (best_val_rrmse - val_eval["rrmse"].item()) > min_delta
        if improved:
            best_val_rrmse = val_eval["rrmse"].item()
            fail = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            fail += 1
            if fail % lr_patience == 0 and lr > min_lr:
                lr = max(lr * 0.5, min_lr)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            if fail >= early_stop_patience:
                break

    model.load_state_dict(best_state)
    train_eval = evaluate_model(model, train_dl, device)
    val_eval = evaluate_model(model, val_dl, device)
    test_eval = evaluate_model(model, test_dl, device)
    test_sr_015, test_n_015 = success_rate_at_threshold(test_eval["pred"], test_eval["true"], 0.15)
    test_sr_020, test_n_020 = success_rate_at_threshold(test_eval["pred"], test_eval["true"], 0.20)
    fold_row = {
        "test_year": test_year,
        "n_train": int(len(tra_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "best_val_hir_rrmse": best_val_rrmse,
        "final_train_hir_rrmse": train_eval["rrmse"].item(),
        "final_val_hir_rrmse": val_eval["rrmse"].item(),
        "final_test_hir_rrmse": test_eval["rrmse"].item(),
        "final_train_hir_rmse": train_eval["rmse"].item(),
        "final_val_hir_rmse": val_eval["rmse"].item(),
        "final_test_hir_rmse": test_eval["rmse"].item(),
        "test_success_rate_0.15": test_sr_015,
        "test_success_rate_0.15_n": test_n_015,
        "test_success_rate_0.20": test_sr_020,
        "test_success_rate_0.20_n": test_n_020,
    }
    for threshold in precision_recall_thresholds:
        threshold_suffix = f"{threshold:.2f}"
        precision, recall, pred_n, actual_n = precision_recall_at_threshold(test_eval["pred"], test_eval["true"], threshold)
        fold_row[f"test_precision_{threshold_suffix}"] = precision
        fold_row[f"test_recall_{threshold_suffix}"] = recall
        fold_row[f"test_predicted_positive_n_{threshold_suffix}"] = pred_n
        fold_row[f"test_actual_positive_n_{threshold_suffix}"] = actual_n
    fold_results.append(fold_row)

fold_df = pd.DataFrame(fold_results).sort_values("test_year")
fold_df.to_csv(os.path.join(run_dir, "loyo_metrics.csv"), index=False)
summary = {
    "run_dir": run_dir,
    "seed": seed,
    "device": str(device),
    "model": "CNNLSTM_LOYO",
    "years": fold_df["test_year"].tolist(),
    "mean_test_hir_rrmse": float(fold_df["final_test_hir_rrmse"].mean()),
    "std_test_hir_rrmse": float(fold_df["final_test_hir_rrmse"].std(ddof=0)),
    "mean_test_hir_rmse": float(fold_df["final_test_hir_rmse"].mean()),
    "std_test_hir_rmse": float(fold_df["final_test_hir_rmse"].std(ddof=0)),
    "precision_recall_thresholds": precision_recall_thresholds,
    "mean_test_precision_0.10": float(fold_df["test_precision_0.10"].mean()),
    "mean_test_recall_0.10": float(fold_df["test_recall_0.10"].mean()),
    "mean_test_precision_0.15": float(fold_df["test_precision_0.15"].mean()),
    "mean_test_recall_0.15": float(fold_df["test_recall_0.15"].mean()),
    "mean_test_precision_0.20": float(fold_df["test_precision_0.20"].mean()),
    "mean_test_recall_0.20": float(fold_df["test_recall_0.20"].mean()),
}
with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
