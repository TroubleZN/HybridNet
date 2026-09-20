import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from Code.DHCorn import dhcorn

from Code.make_g import make_treno
from Code.res_plot import plot_success


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"当前设备: {device}")

seed = int(datetime.now().strftime("%Y%m%d")) % (2**31 - 1)
torch.manual_seed(seed)
np.random.seed(seed % (2**32 - 1))
print(f"随机种子: {seed}")

data_path = "Data/"
pheno_path = data_path + "01_Data_Pheno/"

PD = pd.read_csv(pheno_path + "Planting Dates/PD_2018-2024.csv")
env_ames = pd.read_excel(data_path + "Clean/ames.xlsx")
pheno = pd.read_csv(data_path + "Clean/phenotype_all.csv")
G_raw = pd.read_csv(data_path + "Clean/geno1.csv")

G_mat = torch.tensor(G_raw.iloc[:, 10:].to_numpy()).t() * 2
ID = G_raw.columns[10:]
id_to_idx = {sid: i for i, sid in enumerate(ID)}

pheno = pheno[pheno["SID"].isin(ID)].reset_index(drop=True)
N = len(pheno)

G = torch.zeros(len(pheno), G_mat.shape[1])
for i in range(N):
    G[i, :] = G_mat[id_to_idx[pheno["SID"].iloc[i]], :]
G = G.to(device)

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

env = torch.zeros(N, 130 * 24, 13)
for i in range(N):
    key = f"{pheno.Year[i]}_{pheno.REP[i]}"
    env_i = env_all[key].iloc[:, [2, 9, 10, 11, 12, 5, 13, 14, 15, 3, 4, 6, 8]]
    env[i, :, :] = torch.tensor(env_i.to_numpy())
env = env.to(device)

mg = torch.zeros(N, 2)
mg[:, 1] = 31363 / 44560
for i in range(N):
    year = pheno.Year[i]
    key = f"{pheno.Year[i]}_{pheno.REP[i]}"
    plant = env_all[key].iloc[0, 1]
    harvest = pd.Timestamp(str(year) + "-08-24 09:00:00")
    mg[i, 0] = (harvest - plant).days
mg = mg.to(device)

treno, Utreno, Ltreno = make_treno()
Utreno = torch.tensor(Utreno, dtype=torch.float32, device=device)
Ltreno = torch.tensor(Ltreno, dtype=torch.float32, device=device)

Height = torch.tensor(pheno["PH"].to_numpy(), dtype=torch.float32, device=device)
Height[Height == 0] = torch.nan
Height_mat = torch.full((N, 130), torch.nan, device=device)
for i in range(N):
    if ~Height[i].isnan():
        Height_mat[i, int(mg[i, 0])] = Height[i]

HIR = torch.tensor(pheno["HIR"].to_numpy(), dtype=torch.float32, device=device) / 100
HIR_mat = torch.full((N, 130), torch.nan, device=device)
for i in range(N):
    if ~HIR[i].isnan():
        HIR_mat[i, int(mg[i, 0])] = HIR[i]

FT = torch.tensor(pheno["FT"].to_numpy(), dtype=torch.float32, device=device)
FT_day = FT / 24


def pearson_corr(y_true, y_pred):
    y_true = y_true.float()
    y_pred = y_pred.float()
    vx = y_true - y_true.mean()
    vy = y_pred - y_pred.mean()
    return (vx * vy).sum() / (torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum()))


def loss_fn(pred, true, method="RRMSE"):
    mask = ~torch.isnan(true) & ~torch.isnan(pred)
    pred = pred[mask]
    true = true[mask]

    if pred.numel() == 0:
        return torch.tensor(0.0, device=G.device, requires_grad=True)
    if method == "Pearson":
        return 1 - pearson_corr(true, pred)
    if method == "RMSE":
        return ((pred - true) ** 2).mean().sqrt()
    if method == "RRMSE3":
        return ((pred - true) ** 2).mean().sqrt() / pred.mean().detach().clamp(min=1e-6)

    top_ratio = 0.2
    top_k = max(1, int(len(true) * top_ratio))
    top_idx = torch.topk(true, top_k).indices
    rmse = ((pred - true) ** 2).mean().sqrt()
    top_pred_mean = true[top_idx].mean().detach().clamp(min=1e-6)
    return rmse / top_pred_mean


def predicted_ft_day(stage_curve, temperature=20.0):
    stage_signal = stage_curve[:, 2, :]
    day_axis = torch.arange(stage_signal.shape[1], device=stage_signal.device, dtype=stage_signal.dtype)
    weights = torch.softmax(stage_signal * temperature, dim=1)
    return (weights * day_axis.unsqueeze(0)).sum(dim=1)


def success_rate_at_threshold(pred, true, threshold):
    mask = ~torch.isnan(true) & ~torch.isnan(pred)
    pred = pred[mask]
    true = true[mask]
    if pred.numel() == 0:
        return float("nan"), 0
    selected = pred > threshold
    count = int(selected.sum().item())
    if count == 0:
        return float("nan"), 0
    return (true[selected] > threshold).float().mean().item(), count


perm = torch.randperm(N)
test_size = int(N * 0.3)
test_idx = perm[:test_size]
train_idx = perm[test_size:]

perm2 = torch.randperm(len(train_idx))
validate_size = int(0.1 * N)
val_idx = train_idx[perm2[:validate_size]]
tra_idx = train_idx[perm2[validate_size:]]

beta = torch.randn(G.shape[1], 2, 4 * 89, device=device) / 100
beta.requires_grad_(True)

weight = torch.tensor([0.05, 0.90, 0.00, 0.05], device=device)
lr = 1e-4
max_epochs = 500
early_stop_patience = 30
lr_patience = 10
min_delta = 5e-4
min_lr = 1e-6
fail = 0
loss_val_best = 1e9
beta_best = beta.detach().clone()

result_root = "Result"
os.makedirs(result_root, exist_ok=True)
run_name = f"hir_focus_{seed}"
run_dir = os.path.join(result_root, run_name)
os.makedirs(run_dir, exist_ok=True)
print(f"结果输出目录: {run_dir}")

HIRfig_name = os.path.join(run_dir, "HIR.png")
precisionfig_name = os.path.join(run_dir, "precision.png")
metrics_path = os.path.join(run_dir, "metrics.txt")
config_path = os.path.join(run_dir, "config.json")

optimizer = torch.optim.Adam([beta], lr=lr)

for epoch in range(max_epochs):
    optimizer.zero_grad()
    perm2 = torch.randperm(len(tra_idx))
    train_size = int(0.1 * len(tra_idx))
    tra_idx2 = tra_idx[perm2[:train_size]]

    g = torch.matmul(G, beta[:, 0, :]).reshape(N, 4, 89) + torch.matmul((G == 1) * 1.0, beta[:, 1, :]).reshape(N, 4, 89)
    geno = 1 / (1 + torch.exp(-g)) * (Utreno - Ltreno) + Ltreno
    ph = dhcorn(env, mg, geno, device=device)

    loss_HIR = loss_fn(ph["HIR"][tra_idx2], HIR_mat[tra_idx2], method="RRMSE")
    RMSE_HIR = loss_fn(ph["HIR"][tra_idx2], HIR_mat[tra_idx2], method="RMSE")
    loss_PH = loss_fn(ph["Height"][tra_idx2], Height_mat[tra_idx2], method="RRMSE3")
    loss_TL = torch.tensor(0.0, device=device)
    loss_FT = loss_fn(predicted_ft_day(ph["stage"][tra_idx2]), FT_day[tra_idx2], method="RRMSE3")
    loss = torch.dot(weight, torch.stack([loss_PH, loss_HIR, loss_TL, loss_FT])) + beta.square().mean() * 10
    loss.backward()
    optimizer.step()

    print(
        f"Epoch {epoch + 1:02d} train, "
        f"RRMSE = {loss.item():.4f}, "
        f"\tFT = {loss_FT.item():.4f}, "
        f"\tHIR = {loss_HIR.item():.4f}, "
        f"\tHeight = {loss_PH.item():.4f}, "
        f"\tTassel = {loss_TL.item():.4f}, "
        f"\t Fail = {fail:02d}"
    )

    with torch.no_grad():
        loss_HIR_val = loss_fn(ph["HIR"][val_idx], HIR_mat[val_idx], method="RRMSE")
        RMSE_HIR_val = loss_fn(ph["HIR"][val_idx], HIR_mat[val_idx], method="RMSE")
        loss_PH_val = loss_fn(ph["Height"][val_idx], Height_mat[val_idx], method="RRMSE3")
        loss_TL_val = torch.tensor(0.0, device=device)
        loss_FT_val = loss_fn(predicted_ft_day(ph["stage"][val_idx]), FT_day[val_idx], method="RRMSE3")
        loss_val = torch.dot(weight, torch.stack([loss_PH_val, loss_HIR_val, loss_TL_val, loss_FT_val])) + beta.square().mean() * 10

        loss_HIR_test = loss_fn(ph["HIR"][test_idx], HIR_mat[test_idx], method="RRMSE")
        RMSE_HIR_test = loss_fn(ph["HIR"][test_idx], HIR_mat[test_idx], method="RMSE")
        loss_PH_test = loss_fn(ph["Height"][test_idx], Height_mat[test_idx], method="RRMSE3")
        loss_TL_test = torch.tensor(0.0, device=device)
        loss_FT_test = loss_fn(predicted_ft_day(ph["stage"][test_idx]), FT_day[test_idx], method="RRMSE3")
        loss_test = torch.dot(weight, torch.stack([loss_PH_test, loss_HIR_test, loss_TL_test, loss_FT_test])) + beta.square().mean() * 10

        print(f"Epoch {epoch + 1:02d}   val, RRMSE = {loss_val.item():.4f}, \tFT = {loss_FT_val.item():.4f}, \tHIR = {loss_HIR_val.item():.4f}, \tHeight = {loss_PH_val.item():.4f}, \tTassel = {loss_TL_val.item():.4f}")
        print(f"Epoch {epoch + 1:02d}  test, RRMSE = {loss_test.item():.4f}, \tFT = {loss_FT_test.item():.4f}, \tHIR = {loss_HIR_test.item():.4f}, \tHeight = {loss_PH_test.item():.4f}, \tTassel = {loss_TL_test.item():.4f}\n")

        improved = (loss_val_best - loss_val.item()) > min_delta
        if improved:
            loss_val_best = loss_val
            beta_best = beta.detach().clone()
            fail = 0

            fig, ax = plt.subplots()
            plt.scatter(ph["HIR"][train_idx][~HIR_mat[train_idx].isnan()].detach().cpu().numpy(), HIR_mat[train_idx][~HIR_mat[train_idx].isnan()].detach().cpu().numpy(), label="train")
            plt.scatter(ph["HIR"][test_idx][~HIR_mat[test_idx].isnan()].detach().cpu().numpy(), HIR_mat[test_idx][~HIR_mat[test_idx].isnan()].detach().cpu().numpy(), label="test")
            plt.plot([0, 0.4], [0, 0.4], "red", linestyle="--")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            corr_test = round(loss_fn(ph["HIR"][test_idx], HIR_mat[test_idx], method="Pearson").item(), 3)
            plt.title(
                "Test RMSE: "
                + str(round(RMSE_HIR_test.item(), 3))
                + "  Test RRMSE: "
                + str(round(loss_HIR_test.item(), 3))
                + "  Test Corr: "
                + str(1 - corr_test)
            )
            plt.legend()
            plt.savefig(HIRfig_name, dpi=400)
            plt.close(fig)
            plot_success(ph["HIR"], HIR_mat, train_idx, test_idx, fig_name=precisionfig_name)
        else:
            fail += 1
            if fail % lr_patience == 0 and lr > min_lr:
                lr = max(lr * 0.5, min_lr)
                optimizer = torch.optim.Adam([beta], lr=lr)
                print("lr: " + str(lr))
            if fail >= early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

with torch.no_grad():
    beta.copy_(beta_best)
    g = torch.matmul(G, beta[:, 0, :]).reshape(N, 4, 89) + torch.matmul((G == 1) * 1.0, beta[:, 1, :]).reshape(N, 4, 89)
    geno = 1 / (1 + torch.exp(-g)) * (Utreno - Ltreno) + Ltreno
    ph = dhcorn(env, mg, geno, device=device)

    final_train_loss_HIR = loss_fn(ph["HIR"][tra_idx], HIR_mat[tra_idx], method="RRMSE")
    final_train_rmse_HIR = loss_fn(ph["HIR"][tra_idx], HIR_mat[tra_idx], method="RMSE")
    final_val_loss_HIR = loss_fn(ph["HIR"][val_idx], HIR_mat[val_idx], method="RRMSE")
    final_val_rmse_HIR = loss_fn(ph["HIR"][val_idx], HIR_mat[val_idx], method="RMSE")
    final_test_loss_HIR = loss_fn(ph["HIR"][test_idx], HIR_mat[test_idx], method="RRMSE")
    final_test_rmse_HIR = loss_fn(ph["HIR"][test_idx], HIR_mat[test_idx], method="RMSE")

    config = {
        "run_dir": run_dir,
        "timestamp": run_name,
        "seed": seed,
        "device": str(device),
        "weight_fixed": [float(x) for x in weight.detach().cpu().tolist()],
        "learning_rate_final": lr,
        "status": "finished",
    }

    train_sr_015, train_n_015 = success_rate_at_threshold(ph["HIR"][tra_idx], HIR_mat[tra_idx], 0.15)
    val_sr_015, val_n_015 = success_rate_at_threshold(ph["HIR"][val_idx], HIR_mat[val_idx], 0.15)
    test_sr_015, test_n_015 = success_rate_at_threshold(ph["HIR"][test_idx], HIR_mat[test_idx], 0.15)
    train_sr_020, train_n_020 = success_rate_at_threshold(ph["HIR"][tra_idx], HIR_mat[tra_idx], 0.20)
    val_sr_020, val_n_020 = success_rate_at_threshold(ph["HIR"][val_idx], HIR_mat[val_idx], 0.20)
    test_sr_020, test_n_020 = success_rate_at_threshold(ph["HIR"][test_idx], HIR_mat[test_idx], 0.20)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"run_dir: {run_dir}\n")
        f.write(f"device: {device}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"best_val_loss_during_training: {loss_val_best.item():.6f}\n")
        f.write(f"final_train_HIR_RRMSE: {final_train_loss_HIR.item():.6f}\n")
        f.write(f"final_train_HIR_RMSE: {final_train_rmse_HIR.item():.6f}\n")
        f.write(f"final_val_HIR_RRMSE: {final_val_loss_HIR.item():.6f}\n")
        f.write(f"final_val_HIR_RMSE: {final_val_rmse_HIR.item():.6f}\n")
        f.write(f"final_test_HIR_RRMSE: {final_test_loss_HIR.item():.6f}\n")
        f.write(f"final_test_HIR_RMSE: {final_test_rmse_HIR.item():.6f}\n")
        f.write(f"train_success_rate_0.15: {train_sr_015:.6f} (n={train_n_015})\n")
        f.write(f"val_success_rate_0.15: {val_sr_015:.6f} (n={val_n_015})\n")
        f.write(f"test_success_rate_0.15: {test_sr_015:.6f} (n={test_n_015})\n")
        f.write(f"train_success_rate_0.20: {train_sr_020:.6f} (n={train_n_020})\n")
        f.write(f"val_success_rate_0.20: {val_sr_020:.6f} (n={val_n_020})\n")
        f.write(f"test_success_rate_0.20: {test_sr_020:.6f} (n={test_n_020})\n")
