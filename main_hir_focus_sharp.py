import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

MODEL_VARIANT = os.environ.get("DHCORN_VARIANT", "sharp").strip().lower()
dhcorn_kwargs = {}
if MODEL_VARIANT == "sharp_nostart0":
    from Code.DHCorn_sharp import dhcorn
    model_label = "DHCorn_sharp_no_start0"
    run_prefix = "hir_focus_sharp_nostart0"
    dhcorn_kwargs = {"start0_weight": 0.0}
elif MODEL_VARIANT == "sharp6":
    from Code.DHCorn_sharp6 import dhcorn
    model_label = "DHCorn_sharp6"
    run_prefix = "hir_focus_sharp6"
elif MODEL_VARIANT == "sharp":
    from Code.DHCorn_sharp import dhcorn
    model_label = "DHCorn_sharp"
    run_prefix = "hir_focus_sharp"
else:
    raise ValueError(f"Unsupported DHCORN_VARIANT: {MODEL_VARIANT}")

from Code.make_g import make_treno
from Code.res_plot import plot_success


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"当前设备: {device}")
print(f"HIR 模型变体: {model_label}")
if dhcorn_kwargs:
    print(f"额外 HIR 参数: {dhcorn_kwargs}")

START0_STAGE_IDX = -1
START0_PARAM_IDX = 87
MATERNAL_HEALTH_PARAM_IDX = 77
MATERNAL_REPAIR_PARAM_IDX = 78
default_start0_var_reg = 5.0 if dhcorn_kwargs.get("start0_weight", 1.0) != 0.0 else 0.0
start0_var_reg = float(os.environ.get("DHCORN_START0_VAR_REG", str(default_start0_var_reg)))
print(f"HIBackgroundEffect variance regularization: {start0_var_reg}")

repair_weight_override = os.environ.get("DHCORN_REPAIR_WEIGHT")
if repair_weight_override is not None:
    dhcorn_kwargs["repair_weight"] = float(repair_weight_override)
    print(f"Override repair_weight: {dhcorn_kwargs['repair_weight']}")

s4_background_max = os.environ.get("DHCORN_S4_HIBG_MAX")
if s4_background_max is not None:
    s4_background_max = float(s4_background_max)
    print(f"Override S4 HIBackgroundEffect max: {s4_background_max}")

maternal_health_fixed = os.environ.get("DHCORN_MATERNAL_HEALTH_FIXED")
if maternal_health_fixed is not None:
    maternal_health_fixed = float(maternal_health_fixed)
    print(f"Fix MaternalHealth to: {maternal_health_fixed}")

maternal_repair_fixed = os.environ.get("DHCORN_MATERNAL_REPAIR_FIXED")
if maternal_repair_fixed is not None:
    maternal_repair_fixed = float(maternal_repair_fixed)
    print(f"Fix MaternalRepair to: {maternal_repair_fixed}")

stage_sharpness = 0.10
print(f"阶段切换 sharpness: {stage_sharpness}")

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

if s4_background_max is not None:
    bg_lower = float(Ltreno[0, START0_STAGE_IDX, START0_PARAM_IDX].item())
    capped_max = max(bg_lower, s4_background_max)
    Utreno[:, START0_STAGE_IDX, START0_PARAM_IDX] = capped_max
    Ltreno[:, START0_STAGE_IDX, START0_PARAM_IDX] = bg_lower

if maternal_health_fixed is not None:
    Utreno[:, :, MATERNAL_HEALTH_PARAM_IDX] = maternal_health_fixed
    Ltreno[:, :, MATERNAL_HEALTH_PARAM_IDX] = maternal_health_fixed

if maternal_repair_fixed is not None:
    Utreno[:, :, MATERNAL_REPAIR_PARAM_IDX] = maternal_repair_fixed
    Ltreno[:, :, MATERNAL_REPAIR_PARAM_IDX] = maternal_repair_fixed

n_stages = Utreno.shape[1]

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

TL = torch.tensor(pheno["TL"].to_numpy(), dtype=torch.float32, device=device)
TL[TL == 0] = torch.nan
TL_mat = torch.full((N, 130), torch.nan, device=device)
for i in range(N):
    if ~TL[i].isnan():
        TL_mat[i, int(mg[i, 0])] = TL[i]

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
        return ((pred - true) ** 2).mean().sqrt() / pred.mean().clamp(min=1e-6)

    top_ratio = 0.2
    top_k = max(1, int(len(true) * top_ratio))
    top_idx = torch.topk(true, top_k).indices
    rmse = ((pred - true) ** 2).mean().sqrt()
    top_pred_mean = true[top_idx].mean().clamp(min=1e-6)
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


def precision_recall_at_threshold(pred, true, threshold):
    mask = ~torch.isnan(true) & ~torch.isnan(pred)
    pred = pred[mask]
    true = true[mask]
    if pred.numel() == 0:
        return float("nan"), float("nan"), 0, 0
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


perm = torch.randperm(N)
test_size = int(N * 0.3)
test_idx = perm[:test_size]
train_idx = perm[test_size:]

perm2 = torch.randperm(len(train_idx))
validate_size = int(0.1 * N)
val_idx = train_idx[perm2[:validate_size]]
tra_idx = train_idx[perm2[validate_size:]]

beta = torch.randn(G.shape[1], 2, n_stages * 89, device=device) / 100
beta.requires_grad_(True)

weight = torch.tensor([0.05, 0.90, 0.01, 0.05], device=device)
train_subset_ratio = 1.00
lr = 1e-4
beta_reg = 30
grad_clip_norm = 1.0
max_epochs = 800
early_stop_patience = 50
lr_patience = 10
min_delta = 5e-4
min_lr = 1e-8
fail = 0
loss_val_best = 1e9
beta_best = beta.detach().clone()
history = []

result_root = "Result"
os.makedirs(result_root, exist_ok=True)
run_name = os.environ.get("DHCORN_RUN_NAME", f"{run_prefix}_{seed}")
run_dir = os.path.join(result_root, run_name)
os.makedirs(run_dir, exist_ok=True)
print(f"结果输出目录: {run_dir}")

HIRfig_name = os.path.join(run_dir, "HIR.png")
precisionfig_name = os.path.join(run_dir, "precision.png")
metrics_path = os.path.join(run_dir, "metrics.txt")
config_path = os.path.join(run_dir, "config.json")
history_path = os.path.join(run_dir, "history.csv")
curve_path = os.path.join(run_dir, "training_curves.png")
best_checkpoint_path = os.path.join(run_dir, "best_checkpoint.pt")
final_checkpoint_path = os.path.join(run_dir, "final_checkpoint.pt")
precision_recall_thresholds = [0.10, 0.15, 0.20]

optimizer = torch.optim.Adam([beta], lr=lr)

for epoch in range(max_epochs):
    optimizer.zero_grad()
    perm2 = torch.randperm(len(tra_idx))
    train_size = max(1, int(train_subset_ratio * len(tra_idx)))
    tra_idx2 = tra_idx[perm2[:train_size]]

    g = (
        torch.matmul(G, beta[:, 0, :]).reshape(N, n_stages, 89)
        + torch.matmul((G == 1) * 1.0, beta[:, 1, :]).reshape(N, n_stages, 89)
    )
    geno = torch.sigmoid(g) * (Utreno - Ltreno) + Ltreno
    ph = dhcorn(env, mg, geno, device=device, stage_sharpness=stage_sharpness, **dhcorn_kwargs)
    start0_var_penalty = torch.var(geno[:, START0_STAGE_IDX, START0_PARAM_IDX], unbiased=False)

    loss_HIR = loss_fn(ph["HIR"][tra_idx2], HIR_mat[tra_idx2], method="RRMSE")
    RMSE_HIR = loss_fn(ph["HIR"][tra_idx2], HIR_mat[tra_idx2], method="RMSE")
    loss_PH = loss_fn(ph["Height"][tra_idx2], Height_mat[tra_idx2], method="RRMSE3")
    loss_TL = loss_fn(ph["TasselLength"][tra_idx2], TL_mat[tra_idx2], method="RRMSE3")
    loss_FT = loss_fn(predicted_ft_day(ph["stage"][tra_idx2]), FT_day[tra_idx2], method="RRMSE3")
    loss = (
        torch.dot(weight, torch.stack([loss_PH, loss_HIR, loss_TL, loss_FT]))
        + beta.square().mean() * beta_reg
        + start0_var_reg * start0_var_penalty
    )
    if not torch.isfinite(loss):
        print(f"Epoch {epoch + 1:02d}: non-finite loss, stop training")
        break
    loss.backward()
    torch.nn.utils.clip_grad_norm_([beta], grad_clip_norm)
    optimizer.step()

    print(
        f"Epoch {epoch + 1:02d} train, "
        f"RRMSE = {loss.item():.4f}, "
        f"\tFT = {loss_FT.item():.4f}, "
        f"\tHIR = {loss_HIR.item():.4f}, "
        f"\tHeight = {loss_PH.item():.4f}, "
        f"\tTassel = {loss_TL.item():.4f}, "
        f"\tHIBackgroundEffectVar = {start0_var_penalty.item():.4f}, "
        f"\t Fail = {fail:02d}"
    )

    with torch.no_grad():
        loss_HIR_val = loss_fn(ph["HIR"][val_idx], HIR_mat[val_idx], method="RRMSE")
        RMSE_HIR_val = loss_fn(ph["HIR"][val_idx], HIR_mat[val_idx], method="RMSE")
        loss_PH_val = loss_fn(ph["Height"][val_idx], Height_mat[val_idx], method="RRMSE3")
        loss_TL_val = loss_fn(ph["TasselLength"][val_idx], TL_mat[val_idx], method="RRMSE3")
        loss_FT_val = loss_fn(predicted_ft_day(ph["stage"][val_idx]), FT_day[val_idx], method="RRMSE3")
        loss_val = (
            torch.dot(weight, torch.stack([loss_PH_val, loss_HIR_val, loss_TL_val, loss_FT_val]))
            + beta.square().mean() * beta_reg
            + start0_var_reg * start0_var_penalty
        )

        loss_HIR_test = loss_fn(ph["HIR"][test_idx], HIR_mat[test_idx], method="RRMSE")
        RMSE_HIR_test = loss_fn(ph["HIR"][test_idx], HIR_mat[test_idx], method="RMSE")
        loss_PH_test = loss_fn(ph["Height"][test_idx], Height_mat[test_idx], method="RRMSE3")
        loss_TL_test = loss_fn(ph["TasselLength"][test_idx], TL_mat[test_idx], method="RRMSE3")
        loss_FT_test = loss_fn(predicted_ft_day(ph["stage"][test_idx]), FT_day[test_idx], method="RRMSE3")
        loss_test = (
            torch.dot(weight, torch.stack([loss_PH_test, loss_HIR_test, loss_TL_test, loss_FT_test]))
            + beta.square().mean() * beta_reg
            + start0_var_reg * start0_var_penalty
        )

        beta_l2 = beta.detach().pow(2).mean().sqrt().item()
        history_row = {
            "epoch": epoch + 1,
            "lr": lr,
            "fail": fail,
            "train_total_loss": loss.item(),
            "val_total_loss": loss_val.item(),
            "test_total_loss": loss_test.item(),
            "train_hir_rrmse": loss_HIR.item(),
            "val_hir_rrmse": loss_HIR_val.item(),
            "test_hir_rrmse": loss_HIR_test.item(),
            "train_hir_rmse": RMSE_HIR.item(),
            "val_hir_rmse": RMSE_HIR_val.item(),
            "test_hir_rmse": RMSE_HIR_test.item(),
            "train_ft_rrmse": loss_FT.item(),
            "val_ft_rrmse": loss_FT_val.item(),
            "test_ft_rrmse": loss_FT_test.item(),
            "train_height_rrmse": loss_PH.item(),
            "val_height_rrmse": loss_PH_val.item(),
            "test_height_rrmse": loss_PH_test.item(),
            "start0_var": start0_var_penalty.item(),
            "start0_var_penalty": (start0_var_reg * start0_var_penalty).item(),
            "beta_l2": beta_l2,
        }

        print(f"Epoch {epoch + 1:02d}   val, RRMSE = {loss_val.item():.4f}, \tFT = {loss_FT_val.item():.4f}, \tHIR = {loss_HIR_val.item():.4f}, \tHeight = {loss_PH_val.item():.4f}, \tTassel = {loss_TL_val.item():.4f}")
        print(f"Epoch {epoch + 1:02d}  test, RRMSE = {loss_test.item():.4f}, \tFT = {loss_FT_test.item():.4f}, \tHIR = {loss_HIR_test.item():.4f}, \tHeight = {loss_PH_test.item():.4f}, \tTassel = {loss_TL_test.item():.4f}\n")

        improved = (loss_val_best - loss_HIR_val.item()) > min_delta
        history_row["improved"] = int(improved)
        history.append(history_row)
        if improved:
            loss_val_best = loss_HIR_val.item()
            beta_best = beta.detach().clone()
            fail = 0
            torch.save(
                {
                    "beta": beta_best.detach().cpu(),
                    "Utreno": Utreno.detach().cpu(),
                    "Ltreno": Ltreno.detach().cpu(),
                    "seed": seed,
                    "stage_sharpness": stage_sharpness,
                    "train_subset_ratio": train_subset_ratio,
                    "weight_fixed": weight.detach().cpu(),
                    "dhcorn_kwargs": dhcorn_kwargs,
                    "start0_var_reg": start0_var_reg,
                    "s4_hibg_max": s4_background_max,
                    "maternal_health_fixed": maternal_health_fixed,
                    "maternal_repair_fixed": maternal_repair_fixed,
                    "train_idx": train_idx.detach().cpu(),
                    "val_idx": val_idx.detach().cpu(),
                    "test_idx": test_idx.detach().cpu(),
                    "best_val_hir_rrmse": loss_val_best,
                    "epoch": epoch + 1,
                    "model": model_label,
                },
                best_checkpoint_path,
            )

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

#%%
with torch.no_grad():
    beta.copy_(beta_best)
    g = (
        torch.matmul(G, beta[:, 0, :]).reshape(N, n_stages, 89)
        + torch.matmul((G == 1) * 1.0, beta[:, 1, :]).reshape(N, n_stages, 89)
    )
    geno = torch.sigmoid(g) * (Utreno - Ltreno) + Ltreno
    ph = dhcorn(env, mg, geno, device=device, stage_sharpness=stage_sharpness, **dhcorn_kwargs)

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
        "model": model_label,
        "model_variant": MODEL_VARIANT,
        "dhcorn_kwargs": dhcorn_kwargs,
        "start0_var_reg": start0_var_reg,
        "s4_hibg_max": s4_background_max,
        "maternal_health_fixed": maternal_health_fixed,
        "maternal_repair_fixed": maternal_repair_fixed,
        "start0_regularized_stage": START0_STAGE_IDX,
        "start0_regularized_param": START0_PARAM_IDX,
        "n_stages": int(n_stages),
        "stage_sharpness": stage_sharpness,
        "train_subset_ratio": train_subset_ratio,
        "weight_fixed": [float(x) for x in weight.detach().cpu().tolist()],
        "learning_rate_final": lr,
        "beta_reg": beta_reg,
        "grad_clip_norm": grad_clip_norm,
        "max_epochs": max_epochs,
        "early_stop_patience": early_stop_patience,
        "lr_patience": lr_patience,
        "precision_recall_thresholds": precision_recall_thresholds,
        "status": "finished",
    }

    train_sr_015, train_n_015 = success_rate_at_threshold(ph["HIR"][tra_idx], HIR_mat[tra_idx], 0.15)
    val_sr_015, val_n_015 = success_rate_at_threshold(ph["HIR"][val_idx], HIR_mat[val_idx], 0.15)
    test_sr_015, test_n_015 = success_rate_at_threshold(ph["HIR"][test_idx], HIR_mat[test_idx], 0.15)
    train_sr_020, train_n_020 = success_rate_at_threshold(ph["HIR"][tra_idx], HIR_mat[tra_idx], 0.20)
    val_sr_020, val_n_020 = success_rate_at_threshold(ph["HIR"][val_idx], HIR_mat[val_idx], 0.20)
    test_sr_020, test_n_020 = success_rate_at_threshold(ph["HIR"][test_idx], HIR_mat[test_idx], 0.20)
    precision_recall_results = {"train": {}, "val": {}, "test": {}}
    for threshold in precision_recall_thresholds:
        threshold_key = f"{int(round(threshold * 100)):02d}"
        precision_recall_results["train"][threshold_key] = precision_recall_at_threshold(ph["HIR"][tra_idx], HIR_mat[tra_idx], threshold)
        precision_recall_results["val"][threshold_key] = precision_recall_at_threshold(ph["HIR"][val_idx], HIR_mat[val_idx], threshold)
        precision_recall_results["test"][threshold_key] = precision_recall_at_threshold(ph["HIR"][test_idx], HIR_mat[test_idx], threshold)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    pd.DataFrame(history).to_csv(history_path, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=200)
    history_df = pd.DataFrame(history)
    axes[0, 0].plot(history_df["epoch"], history_df["train_hir_rrmse"], label="train")
    axes[0, 0].plot(history_df["epoch"], history_df["val_hir_rrmse"], label="val")
    axes[0, 0].plot(history_df["epoch"], history_df["test_hir_rrmse"], label="test")
    axes[0, 0].set_title("HIR RRMSE")
    axes[0, 0].legend()

    axes[0, 1].plot(history_df["epoch"], history_df["train_hir_rmse"], label="train")
    axes[0, 1].plot(history_df["epoch"], history_df["val_hir_rmse"], label="val")
    axes[0, 1].plot(history_df["epoch"], history_df["test_hir_rmse"], label="test")
    axes[0, 1].set_title("HIR RMSE")
    axes[0, 1].legend()

    axes[1, 0].plot(history_df["epoch"], history_df["train_total_loss"], label="train")
    axes[1, 0].plot(history_df["epoch"], history_df["val_total_loss"], label="val")
    axes[1, 0].plot(history_df["epoch"], history_df["test_total_loss"], label="test")
    axes[1, 0].set_title("Total Loss")
    axes[1, 0].legend()

    axes[1, 1].plot(history_df["epoch"], history_df["lr"], label="lr")
    axes[1, 1].plot(history_df["epoch"], history_df["beta_l2"], label="beta_l2")
    axes[1, 1].set_title("LR / Beta Scale")
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(curve_path)
    plt.close(fig)

    torch.save(
        {
            "beta": beta.detach().cpu(),
            "geno": geno.detach().cpu(),
            "Utreno": Utreno.detach().cpu(),
            "Ltreno": Ltreno.detach().cpu(),
            "seed": seed,
            "stage_sharpness": stage_sharpness,
            "train_subset_ratio": train_subset_ratio,
            "weight_fixed": weight.detach().cpu(),
            "dhcorn_kwargs": dhcorn_kwargs,
            "start0_var_reg": start0_var_reg,
            "s4_hibg_max": s4_background_max,
            "maternal_health_fixed": maternal_health_fixed,
            "maternal_repair_fixed": maternal_repair_fixed,
            "train_idx": train_idx.detach().cpu(),
            "val_idx": val_idx.detach().cpu(),
            "test_idx": test_idx.detach().cpu(),
            "history_path": history_path,
            "model": model_label,
            "model_variant": MODEL_VARIANT,
        },
        final_checkpoint_path,
    )

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"run_dir: {run_dir}\n")
        f.write(f"device: {device}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"start0_var_reg: {start0_var_reg}\n")
        f.write(f"s4_hibg_max: {s4_background_max}\n")
        f.write(f"maternal_health_fixed: {maternal_health_fixed}\n")
        f.write(f"maternal_repair_fixed: {maternal_repair_fixed}\n")
        f.write(f"repair_weight: {dhcorn_kwargs.get('repair_weight', 0.5)}\n")
        f.write(f"best_val_loss_during_training: {loss_val_best:.6f}\n")
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
        for split in ["train", "val", "test"]:
            for threshold in precision_recall_thresholds:
                threshold_key = f"{int(round(threshold * 100)):02d}"
                precision, recall, pred_n, actual_n = precision_recall_results[split][threshold_key]
                f.write(f"{split}_precision_{threshold:.2f}: {precision:.6f} (pred_n={pred_n}, actual_n={actual_n})\n")
                f.write(f"{split}_recall_{threshold:.2f}: {recall:.6f} (pred_n={pred_n}, actual_n={actual_n})\n")
