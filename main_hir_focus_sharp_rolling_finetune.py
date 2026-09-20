import json
import os
from datetime import datetime
import time

import numpy as np
import pandas as pd
import torch

from Code.DHCorn_sharp import dhcorn
from Code.make_g import make_treno


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"当前设备: {device}")

stage_sharpness = 0.20
seed = int(datetime.now().strftime("%Y%m%d")) % (2**31 - 1)
torch.manual_seed(seed)
np.random.seed(seed % (2**32 - 1))
print(f"随机种子: {seed}")


def loss_fn(pred, true, method="RRMSE"):
    mask = ~torch.isnan(true) & ~torch.isnan(pred)
    pred = pred[mask]
    true = true[mask]
    if pred.numel() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    if method == "RMSE":
        return ((pred - true) ** 2).mean().sqrt()
    if method == "RRMSE3":
        return ((pred - true) ** 2).mean().sqrt() / pred.mean().clamp(min=1e-6)
    top_k = max(1, int(len(true) * 0.2))
    top_idx = torch.topk(true, top_k).indices
    return ((pred - true) ** 2).mean().sqrt() / true[top_idx].mean().clamp(min=1e-6)


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


def format_seconds(seconds):
    seconds = max(0.0, float(seconds))
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


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
theta4_base = np.apply_along_axis(lambda row: np.interp(log4, depths_log[::-1], row[::-1]), axis=1, arr=moistures)
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

env = torch.zeros(N, 130 * 24, 13)
for i in range(N):
    key = f"{pheno.Year[i]}_{pheno.REP[i]}"
    env_i = env_all[key].iloc[:, [2, 9, 10, 11, 12, 5, 13, 14, 15, 3, 4, 6, 8]]
    env[i, :, :] = torch.tensor(env_i.to_numpy())
env = env.to(device)

mg = torch.zeros(N, 2)
mg[:, 1] = 31363 / 44560
for i in range(N):
    key = f"{pheno.Year[i]}_{pheno.REP[i]}"
    plant = env_all[key].iloc[0, 1]
    harvest = pd.Timestamp(str(pheno.Year[i]) + "-08-24 09:00:00")
    mg[i, 0] = (harvest - plant).days
mg = mg.to(device)

treno, Utreno, Ltreno = make_treno()
Utreno = torch.tensor(Utreno, dtype=torch.float32, device=device)
Ltreno = torch.tensor(Ltreno, dtype=torch.float32, device=device)
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

FT = torch.tensor(pheno["FT"].to_numpy(), dtype=torch.float32, device=device)
FT_day = FT / 24

years = sorted(int(y) for y in pheno.loc[~pheno["HIR"].isna(), "Year"].unique())
year_tensor = torch.tensor(pheno["Year"].to_numpy(), device=device)
weight = torch.tensor([0.05, 0.90, 0.00, 0.05], device=device)
train_subset_ratio = 0.30
lr_init = 1e-4
lr_finetune = 5e-5
beta_reg = 50.0
grad_clip_norm = 1.0
max_epochs = 800
early_stop_patience = 50
lr_patience = 15
min_delta = 5e-4
min_lr = 1e-6
min_val_samples = 24
progress_every = 10

run_dir = os.path.join("Result", f"hir_focus_sharp_rolling_finetune_{seed}")
os.makedirs(run_dir, exist_ok=True)
fold_results = []
print(f"结果目录: {run_dir}")

rolling_years = years[1:]
beta_warm = None

for fold_id, test_year in enumerate(rolling_years, start=1):
    test_idx = torch.where((year_tensor == test_year) & ~torch.isnan(HIR))[0]
    train_pool = torch.where((year_tensor < test_year) & ~torch.isnan(HIR))[0]
    if len(test_idx) == 0 or len(train_pool) <= min_val_samples:
        print(f"[{fold_id}/{len(rolling_years)}] 跳过 {test_year}: 可用训练或测试样本不足")
        continue

    perm = torch.randperm(len(train_pool), device=device)
    val_size = max(min_val_samples, int(0.1 * len(train_pool)))
    val_idx = train_pool[perm[:val_size]]
    tra_idx = train_pool[perm[val_size:]]

    if beta_warm is None:
        beta = torch.randn(G.shape[1], 2, n_stages * 89, device=device) / 100
        lr = lr_init
        init_mode = "random"
    else:
        beta = beta_warm.detach().clone().to(device)
        lr = lr_finetune
        init_mode = "warm_start"
    beta.requires_grad_(True)

    beta_best = beta.detach().clone()
    optimizer = torch.optim.Adam([beta], lr=lr)
    fail = 0
    best_val = 1e9
    history = []
    fold_start = time.time()

    print(
        f"[{fold_id}/{len(rolling_years)}] Rolling test year {test_year} "
        f"(train={len(tra_idx)}, val={len(val_idx)}, test={len(test_idx)}, init={init_mode}, lr={lr:.1e})"
    )

    for epoch in range(max_epochs):
        optimizer.zero_grad()
        perm2 = torch.randperm(len(tra_idx), device=device)
        tra_idx2 = tra_idx[perm2[: max(1, int(train_subset_ratio * len(tra_idx)))]]

        g = torch.matmul(G, beta[:, 0, :]).reshape(N, n_stages, 89) + torch.matmul((G == 1) * 1.0, beta[:, 1, :]).reshape(N, n_stages, 89)
        geno = torch.sigmoid(g) * (Utreno - Ltreno) + Ltreno
        ph = dhcorn(env, mg, geno, device=device, stage_sharpness=stage_sharpness)

        loss_hir = loss_fn(ph["HIR"][tra_idx2], HIR_mat[tra_idx2], method="RRMSE")
        loss_ph = loss_fn(ph["Height"][tra_idx2], Height_mat[tra_idx2], method="RRMSE3")
        loss_ft = loss_fn(predicted_ft_day(ph["stage"][tra_idx2]), FT_day[tra_idx2], method="RRMSE3")
        loss = torch.dot(weight, torch.stack([loss_ph, loss_hir, torch.tensor(0.0, device=device), loss_ft])) + beta.square().mean() * beta_reg
        if not torch.isfinite(loss):
            print(f"[{fold_id}/{len(rolling_years)}] year {test_year}: non-finite loss, stop fold")
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_([beta], grad_clip_norm)
        optimizer.step()

        with torch.no_grad():
            val_rrmse = loss_fn(ph["HIR"][val_idx], HIR_mat[val_idx], method="RRMSE").item()
            test_rrmse = loss_fn(ph["HIR"][test_idx], HIR_mat[test_idx], method="RRMSE").item()
            val_rmse = loss_fn(ph["HIR"][val_idx], HIR_mat[val_idx], method="RMSE").item()
            test_rmse = loss_fn(ph["HIR"][test_idx], HIR_mat[test_idx], method="RMSE").item()
            improved = (best_val - val_rrmse) > min_delta
            history.append(
                {
                    "epoch": epoch + 1,
                    "lr": lr,
                    "val_hir_rrmse": val_rrmse,
                    "test_hir_rrmse": test_rrmse,
                    "val_hir_rmse": val_rmse,
                    "test_hir_rmse": test_rmse,
                    "improved": int(improved),
                    "init_mode": init_mode,
                }
            )
            if improved:
                best_val = val_rrmse
                beta_best = beta.detach().clone()
                fail = 0
            else:
                fail += 1
                if fail % lr_patience == 0 and lr > min_lr:
                    lr = max(lr * 0.5, min_lr)
                    optimizer = torch.optim.Adam([beta], lr=lr)
                if fail >= early_stop_patience:
                    break

            if (epoch + 1) % progress_every == 0 or epoch == 0 or improved or (fail >= early_stop_patience):
                elapsed = time.time() - fold_start
                avg_epoch_time = elapsed / (epoch + 1)
                eta = avg_epoch_time * max(max_epochs - epoch - 1, 0)
                flag = " *" if improved else ""
                print(
                    f"  epoch {epoch + 1:4d}/{max_epochs} | "
                    f"val_rrmse={val_rrmse:.4f} | test_rrmse={test_rrmse:.4f} | "
                    f"lr={lr:.2e} | fail={fail:2d} | "
                    f"elapsed={format_seconds(elapsed)} | eta={format_seconds(eta)}{flag}"
                )

    with torch.no_grad():
        beta.copy_(beta_best)
        g = torch.matmul(G, beta[:, 0, :]).reshape(N, n_stages, 89) + torch.matmul((G == 1) * 1.0, beta[:, 1, :]).reshape(N, n_stages, 89)
        geno = torch.sigmoid(g) * (Utreno - Ltreno) + Ltreno
        ph = dhcorn(env, mg, geno, device=device, stage_sharpness=stage_sharpness)
        test_rrmse = loss_fn(ph["HIR"][test_idx], HIR_mat[test_idx], method="RRMSE").item()
        test_rmse = loss_fn(ph["HIR"][test_idx], HIR_mat[test_idx], method="RMSE").item()
        sr015, n015 = success_rate_at_threshold(ph["HIR"][test_idx], HIR_mat[test_idx], 0.15)
        sr020, n020 = success_rate_at_threshold(ph["HIR"][test_idx], HIR_mat[test_idx], 0.20)

    beta_warm = beta_best.detach().clone()

    year_dir = os.path.join(run_dir, f"year_{test_year}")
    os.makedirs(year_dir, exist_ok=True)
    pd.DataFrame(history).to_csv(os.path.join(year_dir, "history.csv"), index=False)
    torch.save(
        {
            "beta": beta_best.detach().cpu(),
            "seed": seed,
            "test_year": test_year,
            "init_mode": init_mode,
            "lr_init_used": lr_finetune if init_mode == "warm_start" else lr_init,
            "stage_sharpness": stage_sharpness,
            "weight_fixed": weight.detach().cpu(),
        },
        os.path.join(year_dir, "best_checkpoint.pt"),
    )
    fold_results.append(
        {
            "test_year": test_year,
            "n_train": int(len(tra_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "init_mode": init_mode,
            "best_val_hir_rrmse": best_val,
            "final_test_hir_rrmse": test_rrmse,
            "final_test_hir_rmse": test_rmse,
            "test_success_rate_0.15": sr015,
            "test_success_rate_0.15_n": n015,
            "test_success_rate_0.20": sr020,
            "test_success_rate_0.20_n": n020,
        }
    )
    print(
        f"完成 {test_year}: best_val_rrmse={best_val:.4f}, "
        f"test_rrmse={test_rrmse:.4f}, test_rmse={test_rmse:.4f}, "
        f"耗时={format_seconds(time.time() - fold_start)}"
    )

fold_df = pd.DataFrame(fold_results).sort_values("test_year")
fold_df.to_csv(os.path.join(run_dir, "rolling_metrics.csv"), index=False)
summary = {
    "run_dir": run_dir,
    "seed": seed,
    "device": str(device),
    "model": "DHCorn_sharp_rolling_finetune",
    "years": fold_df["test_year"].tolist(),
    "mean_test_hir_rrmse": float(fold_df["final_test_hir_rrmse"].mean()),
    "std_test_hir_rrmse": float(fold_df["final_test_hir_rrmse"].std(ddof=0)),
    "mean_test_hir_rmse": float(fold_df["final_test_hir_rmse"].mean()),
    "std_test_hir_rmse": float(fold_df["final_test_hir_rmse"].std(ddof=0)),
    "lr_init": lr_init,
    "lr_finetune": lr_finetune,
    "beta_reg": beta_reg,
    "grad_clip_norm": grad_clip_norm,
}
with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print("全部 rolling 年份完成。")
print(json.dumps(summary, indent=2, ensure_ascii=False))
