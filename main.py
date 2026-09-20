#%%
import os
import json
from datetime import datetime

import pandas as pd
import torch
from Code.DHCorn import dhcorn
# from Code.DHCorn_sharp import dhcorn

from Code.make_g import make_treno
from Code.utils import simulate_growth_curves
import matplotlib.pyplot as plt
import numpy as np
from Code.res_plot import plot_success

#%% Check GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# device = torch.device("mps" if torch.mps.is_available() else "cpu")
device = 'cpu'
print(f"当前设备: {device}")

seed = int(datetime.now().strftime("%Y%m%d")) % (2**31 - 1)
torch.manual_seed(seed)
np.random.seed(seed % (2**32 - 1))
print(f"随机种子: {seed}")

# torch.autograd.set_detect_anomaly(True)

#%% Load data
data_path = 'Data/'
pheno_path = data_path + '01_Data_Pheno/'
env_path = data_path + '02_Data_Env/'
geno_path = data_path + '03_Data_Geno/'

PD = pd.read_csv(pheno_path + 'Planting Dates/PD_2018-2024.csv')
env_ames = pd.read_excel(data_path + 'Clean/ames.xlsx')

pheno = pd.read_csv(data_path + 'Clean/phenotype_all.csv')
# N = len(pheno)

G_raw = pd.read_csv(data_path + 'Clean/geno1.csv')

#%%
G_mat = torch.tensor(G_raw.iloc[:, 10:].to_numpy()).t() * 2
ID = G_raw.columns[10:]
id_to_idx = {sid: i for i, sid in enumerate(ID)}

pheno = pheno[pheno['SID'].isin(ID)].reset_index(drop=True)
N = len(pheno)

G = torch.zeros(len(pheno), G_mat.shape[1])
for i in range(N):
    G[i, :] = G_mat[id_to_idx[pheno['SID'].iloc[i]], :]

G = G.to(device)

#%% create env

# Soil 4 inch
# Step 1: log-depth 插值估算基础湿度
depths_log = np.log([12, 24, 50])
log4 = np.log(4)

# 3层湿度值拼成 3 x N 数组
moistures = env_ames[['SoilMoist2', 'SoilMoist3', 'SoilMoist4']].values

# 插值 (逐列进行)
def interp_row(row):
    return np.interp(log4, depths_log[::-1], row[::-1])  # 反向插值保持深度顺序
theta4_base = np.apply_along_axis(interp_row, axis=1, arr=moistures)


# Step 2: 降雨量换算（inch -> mm）
rain_mm = env_ames['SoilMoist1'] * 25.4

# Step 3: 计算降雨修正项
delta_theta = np.where(
    rain_mm < 1,
    0,
    np.where(
        rain_mm <= 20,
        0.5 * (1 - np.exp(-rain_mm / 3)) * (env_ames['SoilMoist2'] - theta4_base),
        env_ames['SoilMoist2'] - theta4_base
    )
)
env_ames['SoilMoist1'] = theta4_base + delta_theta

#%%
env_all = {}
for i in range(len(PD)):
    year = PD.iloc[i, 0]
    rep1 = PD.iloc[i, 1]
    dt1 = pd.Timestamp(rep1 + ' 09:00:00')
    ht1 = dt1 + pd.Timedelta('130 Days')
    env1 = env_ames[(env_ames.Time >= dt1) & (env_ames.Time < ht1)]
    env_all[str(year) + '_1'] = env1

    rep2 = PD.iloc[i, 2]
    dt2 = pd.Timestamp(rep2 + ' 09:00:00')
    ht2 = dt2 + pd.Timedelta('130 Days')
    env2 = env_ames[(env_ames.Time >= dt2) & (env_ames.Time < ht2)]
    env_all[str(year) + '_2'] = env2

env = torch.zeros(N, 130 * 24, 13)
for i in range(N):
    key = str(pheno.Year[i]) + '_' + str(pheno.REP[i])
    env_i = env_all[key].iloc[:, [2, 9, 10, 11, 12, 5, 13, 14, 15, 3, 4, 6, 8]]
    env[i, :, :] = torch.tensor(env_i.to_numpy())


#%% create mg
mg = torch.zeros(N, 2)
mg[:, 1] = 31363 / 44560
for i in range(N):
    year = pheno.Year[i]
    key = str(pheno.Year[i]) + '_' + str(pheno.REP[i])
    plant = env_all[key].iloc[0, 1]
    harvest = pd.Timestamp(str(year) + '-08-24 09:00:00')
    mg[i, 0] = (harvest - plant).days


# %%
H = 130 * 24
# geno_t = torch.tensor(make_treno(), device=device).requires_grad_()
# geno = geno_t.repeat(N, 1, 1)

treno, Utreno, Ltreno = make_treno()
Utreno = torch.tensor(Utreno, dtype=torch.float32, device=device)
Ltreno = torch.tensor(Ltreno, dtype=torch.float32,  device=device)

#%%
# geno = torch.tensor(treno, device=device).repeat(N, 1, 1).requires_grad_()
# ph = dhcorn(env, mg, geno, device=device)

# G = torch.randint(0, 3, (N, 2000), dtype=torch.float, device=device)
# beta = torch.zeros(2000, 3*76, device=device, requires_grad=True)
# g = torch.matmul(G, beta).reshape(N, 3, 76)
# geno = 1 / (1 + torch.exp(-g)) * (Utreno - Ltreno) + Ltreno

#%%
Height = torch.tensor(pheno['PH'].to_numpy(), dtype=torch.float32, device=device)
Height[Height == 0] = torch.nan
Height_mat = torch.zeros(N, 130, device=device)
Height_mat[:] = torch.nan
for i in range(N):
    if ~Height[i].isnan():
        Height_mat[i, int(mg[i, 0])] = Height[i]
# Height = Height_mat

HIR = torch.tensor(pheno['HIR'].to_numpy(), dtype=torch.float32, device=device) / 100
HIR_mat = torch.zeros(N, 130, device=device)
HIR_mat[:] = torch.nan
for i in range(N):
    if ~HIR[i].isnan():
        HIR_mat[i, int(mg[i, 0])] = HIR[i]
# HIR = HIR_mat

TL = torch.tensor(pheno['TL'].to_numpy(), dtype=torch.float32, device=device)
TL_mat = torch.zeros(N, 130, device=device)
TL_mat[:] = torch.nan
for i in range(N):
    if ~TL[i].isnan():
        TL_mat[i, int(mg[i, 0])] = TL[i]
# TL = TL_mat

FT = torch.tensor(pheno['FT'].to_numpy(), dtype=torch.float32, device=device)
FT_day = FT / 24

# Height, TL, HIR = simulate_growth_curves(nsamples=N, ndays=130, device='cuda')

#%%
def pearson_corr(y_true, y_pred):
    """
    y_true: tensor of shape [N]
    y_pred: tensor of shape [N]
    """
    y_true = y_true.float()
    y_pred = y_pred.float()

    vx = y_true - y_true.mean()
    vy = y_pred - y_pred.mean()

    corr = (vx * vy).sum() / (torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum()))
    return corr

def loss_fn(pred, true, weight=False, method='RRMSE'):
    mask = ~torch.isnan(true) & ~torch.isnan(pred)

    pred = pred[mask]
    true = true[mask]


    if pred.numel() == 0:
        loss = torch.tensor(0.0, device=ph["stage"].device, requires_grad=True)
    elif method == 'weighted':
        weights = true.sqrt() / true.sqrt().mean()
        loss = (((pred - true) ** 2) * weights).mean().sqrt()/pred.max().detach()
    elif method == 'Pearson':
        loss = 1 - pearson_corr(true, pred)
    elif method == 'RMSE':
        loss = ((pred - true) ** 2).mean().sqrt()
    elif method == 'RRMSE2':
        epsilon = 1e-6
        loss = torch.sqrt((((pred - true) / (true + epsilon)) ** 2).mean())
    elif method == 'RRMSE3':
        loss = ((pred - true) ** 2).mean().sqrt() / pred.mean().detach()
    else:
        top_ratio = 0.2
        top_k = int(len(true) * top_ratio)
        top_idx = torch.topk(true, top_k).indices

        rmse = ((pred - true) ** 2).mean().sqrt()

        top_pred_mean = true[top_idx].mean().detach()
        loss = rmse / top_pred_mean
        # loss = ((pred - true) ** 2).mean().sqrt() / true.mean()
    return loss

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

    success = (true[selected] > threshold).float().mean().item()
    return success, count

def predicted_ft_day(stage_curve, temperature=20.0):
    stage_signal = stage_curve[:, 2, :]
    day_axis = torch.arange(stage_signal.shape[1], device=stage_signal.device, dtype=stage_signal.dtype)
    weights = torch.softmax(stage_signal * temperature, dim=1)
    return (weights * day_axis.unsqueeze(0)).sum(dim=1)

def write_run_artifacts(
    run_dir,
    config_path,
    metrics_path,
    config,
    metrics_lines,
    beta,
    geno,
    treno,
    Utreno,
    Ltreno,
    env,
    mg,
    G,
    Height_mat,
    HIR_mat,
    TL_mat,
    FT_day,
    train_idx,
    val_idx,
    test_idx,
    ph,
):
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("\n".join(metrics_lines) + "\n")

    torch.save(geno.detach().cpu(), os.path.join(run_dir, "geno.pt"))
    torch.save(G.detach().cpu(), os.path.join(run_dir, "G.pt"))
    torch.save(beta.detach().cpu(), os.path.join(run_dir, "beta.pt"))
    torch.save(torch.as_tensor(treno), os.path.join(run_dir, "treno.pt"))
    torch.save(Utreno.detach().cpu(), os.path.join(run_dir, "Utreno.pt"))
    torch.save(Ltreno.detach().cpu(), os.path.join(run_dir, "Ltreno.pt"))

    torch.save(env.detach().cpu(), os.path.join(run_dir, "env.pt"))
    torch.save(mg.detach().cpu(), os.path.join(run_dir, "mg.pt"))

    torch.save(Height_mat.detach().cpu(), os.path.join(run_dir, "Height.pt"))
    torch.save(HIR_mat.detach().cpu(), os.path.join(run_dir, "HIR.pt"))
    torch.save(TL_mat.detach().cpu(), os.path.join(run_dir, "TL.pt"))
    torch.save(FT_day.detach().cpu(), os.path.join(run_dir, "FT_day.pt"))
    torch.save(train_idx.detach().cpu(), os.path.join(run_dir, "train_idx.pt"))
    torch.save(val_idx.detach().cpu(), os.path.join(run_dir, "val_idx.pt"))
    torch.save(test_idx.detach().cpu(), os.path.join(run_dir, "test_idx.pt"))
    torch.save(ph["HIR"].detach().cpu(), os.path.join(run_dir, "pred_HIR.pt"))
    torch.save(ph["Height"].detach().cpu(), os.path.join(run_dir, "pred_Height.pt"))
    torch.save(ph["TasselLength"].detach().cpu(), os.path.join(run_dir, "pred_TL.pt"))
    torch.save(ph["stage"].detach().cpu(), os.path.join(run_dir, "pred_FT_stage.pt"))


#%%
perm = torch.randperm(N)
test_size = int(N * 0.3)

test_idx = perm[:test_size]
# train_idx = torch.cat([perm[:cross_id * test_size], perm[(cross_id + 1) * test_size:]])
train_idx = perm[test_size:]

#%%
# G = torch.randint(0, 3, (N, 10000), device=device) * 1.0
beta = torch.randn(G.shape[1], 2, 4*89, device=device, requires_grad=True)
beta = torch.randn(G.shape[1], 2, 4*89, device=device)/100
beta.requires_grad_(True)

weight = torch.tensor([0.01, 0.4, 0.01, 0.58], device=device)

result_root = "Result"
os.makedirs(result_root, exist_ok=True)
run_name = f"run_{seed}"
run_dir = os.path.join(result_root, run_name)
os.makedirs(run_dir, exist_ok=True)
print(f"结果输出目录: {run_dir}")

HIRfig_name = os.path.join(run_dir, "HIR.png")
precisionfig_name = os.path.join(run_dir, "precision.png")
metrics_path = os.path.join(run_dir, "metrics.txt")
config_path = os.path.join(run_dir, "config.json")
checkpoint_path = os.path.join(run_dir, "best_checkpoint.pt")

# %%
perm2 = torch.randperm(len(train_idx))
validate_size = int(0.1 * N)

val_idx = train_idx[perm2[:validate_size]]
tra_idx = train_idx[perm2[validate_size:]]

#%%
beta_best = beta.detach().clone()
loss_val_best = 100000
fail = 0
lr = 1e-4
max_epochs = 500
early_stop_patience = 30
lr_patience = 10
min_delta = 5e-4
min_lr = 1e-6

#%%
optimizer = torch.optim.Adam([beta], lr=lr)

for epoch in range(max_epochs):
    optimizer.zero_grad()

    perm2 = torch.randperm(len(tra_idx))
    train_size = int(0.1 * len(tra_idx))

    tra_idx2 = tra_idx[perm2[:train_size]]

    g = torch.matmul(G, beta[:, 0, :]).reshape(N, 4, 89) + torch.matmul((G == 1) * 1.0, beta[:, 1, :]).reshape(N, 4, 89)
    geno = 1 / (1 + torch.exp(-g)) * (Utreno - Ltreno) + Ltreno

    ph = dhcorn(env, mg, geno, device=device)

    loss_HIR = loss_fn(ph["HIR"][tra_idx2], HIR_mat[tra_idx2], method='RRMSE')
    RMSE_HIR = loss_fn(ph["HIR"][tra_idx2], HIR_mat[tra_idx2], method='RMSE')
    loss_PH = loss_fn(ph["Height"][tra_idx2], Height_mat[tra_idx2], method='RRMSE3')
    loss_TL = loss_fn(ph["TasselLength"][tra_idx2], TL_mat[tra_idx2], method='RRMSE3')
    pred_ft_train = predicted_ft_day(ph["stage"][tra_idx2])
    loss_FT = loss_fn(pred_ft_train, FT_day[tra_idx2], method='RRMSE3')

    loss = torch.dot(weight, torch.stack([loss_PH, loss_HIR, loss_TL, loss_FT])) + beta.square().mean() * 10
    # loss = loss_HIR

    loss.backward()

    # mask = torch.zeros_like(beta)
    # for i in range(4):  # 4个模块
    #     start = i * 89 + (89 - 24)
    #     end = i * 89 + 89
    #     mask[:, :, start:end] = 1
    # with torch.no_grad():
    #     beta.grad *= mask

    optimizer.step()
    print(f"Epoch {epoch + 1:02d} train, "
          f"RRMSE = {loss.item():.4f}, "
          f"\tFT = {loss_FT.item():.4f}, "
          f"\tHIR = {loss_HIR.item():.4f}, "
          f"\tHeight = {loss_PH.item():.4f}, "
          f"\tTassel = {loss_TL.item():.4f}, "
          f"\t Fail = {fail:02d}")

    with torch.no_grad():
        loss_HIR_val = loss_fn(ph["HIR"][val_idx], HIR_mat[val_idx], method='RRMSE')
        RMSE_HIR_val = loss_fn(ph["HIR"][val_idx], HIR_mat[val_idx], method='RMSE')
        loss_PH_val = loss_fn(ph["Height"][val_idx], Height_mat[val_idx], method='RRMSE3')
        loss_TL_val = loss_fn(ph["TasselLength"][val_idx], TL_mat[val_idx], method='RRMSE3')
        pred_ft_val = predicted_ft_day(ph["stage"][val_idx])
        loss_FT_val = loss_fn(pred_ft_val, FT_day[val_idx], method='RRMSE3')

        loss_val = torch.dot(weight, torch.stack([loss_PH_val, loss_HIR_val, loss_TL_val, loss_FT_val])) + beta.square().mean() * 10
        print(f"Epoch {epoch + 1:02d}   val, RRMSE = {loss_val.item():.4f}, \tFT = {loss_FT_val.item():.4f}, \tHIR = {loss_HIR_val.item():.4f}, \tHeight = {loss_PH_val.item():.4f}, \tTassel = {loss_TL_val.item():.4f}")

        loss_HIR_test = loss_fn(ph["HIR"][test_idx], HIR_mat[test_idx], method='RRMSE')
        RMSE_HIR_test = loss_fn(ph["HIR"][test_idx], HIR_mat[test_idx], method='RMSE')
        loss_PH_test = loss_fn(ph["Height"][test_idx], Height_mat[test_idx], method='RRMSE3')
        loss_TL_test = loss_fn(ph["TasselLength"][test_idx], TL_mat[test_idx], method='RRMSE3')
        pred_ft_test = predicted_ft_day(ph["stage"][test_idx])
        loss_FT_test = loss_fn(pred_ft_test, FT_day[test_idx], method='RRMSE3')

        loss_test = torch.dot(weight, torch.stack([loss_PH_test, loss_HIR_test, loss_TL_test, loss_FT_test])) + beta.square().mean() * 10
        print(f"Epoch {epoch + 1:02d}  test, RRMSE = {loss_test.item():.4f}, \tFT = {loss_FT_test.item():.4f}, \tHIR = {loss_HIR_test.item():.4f}, \tHeight = {loss_PH_test.item():.4f}, \tTassel = {loss_TL_test.item():.4f}\n")

        improved = (loss_val_best - loss_val.item()) > min_delta
        if improved:
            loss_val_best = loss_val
            beta_best = beta.detach().clone()
            fail = 0

            fig, ax = plt.subplots()
            plt.scatter(ph["HIR"][train_idx][~HIR_mat[train_idx].isnan()].detach().cpu().numpy(), HIR_mat[train_idx][~HIR_mat[train_idx].isnan().detach().cpu().numpy()].cpu(), label='train')
            plt.scatter(ph["HIR"][test_idx][~HIR_mat[test_idx].isnan()].detach().cpu().numpy(), HIR_mat[test_idx][~HIR_mat[test_idx].isnan().detach().cpu().numpy()].cpu(), label='test')
            plt.plot([0, 0.4], [0, 0.4], 'red', linestyle='--')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            corr_test = round(loss_fn(ph["HIR"][test_idx], HIR_mat[test_idx], method='Pearson').item(), 3)
            plt.title('Test RMSE: ' + str(round(RMSE_HIR_test.item(), 3)) + '  Test RRMSE: ' + str(round(loss_HIR_test.item(), 3)) + '  Test Corr: ' + str(1-corr_test))
            plt.legend()

            plt.savefig(HIRfig_name, dpi=400)
            plt.close(fig)
            plot_success(ph["HIR"], HIR_mat, train_idx, test_idx, fig_name=precisionfig_name)

            best_config = {
                "run_dir": run_dir,
                "timestamp": run_name,
                "seed": seed,
                "device": str(device),
                "data_path": data_path,
                "phenotype_file": os.path.join(data_path, "Clean", "phenotype_all.csv"),
                "genotype_file": os.path.join(data_path, "Clean", "geno1.csv"),
                "environment_file": os.path.join(data_path, "Clean", "ames.xlsx"),
                "learning_rate_current": lr,
                "weight_current": [float(x) for x in weight.detach().cpu().tolist()],
                "test_size": int(test_size),
                "validate_size": int(validate_size),
                "n_samples": int(N),
                "best_epoch": int(epoch + 1),
                "status": "best_so_far",
            }

            best_train_sr_015, best_train_n_015 = success_rate_at_threshold(ph["HIR"][tra_idx], HIR_mat[tra_idx], 0.15)
            best_val_sr_015, best_val_n_015 = success_rate_at_threshold(ph["HIR"][val_idx], HIR_mat[val_idx], 0.15)
            best_test_sr_015, best_test_n_015 = success_rate_at_threshold(ph["HIR"][test_idx], HIR_mat[test_idx], 0.15)
            best_train_sr_020, best_train_n_020 = success_rate_at_threshold(ph["HIR"][tra_idx], HIR_mat[tra_idx], 0.20)
            best_val_sr_020, best_val_n_020 = success_rate_at_threshold(ph["HIR"][val_idx], HIR_mat[val_idx], 0.20)
            best_test_sr_020, best_test_n_020 = success_rate_at_threshold(ph["HIR"][test_idx], HIR_mat[test_idx], 0.20)

            best_metrics_lines = [
                f"status: best_so_far",
                f"epoch: {epoch + 1}",
                f"run_dir: {run_dir}",
                f"device: {device}",
                f"seed: {seed}",
                f"best_val_loss_during_training: {loss_val_best.item():.6f}",
                f"train_loss: {loss.item():.6f}",
                f"train_HIR_RRMSE: {loss_HIR.item():.6f}",
                f"train_HIR_RMSE: {RMSE_HIR.item():.6f}",
                f"val_loss: {loss_val.item():.6f}",
                f"val_HIR_RRMSE: {loss_HIR_val.item():.6f}",
                f"val_HIR_RMSE: {RMSE_HIR_val.item():.6f}",
                f"test_loss: {loss_test.item():.6f}",
                f"test_HIR_RRMSE: {loss_HIR_test.item():.6f}",
                f"test_HIR_RMSE: {RMSE_HIR_test.item():.6f}",
                f"train_success_rate_0.15: {best_train_sr_015:.6f} (n={best_train_n_015})",
                f"val_success_rate_0.15: {best_val_sr_015:.6f} (n={best_val_n_015})",
                f"test_success_rate_0.15: {best_test_sr_015:.6f} (n={best_test_n_015})",
                f"train_success_rate_0.20: {best_train_sr_020:.6f} (n={best_train_n_020})",
                f"val_success_rate_0.20: {best_val_sr_020:.6f} (n={best_val_n_020})",
                f"test_success_rate_0.20: {best_test_sr_020:.6f} (n={best_test_n_020})",
            ]

            write_run_artifacts(
                run_dir,
                config_path,
                metrics_path,
                best_config,
                best_metrics_lines,
                beta,
                geno,
                treno,
                Utreno,
                Ltreno,
                env,
                mg,
                G,
                Height_mat,
                HIR_mat,
                TL_mat,
                FT_day,
                train_idx,
                val_idx,
                test_idx,
                ph,
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "beta": beta.detach().cpu(),
                    "weight": weight.detach().cpu(),
                    "lr": lr,
                    "loss_val_best": loss_val_best.detach().cpu(),
                },
                checkpoint_path,
            )
        else:
            fail += 1
            if fail % lr_patience == 0 and lr > min_lr:
                lr = max(lr * 0.5, min_lr)
                optimizer = torch.optim.Adam([beta], lr=lr)
                print('lr: ' + str(lr))
            if fail >= early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    with torch.no_grad():
        tt = torch.stack([10*loss_PH, 100*loss_HIR, 2*loss_TL, 4*loss_FT])
        # tt = torch.stack([loss_PH, 0*loss_HIR, loss_TL, 4*loss_FT])
        weight = tt / sum(tt)

with torch.no_grad():
    beta.copy_(beta_best)

#%%
g = torch.matmul(G, beta[:, 0, :]).reshape(N, 4, 89) + torch.matmul((G == 1) * 1.0, beta[:, 1, :]).reshape(N, 4, 89)
geno = 1 / (1 + torch.exp(-g)) * (Utreno - Ltreno) + Ltreno
ph = dhcorn(env, mg, geno, device=device)

final_train_loss_HIR = loss_fn(ph["HIR"][tra_idx], HIR_mat[tra_idx], method='RRMSE')
final_train_rmse_HIR = loss_fn(ph["HIR"][tra_idx], HIR_mat[tra_idx], method='RMSE')
final_train_loss_PH = loss_fn(ph["Height"][tra_idx], Height_mat[tra_idx], method='RRMSE3')
final_train_loss_TL = loss_fn(ph["TasselLength"][tra_idx], TL_mat[tra_idx], method='RRMSE3')
final_train_loss_FT = loss_fn(predicted_ft_day(ph["stage"][tra_idx]), FT_day[tra_idx], method='RRMSE3')
final_train_loss = torch.dot(weight, torch.stack([final_train_loss_PH, final_train_loss_HIR, final_train_loss_TL, final_train_loss_FT])) + beta.square().mean() * 10

final_val_loss_HIR = loss_fn(ph["HIR"][val_idx], HIR_mat[val_idx], method='RRMSE')
final_val_rmse_HIR = loss_fn(ph["HIR"][val_idx], HIR_mat[val_idx], method='RMSE')
final_val_loss_PH = loss_fn(ph["Height"][val_idx], Height_mat[val_idx], method='RRMSE3')
final_val_loss_TL = loss_fn(ph["TasselLength"][val_idx], TL_mat[val_idx], method='RRMSE3')
final_val_loss_FT = loss_fn(predicted_ft_day(ph["stage"][val_idx]), FT_day[val_idx], method='RRMSE3')
final_val_loss = torch.dot(weight, torch.stack([final_val_loss_PH, final_val_loss_HIR, final_val_loss_TL, final_val_loss_FT])) + beta.square().mean() * 10

final_test_loss_HIR = loss_fn(ph["HIR"][test_idx], HIR_mat[test_idx], method='RRMSE')
final_test_rmse_HIR = loss_fn(ph["HIR"][test_idx], HIR_mat[test_idx], method='RMSE')
final_test_loss_PH = loss_fn(ph["Height"][test_idx], Height_mat[test_idx], method='RRMSE3')
final_test_loss_TL = loss_fn(ph["TasselLength"][test_idx], TL_mat[test_idx], method='RRMSE3')
final_test_loss_FT = loss_fn(predicted_ft_day(ph["stage"][test_idx]), FT_day[test_idx], method='RRMSE3')
final_test_loss = torch.dot(weight, torch.stack([final_test_loss_PH, final_test_loss_HIR, final_test_loss_TL, final_test_loss_FT])) + beta.square().mean() * 10

config = {
    "run_dir": run_dir,
    "timestamp": run_name,
    "seed": seed,
    "device": str(device),
    "data_path": data_path,
    "phenotype_file": os.path.join(data_path, "Clean", "phenotype_all.csv"),
    "genotype_file": os.path.join(data_path, "Clean", "geno1.csv"),
    "environment_file": os.path.join(data_path, "Clean", "ames.xlsx"),
    "learning_rate_final": lr,
    "weight_final": [float(x) for x in weight.detach().cpu().tolist()],
    "test_size": int(test_size),
    "validate_size": int(validate_size),
    "n_samples": int(N),
    "status": "finished",
}

train_sr_015, train_n_015 = success_rate_at_threshold(ph["HIR"][tra_idx], HIR_mat[tra_idx], 0.15)
val_sr_015, val_n_015 = success_rate_at_threshold(ph["HIR"][val_idx], HIR_mat[val_idx], 0.15)
test_sr_015, test_n_015 = success_rate_at_threshold(ph["HIR"][test_idx], HIR_mat[test_idx], 0.15)
train_sr_020, train_n_020 = success_rate_at_threshold(ph["HIR"][tra_idx], HIR_mat[tra_idx], 0.20)
val_sr_020, val_n_020 = success_rate_at_threshold(ph["HIR"][val_idx], HIR_mat[val_idx], 0.20)
test_sr_020, test_n_020 = success_rate_at_threshold(ph["HIR"][test_idx], HIR_mat[test_idx], 0.20)

final_metrics_lines = [
    f"status: finished",
    f"run_dir: {run_dir}",
    f"device: {device}",
    f"seed: {seed}",
    f"best_val_loss_during_training: {loss_val_best.item():.6f}",
    f"final_train_loss: {final_train_loss.item():.6f}",
    f"final_train_HIR_RRMSE: {final_train_loss_HIR.item():.6f}",
    f"final_train_HIR_RMSE: {final_train_rmse_HIR.item():.6f}",
    f"final_val_loss: {final_val_loss.item():.6f}",
    f"final_val_HIR_RRMSE: {final_val_loss_HIR.item():.6f}",
    f"final_val_HIR_RMSE: {final_val_rmse_HIR.item():.6f}",
    f"final_test_loss: {final_test_loss.item():.6f}",
    f"final_test_HIR_RRMSE: {final_test_loss_HIR.item():.6f}",
    f"final_test_HIR_RMSE: {final_test_rmse_HIR.item():.6f}",
    f"train_success_rate_0.15: {train_sr_015:.6f} (n={train_n_015})",
    f"val_success_rate_0.15: {val_sr_015:.6f} (n={val_n_015})",
    f"test_success_rate_0.15: {test_sr_015:.6f} (n={test_n_015})",
    f"train_success_rate_0.20: {train_sr_020:.6f} (n={train_n_020})",
    f"val_success_rate_0.20: {val_sr_020:.6f} (n={val_n_020})",
    f"test_success_rate_0.20: {test_sr_020:.6f} (n={test_n_020})",
]

write_run_artifacts(
    run_dir,
    config_path,
    metrics_path,
    config,
    final_metrics_lines,
    beta,
    geno,
    treno,
    Utreno,
    Ltreno,
    env,
    mg,
    G,
    Height_mat,
    HIR_mat,
    TL_mat,
    FT_day,
    train_idx,
    val_idx,
    test_idx,
    ph,
)

#%%
# geno = torch.load('geno.pt')
# env = torch.load('env.pt')
# mg = torch.load('mg.pt')
#
# Height = torch.load('Height.pt')
# HIR = torch.load('HIR.pt')
# TL = torch.load('TL.pt')
#
# ph = dhcorn(env, mg, geno)
#
# loss_PH = ((ph["Height"] - Height) ** 2).nanmean().sqrt() / Height.nanmean()
# loss_HIR = ((ph["HIR"] - HIR) ** 2).nanmean().sqrt() / HIR.nanmean()
# loss_TL = ((ph["TasselLength"] - TL) ** 2).nanmean().sqrt() / TL.nanmean()
# loss = 0.6 * loss_HIR + 0.2 * loss_PH + 0.2 * loss_TL
#
