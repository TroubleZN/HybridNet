#%%
import os

import pandas as pd
import torch
from Code.DHCorn_hour import dhcorn
from Code.make_g import make_treno
from Code.utils import simulate_growth_curves
import matplotlib.pyplot as plt
import numpy as np

#%% Check GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# device = 'cpu'

print(f"当前设备: {device}")

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
    mg[i, 0] = (harvest - plant).days * 24


# %%
H = 130 * 24
# geno_t = torch.tensor(make_treno(), device=device).requires_grad_()
# geno = geno_t.repeat(N, 1, 1)

treno, Utreno, Ltreno = make_treno()
Utreno = torch.tensor(Utreno, device=device)
Ltreno = torch.tensor(Ltreno, device=device)

#%%
# geno = torch.tensor(treno, device=device).repeat(N, 1, 1).requires_grad_()
# ph = dhcorn(env, mg, geno, device=device)

# G = torch.randint(0, 3, (N, 2000), dtype=torch.float, device=device)
# beta = torch.zeros(2000, 3*76, device=device, requires_grad=True)
# g = torch.matmul(G, beta).reshape(N, 3, 76)
# geno = 1 / (1 + torch.exp(-g)) * (Utreno - Ltreno) + Ltreno

#%%
Height = torch.tensor(pheno['PH'].to_numpy(), device=device)
Height[Height == 0] = torch.nan
Height_mat = torch.zeros(N, 130*24, device=device)
Height_mat[:] = torch.nan
for i in range(N):
    if ~Height[i].isnan():
        Height_mat[i, int(mg[i, 0])] = Height[i]
# Height = Height_mat

HIR = torch.tensor(pheno['HIR'].to_numpy(), device=device) / 100
HIR_mat = torch.zeros(N, 130*24, device=device)
HIR_mat[:] = torch.nan
for i in range(N):
    if ~HIR[i].isnan():
        HIR_mat[i, int(mg[i, 0])] = HIR[i]
# HIR = HIR_mat

TL = torch.tensor(pheno['TL'].to_numpy(), device=device)
TL_mat = torch.zeros(N, 130*24, device=device)
TL_mat[:] = torch.nan
for i in range(N):
    if ~TL[i].isnan():
        TL_mat[i, int(mg[i, 0])] = TL[i]
# TL = TL_mat

FT = torch.tensor(pheno['FT'].to_numpy(), device=device)
FT_mat = torch.zeros(N, 4, 130*24, device=device)
FT_mat[:] = torch.nan
for i in range(N):
    if ~FT[i].isnan():
        FT_mat[i, :, int(FT[i])] = torch.tensor([0, 0, 1, 0])
        FT_mat[i, :, int(FT[i] + 240)] = torch.tensor([0, 0, 0, 1])
        FT_mat[i, :, int(FT[i] - 240)] = torch.tensor([0, 1, 0, 0])

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

#%%
perm = torch.randperm(N)
train_size = int(0.7 * N)

train_idx = perm[:train_size]
test_idx = perm[train_size:]


#%%
# G = torch.randint(0, 3, (N, 10000), device=device) * 1.0
beta = torch.zeros(G.shape[1], 2, 4*89, device=device, requires_grad=True)

weight = torch.tensor([0.25, 0.25, 0.25, 0.25], device=device)
beta_best = beta.detach().clone()
loss_val_best = 100000
fail = 0

if os.path.isfile("HIR_hour.png"):
    i = 2
    while os.path.isfile("HIR_hour_" + str(i) + ".png"):
        i += 1
    HIRfig_name = "HIR_hour_" + str(i) + ".png"
    successfig_name = 'success_rate_hour_' + str(i) + '.png'
else:
    HIRfig_name = "HIR_hour.png"
    successfig_name = 'success_rate_hour.png'

# %%
import gc
optimizer = torch.optim.Adam([beta], lr=1e-5)

for epoch in range(5000):
    if fail > 10:
        break

    optimizer.zero_grad()

    perm = torch.randperm(train_size)
    validate_size = int(0.1 * train_size)

    val_idx = train_idx[perm[:validate_size]]
    tra_idx = train_idx[perm[validate_size:]]

    g = torch.matmul(G, beta[:, 0, :]).reshape(N, 4, 89) + torch.matmul((G == 1) * 1.0, beta[:, 1, :]).reshape(N, 4, 89)
    geno = 1 / (1 + torch.exp(-g)) * (Utreno - Ltreno) + Ltreno

    ph = dhcorn(env, mg, geno, device=device)

    loss_PH = loss_fn(ph["Height"][tra_idx], Height_mat[tra_idx])
    loss_HIR = loss_fn(ph["HIR"][tra_idx], HIR_mat[tra_idx])
    loss_TL = loss_fn(ph["TasselLength"][tra_idx], TL_mat[tra_idx])
    loss_FT = loss_fn(ph["stage"][tra_idx], FT_mat[tra_idx])

    loss = torch.dot(weight, torch.stack([loss_PH, loss_HIR, loss_TL, loss_FT])) + beta.square().mean() * 10

    loss.backward()

    # mask = torch.zeros_like(beta)
    # for i in range(4):  # 4个模块
    #     start = i * 89 + (89 - 24)
    #     end = i * 89 + 89
    #     mask[:, :, start:end] = 1
    # with torch.no_grad():
    #     beta.grad *= mask

    optimizer.step()
    print(f"Epoch {epoch + 1:02d} train, RRMSE = {loss.item():.4f}, \tFT = {loss_FT.item():.4f}, \tHIR = {loss_HIR.item():.4f}, \tHeight = {loss_PH.item():.4f}, \tTassel = {loss_TL.item():.4f}, \t Fail = {fail:02d}")

    with torch.no_grad():
        loss_PH_val = loss_fn(ph["Height"][val_idx], Height_mat[val_idx])
        loss_HIR_val = loss_fn(ph["HIR"][val_idx], HIR_mat[val_idx])
        loss_TL_val = loss_fn(ph["TasselLength"][val_idx], TL_mat[val_idx])
        loss_FT_val = loss_fn(ph["stage"][val_idx], FT_mat[val_idx])

        loss_val = torch.dot(weight, torch.stack([loss_PH_val, loss_HIR_val, loss_TL_val, loss_FT_val])) + beta.square().mean() * 10
        print(f"Epoch {epoch + 1:02d}   val, RRMSE = {loss_val.item():.4f}, \tFT = {loss_FT_val.item():.4f}, \tHIR = {loss_HIR_val.item():.4f}, \tHeight = {loss_PH_val.item():.4f}, \tTassel = {loss_TL_val.item():.4f}")

        loss_PH_test = loss_fn(ph["Height"][test_idx], Height_mat[test_idx])
        loss_HIR_test = loss_fn(ph["HIR"][test_idx], HIR_mat[test_idx])
        loss_TL_test = loss_fn(ph["TasselLength"][test_idx], TL_mat[test_idx])
        loss_FT_test = loss_fn(ph["stage"][test_idx], FT_mat[test_idx])

        loss_test = torch.dot(weight, torch.stack([loss_PH_test, loss_HIR_test, loss_TL_test, loss_FT_test])) + beta.square().mean() * 10
        print(f"Epoch {epoch + 1:02d}  test, RRMSE = {loss_test.item():.4f}, \tFT = {loss_FT_test.item():.4f}, \tHIR = {loss_HIR_test.item():.4f}, \tHeight = {loss_PH_test.item():.4f}, \tTassel = {loss_TL_test.item():.4f}\n")

        if (loss_HIR_val - loss_HIR)/loss_HIR < 0.1:
            loss_val_best = loss_HIR_val
            beta_best = beta.detach().clone()
            fail = 0
        else:
            with torch.no_grad():
                beta.copy_(beta_best)
            fail += 1
        #
        # fig, ax = plt.subplots()
        # plt.scatter(ph["HIR"][train_idx][~HIR_mat[train_idx].isnan()].detach().cpu().numpy(), HIR_mat[train_idx][~HIR_mat[train_idx].isnan().detach().cpu().numpy()].cpu(), label='train')
        # plt.scatter(ph["HIR"][test_idx][~HIR_mat[test_idx].isnan()].detach().cpu().numpy(), HIR_mat[test_idx][~HIR_mat[test_idx].isnan().detach().cpu().numpy()].cpu(), label='test')
        # plt.plot([0, 0.4], [0, 0.4], 'red', linestyle='--')
        # plt.xlabel('Predicted')
        # plt.ylabel('True')
        # plt.legend()
        # plt.savefig("HIR_server.png")
        # # plt.show()

        fig, ax = plt.subplots()
        plt.scatter(ph["HIR"][train_idx][~HIR_mat[train_idx].isnan()].detach().cpu().numpy(), HIR_mat[train_idx][~HIR_mat[train_idx].isnan().detach().cpu().numpy()].cpu(), label='train')
        plt.scatter(ph["HIR"][test_idx][~HIR_mat[test_idx].isnan()].detach().cpu().numpy(), HIR_mat[test_idx][~HIR_mat[test_idx].isnan().detach().cpu().numpy()].cpu(), label='test')
        plt.plot([0, 0.4], [0, 0.4], 'red', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        corr_test = round(loss_fn(ph["HIR"][test_idx], HIR_mat[test_idx], method='Pearson').item(), 3)
        plt.title('Test RMSE: ' + str(round(RMSE_HIR_test.item(), 3)) + '  Test RRMSE: ' + str(round(loss_HIR_test.item(), 3)) + '  Test Corr: ' + str(1-corr_test))
        plt.legend()

        plt.savefig(HIRfig_name)
        plot_success(ph["HIR"], HIR_mat, train_idx, test_idx, fig_name=successfig_name)



    # # 显式删除变量，释放前一轮图
    # torch.cuda.empty_cache()
    # gc.collect()

    with torch.no_grad():
        weight = torch.stack([loss_PH, loss_HIR, loss_TL, loss_FT]) / sum(torch.tensor([loss_PH, loss_HIR, loss_TL, loss_FT]))


#%%
fig, ax = plt.subplots()
plt.plot(range(130), ph["HIR"].detach().cpu().numpy().cpu(), label='HIR')
plt.plot(range(130), ph["Height"].detach().cpu().numpy().cpu(), label='HIR')
plt.plot(range(130), ph["HIR"].detach().cpu().numpy().cpu(), label='HIR')

plt.xlabel('Day')
plt.ylabel('Value')
plt.legend()
plt.savefig("Server/Pheno.png")
# plt.show()

#%%
y_pred_train = ph["HIR"][train_idx]
y_true_train = HIR_mat[train_idx]

y_pred_test = ph["HIR"][test_idx]
y_true_test = HIR_mat[test_idx]

id = ~y_true_train.isnan()
y_pred_train = y_pred_train[id].detach().cpu().numpy()
y_true_train = y_true_train[id].detach().cpu().numpy()

id = ~y_true_test.isnan()
y_pred_test = y_pred_test[id].detach().cpu().numpy()
y_true_test = y_true_test[id].detach().cpu().numpy()

thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]

def compute_success_rate(y_pred, y_true, thresholds):
    results = []
    for t in thresholds:
        mask = y_pred > t
        if np.sum(mask) == 0:
            continue
        success = y_true[mask] > t
        results.append({
            'Threshold': t,
            'Count': np.sum(mask),
            'SuccessRate': np.mean(success)
        })
    return pd.DataFrame(results)

thresholds = np.round(np.arange(0.05, 0.2, 0.025), 3)
df_train = compute_success_rate(y_pred_train, y_true_train, thresholds)
df_test = compute_success_rate(y_pred_test, y_true_test, thresholds)


# 使用右轴显示 Success Rate
fig, ax1 = plt.subplots(figsize=(8, 5), dpi=400)
ax1.plot(df_train['Threshold'], df_train['SuccessRate'], 'o-', label='Train', color='blue')
ax1.plot(df_test['Threshold'], df_test['SuccessRate'], 'o-', label='Test', color='orange')
ax1.set_xlabel('Predicted HIR > Threshold')
ax1.set_ylabel('Success Rate')
ax1.set_ylim(0, 1)
ax1.set_yticks(np.round(np.arange(0, 1.1, 0.1), 1), [str(int(t*100)) + '%' for t in np.round(np.arange(0, 1.1, 0.1), 1)])
ax1.set_xticks(df_train['Threshold'], [str(t*100) + '%' for t in thresholds])
ax1.tick_params(axis='y')
ax1.legend()

for x, y in zip(df_train['Threshold'], df_train['SuccessRate']):
    ax1.annotate(f'{y*100:.1f}%', xy=(x, y), xytext=(0, 5), textcoords='offset points',
                 ha='center', fontsize=8, color='blue')
for x, y in zip(df_train['Threshold'], df_test['SuccessRate']):
    ax1.annotate(f'{y*100:.1f}%', xy=(x, y), xytext=(0, 5), textcoords='offset points',
                 ha='center', fontsize=8, color='orange')

# 样本数（右轴）
ax2 = ax1.twinx()
bar_width = 0.008
bars_train = ax2.bar(df_train['Threshold'] - 0.005, df_train['Count'],
                     width=bar_width, alpha=0.2, color='blue', label='Train Count')
bars_test = ax2.bar(df_test['Threshold'] + 0.005, df_test['Count'],
                    width=bar_width, alpha=0.2, color='orange', label='Test Count')
ax2.set_ylabel('Sample Count')
ax2.set_ylim(0, 500)
ax2.set_yticks(np.round(np.arange(0, 550, 50), 1))
ax2.tick_params(axis='y')

for bar in bars_train:
    height = bar.get_height()
    ax2.annotate(f'{int(height)}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=8, color='blue')

for bar in bars_test:
    height = bar.get_height()
    ax2.annotate(f'{int(height)}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=8, color='orange')

plt.title('Success Rate by Predicted HIR Threshold (Train vs Test)')
plt.grid(True)
plt.tight_layout()
fig.savefig("Server/success_rate.png")

#%%
# torch.save(geno.detach().cpu(), "Result/geno.pt")
# torch.save(G.detach().cpu(), "Result/G.pt")
# torch.save(beta.detach().cpu(), "Result/beta.pt")
#
# torch.save(env.detach().cpu(), "Result/env.pt")
# torch.save(mg.detach().cpu(), "Result/mg.pt")
#
# torch.save(train_idx.detach().cpu(), "Result/train_idx.pt")
# torch.save(test_idx.detach().cpu(), "Result/test_idx.pt")
#
# torch.save(Height_mat.detach().cpu(), "Result/Height.pt")
# torch.save(HIR_mat.detach().cpu(), "Result/HIR.pt")
# torch.save(TL_mat.detach().cpu(), "Result/TL.pt")
# torch.save(FT_mat.detach().cpu(), "Result/FT.pt")

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
