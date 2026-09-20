import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class StandardizationStats:
    env_mean: torch.Tensor
    env_std: torch.Tensor
    G_mean: torch.Tensor
    G_std: torch.Tensor
    mg_mean: torch.Tensor
    mg_std: torch.Tensor


class StableHIRCNNLSTM(nn.Module):
    def __init__(
        self,
        geno_dim,
        mg_dim=2,
        env_dim=13,
        conv_channels=32,
        kernel_size=5,
        pool_size=4,
        lstm_hidden=64,
        mlp_hidden=128,
        dropout=0.2,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(env_dim, conv_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.pool = nn.MaxPool1d(pool_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
        )
        self.geno_mlp = nn.Sequential(
            nn.Linear(geno_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 64),
            nn.ReLU(),
        )
        self.mg_mlp = nn.Sequential(
            nn.Linear(mg_dim, 16),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden + 64 + 16, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, env, geno, mg):
        x_env = self.conv1(env.permute(0, 2, 1))
        x_env = self.relu(x_env)
        x_env = self.conv2(x_env)
        x_env = self.relu(x_env)
        x_env = self.pool(x_env)
        x_env = x_env.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x_env)
        x_env_last = h_n[-1]
        x_g = self.geno_mlp(geno)
        x_mg = self.mg_mlp(mg)
        x = torch.cat([x_env_last, x_g, x_mg], dim=1)
        return torch.sigmoid(self.head(x)).squeeze(1)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))


def load_base_dataset(data_path="Data/"):
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
        for rep_col, rep in [(1, 1), (2, 2)]:
            dt = pd.Timestamp(PD.iloc[i, rep_col] + " 09:00:00")
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
    label_mask = ~torch.isnan(HIR)
    return pheno, env, G, mg, HIR, label_mask


def compute_standardization_stats(env, G, mg, train_idx):
    return StandardizationStats(
        env_mean=env[train_idx].mean(dim=(0, 1), keepdim=True),
        env_std=env[train_idx].std(dim=(0, 1), keepdim=True).clamp(min=1e-6),
        G_mean=G[train_idx].mean(dim=0, keepdim=True),
        G_std=G[train_idx].std(dim=0, keepdim=True).clamp(min=1e-6),
        mg_mean=mg[train_idx].mean(dim=0, keepdim=True),
        mg_std=mg[train_idx].std(dim=0, keepdim=True).clamp(min=1e-6),
    )


def normalize_inputs(env, G, mg, stats):
    env_n = (env - stats.env_mean) / stats.env_std
    G_n = (G - stats.G_mean) / stats.G_std
    mg_n = (mg - stats.mg_mean) / stats.mg_std
    return env_n, G_n, mg_n


def build_dataloaders(env, G, mg, HIR, train_idx, val_idx, test_idx, batch_size=32):
    train_ds = TensorDataset(env[train_idx], G[train_idx], mg[train_idx], HIR[train_idx])
    val_ds = TensorDataset(env[val_idx], G[val_idx], mg[val_idx], HIR[val_idx])
    test_ds = TensorDataset(env[test_idx], G[test_idx], mg[test_idx], HIR[test_idx])
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


def rmse_metric(y_true, y_pred):
    return torch.sqrt(((y_pred - y_true) ** 2).mean())


def rrmse_metric(y_true, y_pred):
    top_k = max(1, int(len(y_true) * 0.2))
    top_idx = torch.topk(y_true, top_k).indices
    return rmse_metric(y_true, y_pred) / y_true[top_idx].mean().clamp(min=1e-6)


def pearson_corr(y_true, y_pred):
    y_true = y_true.float()
    y_pred = y_pred.float()
    vx = y_true - y_true.mean()
    vy = y_pred - y_pred.mean()
    denom = torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum())
    if denom <= 0:
        return torch.tensor(0.0, device=y_true.device)
    return (vx * vy).sum() / denom


def evaluate_model(model, data_loader, device):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for xb_env, xb_g, xb_mg, yb in data_loader:
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


def success_rate_at_threshold(pred, true, threshold):
    selected = pred > threshold
    count = int(selected.sum().item())
    if count == 0:
        return float("nan"), 0
    return (true[selected] > threshold).float().mean().item(), count


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


def train_cnnlstm_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    lr_init=3e-4,
    weight_decay=1e-4,
    max_epochs=400,
    early_stop_patience=50,
    lr_patience=15,
    min_lr=1e-6,
    min_delta=5e-4,
    grad_clip_norm=1.0,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init, weight_decay=weight_decay)
    criterion = nn.HuberLoss(delta=0.05)
    lr = lr_init
    best_val = float("inf")
    fail = 0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    history = []

    for epoch in range(max_epochs):
        model.train()
        epoch_losses = []
        for xb_env, xb_g, xb_mg, yb in train_loader:
            xb_env = xb_env.to(device)
            xb_g = xb_g.to(device)
            xb_mg = xb_mg.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb_env, xb_g, xb_mg)
            loss = criterion(pred, yb)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            epoch_losses.append(loss.item())

        train_eval = evaluate_model(model, train_loader, device)
        val_eval = evaluate_model(model, val_loader, device)
        test_eval = evaluate_model(model, test_loader, device)

        improved = (best_val - val_eval["rrmse"].item()) > min_delta
        history.append(
            {
                "epoch": epoch + 1,
                "lr": lr,
                "train_total_loss": float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
                "train_hir_rrmse": train_eval["rrmse"].item(),
                "val_hir_rrmse": val_eval["rrmse"].item(),
                "test_hir_rrmse": test_eval["rrmse"].item(),
                "train_hir_rmse": train_eval["rmse"].item(),
                "val_hir_rmse": val_eval["rmse"].item(),
                "test_hir_rmse": test_eval["rmse"].item(),
                "train_corr": train_eval["corr"].item(),
                "val_corr": val_eval["corr"].item(),
                "test_corr": test_eval["corr"].item(),
                "improved": int(improved),
            }
        )

        if improved:
            best_val = val_eval["rrmse"].item()
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            fail = 0
        else:
            fail += 1
            if fail % lr_patience == 0 and lr > min_lr:
                lr = max(lr * 0.5, min_lr)
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            if fail >= early_stop_patience:
                break

    model.load_state_dict(best_state)
    train_eval = evaluate_model(model, train_loader, device)
    val_eval = evaluate_model(model, val_loader, device)
    test_eval = evaluate_model(model, test_loader, device)
    return {
        "model": model,
        "history": history,
        "best_val_hir_rrmse": best_val,
        "best_epoch": best_epoch,
        "final_lr": lr,
        "train_eval": train_eval,
        "val_eval": val_eval,
        "test_eval": test_eval,
    }
