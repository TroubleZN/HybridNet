import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

env = env.float()
G = G.float()
mg = mg.float()
HIR = HIR.float()

# 清洗 NaN 标签
mask_train = ~torch.isnan(HIR[train_idx])
mask_test = ~torch.isnan(HIR[test_idx])

env_train = env[train_idx][mask_train]
G_train = G[train_idx][mask_train]
mg_train = mg[train_idx][mask_train]
HIR_train = HIR[train_idx][mask_train]

env_test = env[test_idx][mask_test]
G_test = G[test_idx][mask_test]
mg_test = mg[test_idx][mask_test]
HIR_test = HIR[test_idx][mask_test]

# 构建 Dataset 与 Dataloader
train_ds = TensorDataset(env_train, G_train, mg_train, HIR_train)
test_ds = TensorDataset(env_test, G_test, mg_test, HIR_test)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=32)

# Create Dataloaders
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=32)

#%%
# 模型定义
class HIRPredictorCNNLSTM(nn.Module):
    def __init__(self, geno_dim=3768, mg_dim=2, env_dim=13, lstm_hidden=64, mlp_hidden=128):
        super().__init__()
        self.conv1 = nn.Conv1d(env_dim, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(4)
        self.lstm = nn.LSTM(input_size=32, hidden_size=lstm_hidden, batch_first=True)
        self.mlp_geno = nn.Sequential(
            nn.Linear(geno_dim, mlp_hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(mlp_hidden, 64)
        )
        self.mlp_mg = nn.Sequential(nn.Linear(mg_dim, 16), nn.ReLU())
        self.final = nn.Sequential(
            nn.Linear(lstm_hidden + 64 + 16, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, env, G, mg):
        x = self.conv1(env.permute(0, 2, 1))  # [B, 13, 3120] → [B, 32, 3120]
        x = self.pool(x)                      # → [B, 32, 780]
        x = x.permute(0, 2, 1)                # → [B, 780, 32]
        _, (h_lstm, _) = self.lstm(x)         # [1, B, H] → 取 h_lstm[-1]
        x_env = h_lstm[-1]
        x_g = self.mlp_geno(G)
        x_mg = self.mlp_mg(mg)
        x_all = torch.cat([x_env, x_g, x_mg], dim=1)
        return self.final(x_all).squeeze(1)

# 初始化模型与训练配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HIRPredictorCNNLSTM().to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)  # 加L2正则

# 辅助函数
def compute_metrics(y_true, y_pred):
    mask = ~torch.isnan(y_true)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    rmse = torch.sqrt(((y_pred - y_true) ** 2).mean()).item()
    rrmse = (torch.sqrt(((y_pred - y_true) ** 2).mean()) / y_true.mean()).item()
    corr = torch.corrcoef(torch.stack([y_true, y_pred]))[0, 1].item()
    return rmse, rrmse, corr

#%% 训练过程
for epoch in range(500):
    model.train()
    all_train_preds = []
    all_train_labels = []
    for xb_env, xb_g, xb_mg, yb in train_dl:
        xb_env, xb_g, xb_mg, yb = xb_env.to(device), xb_g.to(device), xb_mg.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb_env, xb_g, xb_mg)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        all_train_preds.append(pred.detach().cpu())
        all_train_labels.append(yb.detach().cpu())

    model.eval()
    all_test_preds = []
    all_test_labels = []
    with torch.no_grad():
        for xb_env, xb_g, xb_mg, yb in test_dl:
            xb_env, xb_g, xb_mg, yb = xb_env.to(device), xb_g.to(device), xb_mg.to(device), yb.to(device)
            pred = model(xb_env, xb_g, xb_mg)
            all_test_preds.append(pred.cpu())
            all_test_labels.append(yb.cpu())

    # 合并并计算指标
    train_pred = torch.cat(all_train_preds)
    train_true = torch.cat(all_train_labels)
    test_pred = torch.cat(all_test_preds)
    test_true = torch.cat(all_test_labels)

    train_rmse, train_rrmse, train_corr = compute_metrics(train_true, train_pred)
    test_rmse, test_rrmse, test_corr = compute_metrics(test_true, test_pred)

    print(f"Epoch {epoch+1}:")
    print(f"  Train -> RMSE: {train_rmse:.4f}, RRMSE: {train_rrmse:.4f}, Corr: {train_corr:.4f}")
    print(f"  Test  -> RMSE: {test_rmse:.4f}, RRMSE: {test_rrmse:.4f}, Corr: {test_corr:.4f}")

#%%
fig, ax = plt.subplots()
plt.scatter(train_pred, train_true, label='train')
plt.scatter(test_pred, test_true, label='test')
plt.plot([0, 0.4], [0, 0.4], 'red', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('True')
corr_test = test_corr
plt.title('Train RRMSE: ' + str(round(train_rrmse, 3)) + '  Test RRMSE: ' + str(
    round(test_rrmse, 3)) + '  Test Corr: ' + str(round(test_corr, 3)))
plt.legend()

plt.savefig('LSTM.png')

plot_success(
    torch.cat([train_pred, test_pred]),
    torch.cat([train_true, test_true]),
    range(len(train_pred)),
    range(len(train_pred), len(train_pred) + len(test_pred)),
    fig_name='success_LSTM.png'
)