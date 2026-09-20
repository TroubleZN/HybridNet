import torch
import numpy as np
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


#%%
# HIR:      [916]
# env:      [916, 3120, 13]
# mg:       [916, 2]
# G:        [916, 3768]

#%% Step 1: 聚合环境变量（每20天一个窗口 = 480小时）
window_size = 240
num_windows = env.shape[1] // window_size
env_w = env[:, :num_windows * window_size, :].reshape(env.shape[0], num_windows, window_size, env.shape[2])
env_final = torch.cat([
    env_w.mean(dim=2),
    # env_w.std(dim=2),
    env_w.max(dim=2)[0],
    env_w.min(dim=2)[0],
], dim=2).reshape(env.shape[0], -1)  # [916, 312]

#%% Step 2: 拼接所有特征
X_all = torch.cat([mg, G, env_final.reshape(env.shape[0], -1)], dim=1).cpu().numpy()  # shape = [916, 4082]
y_all = HIR.cpu().numpy()

#%% Step 4: Train/Test split
# 对于 train：只保留非 NaN
train_mask = ~np.isnan(y_all[train_idx])
X_train = X_all[train_idx][train_mask]
y_train = y_all[train_idx][train_mask]

# 对于 test：同样过滤 NaN
test_mask = ~np.isnan(y_all[test_idx])
X_test = X_all[test_idx][test_mask]
y_test = y_all[test_idx][test_mask]

#%% Step 5: 模型训练
model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

#%% Step 6: 评估
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
rrmse = rmse/y_pred.mean()
r2 = r2_score(y_test, y_pred)
corr = np.corrcoef(y_test, y_pred)[0, 1]

print(f"RMSE: {rmse:.4f}")
print(f"RRMSE: {rrmse:.4f}")
print(f"R²:   {r2:.4f}")
print(f"Pearson Correlation: {corr:.4f}")

#%%
fig, ax = plt.subplots()
plt.scatter(model.predict(X_train), y_train, label='train')
plt.scatter(model.predict(X_test), y_test, label='test')
plt.plot([0, 0.4], [0, 0.4], 'red', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('True')
corr_test = corr
plt.title('Test RMSE: ' + str(round(rmse, 3)) + '  Test RRMSE: ' + str(
    round(rrmse, 3)) + '  Test Corr: ' + str(round(corr, 3)))
plt.legend()

plt.savefig('XGBoost.png')

plot_success(
    torch.cat([torch.tensor(model.predict(X_train)), torch.tensor(model.predict(X_test))]),
    torch.cat([torch.tensor(y_train), torch.tensor(y_test)]),
    range(len(model.predict(X_train))),
    range(len(model.predict(X_train)), len(model.predict(X_train)) + len(model.predict(X_test))),
    fig_name='success_XGBoost.png'
)

#%% Step 7: 特征重要性图
xgb.plot_importance(model, max_num_features=20, importance_type='gain')
plt.tight_layout()
plt.savefig('xgboost.png')
# plt.show()


# 训练 RR-BLUP = ridge regression
#%% Step 2: 拼接所有特征
X_all = torch.cat([mg, G], dim=1).cpu().numpy()  # shape = [916, 4082]
y_all = HIR.cpu().numpy()

#%% Step 4: Train/Test split
# 对于 train：只保留非 NaN
train_mask = ~np.isnan(y_all[train_idx])
X_train = X_all[train_idx][train_mask]
y_train = y_all[train_idx][train_mask]

# 对于 test：同样过滤 NaN
test_mask = ~np.isnan(y_all[test_idx])
X_test = X_all[test_idx][test_mask]
y_test = y_all[test_idx][test_mask]

#%%
model = Ridge(alpha=12)  # alpha 是正则强度，等价于 σ²/τ²
model.fit(X_train, y_train)

# 预测 & 评估
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
rrmse = rmse/y_pred.mean()
r2 = r2_score(y_test, y_pred)
corr = np.corrcoef(y_test, y_pred)[0, 1]

print(f"RMSE: {rmse:.4f}")
print(f"RRMSE: {rrmse:.4f}")
print(f"R²:   {r2:.4f}")
print(f"Pearson Correlation: {corr:.4f}")


#%%
fig, ax = plt.subplots()
plt.scatter(model.predict(X_train), y_train, label='train')
plt.scatter(model.predict(X_test), y_test, label='test')
plt.plot([0, 0.4], [0, 0.4], 'red', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('True')
corr_test = corr
plt.title('Test RMSE: ' + str(round(rmse, 3)) + '  Test RRMSE: ' + str(
    round(rrmse, 3)) + '  Test Corr: ' + str(round(corr, 3)))
plt.legend()

plt.savefig('RR.png')

plot_success(
    torch.cat([torch.tensor(model.predict(X_train)), torch.tensor(model.predict(X_test))]),
    torch.cat([torch.tensor(y_train), torch.tensor(y_test)]),
    range(len(model.predict(X_train))),
    range(len(model.predict(X_train)), len(model.predict(X_train)) + len(model.predict(X_test))),
    fig_name='success_RR.png'
)
