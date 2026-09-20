import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import torch

from Code.cnnlstm_shared import (
    StableHIRCNNLSTM,
    build_dataloaders,
    compute_standardization_stats,
    evaluate_model,
    get_device,
    load_base_dataset,
    normalize_inputs,
    precision_recall_at_threshold,
    set_seed,
    success_rate_at_threshold,
    train_cnnlstm_model,
)
from Code.res_plot import plot_success


device = get_device()
print(f"当前设备: {device}")

seed = 20260314
set_seed(seed)
print(f"随机种子: {seed}")

pheno, env, G, mg, HIR, label_mask = load_base_dataset()
N = len(pheno)

perm = torch.randperm(N)
test_size = int(N * 0.3)
test_idx = perm[:test_size]
train_idx = perm[test_size:]

perm2 = torch.randperm(len(train_idx))
validate_size = int(0.1 * N)
val_idx = train_idx[perm2[:validate_size]]
tra_idx = train_idx[perm2[validate_size:]]

train_valid_idx = tra_idx[label_mask[tra_idx]]
val_valid_idx = val_idx[label_mask[val_idx]]
test_valid_idx = test_idx[label_mask[test_idx]]

stats = compute_standardization_stats(env, G, mg, train_valid_idx)
env_norm, G_norm, mg_norm = normalize_inputs(env, G, mg, stats)

batch_size = 32
train_dl, val_dl, test_dl = build_dataloaders(
    env_norm,
    G_norm,
    mg_norm,
    HIR,
    train_valid_idx,
    val_valid_idx,
    test_valid_idx,
    batch_size=batch_size,
)

model = StableHIRCNNLSTM(geno_dim=G.shape[1]).to(device)
result = train_cnnlstm_model(
    model=model,
    train_loader=train_dl,
    val_loader=val_dl,
    test_loader=test_dl,
    device=device,
    lr_init=3e-4,
    weight_decay=1e-4,
    max_epochs=400,
    early_stop_patience=50,
    lr_patience=15,
    min_lr=1e-6,
    min_delta=5e-4,
    grad_clip_norm=1.0,
)

train_eval = result["train_eval"]
val_eval = result["val_eval"]
test_eval = result["test_eval"]
precision_recall_thresholds = [0.10, 0.15, 0.20]

train_sr_015, train_n_015 = success_rate_at_threshold(train_eval["pred"], train_eval["true"], 0.15)
val_sr_015, val_n_015 = success_rate_at_threshold(val_eval["pred"], val_eval["true"], 0.15)
test_sr_015, test_n_015 = success_rate_at_threshold(test_eval["pred"], test_eval["true"], 0.15)
train_sr_020, train_n_020 = success_rate_at_threshold(train_eval["pred"], train_eval["true"], 0.20)
val_sr_020, val_n_020 = success_rate_at_threshold(val_eval["pred"], val_eval["true"], 0.20)
test_sr_020, test_n_020 = success_rate_at_threshold(test_eval["pred"], test_eval["true"], 0.20)

precision_recall_results = {"train": {}, "val": {}, "test": {}}
for threshold in precision_recall_thresholds:
    threshold_key = f"{int(round(threshold * 100)):02d}"
    precision_recall_results["train"][threshold_key] = precision_recall_at_threshold(train_eval["pred"], train_eval["true"], threshold)
    precision_recall_results["val"][threshold_key] = precision_recall_at_threshold(val_eval["pred"], val_eval["true"], threshold)
    precision_recall_results["test"][threshold_key] = precision_recall_at_threshold(test_eval["pred"], test_eval["true"], threshold)

result_root = "Result"
os.makedirs(result_root, exist_ok=True)
run_name = f"cnnlstm_{seed}"
run_dir = os.path.join(result_root, run_name)
os.makedirs(run_dir, exist_ok=True)
print(f"结果输出目录: {run_dir}")

metrics_path = os.path.join(run_dir, "metrics.txt")
config_path = os.path.join(run_dir, "config.json")
history_path = os.path.join(run_dir, "history.csv")
curve_path = os.path.join(run_dir, "training_curves.png")
scatter_path = os.path.join(run_dir, "HIR.png")
success_path = os.path.join(run_dir, "success_rate.png")
best_checkpoint_path = os.path.join(run_dir, "best_checkpoint.pt")
final_checkpoint_path = os.path.join(run_dir, "final_checkpoint.pt")

history_df = pd.DataFrame(result["history"])
history_df.to_csv(history_path, index=False)

fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=200)
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
axes[1, 0].set_title("Train Loss")
axes[1, 0].legend()

axes[1, 1].plot(history_df["epoch"], history_df["lr"], label="lr")
axes[1, 1].plot(history_df["epoch"], history_df["val_corr"], label="val corr")
axes[1, 1].plot(history_df["epoch"], history_df["test_corr"], label="test corr")
axes[1, 1].set_title("LR / Correlation")
axes[1, 1].legend()

for ax in axes.flat:
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(curve_path)
plt.close(fig)

fig, ax = plt.subplots()
plt.scatter(train_eval["pred"].detach().cpu().numpy(), train_eval["true"].detach().cpu().numpy(), label="train")
plt.scatter(test_eval["pred"].detach().cpu().numpy(), test_eval["true"].detach().cpu().numpy(), label="test")
plt.plot([0, 0.4], [0, 0.4], "red", linestyle="--")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(
    "Test RMSE: "
    + str(round(test_eval["rmse"].item(), 3))
    + "  Test RRMSE: "
    + str(round(test_eval["rrmse"].item(), 3))
    + "  Test Corr: "
    + str(round(test_eval["corr"].item(), 3))
)
plt.legend()
plt.savefig(scatter_path)
plt.close(fig)

plot_success(
    torch.cat([train_eval["pred"].detach().cpu(), test_eval["pred"].detach().cpu()]),
    torch.cat([train_eval["true"].detach().cpu(), test_eval["true"].detach().cpu()]),
    range(len(train_eval["pred"])),
    range(len(train_eval["pred"]), len(train_eval["pred"]) + len(test_eval["pred"])),
    fig_name=success_path,
)

checkpoint_payload = {
    "model_state_dict": result["model"].state_dict(),
    "seed": seed,
    "geno_dim": int(G.shape[1]),
    "env_mean": stats.env_mean.cpu(),
    "env_std": stats.env_std.cpu(),
    "G_mean": stats.G_mean.cpu(),
    "G_std": stats.G_std.cpu(),
    "mg_mean": stats.mg_mean.cpu(),
    "mg_std": stats.mg_std.cpu(),
    "train_idx": train_idx.cpu(),
    "val_idx": val_idx.cpu(),
    "test_idx": test_idx.cpu(),
    "train_valid_idx": train_valid_idx.cpu(),
    "val_valid_idx": val_valid_idx.cpu(),
    "test_valid_idx": test_valid_idx.cpu(),
    "best_val_hir_rrmse": result["best_val_hir_rrmse"],
    "best_epoch": result["best_epoch"],
    "history_path": history_path,
    "model": "StableCNNLSTM",
}
torch.save(checkpoint_payload, best_checkpoint_path)
torch.save(checkpoint_payload, final_checkpoint_path)

config = {
    "run_dir": run_dir,
    "timestamp": run_name,
    "seed": seed,
    "device": str(device),
    "batch_size": batch_size,
    "learning_rate_final": result["final_lr"],
    "weight_decay": 1e-4,
    "max_epochs": 400,
    "early_stop_patience": 50,
    "lr_patience": 15,
    "best_epoch": result["best_epoch"],
    "precision_recall_thresholds": precision_recall_thresholds,
    "status": "finished",
}

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

with open(metrics_path, "w", encoding="utf-8") as f:
    f.write(f"run_dir: {run_dir}\n")
    f.write(f"device: {device}\n")
    f.write(f"seed: {seed}\n")
    f.write(f"best_val_loss_during_training: {result['best_val_hir_rrmse']:.6f}\n")
    f.write(f"final_train_HIR_RRMSE: {train_eval['rrmse'].item():.6f}\n")
    f.write(f"final_train_HIR_RMSE: {train_eval['rmse'].item():.6f}\n")
    f.write(f"final_val_HIR_RRMSE: {val_eval['rrmse'].item():.6f}\n")
    f.write(f"final_val_HIR_RMSE: {val_eval['rmse'].item():.6f}\n")
    f.write(f"final_test_HIR_RRMSE: {test_eval['rrmse'].item():.6f}\n")
    f.write(f"final_test_HIR_RMSE: {test_eval['rmse'].item():.6f}\n")
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
