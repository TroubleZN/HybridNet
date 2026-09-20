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
    get_device,
    load_base_dataset,
    normalize_inputs,
    precision_recall_at_threshold,
    set_seed,
    success_rate_at_threshold,
    train_cnnlstm_model,
)


def save_fold_curves(history_df, output_path):
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
    fig.savefig(output_path)
    plt.close(fig)


def save_run_summary_plot(fold_df, output_path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), dpi=220, sharex=True)

    axes[0].plot(fold_df["test_year"], fold_df["final_test_hir_rmse"], marker="o", label="test RMSE")
    axes[0].plot(fold_df["test_year"], fold_df["best_val_hir_rrmse"], marker="s", label="best val RRMSE")
    axes[0].set_ylabel("Metric")
    axes[0].set_title("CNNLSTM Rolling-Year Performance")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].plot(fold_df["test_year"], fold_df["test_precision_0.15"], marker="o", label="precision@0.15")
    axes[1].plot(fold_df["test_year"], fold_df["test_recall_0.15"], marker="s", label="recall@0.15")
    axes[1].plot(fold_df["test_year"], fold_df["test_precision_0.20"], marker="^", label="precision@0.20")
    axes[1].plot(fold_df["test_year"], fold_df["test_recall_0.20"], marker="d", label="recall@0.20")
    axes[1].set_xlabel("Test Year")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


device = get_device()
print(f"当前设备: {device}")

seed = 20260314
set_seed(seed)
print(f"随机种子: {seed}")

pheno, env, G, mg, HIR, label_mask = load_base_dataset()
year_tensor = torch.tensor(pheno["Year"].to_numpy(), dtype=torch.int64)
years = sorted(int(y) for y in pheno.loc[label_mask.cpu().numpy(), "Year"].unique())

result_root = "Result"
os.makedirs(result_root, exist_ok=True)
run_name = f"cnnlstm_rolling_{seed}"
run_dir = os.path.join(result_root, run_name)
os.makedirs(run_dir, exist_ok=True)
print(f"结果输出目录: {run_dir}")

rolling_metrics_path = os.path.join(run_dir, "rolling_metrics.csv")
summary_path = os.path.join(run_dir, "summary.json")
config_path = os.path.join(run_dir, "config.json")
summary_plot_path = os.path.join(run_dir, "rolling_summary.png")

batch_size = 32
precision_recall_thresholds = [0.10, 0.15, 0.20]
fold_results = []

for fold_id, test_year in enumerate(years[1:], start=1):
    test_idx = torch.where((year_tensor == test_year) & label_mask)[0]
    train_pool = torch.where((year_tensor < test_year) & label_mask)[0]
    if len(test_idx) == 0 or len(train_pool) < 30:
        continue

    perm = torch.randperm(len(train_pool))
    proposed_val_size = max(24, int(0.1 * len(train_pool)))
    val_size = min(proposed_val_size, max(1, len(train_pool) - 1))
    val_idx = train_pool[perm[:val_size]]
    tra_idx = train_pool[perm[val_size:]]
    if len(tra_idx) == 0:
        continue

    stats = compute_standardization_stats(env, G, mg, tra_idx)
    env_norm, G_norm, mg_norm = normalize_inputs(env, G, mg, stats)

    train_dl, val_dl, test_dl = build_dataloaders(
        env_norm,
        G_norm,
        mg_norm,
        HIR,
        tra_idx,
        val_idx,
        test_idx,
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

    fold_dir = os.path.join(run_dir, f"year_{test_year}")
    os.makedirs(fold_dir, exist_ok=True)
    history_path = os.path.join(fold_dir, "history.csv")
    curve_path = os.path.join(fold_dir, "training_curves.png")
    best_checkpoint_path = os.path.join(fold_dir, "best_checkpoint.pt")
    final_checkpoint_path = os.path.join(fold_dir, "final_checkpoint.pt")

    history_df = pd.DataFrame(result["history"])
    history_df.to_csv(history_path, index=False)
    save_fold_curves(history_df, curve_path)

    checkpoint_payload = {
        "model_state_dict": result["model"].state_dict(),
        "seed": seed,
        "test_year": int(test_year),
        "geno_dim": int(G.shape[1]),
        "env_mean": stats.env_mean.cpu(),
        "env_std": stats.env_std.cpu(),
        "G_mean": stats.G_mean.cpu(),
        "G_std": stats.G_std.cpu(),
        "mg_mean": stats.mg_mean.cpu(),
        "mg_std": stats.mg_std.cpu(),
        "train_idx": tra_idx.cpu(),
        "val_idx": val_idx.cpu(),
        "test_idx": test_idx.cpu(),
        "best_val_hir_rrmse": result["best_val_hir_rrmse"],
        "best_epoch": result["best_epoch"],
        "history_path": history_path,
        "model": "StableCNNLSTM",
    }
    torch.save(checkpoint_payload, best_checkpoint_path)
    torch.save(checkpoint_payload, final_checkpoint_path)

    fold_row = {
        "fold_id": fold_id,
        "test_year": int(test_year),
        "n_train": int(len(tra_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "best_epoch": int(result["best_epoch"]),
        "best_val_hir_rrmse": float(result["best_val_hir_rrmse"]),
        "final_train_hir_rrmse": float(train_eval["rrmse"].item()),
        "final_train_hir_rmse": float(train_eval["rmse"].item()),
        "final_val_hir_rrmse": float(val_eval["rrmse"].item()),
        "final_val_hir_rmse": float(val_eval["rmse"].item()),
        "final_test_hir_rrmse": float(test_eval["rrmse"].item()),
        "final_test_hir_rmse": float(test_eval["rmse"].item()),
        "final_test_corr": float(test_eval["corr"].item()),
        "learning_rate_final": float(result["final_lr"]),
        "history_path": history_path,
    }

    for threshold in [0.15, 0.20]:
        success_rate, selected_n = success_rate_at_threshold(test_eval["pred"], test_eval["true"], threshold)
        threshold_suffix = f"{threshold:.2f}"
        fold_row[f"test_success_rate_{threshold_suffix}"] = success_rate
        fold_row[f"test_success_rate_n_{threshold_suffix}"] = selected_n

    for threshold in precision_recall_thresholds:
        precision, recall, pred_n, actual_n = precision_recall_at_threshold(test_eval["pred"], test_eval["true"], threshold)
        threshold_suffix = f"{threshold:.2f}"
        fold_row[f"test_precision_{threshold_suffix}"] = precision
        fold_row[f"test_recall_{threshold_suffix}"] = recall
        fold_row[f"test_predicted_positive_n_{threshold_suffix}"] = pred_n
        fold_row[f"test_actual_positive_n_{threshold_suffix}"] = actual_n

    fold_results.append(fold_row)

if not fold_results:
    raise RuntimeError("No valid rolling folds were created for CNNLSTM.")

fold_df = pd.DataFrame(fold_results).sort_values("test_year").reset_index(drop=True)
fold_df.to_csv(rolling_metrics_path, index=False)
save_run_summary_plot(fold_df, summary_plot_path)

summary = {
    "run_dir": run_dir,
    "seed": seed,
    "device": str(device),
    "model": "StableCNNLSTM_rolling",
    "years": fold_df["test_year"].tolist(),
    "precision_recall_thresholds": precision_recall_thresholds,
    "mean_test_hir_rrmse": float(fold_df["final_test_hir_rrmse"].mean()),
    "std_test_hir_rrmse": float(fold_df["final_test_hir_rrmse"].std(ddof=0)),
    "mean_test_hir_rmse": float(fold_df["final_test_hir_rmse"].mean()),
    "std_test_hir_rmse": float(fold_df["final_test_hir_rmse"].std(ddof=0)),
    "mean_test_precision_0.10": float(fold_df["test_precision_0.10"].mean()),
    "mean_test_recall_0.10": float(fold_df["test_recall_0.10"].mean()),
    "mean_test_precision_0.15": float(fold_df["test_precision_0.15"].mean()),
    "mean_test_recall_0.15": float(fold_df["test_recall_0.15"].mean()),
    "mean_test_precision_0.20": float(fold_df["test_precision_0.20"].mean()),
    "mean_test_recall_0.20": float(fold_df["test_recall_0.20"].mean()),
}

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

config = {
    "run_dir": run_dir,
    "timestamp": run_name,
    "seed": seed,
    "device": str(device),
    "batch_size": batch_size,
    "learning_rate_init": 3e-4,
    "weight_decay": 1e-4,
    "max_epochs": 400,
    "early_stop_patience": 50,
    "lr_patience": 15,
    "precision_recall_thresholds": precision_recall_thresholds,
    "status": "finished",
}

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
