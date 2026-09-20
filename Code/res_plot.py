import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_hir_scatter_plot(
    train_pred,
    train_true,
    test_pred,
    test_true,
    rmse,
    rrmse,
    corr,
    fig_name,
    dpi=400,
):
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=dpi)
    ax.scatter(train_pred, train_true, label="train", s=18, alpha=0.75)
    ax.scatter(test_pred, test_true, label="test", s=18, alpha=0.80)
    ax.plot([0, 0.4], [0, 0.4], color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xlim(0, 0.4)
    ax.set_ylim(0, 0.4)
    ax.grid(True, alpha=0.25)
    ax.set_title(
        f"Test RMSE: {rmse:.3f}  Test RRMSE: {rrmse:.3f}  Test Corr: {corr:.3f}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_name, dpi=dpi)
    plt.close(fig)


def save_training_curves_plot(history_df, fig_name, dpi=400):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=dpi)

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
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(fig_name, dpi=dpi)
    plt.close(fig)


# %%
def plot_success(ph, HIR_mat, train_idx, test_idx, fig_name='success_rate_day.png'):
    y_pred_train = ph[train_idx]
    y_true_train = HIR_mat[train_idx]

    y_pred_test = ph[test_idx]
    y_true_test = HIR_mat[test_idx]

    id = ~y_true_train.isnan()
    y_pred_train = y_pred_train[id].detach().cpu().numpy()
    y_true_train = y_true_train[id].detach().cpu().numpy()

    id = ~y_true_test.isnan()
    y_pred_test = y_pred_test[id].detach().cpu().numpy()
    y_true_test = y_true_test[id].detach().cpu().numpy()

    def compute_precision(y_pred, y_true, thresholds):
        results = []
        for t in thresholds:
            mask = y_pred > t
            if np.sum(mask) == 0:
                continue
            precision = np.mean(y_true[mask] > t)
            results.append({
                'Threshold': t,
                'Count': np.sum(mask),
                'Precision': precision
            })
        return pd.DataFrame(results, columns=['Threshold', 'Count', 'Precision'])

    thresholds = np.round(np.arange(0.05, 0.2, 0.025), 3)
    df_train = compute_precision(y_pred_train, y_true_train, thresholds)
    df_test = compute_precision(y_pred_test, y_true_test, thresholds)

    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=400)
    ax1.plot(df_train['Threshold'], df_train['Precision'], 'o-', label='Train', color='blue')
    ax1.plot(df_test['Threshold'], df_test['Precision'], 'o-', label='Test', color='orange')
    ax1.set_xlabel('Predicted HIR > Threshold')
    ax1.set_ylabel('Precision')
    ax1.set_ylim(0, 1)
    ax1.set_yticks(np.round(np.arange(0, 1.1, 0.1), 1),
                   [str(int(t * 100)) + '%' for t in np.round(np.arange(0, 1.1, 0.1), 1)])
    ax1.set_xticks(thresholds, [str(t * 100) + '%' for t in thresholds])
    ax1.tick_params(axis='y')
    ax1.legend()

    for x, y in zip(df_train['Threshold'], df_train['Precision']):
        ax1.annotate(f'{y * 100:.1f}%', xy=(x, y), xytext=(0, 5), textcoords='offset points',
                     ha='center', fontsize=8, color='blue')
    for x, y in zip(df_test['Threshold'], df_test['Precision']):
        ax1.annotate(f'{y * 100:.1f}%', xy=(x, y), xytext=(0, 5), textcoords='offset points',
                     ha='center', fontsize=8, color='orange')

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

    plt.title('Precision by Predicted HIR Threshold (Train vs Test)')
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(fig_name, dpi=400)
    plt.close(fig)
