import matplotlib.pyplot as plt
#\\TODO change out_dir to fp in mooreRNN

def plot_loss_per_epoch(train_loss, fp, model_name=None, dataset=None):
    ###
    # train_loss : per-step (per-batch) training loss, not averaged per epoch
    # out_dir : path to output directory
    ###

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_loss, label="Train MSE")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("MSE Loss")
    title = "Training Loss Over Training Steps"
    if model_name and dataset:
        title = f"{model_name} on {dataset} — {title}"
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    
    plt.close(fig)


def plot_loss_convergence(train_loss, steps_per_epoch, fp, model_name=None, dataset=None):
    ###
    # train_loss : per-step (per-batch) training loss, not averaged per epoch
    # steps_per_epoch : number of training steps (batches) in one epoch,
    #                    e.g. len(train_loader), used to place epoch ticks
    # fp : path to save the figure
    ###
    num_epochs = len(train_loss) // steps_per_epoch
    epoch_avg = [
        sum(train_loss[e * steps_per_epoch:(e + 1) * steps_per_epoch]) / steps_per_epoch
        for e in range(num_epochs)
    ]
    convergence_value = epoch_avg[-1] if epoch_avg else train_loss[-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_loss, linewidth=0.8, alpha=0.6, label="Per-step MSE")

    epoch_boundaries = [e * steps_per_epoch for e in range(1, num_epochs)]
    for boundary in epoch_boundaries:
        ax.axvline(boundary, color="gray", linestyle=":", linewidth=0.8)

    epoch_centers = [e * steps_per_epoch + steps_per_epoch / 2 for e in range(num_epochs)]
    ax.plot(epoch_centers, epoch_avg, "o-", color="tab:red", label="Per-epoch avg MSE")

    ax.axhline(convergence_value, color="tab:green", linestyle="--", linewidth=1)
    ax.annotate(
        f"convergence ≈ {convergence_value:.6g}",
        xy=(len(train_loss), convergence_value),
        xytext=(-5, 8),
        textcoords="offset points",
        ha="right",
        color="tab:green",
    )

    ax.set_xlabel("Training Step")
    ax.set_ylabel("MSE Loss")

    top_ax = ax.secondary_xaxis("top")
    top_ax.set_xticks(epoch_centers)
    top_ax.set_xticklabels([str(e + 1) for e in range(num_epochs)])
    top_ax.set_xlabel("Epoch")

    title = "Training Loss Convergence"
    if model_name and dataset:
        title = f"{model_name} on {dataset} — {title}"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(fp, dpi=150, bbox_inches="tight")

    plt.close(fig)


def plot_grad_hist(grad_history, fp):
    ###
    # grad_history : gradient norm logged once per training step (batch)
    #
    ###
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, norms in grad_history.items():
        ax.plot(norms, label=name)
    # ax.set_yscale("log")  # optional: gradients might benefit from log scale?
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norms per Parameter Across Training Steps")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
