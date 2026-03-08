import matplotlib.pyplot as plt

def plot_loss_per_epoch(train_loss, out_dir):
    ###
    # train_loss : list of stored average epoch losses
    # out_dir : path to output directory
    ###
    out_dir.mkdir(parents=True, exist_ok=True) # create dir if it doesn't exist

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_loss, label="Train MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training Loss Over Epochs")
    ax.legend()

    fig.tight_layout()

    save_path = out_dir / "train_loss_curve.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    
    plt.close(fig)


def plot_grad_hist(grad_history, out_dir):
    ###
    # grad_history : History of gradients over training epochs
    #
    ###
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, norms in grad_history.items():
        ax.plot(norms, label=name)
    # ax.set_yscale("log")  # optional: gradients might benefit from log scale?
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norms per Parameter Across Epochs") #TODO check
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    save_path = out_dir / "gradients.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
