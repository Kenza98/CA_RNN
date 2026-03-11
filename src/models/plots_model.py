import matplotlib.pyplot as plt
#\\TODO change out_dir to fp in mooreRNN

def plot_loss_per_epoch(train_loss, fp):
    ###
    # train_loss : list of stored average epoch losses
    # out_dir : path to output directory
    ###
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_loss, label="Train MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training Loss Over Epochs")
    ax.legend()

    fig.tight_layout()
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    
    plt.close(fig)


def plot_grad_hist(grad_history, fp):
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
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
