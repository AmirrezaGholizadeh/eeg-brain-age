
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_curves(
    train_losses,
    val_losses,
    train_accuracies,
    val_accuracies,
    val_macro_f1s=None,
    val_weighted_f1s=None,
    save_path=None,
):
    """
    Plot training and validation metrics across epochs.

    Plots:
        1. Training and validation loss
        2. Training and validation accuracy
        3. Validation macro F1 and weighted F1 (if provided)
    """
    epochs = range(1, len(train_losses) + 1)

    plot_f1 = (
        val_macro_f1s is not None
        and val_weighted_f1s is not None
    )

    if plot_f1:
        if len(val_macro_f1s) != len(train_losses):
            raise ValueError(
                "val_macro_f1s must have the same length as train_losses."
            )
        if len(val_weighted_f1s) != len(train_losses):
            raise ValueError(
                "val_weighted_f1s must have the same length as train_losses."
            )

    if len(val_losses) != len(train_losses):
        raise ValueError(
            "train_losses and val_losses must have the same length."
        )

    if len(train_accuracies) != len(train_losses):
        raise ValueError(
            "train_accuracies must have the same length as train_losses."
        )

    if len(val_accuracies) != len(train_losses):
        raise ValueError(
            "val_accuracies must have the same length as train_losses."
        )

    n_plots = 3 if plot_f1 else 2
    fig, axes = plt.subplots(
        1, n_plots,
        figsize=(6 * n_plots, 5)
    )

    # 1. Loss
    axes[0].plot(
        epochs, train_losses,
        label="Train Loss", marker="o"
    )
    axes[0].plot(
        epochs, val_losses,
        label="Validation Loss", marker="o"
    )
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 2. Accuracy
    axes[1].plot(
        epochs, train_accuracies,
        label="Train Accuracy", marker="o"
    )
    axes[1].plot(
        epochs, val_accuracies,
        label="Validation Accuracy", marker="o"
    )
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # 3. Validation F1 scores
    if plot_f1:
        axes[2].plot(
            epochs, val_macro_f1s,
            label="Validation Macro F1", marker="o"
        )
        axes[2].plot(
            epochs, val_weighted_f1s,
            label="Validation Weighted F1", marker="o"
        )
        axes[2].set_xlabel("Epochs")
        axes[2].set_ylabel("F1 Score")
        axes[2].set_title("Validation F1 Scores")
        axes[2].legend()
        axes[2].grid(alpha=0.3)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return fig


def plot_confusion_matrix(
    matrix,
    title="Confusion Matrix",
    save_path=None,
):
    """Plot a confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return fig
