"""Training utilities for EEG brain-age classification."""

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    criterion,
    optimizer,
    num_epochs=50,
    checkpoint_path="checkpoints/best_model.pth",
):
    """Train the model and save the checkpoint with best validation Macro-F1."""

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_macro_f1 = -1.0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    val_macro_f1s = []
    val_weighted_f1s = []

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )
            optimizer.step()

            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_dataloader)
        epoch_acc = correct / total

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # Validation
        model.eval()

        val_correct = 0
        val_total = 0
        val_running_loss = 0.0
        val_all_preds = []
        val_all_labels = []

        with torch.no_grad():
            for val_inputs, val_labels in val_dataloader:
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)

                val_output = model(val_inputs)
                val_loss = criterion(val_output, val_labels)

                val_predicted = val_output.argmax(dim=1)

                val_total += val_labels.size(0)
                val_correct += (
                    val_predicted == val_labels
                ).sum().item()
                val_running_loss += val_loss.item()

                val_all_preds.extend(
                    val_predicted.cpu().numpy()
                )
                val_all_labels.extend(
                    val_labels.cpu().numpy()
                )

        val_epoch_loss = val_running_loss / len(val_dataloader)
        val_epoch_acc = val_correct / val_total

        val_epoch_macro_f1 = f1_score(
            val_all_labels,
            val_all_preds,
            average="macro",
            zero_division=0,
        )
        val_epoch_weighted_f1 = f1_score(
            val_all_labels,
            val_all_preds,
            average="weighted",
            zero_division=0,
        )

        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc)
        val_macro_f1s.append(val_epoch_macro_f1)
        val_weighted_f1s.append(val_epoch_weighted_f1)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {epoch_loss:.4f}, "
            f"Train Acc: {epoch_acc:.4f} | "
            f"Val Loss: {val_epoch_loss:.4f}, "
            f"Val Acc: {val_epoch_acc:.4f}, "
            f"Val Macro-F1: {val_epoch_macro_f1:.4f}, "
            f"Val Weighted-F1: {val_epoch_weighted_f1:.4f}"
        )

        if val_epoch_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_epoch_macro_f1
            torch.save(model.state_dict(), checkpoint_path)
            print(
                "  -> Best model saved "
                f"(Val Macro-F1 = {val_epoch_macro_f1:.4f})"
            )

    return (
        train_losses,
        train_accuracies,
        val_losses,
        val_accuracies,
        val_macro_f1s,
        val_weighted_f1s,
    )



def compute_class_weights(train_samples, num_classes=9):
    """Compute class weights using unique subjects in the training split."""
    import torch

    subject_labels = {}

    for sample in train_samples:
        subject_id = sample["subject_id"]
        label = int(sample["label"])

        if subject_id in subject_labels:
            if subject_labels[subject_id] != label:
                raise ValueError(
                    f"Inconsistent labels for subject {subject_id}"
                )
        else:
            subject_labels[subject_id] = label

    labels = torch.tensor(
        list(subject_labels.values()),
        dtype=torch.long,
    )

    if labels.numel() == 0:
        raise ValueError("The training split contains no subjects.")

    if torch.any(labels < 0) or torch.any(labels >= num_classes):
        raise ValueError("Found a label outside the expected class range.")

    class_counts = torch.bincount(
        labels,
        minlength=num_classes,
    ).to(torch.float32)

    if torch.any(class_counts == 0):
        missing = torch.where(class_counts == 0)[0].tolist()
        raise ValueError(
            f"Missing classes in training split: {missing}"
        )

    total_subjects = class_counts.sum()

    class_weights = total_subjects / (
        num_classes * class_counts
    )

    class_weights = torch.clamp(
        class_weights,
        min=0.7,
        max=2.0,
    )

    return class_counts, class_weights
