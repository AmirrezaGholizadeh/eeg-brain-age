"""Evaluation utilities for EEG brain-age classification."""

from collections import Counter

import numpy as np
import torch

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_full(
    model,
    dataloader,
    criterion,
    title="Evaluation",
):
    """Evaluate a model and return classification metrics."""

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            predicted = outputs.argmax(dim=1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if not all_labels:
        raise ValueError("The dataloader produced no samples.")

    final_loss = running_loss / len(dataloader)

    all_preds = np.asarray(all_preds)
    all_labels = np.asarray(all_labels)

    accuracy = (all_preds == all_labels).mean()

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0,
    )
    macro_precision = precision_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )
    macro_recall = recall_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        all_labels,
        all_preds,
        zero_division=0,
        digits=3,
    )

    cm = confusion_matrix(all_labels, all_preds)

    predicted_distribution = dict(
        sorted(Counter(all_preds.tolist()).items())
    )
    true_distribution = dict(
        sorted(Counter(all_labels.tolist()).items())
    )

    print(f"\\n===== {title} =====")
    print(f"Loss: {final_loss:.4f} | Accuracy: {accuracy:.4f}")
    print(
        f"Macro F1: {macro_f1:.4f} | "
        f"Weighted F1: {weighted_f1:.4f}"
    )
    print(
        f"Macro Precision: {macro_precision:.4f} | "
        f"Macro Recall: {macro_recall:.4f}"
    )

    print("\\nPer-class report:")
    print(report)

    print("Predicted distribution:", predicted_distribution)
    print("True distribution:     ", true_distribution)

    return {
        "loss": final_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "preds": all_preds,
        "labels": all_labels,
        "confusion_matrix": cm,
        "classification_report": report,
        "predicted_distribution": predicted_distribution,
        "true_distribution": true_distribution,
    }
