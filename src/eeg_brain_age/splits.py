"""Subject-wise train, validation, and test splitting."""

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def split_subjects(
    samples,
    test_size=0.30,
    random_state=42,
):
    """Split samples into train and temporary sets by subject.

    This preserves the original notebook's first split:
    70% train and 30% temporary data, grouped by subject.
    """
    if not samples:
        raise ValueError("samples is empty.")

    sample_labels = [sample["label"] for sample in samples]
    sample_subjects = [
        sample["subject_id"] for sample in samples
    ]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    train_idx, temp_idx = next(
        splitter.split(
            X=np.zeros(len(samples)),
            y=sample_labels,
            groups=sample_subjects,
        )
    )

    train_samples = [samples[i] for i in train_idx]
    temp_samples = [samples[i] for i in temp_idx]

    return train_samples, temp_samples


def split_validation_test(
    temp_samples,
    test_size=0.50,
    random_state=42,
):
    """Split temporary samples into validation and test sets.

    This preserves the original notebook's second split:
    half of the temporary data for validation and half for test,
    grouped by subject.
    """
    if not temp_samples:
        raise ValueError("temp_samples is empty.")

    temp_labels = [
        sample["label"] for sample in temp_samples
    ]
    temp_subjects = [
        sample["subject_id"] for sample in temp_samples
    ]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    val_idx, test_idx = next(
        splitter.split(
            X=np.zeros(len(temp_samples)),
            y=temp_labels,
            groups=temp_subjects,
        )
    )

    val_samples = [temp_samples[i] for i in val_idx]
    test_samples = [temp_samples[i] for i in test_idx]

    return val_samples, test_samples


def split_train_validation_test(
    samples,
    train_temp_size=0.30,
    val_test_size=0.50,
    random_state=42,
):
    """Perform the original two-stage subject-wise split."""
    train_samples, temp_samples = split_subjects(
        samples=samples,
        test_size=train_temp_size,
        random_state=random_state,
    )

    val_samples, test_samples = split_validation_test(
        temp_samples=temp_samples,
        test_size=val_test_size,
        random_state=random_state,
    )

    return train_samples, val_samples, test_samples
