"""Data loading and sample construction for the LEMON EEG project."""

import glob
import os
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def load_metadata(metadata_path):
    """Load the LEMON metadata CSV."""
    metadata_path = Path(metadata_path)

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}"
        )

    return pd.read_csv(metadata_path)


def create_age_labels(metadata):
    """Create integer labels from the age ranges in the metadata.

    The mapping preserves the original notebook's behavior:
    labels follow the order of unique values in metadata['Age'].
    """
    metadata = metadata.copy()

    unique_ranges = metadata["Age"].unique()
    age_to_label = {
        age_range: idx
        for idx, age_range in enumerate(unique_ranges)
    }

    metadata["Label"] = metadata["Age"].map(age_to_label)

    # Preserve the original notebook's assumption that the first
    # metadata column contains the subject ID.
    subject_id_column = metadata.columns[0]
    label_map = dict(
        zip(metadata[subject_id_column], metadata["Label"])
    )

    label_to_age = {
        label: age_range
        for age_range, label in age_to_label.items()
    }

    return metadata, age_to_label, label_to_age, label_map


def discover_eeg_files(base_dir, pattern="sub-*/*_EC.set"):
    """Find EEG files matching the original notebook's pattern."""
    base_dir = Path(base_dir)

    if not base_dir.is_dir():
        raise FileNotFoundError(
            f"EEG data directory not found: {base_dir}"
        )

    file_paths = glob.glob(
        os.path.join(str(base_dir), pattern)
    )

    if not file_paths:
        raise FileNotFoundError(
            f"No EEG files matched {pattern!r} in {base_dir}"
        )

    return file_paths


def find_common_channels(file_paths):
    """Return channel names shared by all EEG files."""
    if not file_paths:
        raise ValueError("file_paths is empty.")

    all_channel_lists = []

    for filepath in tqdm(file_paths, desc="Reading channel names"):
        raw = mne.io.read_raw_eeglab(
            filepath,
            preload=False,
            verbose="ERROR",
        )
        all_channel_lists.append(raw.ch_names)

    common_channels = list(
        set.intersection(*map(set, all_channel_lists))
    )

    if not common_channels:
        raise ValueError("No common EEG channels were found.")

    return common_channels


def build_samples(
    file_paths,
    label_map,
    sfreq=250,
    block_duration=30,
    blocks_per_sample=5,
):
    """Create sample metadata using consecutive fixed-length windows.

    Each sample records its file path, subject ID, label, and start
    sample index. EEG signals are not loaded into memory here.
    """
    sample_duration = block_duration * blocks_per_sample
    sample_size = sample_duration * sfreq

    samples = []

    for filepath in tqdm(file_paths, desc="Building samples"):
        subject_id = os.path.basename(
            os.path.dirname(filepath)
        )

        if subject_id not in label_map:
            raise KeyError(
                f"No age label found for subject {subject_id!r}"
            )

        label = label_map[subject_id]

        raw = mne.io.read_raw_eeglab(
            filepath,
            preload=False,
            verbose="ERROR",
        )

        n_timepoints = raw.n_times

        for start in range(
            0,
            n_timepoints - sample_size + 1,
            sample_size,
        ):
            samples.append(
                {
                    "filepath": filepath,
                    "subject_id": subject_id,
                    "label": label,
                    "start": start,
                }
            )

    return samples
