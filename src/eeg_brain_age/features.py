"""Mel-spectrogram feature extraction and caching."""

import os
from pathlib import Path

import librosa
import numpy as np


def compute_mel_spectrogram(
    eeg_channel_data,
    sr=250,
    n_mels=64,
):
    """Convert one EEG channel into a Mel spectrogram in dB."""
    spectrogram = librosa.feature.melspectrogram(
        y=eeg_channel_data,
        sr=sr,
        n_mels=n_mels,
    )

    spectrogram_db = librosa.power_to_db(
        spectrogram,
        ref=np.max,
    )

    return spectrogram_db


def eeg_to_mel_image(
    eeg_data,
    sr=250,
    n_mels=64,
):
    """Compute a Mel spectrogram for every channel."""
    mel_spectrograms = [
        compute_mel_spectrogram(
            channel,
            sr=sr,
            n_mels=n_mels,
        )
        for channel in eeg_data
    ]

    return np.stack(mel_spectrograms, axis=0)


def get_cache_path(sample, cache_dir):
    """Build a cache filename for a sample."""
    filename = f"{sample['subject_id']}_{sample['start']}.npy"
    return os.path.join(str(cache_dir), filename)


def load_or_compute(
    sample,
    raw,
    cache_dir,
    sr=250,
    n_mels=64,
    sample_size=37500,
    block_size=7500,
    n_blocks=5,
):
    """Load cached features or compute and save them."""
    cache_path = get_cache_path(sample, cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    if os.path.exists(cache_path):
        return np.load(cache_path)

    eeg = raw.get_data(
        start=sample["start"],
        stop=sample["start"] + sample_size,
    )

    mel_blocks = []

    for block_index in range(n_blocks):
        block_start = block_index * block_size
        block_end = block_start + block_size

        block = eeg[:, block_start:block_end]

        mel = eeg_to_mel_image(
            block,
            sr=sr,
            n_mels=n_mels,
        )

        mel_blocks.append(mel)

    mel_blocks = np.stack(mel_blocks, axis=0)

    np.save(cache_path, mel_blocks)

    return mel_blocks
