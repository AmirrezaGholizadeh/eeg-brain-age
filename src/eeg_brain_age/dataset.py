"""PyTorch Dataset for EEG brain-age classification."""

import mne
import torch
from torch.utils.data import Dataset

from eeg_brain_age.features import eeg_to_mel_image, load_or_compute


class EEGDataset(Dataset):
    def __init__(
        self,
        samples,
        common_channels,
        use_cache=True,
        train=True,
        cache_dir="./cache",
    ):
        self.samples = samples
        self.common_channels = common_channels
        self.use_cache = use_cache
        self.train = train
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        label = torch.tensor(
            sample["label"],
            dtype=torch.long,
        )

        raw = mne.io.read_raw_eeglab(
            sample["filepath"],
            preload=False,
            verbose=False,
        )

        raw.pick(self.common_channels)

        if self.use_cache:
            eeg = load_or_compute(
                sample=sample,
                raw=raw,
                cache_dir=self.cache_dir,
            )
        else:
            eeg = raw.get_data(
                start=sample["start"],
                stop=sample["start"] + 37500,
            )
            eeg = eeg_to_mel_image(eeg)

        eeg = torch.tensor(eeg, dtype=torch.float32)

        return eeg, label
