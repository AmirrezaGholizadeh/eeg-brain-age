"""DataLoader construction for EEG brain-age experiments."""

from torch.utils.data import DataLoader

from eeg_brain_age.dataset import EEGDataset


def create_dataloaders(
    train_samples,
    val_samples,
    test_samples,
    common_channels,
    batch_size=32,
    use_cache=True,
    cache_dir="./cache",
    num_workers=0,
    pin_memory=True,
):
    """Create train, validation, and test DataLoaders."""

    train_dataset = EEGDataset(
        samples=train_samples,
        common_channels=common_channels,
        use_cache=use_cache,
        train=True,
        cache_dir=cache_dir,
    )

    val_dataset = EEGDataset(
        samples=val_samples,
        common_channels=common_channels,
        use_cache=use_cache,
        train=False,
        cache_dir=cache_dir,
    )

    test_dataset = EEGDataset(
        samples=test_samples,
        common_channels=common_channels,
        use_cache=use_cache,
        train=False,
        cache_dir=cache_dir,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
