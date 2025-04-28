# utils/data.py
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Resize, Grayscale, ToTensor, Normalize
from torch.utils.data import DataLoader, random_split

def get_dataloaders(
    root: str = "data",
    img_size: int = 224,
    batch_size_train: int = 32,
    batch_size_test: int = 64,
    val_split: float = 0.1,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 0
):
    """
    Returns (train_loader, val_loader, test_loader) for MNIST → 3×224×224.
    Splits off `val_split` fraction of the training set for threshold calibration.
    """
    # 1. Transforms
    transform = Compose([
        Resize(img_size),
        Grayscale(num_output_channels=3),
        ToTensor(),
        Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    # 2. Datasets
    full_train = MNIST(root, train=True, download=True, transform=transform)
    test_set   = MNIST(root, train=False, download=True, transform=transform)

    # 3. Train / Val split
    val_size   = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # 4. DataLoaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader
