import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

NUM_WORKERS = os.cpu_count()


def create_dataloaders(train_dir: str, val_dir: str, transform: transforms.Compose, batch_size: int,
                       num_workers: int = NUM_WORKERS):
    """
    Creates dataloaders for training and validation sets.
    返回 :A tuple of (train_dataloader, test_dataloader, class_names)
    class_names是文件夹中类别的名称
    """
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    val_data = datasets.ImageFolder(val_dir, transform=transform)
    class_names = train_data.classes

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, class_names
