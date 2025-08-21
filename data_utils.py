import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_orientation_loaders(
    data_dir: str,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4
):
    """
    Prepare PyTorch DataLoaders for orientation classification.
    Assumes `data_dir/train` and `data_dir/val` with class subfolders.
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))
    ])

    train_folder = os.path.join(os.getcwd(), 'train')
    val_folder   = os.path.join(os.getcwd(), 'val')

    train_ds = datasets.ImageFolder(train_folder, transform=transform)
    val_ds   = datasets.ImageFolder(val_folder,   transform=transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    classes = train_ds.classes
    print(f"[Data] classes={classes}, train={len(train_ds)}, val={len(val_ds)}")
    return train_loader, val_loader, classes


def get_orientation_loaders_evaluation(
    data_dir: str,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4
):
    """
    Prepare PyTorch DataLoaders for orientation classification.
    Assumes `data_dir/train` and `data_dir/val` with class subfolders.
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))
    ])

    # train_folder = os.path.join(os.getcwd(), 'train')
    val_folder   = os.path.join(os.getcwd(), 'val')

    # train_ds = datasets.ImageFolder(train_folder, transform=transform)
    val_ds   = datasets.ImageFolder(val_folder,   transform=transform)

    # train_loader = DataLoader(
    #     train_ds, batch_size=batch_size, shuffle=True,
    #     num_workers=num_workers, pin_memory=True
    # )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    classes = val_ds.classes
    print(f"[Data] classes={classes}, val={len(val_ds)}")
    return val_loader, classes