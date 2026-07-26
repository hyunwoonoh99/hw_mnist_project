import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def get_transforms(augment=False):
    if augment:
        return transforms.Compose([
            transforms.RandomRotation(10),                          # 회전은 유지 (±10도 적절)
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),  # 이동·확대 완화
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])


def get_dataloaders(batch_size=128, data_dir="./data", num_workers=2,
                    val_ratio=0.1, seed=42, train_subset=None, augment=False):
    train_transform = get_transforms(augment=augment)
    eval_transform = get_transforms(augment=False)

    # train/val 인덱스 분할 (transform 없이 인덱스만 먼저)
    base = datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)
    val_size = int(len(base) * val_ratio)
    train_size = len(base) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_split, val_split = random_split(base, [train_size, val_size], generator=generator)

    train_indices = train_split.indices
    val_indices = val_split.indices

    # train: augment transform / val: eval transform (원본) — 각각 다른 dataset 인스턴스
    train_base = datasets.MNIST(root=data_dir, train=True, transform=train_transform)
    val_base   = datasets.MNIST(root=data_dir, train=True, transform=eval_transform)
    train_dataset = Subset(train_base, train_indices)
    val_dataset   = Subset(val_base, val_indices)

    # 데이터 축소
    if train_subset is not None:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(train_dataset), generator=g)[:train_subset]
        train_dataset = Subset(train_dataset, idx.tolist())

    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader