import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])


def get_dataloaders(batch_size=128, data_dir="./data", num_workers=2,
                    val_ratio=0.1, seed=42):
    """train/val/test DataLoader 반환.
    train 6만 장을 (1-val_ratio):val_ratio 로 train/val 분할.
    """
    transform = get_transforms()

    full_train = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    val_size = int(len(full_train) * val_ratio)      # 6000
    train_size = len(full_train) - val_size          # 54000
    generator = torch.Generator().manual_seed(seed)  # 분할 고정 (재현성)
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128)
    print(f"train batch 수: {len(train_loader)}")   # 54000/128 ≈ 422
    print(f"val batch 수:   {len(val_loader)}")     # 6000/128 ≈ 47
    print(f"test batch 수:  {len(test_loader)}")    # 10000/128 ≈ 79
    images, labels = next(iter(train_loader))
    print(f"이미지 shape: {images.shape}")