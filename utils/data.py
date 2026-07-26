import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset

# MNIST 전체 데이터셋의 픽셀 통계값 (정규화용 상수)
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def get_transforms(augment=False):
    """전처리 파이프라인. augment=True면 학습용 augmentation 추가 (MNIST 맞춤)."""
    if augment:
        return transforms.Compose([
            transforms.RandomRotation(10),                                          # ±10도 회전
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),  # 이동·확대
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])


def get_dataloaders(batch_size=128, data_dir="./data", num_workers=2,
                    val_ratio=0.1, seed=42,
                    train_subset=None, augment=False,
                    imbalance_classes=None, imbalance_ratio=0.1):
    """MNIST train/val/test DataLoader 생성.

    옵션:
    - train_subset: 정수면 train을 그 개수로 축소 (데이터 부족 실험용).
    - augment: True면 train에만 augmentation 적용.
    - imbalance_classes: 줄일 클래스 리스트 (예: [3, 8]). None이면 균형.
    - imbalance_ratio: 해당 클래스를 원본의 몇 배로 남길지 (0.1=10%, 0.01=1%).

    불균형/축소/augment는 모두 train에만 적용, val/test는 항상 원본 균형.
    """
    train_transform = get_transforms(augment=augment)
    eval_transform = get_transforms(augment=False)

    # train/val 인덱스 분할 (seed 고정)
    base = datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)
    val_size = int(len(base) * val_ratio)
    train_size = len(base) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_split, val_split = random_split(base, [train_size, val_size], generator=generator)

    train_indices = list(train_split.indices)
    val_indices = list(val_split.indices)

    # train은 augment transform, val은 원본 transform (평가 공정성)
    train_base = datasets.MNIST(root=data_dir, train=True, transform=train_transform)
    val_base   = datasets.MNIST(root=data_dir, train=True, transform=eval_transform)

    # --- 클래스 불균형: 지정 클래스의 train 샘플을 imbalance_ratio만큼만 유지 ---
    if imbalance_classes is not None:
        targets = train_base.targets
        g = torch.Generator().manual_seed(seed)
        kept = []
        for idx in train_indices:
            label = int(targets[idx])
            if label in imbalance_classes:
                if torch.rand(1, generator=g).item() < imbalance_ratio:
                    kept.append(idx)      # 확률적으로 일부만 유지
            else:
                kept.append(idx)          # 나머지 클래스는 전부 유지
        train_indices = kept

    train_dataset = Subset(train_base, train_indices)
    val_dataset   = Subset(val_base, val_indices)

    # --- 전체 축소 (imbalance와 별개, 데이터 부족 실험용) ---
    if train_subset is not None:
        g2 = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(train_dataset), generator=g2)[:train_subset]
        train_dataset = Subset(train_dataset, idx.tolist())

    # test는 항상 원본 전체 (균형)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    from collections import Counter

    # 균형 확인
    tr, va, te = get_dataloaders(batch_size=128)
    print(f"[균형] train batch: {len(tr)}, val: {len(va)}, test: {len(te)}")

    # 불균형 확인 (3, 8을 1%로)
    tr2, _, _ = get_dataloaders(imbalance_classes=[3, 8], imbalance_ratio=0.01)
    labels = []
    for _, y in tr2:
        labels.extend(y.tolist())
    dist = Counter(labels)
    print("[불균형 1%] 클래스 분포:")
    for c in range(10):
        print(f"  숫자 {c}: {dist[c]}")