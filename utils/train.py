import os
import copy
import torch
import torch.nn as nn
import random
import numpy as np

def set_seed(seed=42):
    """재현성을 위해 모든 랜덤 시드 고정."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def train_one_epoch(model, loader, criterion, optimizer, device):
    """한 epoch 학습. 평균 loss와 정확도 반환."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """평가 (val 또는 test). 평균 loss와 정확도 반환."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def fit(model, train_loader, val_loader, test_loader,
        epochs=15, lr=0.001, device="cuda", save_path=None):
    """전체 학습 실행.
    - val로 매 epoch 모니터링
    - val_loss 최저인 모델(best)을 기억
    - 학습 후 best model로 test 최종 평가
    반환: history(딕셔너리), best_epoch, test 결과
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # val_loss가 지금까지 최저면 이 모델을 best로 기억
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())  # 가중치 복사본 저장
            best_epoch = epoch
            marker = "  <- best"

        print(f"Epoch {epoch:2d}/{epochs} | "
              f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val_loss {val_loss:.4f} acc {val_acc:.4f}{marker}")

    # best model 복원 후 test 평가
    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n>> Best epoch: {best_epoch} (val_loss {best_val_loss:.4f})")
    print(f">> Test  | loss {test_loss:.4f} acc {test_acc:.4f}")

    # best model 파일로 저장 (선택)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(best_state, save_path)
        print(f">> Saved best model to {save_path}")

    return history, best_epoch, (test_loss, test_acc)


# 자체 테스트
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.data import get_dataloaders
    from models.mlp import MLP

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128)
    model = MLP()
    history, best_epoch, test_result = fit(
        model, train_loader, val_loader, test_loader,
        epochs=15, device=device,
        save_path="./results/mlp_best.pth"
    )