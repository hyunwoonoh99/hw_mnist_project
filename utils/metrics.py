import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix,
    classification_report,
)


@torch.no_grad()
def collect_predictions(model, loader, device):
    """전체 loader에 대해 예측값과 정답을 수집.
    반환: (y_true, y_pred) numpy 배열
    """
    model.eval()
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)          # 예측 클래스
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    return y_true, y_pred


def compute_metrics(y_true, y_pred, num_classes=10):
    """여러 평가 지표를 계산해 딕셔너리로 반환."""
    metrics = {}

    # 1) Accuracy
    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    # 2) F1 — 세 가지 평균 방식
    metrics["f1_micro"] = f1_score(y_true, y_pred, average="micro")
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro")
    metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted")

    # 3) Precision/Recall/F1 (macro 기준 대표값)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    metrics["precision_macro"] = p
    metrics["recall_macro"] = r

    # 4) Per-class F1 (클래스별)
    per_class_f1 = f1_score(y_true, y_pred, average=None)  # 배열 반환
    metrics["per_class_f1"] = per_class_f1

    return metrics


def print_metrics_report(y_true, y_pred):
    """지표들을 보기 좋게 출력."""
    m = compute_metrics(y_true, y_pred)

    print("=" * 50)
    print(f"Accuracy      : {m['accuracy']:.4f}")
    print(f"F1 (micro)    : {m['f1_micro']:.4f}   <- accuracy와 비교해봐")
    print(f"F1 (macro)    : {m['f1_macro']:.4f}")
    print(f"F1 (weighted) : {m['f1_weighted']:.4f}")
    print(f"Precision(mac): {m['precision_macro']:.4f}")
    print(f"Recall (mac)  : {m['recall_macro']:.4f}")
    print("-" * 50)
    print("Per-class F1:")
    for cls, f1 in enumerate(m["per_class_f1"]):
        print(f"  숫자 {cls}: {f1:.4f}")
    print("=" * 50)

    # sklearn의 종합 리포트 (precision/recall/f1/support 한 번에)
    print("\n[sklearn classification_report]")
    print(classification_report(y_true, y_pred, digits=4))


def get_confusion_matrix(y_true, y_pred, num_classes=10):
    """confusion matrix 반환 (시각화는 visualize.py에서)."""
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


# 자체 테스트 — 학습된 MLP로 실제 지표 계산
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.data import get_dataloaders
    from models.mlp import MLP

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 저장해둔 best model 불러오기
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128)
    model = MLP().to(device)
    model.load_state_dict(torch.load("./results/mlp_best.pth", map_location=device))
    print("Loaded best MLP model.\n")

    # test set에 대해 예측 수집 + 지표 계산
    y_true, y_pred = collect_predictions(model, test_loader, device)
    print_metrics_report(y_true, y_pred)

    # confusion matrix 출력 (숫자로)
    cm = get_confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix (행=정답, 열=예측):")
    print(cm)