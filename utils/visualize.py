import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, num_classes=10, title="Confusion Matrix",
                          normalize=False, save_path=None):
    """혼동 행렬 heatmap. normalize=True면 행(정답)별 비율로."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        fmt = "d"

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_learning_curve(history, title="Learning Curve", save_path=None):
    """train/val loss·accuracy를 epoch별로 그린다."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, history["train_loss"], label="train_loss")
    ax1.plot(epochs, history["val_loss"], label="val_loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="train_acc")
    ax2.plot(epochs, history["val_acc"], label="val_acc")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.set_title("Accuracy")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


@torch.no_grad()
def extract_features(model, loader, device, max_samples=2000):
    """모델의 return_features=True를 이용해 penultimate feature 추출.
    max_samples: t-SNE는 느리니 이 개수만.
    반환: (features, labels) numpy 배열
    """
    model.eval()
    features, labels = [], []
    count = 0

    for images, targets in loader:
        images = images.to(device)
        _, feat = model(images, return_features=True)   # (logit, feature)
        features.append(feat.cpu().numpy())
        labels.append(targets.numpy())
        count += images.size(0)
        if count >= max_samples:
            break

    features = np.concatenate(features)[:max_samples]
    labels = np.concatenate(labels)[:max_samples]
    return features, labels


def plot_tsne(features, labels, num_classes=10, title="t-SNE of Features",
              save_path=None, perplexity=30, seed=42):
    """고차원 feature를 t-SNE로 2D 투영해 산점도로."""
    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=seed, init="pca")
    emb = tsne.fit_transform(features)

    plt.figure(figsize=(9, 8))
    scatter = plt.scatter(emb[:, 0], emb[:, 1], c=labels,
                          cmap="tab10", s=8, alpha=0.7)
    plt.colorbar(scatter, ticks=range(num_classes), label="digit")
    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_model_comparison_curves(all_history, metric="val_acc", title=None, save_path=None):
    """여러 모델의 learning curve를 한 그림에 겹쳐 그린다."""
    plt.figure(figsize=(9, 6))
    for name, hist in all_history.items():
        epochs = range(1, len(hist[metric]) + 1)
        plt.plot(epochs, hist[metric], marker="o", markersize=3, label=name)
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(title or f"Model Comparison ({metric})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_accuracy_bar(model_accs, title="Test Accuracy Comparison", save_path=None):
    """모델별 test accuracy를 막대그래프로."""
    names = list(model_accs.keys())
    accs = list(model_accs.values())

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, accs, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
    plt.ylabel("Test Accuracy")
    plt.title(title)
    plt.ylim(0.95, 1.0)
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width()/2, acc + 0.0005,
                 f"{acc:.4f}", ha="center", va="bottom", fontsize=10)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()