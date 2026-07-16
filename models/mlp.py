import torch
import torch.nn as nn


class MLP(nn.Module):
    """이미지를 펴서 완전연결층으로 분류.
    구조를 feature_extractor(특징 추출) + classifier(분류)로 분리.
    """

    def __init__(self, input_size=784, hidden_sizes=(256, 128), num_classes=10):
        super().__init__()

        self.flatten = nn.Flatten()  # [B,1,28,28] → [B,784]

        # feature_extractor: 입력 → penultimate feature (128차원)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),      # 784 → 256
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]), # 256 → 128
            nn.ReLU(),
        )

        # classifier: feature → 클래스 logit (마지막 층만)
        self.classifier = nn.Linear(hidden_sizes[1], num_classes)  # 128 → 10

    def forward(self, x, return_features=False):
        """x: [B,1,28,28] → logit [B,10]
        return_features=True면 (logit, feature) 튜플 반환.
        """
        x = self.flatten(x)
        feat = self.feature_extractor(x)   # [B, 128]  ← t-SNE에 쓸 feature
        logit = self.classifier(feat)      # [B, 10]
        if return_features:
            return logit, feat
        return logit


# 자체 테스트
if __name__ == "__main__":
    model = MLP()
    print(model)

    dummy = torch.randn(128, 1, 28, 28)

    # 일반 forward
    logit = model(dummy)
    print(f"\nlogit shape: {logit.shape}")          # [128, 10]

    # feature 반환 forward
    logit, feat = model(dummy, return_features=True)
    print(f"feature shape: {feat.shape}")           # [128, 128]

    total_params = sum(p.numel() for p in model.parameters())
    print(f"학습 파라미터 수: {total_params:,}")       # 235,146 (구조 나눠도 총량 동일)