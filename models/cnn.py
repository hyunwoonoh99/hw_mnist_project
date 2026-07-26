import torch
import torch.nn as nn


class CNN(nn.Module):
    """합성곱 신경망: conv+pooling으로 특징 추출 후 FC로 분류.
    MLP와 동일하게 feature_extractor + classifier 2단 구조.
    """

    def __init__(self, num_classes=10):
        super().__init__()

        # feature_extractor: conv 층들로 공간적 특징 추출
        self.conv_layers = nn.Sequential(
            # 1번째 conv 블록: 1채널 → 32채널
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),   # 28×28 → 14×14

            # 2번째 conv 블록: 32채널 → 64채널
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),   # 14×14 → 7×7
        )

        self.flatten = nn.Flatten()   # [B, 64, 7, 7] → [B, 3136]

        # penultimate feature layer (t-SNE에 쓸 부분)
        self.fc_feature = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
        )

        # classifier
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x, return_features=False):
        """x: [B, 1, 28, 28] → logit [B, 10]"""
        x = self.conv_layers(x)      # [B, 64, 7, 7]
        x = self.flatten(x)          # [B, 3136]
        feat = self.fc_feature(x)    # [B, 128]  ← feature
        logit = self.classifier(feat)  # [B, 10]
        if return_features:
            return logit, feat
        return logit


# 자체 테스트
if __name__ == "__main__":
    model = CNN()
    print(model)

    dummy = torch.randn(128, 1, 28, 28)
    logit = model(dummy)
    print(f"\nlogit shape: {logit.shape}")            # [128, 10]

    logit, feat = model(dummy, return_features=True)
    print(f"feature shape: {feat.shape}")             # [128, 128]

    total_params = sum(p.numel() for p in model.parameters())
    print(f"학습 파라미터 수: {total_params:,}")