import torch
import torch.nn as nn


class CNN(nn.Module):
    """Ablation 가능한 CNN.
    use_bn, use_dropout 옵션으로 BatchNorm/Dropout을 켜고 끌 수 있다.
    """

    def __init__(self, num_classes=10, use_bn=False, use_dropout=False, dropout_p=0.25):
        super().__init__()
        self.use_bn = use_bn
        self.use_dropout = use_dropout

        # --- Conv block 1 ---
        layers1 = [nn.Conv2d(1, 32, kernel_size=3, padding=1)]
        if use_bn:
            layers1.append(nn.BatchNorm2d(32))   # conv 다음, ReLU 전
        layers1.append(nn.ReLU())
        layers1.append(nn.MaxPool2d(2))          # 28 → 14
        self.conv1 = nn.Sequential(*layers1)

        # --- Conv block 2 ---
        layers2 = [nn.Conv2d(32, 64, kernel_size=3, padding=1)]
        if use_bn:
            layers2.append(nn.BatchNorm2d(64))
        layers2.append(nn.ReLU())
        layers2.append(nn.MaxPool2d(2))          # 14 → 7
        self.conv2 = nn.Sequential(*layers2)

        self.flatten = nn.Flatten()

        # --- FC feature ---
        fc_layers = [nn.Linear(64 * 7 * 7, 128), nn.ReLU()]
        if use_dropout:
            fc_layers.append(nn.Dropout(dropout_p))   # FC 다음
        self.fc_feature = nn.Sequential(*fc_layers)

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x, return_features=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        feat = self.fc_feature(x)
        logit = self.classifier(feat)
        if return_features:
            return logit, feat
        return logit


if __name__ == "__main__":
    # 네 가지 조합 테스트
    for bn, dp in [(False, False), (True, False), (False, True), (True, True)]:
        m = CNN(use_bn=bn, use_dropout=dp)
        params = sum(p.numel() for p in m.parameters())
        print(f"use_bn={bn}, use_dropout={dp} → params={params:,}")