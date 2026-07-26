import torch
import torch.nn as nn


class RNN(nn.Module):
    """LSTM 기반 순환 신경망: 이미지를 row 단위 sequence로 처리.
    이미지에 부적합한 구조를 의도적으로 적용 (발표 논점).
    """

    def __init__(self, input_size=28, hidden_size=128, num_layers=2, num_classes=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM: 각 스텝에서 28차원(한 행)을 받아 hidden_size 차원 기억으로
        self.lstm = nn.LSTM(
            input_size=input_size,      # 28 (한 행의 픽셀 수)
            hidden_size=hidden_size,    # 128 (기억 차원)
            num_layers=num_layers,      # 2층 LSTM
            batch_first=True,           # 입력 shape을 [B, seq, feature]로
        )

        # classifier: 마지막 hidden state → 클래스
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x, return_features=False):
        """x: [B, 1, 28, 28] → logit [B, 10]"""
        # 이미지를 sequence로 변환: [B,1,28,28] → [B,28,28]
        x = x.squeeze(1)   # 채널 차원 제거 → [B, 28, 28] (28 timesteps × 28 features)

        # LSTM 통과
        # out: [B, 28, hidden] (모든 스텝의 hidden state)
        # (h_n, c_n): 마지막 스텝의 hidden/cell state
        out, (h_n, c_n) = self.lstm(x)

        # 마지막 timestep의 hidden state를 feature로 사용
        feat = out[:, -1, :]   # [B, hidden] ← sequence 전체 요약

        logit = self.classifier(feat)   # [B, 10]
        if return_features:
            return logit, feat
        return logit


# 자체 테스트
if __name__ == "__main__":
    model = RNN()
    print(model)

    dummy = torch.randn(128, 1, 28, 28)
    logit = model(dummy)
    print(f"\nlogit shape: {logit.shape}")            # [128, 10]

    logit, feat = model(dummy, return_features=True)
    print(f"feature shape: {feat.shape}")             # [128, 128]

    total_params = sum(p.numel() for p in model.parameters())
    print(f"학습 파라미터 수: {total_params:,}")