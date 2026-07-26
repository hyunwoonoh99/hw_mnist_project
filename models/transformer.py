import torch
import torch.nn as nn


class ViT(nn.Module):
    """간단한 Vision Transformer: 이미지를 패치로 쪼개 self-attention으로 분류."""

    def __init__(self, img_size=28, patch_size=7, in_channels=1,
                 embed_dim=128, num_heads=4, num_layers=2, num_classes=10):
        super().__init__()

        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2   # (28//7)^2 = 16
        patch_dim = in_channels * patch_size * patch_size  # 1*7*7 = 49

        # 1) 패치 embedding: 각 패치(49) → embed_dim(128)
        self.patch_embed = nn.Linear(patch_dim, embed_dim)

        # 2) CLS 토큰 (학습 가능한 대표 토큰)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # 3) position embedding (패치 위치 정보, +1은 CLS 토큰 자리)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        # 4) Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            batch_first=True,   # [B, seq, dim] 순서
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5) classifier
        self.classifier = nn.Linear(embed_dim, num_classes)

    def patchify(self, x):
        """이미지를 패치 시퀀스로 변환.
        [B, 1, 28, 28] → [B, 16, 49]
        """
        B, C, H, W = x.shape
        p = self.patch_size
        # [B,C,H,W] → 패치 격자로 나누고 평탄화
        x = x.unfold(2, p, p).unfold(3, p, p)      # [B, C, H/p, W/p, p, p]
        x = x.contiguous().view(B, C, -1, p * p)    # [B, C, num_patches, p*p]
        x = x.permute(0, 2, 1, 3).contiguous()      # [B, num_patches, C, p*p]
        x = x.view(B, x.size(1), -1)                # [B, num_patches, C*p*p]
        return x

    def forward(self, x, return_features=False):
        """x: [B, 1, 28, 28] → logit [B, 10]"""
        B = x.size(0)

        # 패치로 쪼개고 embedding
        x = self.patchify(x)              # [B, 16, 49]
        x = self.patch_embed(x)           # [B, 16, 128]

        # CLS 토큰을 앞에 붙이기
        cls = self.cls_token.expand(B, -1, -1)   # [B, 1, 128]
        x = torch.cat([cls, x], dim=1)           # [B, 17, 128]

        # position embedding 더하기
        x = x + self.pos_embed            # [B, 17, 128]

        # Transformer 통과
        x = self.transformer(x)           # [B, 17, 128]

        # CLS 토큰(첫 번째)만 취해서 분류
        feat = x[:, 0]                    # [B, 128] ← 이미지 요약 feature
        logit = self.classifier(feat)     # [B, 10]

        if return_features:
            return logit, feat
        return logit


# 자체 테스트
if __name__ == "__main__":
    model = ViT()
    print(model)

    dummy = torch.randn(128, 1, 28, 28)
    logit = model(dummy)
    print(f"\nlogit shape: {logit.shape}")            # [128, 10]

    logit, feat = model(dummy, return_features=True)
    print(f"feature shape: {feat.shape}")             # [128, 128]

    total_params = sum(p.numel() for p in model.parameters())
    print(f"학습 파라미터 수: {total_params:,}")