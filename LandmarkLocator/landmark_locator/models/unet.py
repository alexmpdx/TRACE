"""ResNet18-encoder U-Net for heatmap-based landmark detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DecoderBlock(nn.Module):
    """Upsample → concat skip → two 3×3 conv+BN+ReLU."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        """Initialize decoder block."""
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsample, concat skip, apply convolutions."""
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class LandmarkUNet(nn.Module):
    """U-Net with pretrained ResNet18 encoder for landmark heatmap prediction."""

    def __init__(self, num_landmarks: int = 5, pretrained: bool = True) -> None:
        """Initialize encoder and decoder."""
        super().__init__()
        self.num_landmarks = num_landmarks

        # Encoder: pretrained ResNet18
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        # Extract encoder stages
        # Stage 0: conv1 + bn1 + relu → stride 2, 64 channels
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        # Stage 1: maxpool + layer1 → stride 4, 64 channels
        self.enc1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        # Stage 2: layer2 → stride 8, 128 channels
        self.enc2 = resnet.layer2
        # Stage 3: layer3 → stride 16, 256 channels
        self.enc3 = resnet.layer3
        # Stage 4: layer4 → stride 32, 512 channels
        self.enc4 = resnet.layer4

        # Decoder blocks (bottom-up)
        self.dec4 = DecoderBlock(512, 256, 256)  # stride 32 → 16
        self.dec3 = DecoderBlock(256, 128, 128)  # stride 16 → 8
        self.dec2 = DecoderBlock(128, 64, 64)  # stride 8 → 4
        self.dec1 = DecoderBlock(64, 64, 32)  # stride 4 → 2

        # Final upsample to full resolution + 1×1 conv
        self.final_conv = nn.Conv2d(32, num_landmarks, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encoder → decoder → heatmaps."""
        # Encoder
        e0 = self.enc0(x)  # stride 2, 64ch
        e1 = self.enc1(e0)  # stride 4, 64ch
        e2 = self.enc2(e1)  # stride 8, 128ch
        e3 = self.enc3(e2)  # stride 16, 256ch
        e4 = self.enc4(e3)  # stride 32, 512ch

        # Decoder with skip connections
        d4 = self.dec4(e4, e3)  # 256ch
        d3 = self.dec3(d4, e2)  # 128ch
        d2 = self.dec2(d3, e1)  # 64ch
        d1 = self.dec1(d2, e0)  # 32ch

        # Upsample to input resolution and predict
        d0 = F.interpolate(d1, size=x.shape[2:], mode="bilinear", align_corners=False)
        out = self.final_conv(d0)

        return out

    def freeze_encoder(self) -> None:
        """Freeze all encoder parameters."""
        for module in [self.enc0, self.enc1, self.enc2, self.enc3, self.enc4]:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze all encoder parameters."""
        for module in [self.enc0, self.enc1, self.enc2, self.enc3, self.enc4]:
            for param in module.parameters():
                param.requires_grad = True

    def get_param_groups(self, base_lr: float, encoder_lr_factor: float = 0.1) -> list[dict]:
        """Return param groups with differential LR for encoder vs decoder."""
        encoder_params = []
        for module in [self.enc0, self.enc1, self.enc2, self.enc3, self.enc4]:
            encoder_params.extend(module.parameters())

        decoder_params = []
        for module in [self.dec4, self.dec3, self.dec2, self.dec1, self.final_conv]:
            decoder_params.extend(module.parameters())

        return [
            {"params": encoder_params, "lr": base_lr * encoder_lr_factor},
            {"params": decoder_params, "lr": base_lr},
        ]
