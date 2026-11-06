"""Two-stream late-fusion SegNet for Sentinel-1 SAR + Sentinel-2 optical inputs.

Sentinel-1 and Sentinel-2 are separate sensor modalities. This model does not
interpret them as temporal observations.
"""
from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, layers: int = 2):
        modules = []
        for index in range(layers):
            source = in_channels if index == 0 else out_channels
            modules.extend([
                nn.Conv2d(source, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
        super().__init__(*modules)


class SegNetEncoder(nn.Module):
    """VGG-style encoder that retains max-pool indices for SegNet decoding."""
    def __init__(self, in_channels: int, channels: tuple[int, ...]):
        super().__init__()
        depths = (2, 2, 3, 3, 3)[:len(channels)]
        blocks, current = [], in_channels
        for width, depth in zip(channels, depths):
            blocks.append(ConvBlock(current, width, depth))
            current = width
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x: torch.Tensor):
        indices, pre_pool_shapes = [], []
        for block in self.blocks:
            x = block(x)
            pre_pool_shapes.append(x.size())
            x, index = self.pool(x)
            indices.append(index)
        return x, indices, pre_pool_shapes


class SegNetDecoder(nn.Module):
    def __init__(self, channels: tuple[int, ...]):
        super().__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        blocks = []
        for level in range(len(channels) - 1, -1, -1):
            source = channels[level]
            target = channels[level - 1] if level > 0 else channels[0]
            depth = 3 if level >= 3 else 2
            blocks.append(ConvBlock(source, target, depth))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, indices, pre_pool_shapes):
        for block, level in zip(self.blocks, range(len(indices) - 1, -1, -1)):
            x = self.unpool(x, indices[level], output_size=pre_pool_shapes[level])
            x = block(x)
        return x


class LateFusionSegNet(nn.Module):
    """Independent modality encoders/decoders fused at decoded feature level.

    Args:
        s1_channels: Sentinel-1 band count (the source research notebook uses 2).
        s2_channels: Sentinel-2 band count (the source research notebook uses 4).
        base_channels: Width of the first encoder stage. 32 is sized for Colab GPUs.
    Returns one raw binary logit per pixel; use BCEWithLogits/Dice during training.
    """
    def __init__(self, s1_channels: int = 2, s2_channels: int = 4,
                 base_channels: int = 32):
        super().__init__()
        channels = (base_channels, base_channels * 2, base_channels * 4,
                    base_channels * 8, base_channels * 8)
        self.s1_encoder = SegNetEncoder(s1_channels, channels)
        self.s1_decoder = SegNetDecoder(channels)
        self.s2_encoder = SegNetEncoder(s2_channels, channels)
        self.s2_decoder = SegNetDecoder(channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(channels[0] * 2, channels[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[0], 1, kernel_size=1),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, sentinel1: torch.Tensor, sentinel2: torch.Tensor) -> torch.Tensor:
        if sentinel1.shape[-2:] != sentinel2.shape[-2:]:
            raise ValueError("Sentinel-1 and Sentinel-2 tensors must share spatial dimensions")
        h, w = sentinel1.shape[-2:]
        if h % 32 or w % 32:
            raise ValueError("Input height and width must be divisible by 32 (e.g. 256x256)")
        z1, i1, s1 = self.s1_encoder(sentinel1)
        z2, i2, s2 = self.s2_encoder(sentinel2)
        d1 = self.s1_decoder(z1, i1, s1)
        d2 = self.s2_decoder(z2, i2, s2)
        return self.fusion(torch.cat((d2, d1), dim=1))
