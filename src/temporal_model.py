import torch
import torch.nn as nn
import torch.nn.functional as F

TEMPORAL_CHECKPOINT_PATH = "models/best_temporal_model.pth"

class DoubleConv(nn.Module):
    """(Convolution -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class TemporalUNet(nn.Module):
    """
    Temporal U-Net Architecture for Deforestation Detection.
    
    Input: (B, 6, H, W) Concatenated T1 and T2 Satellite Imagery
    Output: (B, 1, H, W) Logits for Deforestation Change (1) vs No Change (0)
    """
    def __init__(self, in_channels=6, out_channels=1, init_features=32):
        super(TemporalUNet, self).__init__()
        features = init_features

        # Encoder
        self.inc = DoubleConv(in_channels, features)  # 6 -> 32
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(features, features * 2)         # 32 -> 64
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(features * 2, features * 4)     # 64 -> 128
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(features * 4, features * 8)     # 128 -> 256
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(features * 8, features * 16)    # 256 -> 512
        )

        # Decoder
        self.up1 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(features * 16, features * 8)

        self.up2 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(features * 8, features * 4)

        self.up3 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(features * 4, features * 2)

        self.up4 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(features * 2, features)

        # Output Layer
        self.outc = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x_up = self.up1(x5)
        x_up = torch.cat([x4, x_up], dim=1)
        x_up = self.conv_up1(x_up)

        x_up = self.up2(x_up)
        x_up = torch.cat([x3, x_up], dim=1)
        x_up = self.conv_up2(x_up)

        x_up = self.up3(x_up)
        x_up = torch.cat([x2, x_up], dim=1)
        x_up = self.conv_up3(x_up)

        x_up = self.up4(x_up)
        x_up = torch.cat([x1, x_up], dim=1)
        x_up = self.conv_up4(x_up)

        logits = self.outc(x_up)
        return logits

if __name__ == "__main__":
    print("=" * 50)
    print("TEMPORAL U-NET MODEL ARCHITECTURE TEST")
    print("=" * 50)

    model = TemporalUNet(in_channels=6, out_channels=1, init_features=32)
    dummy_input = torch.randn(2, 6, 256, 256)
    output = model(dummy_input)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Input shape       : {dummy_input.shape}")
    print(f"Output shape      : {output.shape}")
    print(f"Trainable params  : {total_params:,}")
    print("Temporal model architecture verification successful!")
