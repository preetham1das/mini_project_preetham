import torch
import torch.nn as nn
import torch.nn.functional as F

TEMPORAL_CHECKPOINT_PATH = "models/best_temporal_model.pth"


# --------------------------------------------------
# Building Blocks
# --------------------------------------------------

class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) x 2"""
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0.0:
            layers.append(nn.Dropout2d(p=dropout_p))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# --------------------------------------------------
# Loss Functions
# --------------------------------------------------

class DiceLoss(nn.Module):
    """Soft Dice Loss (operates on probabilities, not logits)."""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        if targets.ndim == logits.ndim - 1:
            targets = targets.unsqueeze(1)
        if logits.shape != targets.shape:
            raise ValueError(f"Logits and targets must have matching shapes; got {logits.shape} and {targets.shape}.")
        probs = torch.sigmoid(logits).flatten(start_dim=1)
        targets = targets.float().flatten(start_dim=1)
        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


def _match_target_shape(logits, targets):
    if targets.ndim == logits.ndim - 1:
        targets = targets.unsqueeze(1)
    if logits.shape != targets.shape:
        raise ValueError(f"Logits and targets must have matching shapes; got {logits.shape} and {targets.shape}.")
    return targets.float()


class FocalLoss(nn.Module):
    """
    Focal Loss for binary segmentation (Sigmoid-based).
    Ref: Lin et al. 2017 -- "Focal Loss for Dense Object Detection"

    gamma: focusing parameter (0 = BCE, 2 = standard focal)
    alpha: weight for positive class (None = no weight)
    """
    def __init__(self, gamma=2.0, alpha=0.75, reduction="mean"):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        if alpha is not None and not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be None or between 0 and 1")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = _match_target_shape(logits, targets)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedBCEDiceLoss(nn.Module):
    """Weighted BCE + Dice (original approach, improved)."""
    def __init__(self, pos_weight_val=20.0, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        pos_w = torch.tensor([pos_weight_val])
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        targets = _match_target_shape(logits, targets)
        # Move pos_weight to same device as logits
        self.bce.pos_weight = self.bce.pos_weight.to(logits.device)
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)


class FocalDiceLoss(nn.Module):
    """Focal Loss + Dice Loss (best for highly imbalanced change detection)."""
    def __init__(self, focal_weight=0.5, dice_weight=0.5, gamma=2.0, alpha=0.75):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal = FocalLoss(gamma=gamma, alpha=alpha)
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        targets = targets.float()
        return self.focal_weight * self.focal(logits, targets) + self.dice_weight * self.dice(logits, targets)


# --------------------------------------------------
# Model Architecture
# --------------------------------------------------

class TemporalUNet(nn.Module):
    """
    Temporal U-Net for Deforestation Change Detection.

    Supports:
      - 6-channel input  [A_RGB, B_RGB]               (diff_channels=False)
      - 9-channel input  [A_RGB, B_RGB, |A-B|_RGB]    (diff_channels=True)

    Args:
        in_channels:    6 or 9
        out_channels:   1 (binary mask)
        init_features:  base feature count (32 = standard, 48 = larger)
        bottleneck_dropout: dropout probability in bottleneck (0.0 = off)
    """
    def __init__(
        self,
        in_channels=6,
        out_channels=1,
        init_features=32,
        bottleneck_dropout=0.0,
    ):
        super().__init__()
        f = init_features

        # Encoder
        self.inc    = DoubleConv(in_channels, f)
        self.down1  = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f,     f * 2))
        self.down2  = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 2, f * 4))
        self.down3  = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 4, f * 8))
        self.down4  = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(f * 8, f * 16, dropout_p=bottleneck_dropout),
        )

        # Decoder
        self.up1     = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(f * 16, f * 8)

        self.up2     = nn.ConvTranspose2d(f * 8,  f * 4, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(f * 8,  f * 4)

        self.up3     = nn.ConvTranspose2d(f * 4,  f * 2, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(f * 4,  f * 2)

        self.up4     = nn.ConvTranspose2d(f * 2,  f,     kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(f * 2,  f)

        self.outc = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        expected_channels = self.inc.block[0].in_channels
        if x.ndim != 4 or x.shape[1] != expected_channels:
            raise ValueError(f"Expected input shaped (N, {expected_channels}, H, W), got {tuple(x.shape)}")
        height, width = x.shape[-2:]
        if height == 0 or width == 0:
            raise ValueError("Input height and width must be positive.")
        pad_h, pad_w = (-height) % 16, (-width) % 16
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        u = self.up1(x5);  u = self.conv_up1(torch.cat([x4, u], dim=1))
        u = self.up2(u);   u = self.conv_up2(torch.cat([x3, u], dim=1))
        u = self.up3(u);   u = self.conv_up3(torch.cat([x2, u], dim=1))
        u = self.up4(u);   u = self.conv_up4(torch.cat([x1, u], dim=1))

        logits = self.outc(u)
        return logits[..., :height, :width]


if __name__ == "__main__":
    for ch in [6, 9]:
        model = TemporalUNet(in_channels=ch, out_channels=1, init_features=32)
        dummy = torch.randn(2, ch, 256, 256)
        out = model(dummy)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"in_channels={ch}: output={out.shape}, params={params:,}")
