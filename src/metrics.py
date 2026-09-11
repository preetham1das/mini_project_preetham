import torch
import torch.nn as nn

THRESHOLD = 0.5


class DiceLoss(nn.Module):
    """
    Dice Loss for Binary Semantic Segmentation.
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1).float()

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice


class CombinedBCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy (BCEWithLogitsLoss) + Dice Loss.
    Optimal loss function for satellite image binary semantic segmentation.
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super(CombinedBCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        # Flatten targets if shape is (B, H, W) to match (B, 1, H, W)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        
        bce = self.bce_loss(logits, targets.float())
        dice = self.dice_loss(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice


def calculate_confusion_counts(predictions, targets, threshold=THRESHOLD, is_logits=True):
    """Return foreground confusion counts for a batch of predictions."""
    if is_logits:
        predictions = torch.sigmoid(predictions)

    preds = (predictions > threshold).float()
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)
    targets = targets.float()

    preds_flat = preds.reshape(-1)
    targets_flat = targets.reshape(-1)

    return {
        "tp": int((preds_flat * targets_flat).sum().item()),
        "fp": int((preds_flat * (1.0 - targets_flat)).sum().item()),
        "fn": int(((1.0 - preds_flat) * targets_flat).sum().item()),
        "tn": int(((1.0 - preds_flat) * (1.0 - targets_flat)).sum().item()),
    }


def metrics_from_counts(counts):
    """Calculate exact foreground metrics from accumulated confusion counts."""
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]

    iou_denominator = tp + fp + fn
    dice_denominator = 2 * tp + fp + fn
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    accuracy_denominator = tp + tn + fp + fn

    return {
        "iou": tp / iou_denominator if iou_denominator else 1.0,
        "dice": (2 * tp) / dice_denominator if dice_denominator else 1.0,
        "precision": tp / precision_denominator if precision_denominator else 1.0,
        "recall": tp / recall_denominator if recall_denominator else 1.0,
        "accuracy": (tp + tn) / accuracy_denominator if accuracy_denominator else 1.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def calculate_metrics(predictions, targets, threshold=THRESHOLD, is_logits=True):
    """
    Calculates binary segmentation evaluation metrics:
    - IoU (Jaccard Index)
    - Dice Coefficient (F1 Score)
    - Precision
    - Recall
    - Pixel Accuracy

    Args:
        predictions (torch.Tensor): Model outputs (logits or probabilities)
        targets (torch.Tensor): Ground truth binary mask (0 or 1)
        threshold (float): Classification threshold (default: 0.5)
        is_logits (bool): If True, applies sigmoid to predictions first

    Returns:
        dict: Dictionary containing metric scores float values
    """
    return metrics_from_counts(
        calculate_confusion_counts(predictions, targets, threshold=threshold, is_logits=is_logits)
    )


if __name__ == "__main__":
    print("=" * 50)
    print("METRICS & LOSS FUNCTIONS TEST (src/metrics.py)")
    print("=" * 50)

    dummy_logits = torch.randn(2, 1, 512, 512)
    dummy_targets = (torch.rand(2, 512, 512) > 0.5).long()

    criterion = CombinedBCEDiceLoss()
    loss = criterion(dummy_logits, dummy_targets)

    metrics = calculate_metrics(dummy_logits, dummy_targets, threshold=0.5)

    print(f"Combined BCE + Dice Loss : {loss.item():.4f}")
    print(f"Calculated IoU           : {metrics['iou']:.4f}")
    print(f"Calculated Dice (F1)     : {metrics['dice']:.4f}")
    print(f"Calculated Precision    : {metrics['precision']:.4f}")
    print(f"Calculated Recall       : {metrics['recall']:.4f}")
    print(f"Calculated Pixel Acc     : {metrics['accuracy']:.4f}")
    print("Metrics test completed successfully!")
