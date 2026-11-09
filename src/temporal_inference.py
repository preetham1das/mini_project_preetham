import numpy as np
import torch


def calculate_deforestation_mask(forest_mask_t1, forest_mask_t2):
    forest_mask_t1 = np.asarray(forest_mask_t1)
    forest_mask_t2 = np.asarray(forest_mask_t2)
    if forest_mask_t1.shape != forest_mask_t2.shape:
        raise ValueError("T1 and T2 forest masks must have identical shapes.")

    return ((forest_mask_t1 == 1) & (forest_mask_t2 == 0)).astype(np.uint8)


def predict_temporal_forest_masks(model, image_t1, image_t2, device, threshold):
    if image_t1.shape != image_t2.shape:
        raise ValueError("T1 and T2 images must have identical tensor shapes.")

    model.eval()
    with torch.no_grad():
        logits_t1 = model(image_t1.to(device))
        logits_t2 = model(image_t2.to(device))
        forest_mask_t1 = (
            (torch.sigmoid(logits_t1) > threshold)[0, 0]
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        forest_mask_t2 = (
            (torch.sigmoid(logits_t2) > threshold)[0, 0]
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

    deforestation_mask = calculate_deforestation_mask(
        forest_mask_t1, forest_mask_t2
    )
    return forest_mask_t1, forest_mask_t2, deforestation_mask
