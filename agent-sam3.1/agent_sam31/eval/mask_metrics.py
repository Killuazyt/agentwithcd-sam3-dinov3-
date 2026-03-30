from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


def load_binary_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {mask_path}")
    return (mask > 0).astype(np.uint8)


def compute_binary_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict[str, float]:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    tn = float(np.logical_and(~pred, ~gt).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    oa = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "oa": oa,
    }
