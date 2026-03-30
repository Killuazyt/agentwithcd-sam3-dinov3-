from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from agent_sam31.types import InstancePrediction, RegionProposal


def clean_binary_mask(mask: np.ndarray, kernel_size: int = 3, min_area: int = 0) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    if kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    if min_area <= 1:
        return mask_u8
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    cleaned = np.zeros_like(mask_u8)
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == label_idx] = 1
    return cleaned


def extract_regions(
    probability_map: np.ndarray,
    binary_mask: np.ndarray,
    min_area: int,
    expand_ratio: float,
    max_regions: int,
) -> list[RegionProposal]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=8)
    regions: list[RegionProposal] = []
    img_h, img_w = binary_mask.shape[:2]
    for label_idx in range(1, num_labels):
        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        region_mask = (labels == label_idx).astype(np.uint8)
        bbox_xyxy = [x, y, x + w, y + h]
        expanded_bbox_xyxy = _expand_box(bbox_xyxy, img_w=img_w, img_h=img_h, expand_ratio=expand_ratio)
        score = float(probability_map[region_mask.astype(bool)].mean())
        regions.append(
            RegionProposal(
                region_id=label_idx,
                bbox_xyxy=bbox_xyxy,
                expanded_bbox_xyxy=expanded_bbox_xyxy,
                area=area,
                score=score,
                mask=region_mask,
            )
        )
    regions.sort(key=lambda item: (item.score, item.area), reverse=True)
    return regions[:max_regions]


def refine_instance_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    if kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    return mask_u8


def binary_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    intersection = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    if union <= 0:
        return 0.0
    return intersection / union


def mask_to_bbox_xyxy(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def dedupe_instances(
    instances: Iterable[InstancePrediction], iou_threshold: float
) -> list[InstancePrediction]:
    kept: list[InstancePrediction] = []
    ordered = sorted(instances, key=lambda item: (item.score, item.area), reverse=True)
    for instance in ordered:
        duplicate = False
        for kept_instance in kept:
            if binary_iou(instance.mask, kept_instance.mask) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(instance)
    kept.sort(key=lambda item: item.instance_id)
    return kept


def aggregate_instance_masks(instances: Iterable[InstancePrediction], shape: tuple[int, int]) -> np.ndarray:
    union_mask = np.zeros(shape, dtype=np.uint8)
    for instance in instances:
        union_mask = np.maximum(union_mask, (instance.mask > 0).astype(np.uint8))
    return union_mask


def _expand_box(
    bbox_xyxy: list[int], img_w: int, img_h: int, expand_ratio: float
) -> list[int]:
    x0, y0, x1, y1 = bbox_xyxy
    box_w = x1 - x0
    box_h = y1 - y0
    dx = int(round(box_w * expand_ratio))
    dy = int(round(box_h * expand_ratio))
    return [
        max(0, x0 - dx),
        max(0, y0 - dy),
        min(img_w, x1 + dx),
        min(img_h, y1 + dy),
    ]
