from __future__ import annotations

import numpy as np

from agent_sam31.types import PromptHints


LOCATION_GRID = ["top", "middle", "bottom"]
LOCATION_COLS = ["left", "center", "right"]


def build_prompt_hints(
    pre_image: np.ndarray,
    post_image: np.ndarray,
    mask: np.ndarray,
    bbox_xyxy: list[int],
    segmentation_reference: str,
) -> PromptHints:
    mask_bool = mask.astype(bool)
    x0, y0, x1, y1 = bbox_xyxy
    box_w = max(1, x1 - x0)
    box_h = max(1, y1 - y0)
    area = max(1, int(mask_bool.sum()))
    bbox_fill_ratio = float(area / (box_w * box_h))
    aspect_ratio = float(max(box_w / box_h, box_h / box_w))
    location_hint = _build_location_hint(mask.shape[1], mask.shape[0], bbox_xyxy)
    object_hint = _build_object_hint(area, mask.shape[0] * mask.shape[1], bbox_fill_ratio, aspect_ratio)
    mean_abs_diff = _masked_mean_abs_diff(pre_image, post_image, mask_bool, bbox_xyxy)
    change_hint = _build_change_hint(mean_abs_diff)
    return PromptHints(
        location_hint=location_hint,
        object_hint=object_hint,
        change_hint=change_hint,
        reference_hint=segmentation_reference,
        mean_abs_diff=mean_abs_diff,
        bbox_fill_ratio=bbox_fill_ratio,
        aspect_ratio=aspect_ratio,
    )


def _build_location_hint(img_w: int, img_h: int, bbox_xyxy: list[int]) -> str:
    x0, y0, x1, y1 = bbox_xyxy
    cx = (x0 + x1) / 2 / max(1, img_w)
    cy = (y0 + y1) / 2 / max(1, img_h)
    row = min(2, int(cy * 3))
    col = min(2, int(cx * 3))
    return f"{LOCATION_GRID[row]}-{LOCATION_COLS[col]}"


def _build_object_hint(
    area: int, image_area: int, bbox_fill_ratio: float, aspect_ratio: float
) -> str:
    area_ratio = area / max(1, image_area)
    if aspect_ratio >= 4.0:
        return "linear_feature"
    if area_ratio >= 0.1:
        return "large_region"
    if bbox_fill_ratio >= 0.55:
        return "compact_object"
    if bbox_fill_ratio <= 0.2:
        return "sparse_region"
    return "changed_region"


def _build_change_hint(mean_abs_diff: float) -> str:
    if mean_abs_diff >= 45.0:
        return "strong_change"
    if mean_abs_diff >= 20.0:
        return "moderate_change"
    return "subtle_change"


def _masked_mean_abs_diff(
    pre_image: np.ndarray,
    post_image: np.ndarray,
    mask_bool: np.ndarray,
    bbox_xyxy: list[int],
) -> float:
    diff = np.abs(pre_image.astype(np.float32) - post_image.astype(np.float32)).mean(axis=2)
    if mask_bool.any():
        return float(diff[mask_bool].mean())
    x0, y0, x1, y1 = bbox_xyxy
    return float(diff[y0:y1, x0:x1].mean()) if y1 > y0 and x1 > x0 else 0.0
