from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from agent_sam31.types import InstancePrediction


def save_visualizations(
    output_dir: Path,
    reference_image: np.ndarray,
    probability_map: np.ndarray,
    binary_mask: np.ndarray,
    instances: list[InstancePrediction],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    instances_dir = output_dir / "instances"
    instances_dir.mkdir(parents=True, exist_ok=True)

    probability_map_path = output_dir / "change_probability.png"
    binary_mask_path = output_dir / "change_binary.png"
    coarse_overlay_path = output_dir / "coarse_overlay.png"
    instance_overlay_path = output_dir / "instance_overlay.png"

    probability_u8 = np.clip(probability_map * 255.0, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(probability_u8, cv2.COLORMAP_JET)
    cv2.imwrite(str(probability_map_path), heatmap)
    cv2.imwrite(str(binary_mask_path), (binary_mask > 0).astype(np.uint8) * 255)
    cv2.imwrite(str(coarse_overlay_path), _overlay_mask(reference_image, binary_mask, (0, 255, 255), 0.35))

    overlay = reference_image.copy()
    for instance in instances:
        color = _instance_color(instance.instance_id)
        overlay = _overlay_mask(overlay, instance.mask, color, 0.35)
        x0, y0, x1, y1 = instance.bbox_xyxy
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)
        cv2.putText(
            overlay,
            f"#{instance.instance_id}",
            (x0, max(12, y0 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
        instance_mask_path = instances_dir / f"instance_{instance.instance_id:03d}_mask.png"
        cv2.imwrite(str(instance_mask_path), (instance.mask > 0).astype(np.uint8) * 255)
        instance.mask_path = str(instance_mask_path.relative_to(output_dir))
    cv2.imwrite(str(instance_overlay_path), overlay)
    return {
        "probability_map_path": probability_map_path,
        "binary_mask_path": binary_mask_path,
        "coarse_overlay_path": coarse_overlay_path,
        "instance_overlay_path": instance_overlay_path,
    }


def _overlay_mask(
    image_bgr: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float
) -> np.ndarray:
    overlay = image_bgr.copy()
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = (
        (1 - alpha) * overlay[mask_bool] + alpha * np.asarray(color, dtype=np.float32)
    ).astype(np.uint8)
    return overlay


def _instance_color(instance_id: int) -> tuple[int, int, int]:
    base = np.random.default_rng(seed=instance_id).integers(80, 255, size=3)
    return int(base[0]), int(base[1]), int(base[2])
