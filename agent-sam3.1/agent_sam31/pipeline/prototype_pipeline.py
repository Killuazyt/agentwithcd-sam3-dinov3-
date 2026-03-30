from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from agent_sam31.adapters.changevit_adapter import ChangeVitAdapter
from agent_sam31.adapters.sam3_adapter import Sam3Adapter
from agent_sam31.eval.mask_metrics import compute_binary_metrics, load_binary_mask
from agent_sam31.export.json_summary import save_json_summary
from agent_sam31.export.visualize import save_visualizations
from agent_sam31.interpret.prompts import build_prompt_hints
from agent_sam31.interpret.templates import build_short_description_zh
from agent_sam31.postprocess.regions import (
    aggregate_instance_masks,
    binary_iou,
    clean_binary_mask,
    dedupe_instances,
    extract_regions,
    mask_to_bbox_xyxy,
    refine_instance_mask,
)
from agent_sam31.types import InstancePrediction, PrototypeResult, RuntimeConfig


def run_prototype(config: RuntimeConfig) -> PrototypeResult:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    pre_image = ChangeVitAdapter.load_image(config.pre_image)
    post_image = ChangeVitAdapter.load_image(config.post_image)
    if pre_image.shape[:2] != post_image.shape[:2]:
        raise ValueError("The bi-temporal images must share the same height and width")

    changevit = ChangeVitAdapter(
        repo_dir=config.changevit_repo,
        checkpoint_path=config.changevit_checkpoint,
        model_type=config.changevit_model_type,
        device=config.device,
        input_size=config.changevit_input_size,
        norm_profile=config.changevit_norm_profile,
    )
    change_outputs = changevit.predict_arrays(
        pre_image,
        post_image,
        threshold=config.change_threshold,
    )
    probability_map = change_outputs["probability_map"]
    binary_mask = clean_binary_mask(
        change_outputs["binary_mask"],
        kernel_size=config.mask_morph_kernel,
        min_area=config.min_region_area,
    )
    regions = extract_regions(
        probability_map=probability_map,
        binary_mask=binary_mask,
        min_area=config.min_region_area,
        expand_ratio=config.region_expand_ratio,
        max_regions=config.max_regions,
    )

    reference_image = post_image if config.segmentation_reference == "post" else pre_image
    sam3 = None
    if regions:
        sam3 = Sam3Adapter(
            repo_dir=config.sam3_repo,
            checkpoint_path=config.sam3_checkpoint,
            device=config.device,
            confidence_threshold=config.sam_confidence_threshold,
            bpe_path=config.sam3_bpe_path,
            load_from_hf=config.sam3_load_from_hf,
        )
        sam3.set_image(reference_image)

    instances: list[InstancePrediction] = []
    image_area = reference_image.shape[0] * reference_image.shape[1]
    for next_id, region in enumerate(regions, start=1):
        sam_result = sam3.segment_from_box(region.expanded_bbox_xyxy) if sam3 is not None else None
        final_mask, source, score, overlap, extra = _resolve_instance_mask(
            region_mask=region.mask,
            region_bbox=region.expanded_bbox_xyxy,
            sam_result=sam_result,
            min_mask_overlap=config.min_mask_overlap,
            max_mask_area_ratio=config.max_mask_area_ratio,
            image_area=image_area,
            morphology_kernel=config.mask_morph_kernel,
        )
        if final_mask.sum() < config.min_region_area:
            continue
        bbox_xyxy = mask_to_bbox_xyxy(final_mask)
        hints = build_prompt_hints(
            pre_image=pre_image,
            post_image=post_image,
            mask=final_mask,
            bbox_xyxy=bbox_xyxy,
            segmentation_reference=config.segmentation_reference,
        )
        instance = InstancePrediction(
            instance_id=next_id,
            region_id=region.region_id,
            source=source,
            score=score,
            area=int(final_mask.sum()),
            bbox_xyxy=bbox_xyxy,
            mask=final_mask,
            coarse_overlap=overlap,
            prompt_hints=hints,
            short_description_zh=build_short_description_zh(hints),
            extra=extra,
        )
        instances.append(instance)

    instances = dedupe_instances(instances, config.dedupe_iou)
    for new_id, instance in enumerate(instances, start=1):
        instance.instance_id = new_id

    visualization_paths = save_visualizations(
        output_dir=config.output_dir,
        reference_image=reference_image,
        probability_map=probability_map,
        binary_mask=binary_mask,
        instances=instances,
    )

    metrics = None
    if config.gt_mask is not None:
        gt_mask = load_binary_mask(config.gt_mask)
        pred_union = aggregate_instance_masks(instances, gt_mask.shape)
        metrics = compute_binary_metrics(pred_union, gt_mask)

    summary_json_path = save_json_summary(
        output_path=config.output_dir / "summary.json",
        config=config,
        instances=instances,
        metrics=metrics,
    )
    return PrototypeResult(
        config=config,
        probability_map_path=visualization_paths["probability_map_path"],
        binary_mask_path=visualization_paths["binary_mask_path"],
        overlay_path=visualization_paths["instance_overlay_path"],
        summary_json_path=summary_json_path,
        metrics=metrics,
        instances=instances,
    )


def _resolve_instance_mask(
    region_mask: np.ndarray,
    region_bbox: list[int],
    sam_result: dict[str, np.ndarray | float | list[float]] | None,
    min_mask_overlap: float,
    max_mask_area_ratio: float,
    image_area: int,
    morphology_kernel: int,
) -> tuple[np.ndarray, str, float, float, dict[str, float | int | str]]:
    fallback_mask = refine_instance_mask(region_mask, morphology_kernel)
    if sam_result is None:
        return fallback_mask, "coarse_fallback", 0.0, 1.0, {"reason": "sam3_no_candidate"}

    sam_mask = refine_instance_mask(np.asarray(sam_result["mask"], dtype=np.uint8), morphology_kernel)
    overlap = _coarse_overlap(sam_mask, region_mask)
    score = float(sam_result["score"])
    mask_area_ratio = float(sam_mask.sum() / max(1, image_area))
    if sam_mask.sum() == 0:
        return fallback_mask, "coarse_fallback", 0.0, 1.0, {"reason": "sam3_empty_mask"}
    if overlap < min_mask_overlap:
        return fallback_mask, "coarse_fallback", score, overlap, {"reason": "low_overlap", "sam_score": score}
    if mask_area_ratio > max_mask_area_ratio:
        return fallback_mask, "coarse_fallback", score, overlap, {"reason": "mask_too_large", "sam_score": score}
    return sam_mask, "sam3", score, overlap, {
        "sam_score": score,
        "num_candidates": int(sam_result.get("num_candidates", 1)),
        "prompt_box_xyxy": region_bbox,
    }


def _coarse_overlap(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    region_area = float((mask_b > 0).sum())
    if region_area <= 0:
        return 0.0
    intersection = float(np.logical_and(mask_a > 0, mask_b > 0).sum())
    return intersection / region_area
