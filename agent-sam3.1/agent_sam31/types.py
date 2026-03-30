from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class RuntimeConfig:
    pre_image: Path
    post_image: Path
    output_dir: Path
    changevit_repo: Path
    changevit_checkpoint: Path
    sam3_repo: Path
    sam3_checkpoint: Path | None = None
    sam3_bpe_path: Path | None = None
    gt_mask: Path | None = None
    device: str = "cuda"
    changevit_model_type: str = "small"
    changevit_input_size: int = 256
    change_threshold: float = 0.5
    changevit_norm_profile: str = "eval"
    segmentation_reference: str = "post"
    min_region_area: int = 64
    region_expand_ratio: float = 0.08
    max_regions: int = 64
    sam_confidence_threshold: float = 0.2
    min_mask_overlap: float = 0.15
    max_mask_area_ratio: float = 0.35
    mask_morph_kernel: int = 3
    dedupe_iou: float = 0.75
    sam3_load_from_hf: bool = False


@dataclass(slots=True)
class RegionProposal:
    region_id: int
    bbox_xyxy: list[int]
    expanded_bbox_xyxy: list[int]
    area: int
    score: float
    mask: np.ndarray = field(repr=False)


@dataclass(slots=True)
class PromptHints:
    location_hint: str
    object_hint: str
    change_hint: str
    reference_hint: str
    mean_abs_diff: float
    bbox_fill_ratio: float
    aspect_ratio: float
    hint_source: str = "heuristic"


@dataclass(slots=True)
class InstancePrediction:
    instance_id: int
    region_id: int
    source: str
    score: float
    area: int
    bbox_xyxy: list[int]
    mask: np.ndarray = field(repr=False)
    coarse_overlap: float
    mask_path: str | None = None
    category: str | None = None
    category_score: float | None = None
    prompt_hints: PromptHints | None = None
    short_description_zh: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PrototypeResult:
    config: RuntimeConfig
    probability_map_path: Path
    binary_mask_path: Path
    overlay_path: Path
    summary_json_path: Path
    metrics: dict[str, float] | None
    instances: list[InstancePrediction]
