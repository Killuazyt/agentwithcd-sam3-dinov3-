from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from agent_sam31.interpret.templates import build_llm_ready_prompt
from agent_sam31.types import InstancePrediction, PrototypeResult, RuntimeConfig


def save_json_summary(
    output_path: Path,
    config: RuntimeConfig,
    instances: list[InstancePrediction],
    metrics: dict[str, float] | None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "pre_image": str(config.pre_image),
        "post_image": str(config.post_image),
        "segmentation_reference": config.segmentation_reference,
        "change_threshold": config.change_threshold,
        "metrics": metrics,
        "instances": [_serialize_instance(instance) for instance in instances],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _serialize_instance(instance: InstancePrediction) -> dict:
    data = {
        "instance_id": instance.instance_id,
        "region_id": instance.region_id,
        "source": instance.source,
        "score": round(float(instance.score), 6),
        "area": int(instance.area),
        "bbox_xyxy": instance.bbox_xyxy,
        "coarse_overlap": round(float(instance.coarse_overlap), 6),
        "mask_path": instance.mask_path,
        "category": instance.category,
        "category_score": instance.category_score,
        "extra": instance.extra,
    }
    if instance.prompt_hints is not None:
        data["prompt_hints"] = asdict(instance.prompt_hints)
        data["llm_ready_prompt"] = build_llm_ready_prompt(instance.prompt_hints)
    if instance.short_description_zh is not None:
        data["short_description_zh"] = instance.short_description_zh
    return data
