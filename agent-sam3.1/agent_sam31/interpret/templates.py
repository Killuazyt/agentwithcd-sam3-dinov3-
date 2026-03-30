from __future__ import annotations

from agent_sam31.types import PromptHints


LOCATION_ZH = {
    "top-left": "左上区域",
    "top-center": "上方区域",
    "top-right": "右上区域",
    "middle-left": "左侧区域",
    "middle-center": "中心区域",
    "middle-right": "右侧区域",
    "bottom-left": "左下区域",
    "bottom-center": "下方区域",
    "bottom-right": "右下区域",
}

OBJECT_ZH = {
    "linear_feature": "线状变化目标",
    "large_region": "大范围变化区域",
    "compact_object": "紧凑型变化目标",
    "sparse_region": "稀疏变化区域",
    "changed_region": "变化目标",
}

CHANGE_ZH = {
    "strong_change": "明显变化",
    "moderate_change": "中等变化",
    "subtle_change": "轻微变化",
}

REFERENCE_ZH = {
    "post": "以后时相轮廓为主",
    "pre": "以前时相轮廓为主",
}


def build_short_description_zh(hints: PromptHints) -> str:
    location = LOCATION_ZH.get(hints.location_hint, hints.location_hint)
    obj = OBJECT_ZH.get(hints.object_hint, hints.object_hint)
    change = CHANGE_ZH.get(hints.change_hint, hints.change_hint)
    reference = REFERENCE_ZH.get(hints.reference_hint, hints.reference_hint)
    return f"在{location}检测到一个{obj}，呈现{change}，当前结果以{reference}。"


def build_llm_ready_prompt(hints: PromptHints) -> dict[str, str | float]:
    return {
        "location_hint": hints.location_hint,
        "object_hint": hints.object_hint,
        "change_hint": hints.change_hint,
        "reference_hint": hints.reference_hint,
        "mean_abs_diff": round(hints.mean_abs_diff, 4),
        "bbox_fill_ratio": round(hints.bbox_fill_ratio, 4),
        "aspect_ratio": round(hints.aspect_ratio, 4),
    }
