from __future__ import annotations

import argparse
from pathlib import Path

from agent_sam31.pipeline.prototype_pipeline import run_prototype
from agent_sam31.types import RuntimeConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ChangeViT + SAM3 变化实例原型")
    parser.add_argument("--pre-image", required=True, help="前时相图像路径")
    parser.add_argument("--post-image", required=True, help="后时相图像路径")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--changevit-repo", required=True, help="ChangeViT 仓库目录")
    parser.add_argument("--changevit-checkpoint", required=True, help="ChangeViT 训练权重路径")
    parser.add_argument("--sam3-repo", required=True, help="sam3 仓库目录")
    parser.add_argument("--sam3-checkpoint", default=None, help="sam3 本地权重路径，可为空")
    parser.add_argument("--sam3-load-from-hf", action="store_true", help="若未提供本地权重，则尝试从 Hugging Face 下载")
    parser.add_argument("--sam3-bpe-path", default=None, help="sam3 BPE 词表路径，可为空")
    parser.add_argument("--device", default="cuda", help="运行设备，如 cuda 或 cpu")
    parser.add_argument("--changevit-model-type", default="small", choices=["tiny", "small"], help="ChangeViT 模型类型")
    parser.add_argument("--changevit-input-size", type=int, default=256, help="ChangeViT 推理输入尺寸")
    parser.add_argument("--change-threshold", type=float, default=0.5, help="变化二值化阈值")
    parser.add_argument("--changevit-norm-profile", default="eval", choices=["eval", "train"], help="ChangeViT 归一化配置")
    parser.add_argument("--segmentation-reference", default="post", choices=["pre", "post"], help="SAM3 精修使用的参考时相")
    parser.add_argument("--min-region-area", type=int, default=64, help="连通域最小面积")
    parser.add_argument("--region-expand-ratio", type=float, default=0.08, help="候选框外扩比例")
    parser.add_argument("--max-regions", type=int, default=64, help="最大候选实例数")
    parser.add_argument("--sam-confidence-threshold", type=float, default=0.2, help="SAM3 候选保留阈值")
    parser.add_argument("--min-mask-overlap", type=float, default=0.15, help="SAM 掩码与粗变化区域的最小重叠比例")
    parser.add_argument("--max-mask-area-ratio", type=float, default=0.35, help="单实例掩码最大面积占比")
    parser.add_argument("--mask-morph-kernel", type=int, default=3, help="实例掩码形态学核大小")
    parser.add_argument("--dedupe-iou", type=float, default=0.75, help="重复实例去重阈值")
    parser.add_argument("--gt-mask", default=None, help="可选像素级 GT 路径")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = RuntimeConfig(
        pre_image=Path(args.pre_image),
        post_image=Path(args.post_image),
        output_dir=Path(args.output_dir),
        changevit_repo=Path(args.changevit_repo),
        changevit_checkpoint=Path(args.changevit_checkpoint),
        sam3_repo=Path(args.sam3_repo),
        sam3_checkpoint=Path(args.sam3_checkpoint) if args.sam3_checkpoint else None,
        sam3_bpe_path=Path(args.sam3_bpe_path) if args.sam3_bpe_path else None,
        gt_mask=Path(args.gt_mask) if args.gt_mask else None,
        device=args.device,
        changevit_model_type=args.changevit_model_type,
        changevit_input_size=args.changevit_input_size,
        change_threshold=args.change_threshold,
        changevit_norm_profile=args.changevit_norm_profile,
        segmentation_reference=args.segmentation_reference,
        min_region_area=args.min_region_area,
        region_expand_ratio=args.region_expand_ratio,
        max_regions=args.max_regions,
        sam_confidence_threshold=args.sam_confidence_threshold,
        min_mask_overlap=args.min_mask_overlap,
        max_mask_area_ratio=args.max_mask_area_ratio,
        mask_morph_kernel=args.mask_morph_kernel,
        dedupe_iou=args.dedupe_iou,
        sam3_load_from_hf=args.sam3_load_from_hf,
    )
    result = run_prototype(config)
    print(f"完成，实例数: {len(result.instances)}")
    print(f"JSON 摘要: {result.summary_json_path}")
    print(f"实例叠加图: {result.overlay_path}")


if __name__ == "__main__":
    main()
