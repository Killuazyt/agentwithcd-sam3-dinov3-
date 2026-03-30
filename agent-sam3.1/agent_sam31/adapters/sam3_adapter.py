from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


class Sam3Adapter:
    def __init__(
        self,
        repo_dir: Path,
        checkpoint_path: Path | None,
        device: str = "cuda",
        confidence_threshold: float = 0.2,
        bpe_path: Path | None = None,
        load_from_hf: bool = False,
    ) -> None:
        self.repo_dir = repo_dir.resolve()
        self.checkpoint_path = checkpoint_path.resolve() if checkpoint_path else None
        self.bpe_path = bpe_path.resolve() if bpe_path else None
        if not self.repo_dir.exists():
            raise FileNotFoundError(f"sam3 repo not found: {self.repo_dir}")
        if self.checkpoint_path is None and not load_from_hf:
            raise ValueError("Provide `sam3_checkpoint` or enable `sam3_load_from_hf`")
        if self.checkpoint_path is not None and not self.checkpoint_path.exists():
            raise FileNotFoundError(f"sam3 checkpoint not found: {self.checkpoint_path}")
        self.device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"

        self._ensure_repo_importable()
        model_builder = importlib.import_module("sam3.model_builder")
        processor_module = importlib.import_module("sam3.model.sam3_image_processor")
        self.model = model_builder.build_sam3_image_model(
            bpe_path=str(self.bpe_path) if self.bpe_path else None,
            device=self.device,
            eval_mode=True,
            checkpoint_path=str(self.checkpoint_path) if self.checkpoint_path else None,
            load_from_HF=load_from_hf,
            enable_segmentation=True,
            enable_inst_interactivity=False,
        )
        self.processor = processor_module.Sam3Processor(
            self.model,
            device=self.device,
            confidence_threshold=confidence_threshold,
        )
        self._state: dict | None = None
        self._image_shape: tuple[int, int] | None = None

    def _ensure_repo_importable(self) -> None:
        repo = str(self.repo_dir)
        if repo not in sys.path:
            sys.path.insert(0, repo)

    def set_image(self, image_bgr: np.ndarray) -> None:
        image_rgb = image_bgr[:, :, ::-1]
        self._image_shape = image_bgr.shape[:2]
        self._state = self.processor.set_image(Image.fromarray(image_rgb))

    def segment_from_box(self, box_xyxy: list[int]) -> dict[str, np.ndarray | float | list[float]] | None:
        if self._state is None or self._image_shape is None:
            raise RuntimeError("Call set_image() before segment_from_box()")
        img_h, img_w = self._image_shape
        x0, y0, x1, y1 = [int(v) for v in box_xyxy]
        x0 = max(0, min(x0, img_w - 1))
        y0 = max(0, min(y0, img_h - 1))
        x1 = max(x0 + 1, min(x1, img_w))
        y1 = max(y0 + 1, min(y1, img_h))
        prompt = [
            (x0 + x1) / 2 / img_w,
            (y0 + y1) / 2 / img_h,
            (x1 - x0) / img_w,
            (y1 - y0) / img_h,
        ]
        self.processor.reset_all_prompts(self._state)
        state = self.processor.add_geometric_prompt(box=prompt, label=True, state=self._state)
        scores = state["scores"].detach().cpu().numpy().astype(np.float32)
        if scores.size == 0:
            return None
        masks = state["masks"].detach().cpu().numpy().astype(bool)
        boxes = state["boxes"].detach().cpu().numpy().astype(np.float32)
        best_index = self._select_best_index(scores, boxes, np.asarray([x0, y0, x1, y1], dtype=np.float32))
        best_mask = np.squeeze(masks[best_index]).astype(np.uint8)
        best_box = boxes[best_index].tolist()
        return {
            "mask": best_mask,
            "score": float(scores[best_index]),
            "box_xyxy": [float(v) for v in best_box],
            "num_candidates": float(scores.size),
        }

    @staticmethod
    def _select_best_index(
        scores: np.ndarray, boxes_xyxy: np.ndarray, prompt_box_xyxy: np.ndarray
    ) -> int:
        prompt_area = max(
            1.0,
            (prompt_box_xyxy[2] - prompt_box_xyxy[0])
            * (prompt_box_xyxy[3] - prompt_box_xyxy[1]),
        )
        box_scores: list[float] = []
        for idx in range(len(scores)):
            box = boxes_xyxy[idx]
            inter_x0 = max(box[0], prompt_box_xyxy[0])
            inter_y0 = max(box[1], prompt_box_xyxy[1])
            inter_x1 = min(box[2], prompt_box_xyxy[2])
            inter_y1 = min(box[3], prompt_box_xyxy[3])
            inter_w = max(0.0, inter_x1 - inter_x0)
            inter_h = max(0.0, inter_y1 - inter_y0)
            inter = inter_w * inter_h
            box_area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
            union = prompt_area + box_area - inter
            iou = inter / union
            box_scores.append(float(scores[idx]) * (0.5 + 0.5 * iou))
        return int(np.argmax(np.asarray(box_scores, dtype=np.float32)))
