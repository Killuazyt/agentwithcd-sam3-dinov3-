from __future__ import annotations

import contextlib
import importlib
import sys
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch


NORM_PROFILES: dict[str, tuple[list[float], list[float]]] = {
    "eval": ([0.5] * 6, [0.5] * 6),
    "train": (
        [0.406, 0.456, 0.485, 0.406, 0.456, 0.485],
        [0.225, 0.224, 0.229, 0.225, 0.224, 0.229],
    ),
}


@contextlib.contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    try:
        path.mkdir(parents=True, exist_ok=True)
        os_chdir = __import__("os").chdir
        os_chdir(path)
        yield
    finally:
        os_chdir(previous)


class ChangeVitAdapter:
    def __init__(
        self,
        repo_dir: Path,
        checkpoint_path: Path,
        model_type: str = "small",
        device: str = "cuda",
        input_size: int = 256,
        norm_profile: str = "eval",
    ) -> None:
        self.repo_dir = repo_dir.resolve()
        self.checkpoint_path = checkpoint_path.resolve()
        if not self.repo_dir.exists():
            raise FileNotFoundError(f"ChangeViT repo not found: {self.repo_dir}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"ChangeViT checkpoint not found: {self.checkpoint_path}")
        self.input_size = input_size

        self.device = torch.device(
            device if device != "cuda" or torch.cuda.is_available() else "cpu"
        )
        if norm_profile not in NORM_PROFILES:
            raise ValueError(f"Unsupported ChangeViT norm profile: {norm_profile}")
        self.mean, self.std = NORM_PROFILES[norm_profile]
        self._ensure_repo_importable()
        trainer_cls = importlib.import_module("model.trainer").Trainer
        with _working_directory(self.repo_dir):
            self.model = trainer_cls(model_type).float()
        checkpoint = torch.load(str(self.checkpoint_path), map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

    def _ensure_repo_importable(self) -> None:
        repo = str(self.repo_dir)
        if repo not in sys.path:
            sys.path.insert(0, repo)

    def _preprocess_pair(
        self, pre_image: np.ndarray, post_image: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = np.concatenate([pre_image, post_image], axis=2).astype(np.float32)
        combined /= 255.0
        for idx in range(6):
            combined[:, :, idx] = (combined[:, :, idx] - self.mean[idx]) / self.std[idx]
        resized = cv2.resize(combined, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        resized = resized[:, :, ::-1].copy()
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).unsqueeze(0).float()
        pre_tensor = tensor[:, 0:3].to(self.device)
        post_tensor = tensor[:, 3:6].to(self.device)
        return pre_tensor, post_tensor

    @staticmethod
    def load_image(image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        return image

    def predict_paths(
        self, pre_image_path: Path, post_image_path: Path, threshold: float = 0.5
    ) -> dict[str, np.ndarray]:
        pre_image = self.load_image(pre_image_path)
        post_image = self.load_image(post_image_path)
        return self.predict_arrays(pre_image, post_image, threshold=threshold)

    def predict_arrays(
        self, pre_image: np.ndarray, post_image: np.ndarray, threshold: float = 0.5
    ) -> dict[str, np.ndarray]:
        if pre_image.shape[:2] != post_image.shape[:2]:
            raise ValueError("The bi-temporal images must share the same height and width")
        pre_tensor, post_tensor = self._preprocess_pair(pre_image, post_image)
        with torch.inference_mode():
            probability = self.model(pre_tensor, post_tensor)
        probability_map = probability.squeeze().detach().cpu().numpy().astype(np.float32)
        probability_map = cv2.resize(
            probability_map,
            (pre_image.shape[1], pre_image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        probability_map = np.clip(probability_map, 0.0, 1.0)
        binary_mask = (probability_map >= threshold).astype(np.uint8)
        return {
            "probability_map": probability_map,
            "binary_mask": binary_mask,
        }
