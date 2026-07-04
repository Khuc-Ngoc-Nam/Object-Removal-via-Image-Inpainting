from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch


class SAM2SegmenterUnavailable(RuntimeError):
    pass


class SAM2ClickSegmenter:
    def __init__(
        self,
        checkpoint_path: Path,
        model_cfg: str = "configs/sam2.1/sam2.1_hiera_s.yaml",
        device: str | None = None,
    ) -> None:
        if not checkpoint_path.exists():
            raise SAM2SegmenterUnavailable(f"SAM2 checkpoint not found: {checkpoint_path}")

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as exc:  # pragma: no cover - depends on local SAM2 install.
            raise SAM2SegmenterUnavailable(
                "SAM2 is not installed. Install it with: "
                "pip install git+https://github.com/facebookresearch/sam2.git"
            ) from exc

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_sam2(model_cfg, str(checkpoint_path), device=self.device)
        self.predictor = SAM2ImagePredictor(self.model)

    @torch.inference_mode()
    def segment_from_point(
        self,
        image_rgb: np.ndarray,
        x: int,
        y: int,
        multimask_output: bool = True,
    ) -> tuple[np.ndarray, float]:
        height, width = image_rgb.shape[:2]
        point = np.array([[np.clip(x, 0, width - 1), np.clip(y, 0, height - 1)]], dtype=np.float32)
        label = np.array([1], dtype=np.int32)

        self.predictor.set_image(image_rgb)
        masks, scores, _ = self.predictor.predict(
            point_coords=point,
            point_labels=label,
            multimask_output=multimask_output,
        )

        if masks is None or len(masks) == 0:
            raise RuntimeError("SAM2 returned no masks for this click.")

        best_idx = int(np.argmax(scores)) if scores is not None and len(scores) else 0
        mask = (masks[best_idx] > 0).astype(np.uint8) * 255
        score = float(scores[best_idx]) if scores is not None and len(scores) else 0.0
        return mask, score


@lru_cache(maxsize=1)
def get_sam2_click_segmenter(
    checkpoint_path: str,
    model_cfg: str,
    device: str | None,
) -> SAM2ClickSegmenter:
    return SAM2ClickSegmenter(
        checkpoint_path=Path(checkpoint_path),
        model_cfg=model_cfg,
        device=device,
    )
