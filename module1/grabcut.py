from __future__ import annotations

import cv2
import numpy as np


DEFAULT_NUM_ITERS = 5
DEFAULT_MAX_SIDE = 1024


def _binary(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.zeros(shape, dtype=np.uint8)

    if mask.shape[:2] != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)

    return (mask > 127).astype(np.uint8)


def _scale_for_inference(
    image_rgb: np.ndarray,
    fg_mask: np.ndarray,
    bg_mask: np.ndarray,
    rect: tuple[int, int, int, int],
    max_side: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int], float]:
    height, width = image_rgb.shape[:2]
    if max_side <= 0 or max(height, width) <= max_side:
        return image_rgb, fg_mask, bg_mask, rect, 1.0

    scale = max_side / max(height, width)
    scaled_w = max(2, int(round(width * scale)))
    scaled_h = max(2, int(round(height * scale)))

    scaled_image = cv2.resize(image_rgb, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
    scaled_fg = cv2.resize(fg_mask, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
    scaled_bg = cv2.resize(bg_mask, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
    x, y, w, h = rect
    scaled_rect = (
        int(round(x * scale)),
        int(round(y * scale)),
        max(1, int(round(w * scale))),
        max(1, int(round(h * scale))),
    )
    return scaled_image, scaled_fg, scaled_bg, scaled_rect, scale


def _clip_rect(rect: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = rect
    x = max(0, min(int(x), width - 2))
    y = max(0, min(int(y), height - 2))
    w = max(1, min(int(w), width - x))
    h = max(1, min(int(h), height - y))
    return x, y, w, h


def run_grabcut(
    image_rgb: np.ndarray,
    rect: tuple[int, int, int, int],
    fg_mask: np.ndarray | None = None,
    bg_mask: np.ndarray | None = None,
    num_iters: int = DEFAULT_NUM_ITERS,
    max_side: int = DEFAULT_MAX_SIDE,
) -> np.ndarray:
    """Run the same GrabCut interaction model as the notebook: rect first, strokes refine."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("GrabCut expects an RGB image.")

    height, width = image_rgb.shape[:2]
    rect = _clip_rect(rect, width=width, height=height)
    if rect[2] < 2 or rect[3] < 2:
        raise ValueError("GrabCut rectangle is too small.")

    fg = _binary(fg_mask, (height, width))
    bg = _binary(bg_mask, (height, width))
    infer_image, infer_fg, infer_bg, infer_rect, scale = _scale_for_inference(
        image_rgb=image_rgb,
        fg_mask=fg,
        bg_mask=bg,
        rect=rect,
        max_side=max_side,
    )

    infer_h, infer_w = infer_image.shape[:2]
    infer_rect = _clip_rect(infer_rect, width=infer_w, height=infer_h)

    gc_mask = np.full((infer_h, infer_w), cv2.GC_BGD, dtype=np.uint8)
    x, y, w, h = infer_rect
    gc_mask[y : y + h, x : x + w] = cv2.GC_PR_FGD
    gc_mask[infer_bg > 0] = cv2.GC_BGD
    gc_mask[infer_fg > 0] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    image_bgr = cv2.cvtColor(infer_image, cv2.COLOR_RGB2BGR)

    cv2.grabCut(
        image_bgr,
        gc_mask,
        infer_rect,
        bgd_model,
        fgd_model,
        max(1, int(num_iters)),
        cv2.GC_INIT_WITH_MASK,
    )

    binary = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)

    if scale != 1.0:
        binary = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)

    return binary
