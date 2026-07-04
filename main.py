from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

try:
    from model import LaMaGenerator
except Exception as exc:  # pragma: no cover - import error is surfaced by /health and /api/inpaint.
    LaMaGenerator = None
    MODEL_IMPORT_ERROR = exc
else:
    MODEL_IMPORT_ERROR = None

try:
    from lasso_core import LassoCore
except Exception as exc:  # pragma: no cover - import error is surfaced by /health and segment endpoints.
    LassoCore = None
    LASSO_IMPORT_ERROR = exc
else:
    LASSO_IMPORT_ERROR = None

try:
    from module1.grabcut import run_grabcut
except Exception as exc:  # pragma: no cover - import error is surfaced by /health and /api/segment/grabcut.
    run_grabcut = None
    GRABCUT_IMPORT_ERROR = exc
else:
    GRABCUT_IMPORT_ERROR = None

try:
    from module1.sam2_click import (
        SAM2SegmenterUnavailable,
        get_sam2_click_segmenter,
    )
except Exception as exc:  # pragma: no cover - import error is surfaced by /health and /api/segment/sam2.
    SAM2SegmenterUnavailable = RuntimeError
    get_sam2_click_segmenter = None
    SAM2_IMPORT_ERROR = exc
else:
    SAM2_IMPORT_ERROR = None


ROOT_DIR = Path(__file__).resolve().parent
LAMA_CHECKPOINT_PATH = ROOT_DIR / "results" / "checkpoints" / "best_lama_generator_only.pth"
SAM2_CHECKPOINT_PATH = ROOT_DIR / "module1" / "checkpoint_3.pt"
LAMA_DEVICE = torch.device(os.getenv("LAMA_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
SAM2_DEVICE = os.getenv("SAM2_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
SAM2_MODEL_CFG = os.getenv("SAM2_MODEL_CFG", "configs/sam2.1/sam2.1_hiera_s.yaml")

app = FastAPI(title="Object Removal via Image Inpainting API")

cors_origins = os.getenv(
    "CORS_ALLOW_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in cors_origins.split(",") if origin.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded image.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def decode_mask_bytes(mask_bytes: bytes, target_size: tuple[int, int]) -> np.ndarray:
    arr = np.frombuffer(mask_bytes, np.uint8)
    mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded mask.")

    target_w, target_h = target_size
    if mask.shape[:2] != (target_h, target_w):
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    return ((mask > 127).astype(np.uint8) * 255)


def encode_png_bytes(image: np.ndarray) -> bytes:
    image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        encoded_image = image
    elif image.ndim == 3 and image.shape[2] == 3:
        encoded_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        raise HTTPException(status_code=500, detail="Unsupported image shape for PNG encoding.")

    ok, buffer = cv2.imencode(".png", encoded_image)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode result image.")
    return buffer.tobytes()


def png_data_url(image: np.ndarray) -> str:
    image_b64 = base64.b64encode(encode_png_bytes(image)).decode("utf-8")
    return f"data:image/png;base64,{image_b64}"


def normalize_point(item: Any) -> dict[str, int]:
    if isinstance(item, dict):
        return {"x": int(round(float(item["x"]))), "y": int(round(float(item["y"])))}

    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return {"x": int(round(float(item[0]))), "y": int(round(float(item[1])))}

    raise ValueError("Each point must be {'x': number, 'y': number} or [x, y].")


def parse_points(points_json: str | None) -> list[dict[str, int]]:
    if not points_json:
        return []

    try:
        payload = json.loads(points_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid points JSON: {exc}") from exc

    if isinstance(payload, dict):
        payload = payload.get("points") or payload.get("anchors") or payload.get("path")

    if not isinstance(payload, list):
        raise HTTPException(status_code=400, detail="points must be a JSON array.")

    try:
        return [normalize_point(point) for point in payload]
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def parse_rect(rect_json: str | None, width: int, height: int) -> tuple[int, int, int, int]:
    if not rect_json:
        raise HTTPException(status_code=400, detail="rect is required.")

    try:
        payload = json.loads(rect_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid rect JSON: {exc}") from exc

    if isinstance(payload, dict):
        x = int(round(float(payload["x"])))
        y = int(round(float(payload["y"])))
        w = int(round(float(payload.get("width", payload.get("w")))))
        h = int(round(float(payload.get("height", payload.get("h")))))
    elif isinstance(payload, (list, tuple)) and len(payload) >= 4:
        x = int(round(float(payload[0])))
        y = int(round(float(payload[1])))
        w = int(round(float(payload[2])))
        h = int(round(float(payload[3])))
    else:
        raise HTTPException(status_code=400, detail="rect must be {x,y,width,height} or [x,y,w,h].")

    x = max(0, min(x, width - 2))
    y = max(0, min(y, height - 2))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    if w < 2 or h < 2:
        raise HTTPException(status_code=400, detail="rect is too small.")

    return x, y, w, h


def clamp_points(points: Iterable[dict[str, int]], width: int, height: int) -> list[dict[str, int]]:
    return [
        {
            "x": max(0, min(int(point["x"]), width - 1)),
            "y": max(0, min(int(point["y"]), height - 1)),
        }
        for point in points
    ]


def line_points(start: dict[str, int], end: dict[str, int]) -> list[dict[str, int]]:
    dist = max(abs(end["x"] - start["x"]), abs(end["y"] - start["y"]))
    if dist <= 0:
        return [dict(end)]

    xs = np.linspace(start["x"], end["x"], dist + 1, dtype=np.int32)
    ys = np.linspace(start["y"], end["y"], dist + 1, dtype=np.int32)
    return [{"x": int(x), "y": int(y)} for x, y in zip(xs, ys)]


def polygon_mask(points: list[dict[str, int]], width: int, height: int) -> np.ndarray:
    if len(points) < 3:
        raise HTTPException(status_code=400, detail="At least 3 points are required.")

    mask = np.zeros((height, width), dtype=np.uint8)
    contour = np.array([[p["x"], p["y"]] for p in points], dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [contour], 255)
    return mask


def dilate_mask(mask: np.ndarray, dilation_px: int) -> np.ndarray:
    mask = ((mask > 127).astype(np.uint8) * 255)
    if dilation_px <= 0:
        return mask

    kernel_size = int(dilation_px) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=1)


def mask_overlay(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image_rgb.copy().astype(np.float32)
    red = np.zeros_like(overlay)
    red[:, :, 0] = 255
    alpha = ((mask > 0).astype(np.float32) * 0.45)[:, :, None]
    return np.clip(overlay * (1.0 - alpha) + red * alpha, 0, 255).astype(np.uint8)


def build_mask_from_path(image_bytes: bytes, image_rgb: np.ndarray, points: list[dict[str, int]]) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    points = clamp_points(points, width, height)

    if len(points) < 3:
        raise HTTPException(status_code=400, detail="At least 3 points are required.")

    if LassoCore is None:
        return polygon_mask(points, width, height)

    try:
        core = LassoCore()
        core.apply_image(image_bytes)
        return core.generate_mask(points)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Mask generation failed: {exc}") from exc


def build_mask_from_anchors(image_bytes: bytes, image_rgb: np.ndarray, anchors: list[dict[str, int]]) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    anchors = clamp_points(anchors, width, height)

    if len(anchors) < 3:
        raise HTTPException(status_code=400, detail="At least 3 anchors are required.")

    if LassoCore is None:
        return polygon_mask(anchors, width, height)

    try:
        core = LassoCore()
        core.apply_image(image_bytes)

        path_points: list[dict[str, int]] = []
        previous = anchors[0]
        core.build_map([previous["x"], previous["y"]])

        for anchor in anchors[1:]:
            segment = core.get_smart_contour(
                [previous["x"], previous["y"]],
                [anchor["x"], anchor["y"]],
            )
            path_points.extend(segment or line_points(previous, anchor))
            core.build_map([anchor["x"], anchor["y"]])
            previous = anchor

        closing_segment = core.get_smart_contour(
            [previous["x"], previous["y"]],
            [anchors[0]["x"], anchors[0]["y"]],
        )
        path_points.extend(closing_segment or line_points(previous, anchors[0]))

        return core.generate_mask(path_points)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Intelligent Scissors failed: {exc}") from exc


def checkpoint_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint is not a dictionary/state_dict.")

    for key in ("generator_state_dict", "state_dict", "model_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    return checkpoint


def strip_state_dict_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    prefixes = ("module.", "_orig_mod.", "generator.")

    for key, value in state_dict.items():
        clean_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix) :]
                    changed = True
        cleaned[clean_key] = value

    return cleaned


def instantiate_lama() -> torch.nn.Module:
    if LaMaGenerator is None:
        raise RuntimeError(
            "Could not import LaMaGenerator from model.py. "
            f"Original error: {MODEL_IMPORT_ERROR}"
        )

    return LaMaGenerator(input_channels=4, output_channels=3, num_ffc_blocks=9)


@lru_cache(maxsize=1)
def get_lama_model() -> torch.nn.Module:
    if not LAMA_CHECKPOINT_PATH.exists():
        raise RuntimeError(f"Checkpoint not found: {LAMA_CHECKPOINT_PATH}")

    model = instantiate_lama().to(LAMA_DEVICE)
    try:
        checkpoint = torch.load(LAMA_CHECKPOINT_PATH, map_location=LAMA_DEVICE, weights_only=True)
    except TypeError:
        checkpoint = torch.load(LAMA_CHECKPOINT_PATH, map_location=LAMA_DEVICE)

    state_dict = strip_state_dict_prefixes(checkpoint_state_dict(checkpoint))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def first_tensor_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (list, tuple)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item

    if isinstance(output, dict):
        for key in ("image", "inpainted", "output", "pred", "x_hat"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value
        for value in output.values():
            if isinstance(value, torch.Tensor):
                return value

    raise RuntimeError("LaMaGenerator did not return an image tensor.")


def resize_for_limit(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    max_side: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_side <= 0 or max(image_rgb.shape[:2]) <= max_side:
        return image_rgb, mask

    height, width = image_rgb.shape[:2]
    scale = max_side / max(height, width)
    new_w = max(8, int(round(width * scale / 8)) * 8)
    new_h = max(8, int(round(height * scale / 8)) * 8)
    resized_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return resized_rgb, resized_mask


def pad_to_multiple_of_8(
    image_rgb: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    height, width = image_rgb.shape[:2]
    pad_h = (8 - height % 8) % 8
    pad_w = (8 - width % 8) % 8

    if pad_h == 0 and pad_w == 0:
        return image_rgb, mask, (height, width)

    border_type = cv2.BORDER_REFLECT_101 if height > 1 and width > 1 else cv2.BORDER_REPLICATE
    image_padded = cv2.copyMakeBorder(image_rgb, 0, pad_h, 0, pad_w, borderType=border_type)
    mask_padded = cv2.copyMakeBorder(
        mask,
        0,
        pad_h,
        0,
        pad_w,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )
    return image_padded, mask_padded, (height, width)


@torch.inference_mode()
def run_lama_on_image(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    max_side: int = 0,
) -> np.ndarray:
    original_h, original_w = image_rgb.shape[:2]
    binary_mask = ((mask > 127).astype(np.uint8) * 255)
    infer_rgb, infer_mask = resize_for_limit(image_rgb, binary_mask, max_side=max_side)
    infer_rgb, infer_mask, unpadded_size = pad_to_multiple_of_8(infer_rgb, infer_mask)

    mask_01 = (infer_mask > 127).astype(np.float32)
    image_01 = infer_rgb.astype(np.float32) / 255.0
    image_norm = image_01 * 2.0 - 1.0
    masked_norm = image_norm * (1.0 - mask_01[..., None])

    input_np = np.concatenate(
        [masked_norm.transpose(2, 0, 1), mask_01[None, :, :]],
        axis=0,
    )
    input_tensor = torch.from_numpy(input_np).unsqueeze(0).to(LAMA_DEVICE, dtype=torch.float32)

    model = get_lama_model()
    generated_batch = first_tensor_output(model(input_tensor))
    generated = generated_batch[0].detach().cpu()
    generated_01 = (generated * 0.5 + 0.5).clamp(0.0, 1.0)
    generated_rgb = (generated_01.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)

    unpadded_h, unpadded_w = unpadded_size
    generated_rgb = generated_rgb[:unpadded_h, :unpadded_w]

    if generated_rgb.shape[:2] != (original_h, original_w):
        generated_rgb = cv2.resize(generated_rgb, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

    repaired = image_rgb.copy()
    repaired[binary_mask > 127] = generated_rgb[binary_mask > 127]
    return repaired


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise HTTPException(status_code=400, detail="The mask is empty.")

    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def centered_crop_box(
    mask: np.ndarray,
    width: int,
    height: int,
    crop_size: int = 256,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = mask_bbox(mask)
    cx = int(round((x1 + x2) / 2))
    cy = int(round((y1 + y2) / 2))

    left = max(0, min(cx - crop_size // 2, width - crop_size))
    top = max(0, min(cy - crop_size // 2, height - crop_size))
    return left, top, left + crop_size, top + crop_size


def inpaint_center_crop_256(
    image_rgb: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = centered_crop_box(mask, width=width, height=height, crop_size=256)

    crop_rgb = image_rgb[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2].copy()
    if not np.any(crop_mask):
        raise HTTPException(status_code=400, detail="The 256x256 crop does not contain the mask.")

    repaired_crop = run_lama_on_image(crop_rgb, crop_mask, max_side=0)
    result = image_rgb.copy()
    result[y1:y2, x1:x2] = repaired_crop
    return result, (x1, y1, x2, y2)


def inpaint_with_strategy(image_rgb: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    height, width = image_rgb.shape[:2]
    binary_mask = ((mask > 127).astype(np.uint8) * 255)
    if not np.any(binary_mask):
        raise HTTPException(status_code=400, detail="The mask is empty.")

    if height < 256 or width < 256:
        full_result = run_lama_on_image(image_rgb, binary_mask, max_side=0)
        return {
            "strategy": "full_only_small_image",
            "crop_bbox": None,
            "outputs": [
                {
                    "id": "full",
                    "label": "Full image inpaint",
                    "image": full_result,
                }
            ],
        }

    crop_result, crop_bbox = inpaint_center_crop_256(image_rgb, binary_mask)
    full_result = run_lama_on_image(image_rgb, binary_mask, max_side=0)
    return {
        "strategy": "crop_256_and_full_image",
        "crop_bbox": crop_bbox,
        "outputs": [
            {
                "id": "crop_256",
                "label": "256x256 centered crop + paste",
                "image": crop_result,
            },
            {
                "id": "full",
                "label": "Full image inpaint",
                "image": full_result,
            },
        ],
    }


def mask_response(image_rgb: np.ndarray, mask: np.ndarray, extra: dict[str, Any] | None = None) -> JSONResponse:
    payload = {
        "mask": png_data_url(mask),
        "overlay": png_data_url(mask_overlay(image_rgb, mask)),
        "width": int(image_rgb.shape[1]),
        "height": int(image_rgb.shape[0]),
    }
    if extra:
        payload.update(extra)
    return JSONResponse(payload)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "lama_device": str(LAMA_DEVICE),
        "lama_checkpoint_exists": LAMA_CHECKPOINT_PATH.exists(),
        "lama_model_import_ok": LaMaGenerator is not None,
        "lasso_import_ok": LassoCore is not None,
        "grabcut_import_ok": GRABCUT_IMPORT_ERROR is None,
        "sam2_wrapper_import_ok": SAM2_IMPORT_ERROR is None,
        "sam2_package_available": importlib.util.find_spec("sam2") is not None,
        "sam2_checkpoint_exists": SAM2_CHECKPOINT_PATH.exists(),
        "sam2_model_cfg": SAM2_MODEL_CFG,
    }


@app.post("/api/segment/scissors")
async def segment_scissors_endpoint(
    image: UploadFile = File(...),
    points: str = Form(...),
    dilation_px: int = Form(3),
):
    image_bytes = await image.read()
    image_rgb = decode_image_bytes(image_bytes)
    anchors = parse_points(points)
    mask = build_mask_from_anchors(image_bytes, image_rgb, anchors)
    mask = dilate_mask(mask, dilation_px=dilation_px)
    return mask_response(image_rgb, mask)


@app.post("/api/segment/grabcut")
async def segment_grabcut_endpoint(
    image: UploadFile = File(...),
    rect: str = Form(...),
    fg_mask: UploadFile | None = File(None),
    bg_mask: UploadFile | None = File(None),
    num_iters: int = Form(5),
    max_side: int = Form(1024),
    dilation_px: int = Form(0),
):
    if run_grabcut is None:
        raise HTTPException(status_code=501, detail=f"GrabCut integration is unavailable: {GRABCUT_IMPORT_ERROR}")

    image_bytes = await image.read()
    image_rgb = decode_image_bytes(image_bytes)
    height, width = image_rgb.shape[:2]
    parsed_rect = parse_rect(rect, width=width, height=height)

    fg_binary = None
    bg_binary = None
    if fg_mask is not None:
        fg_binary = decode_mask_bytes(await fg_mask.read(), target_size=(width, height))
    if bg_mask is not None:
        bg_binary = decode_mask_bytes(await bg_mask.read(), target_size=(width, height))

    try:
        mask = run_grabcut(
            image_rgb=image_rgb,
            rect=parsed_rect,
            fg_mask=fg_binary,
            bg_mask=bg_binary,
            num_iters=num_iters,
            max_side=max_side,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"GrabCut segmentation failed: {exc}") from exc

    mask = dilate_mask(mask, dilation_px=dilation_px)
    return mask_response(
        image_rgb,
        mask,
        extra={
            "rect": {
                "x": parsed_rect[0],
                "y": parsed_rect[1],
                "width": parsed_rect[2],
                "height": parsed_rect[3],
            }
        },
    )


@app.post("/api/segment/sam2")
async def segment_sam2_endpoint(
    image: UploadFile = File(...),
    x: float = Form(...),
    y: float = Form(...),
    dilation_px: int = Form(3),
    multimask_output: bool = Form(True),
):
    if get_sam2_click_segmenter is None:
        raise HTTPException(status_code=501, detail=f"SAM2 integration is unavailable: {SAM2_IMPORT_ERROR}")

    image_bytes = await image.read()
    image_rgb = decode_image_bytes(image_bytes)
    height, width = image_rgb.shape[:2]
    click_x = max(0, min(int(round(x)), width - 1))
    click_y = max(0, min(int(round(y)), height - 1))

    try:
        segmenter = get_sam2_click_segmenter(
            str(SAM2_CHECKPOINT_PATH),
            SAM2_MODEL_CFG,
            SAM2_DEVICE,
        )
        mask, score = segmenter.segment_from_point(
            image_rgb,
            click_x,
            click_y,
            multimask_output=multimask_output,
        )
    except SAM2SegmenterUnavailable as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"SAM2 segmentation failed: {exc}") from exc

    mask = dilate_mask(mask, dilation_px=dilation_px)
    return mask_response(image_rgb, mask, extra={"score": score, "point": {"x": click_x, "y": click_y}})


@app.post("/api/inpaint")
async def inpaint_endpoint(
    image: UploadFile = File(...),
    mask: UploadFile | None = File(None),
    points: str | None = Form(None),
    selection_mode: str = Form("path"),
    dilation_px: int = Form(0),
    response_format: str = Form("json"),
):
    image_bytes = await image.read()
    image_rgb = decode_image_bytes(image_bytes)
    height, width = image_rgb.shape[:2]

    if mask is not None:
        mask_bytes = await mask.read()
        binary_mask = decode_mask_bytes(mask_bytes, target_size=(width, height))
    else:
        parsed_points = parse_points(points)
        if not parsed_points:
            raise HTTPException(status_code=400, detail="Send either a binary mask file or points JSON.")

        mode = selection_mode.lower().strip()
        if mode in {"anchor", "anchors", "intelligent_scissors", "scissors"}:
            binary_mask = build_mask_from_anchors(image_bytes, image_rgb, parsed_points)
        elif mode in {"path", "polygon", "points", "contour"}:
            binary_mask = build_mask_from_path(image_bytes, image_rgb, parsed_points)
        else:
            raise HTTPException(status_code=400, detail="selection_mode must be 'path' or 'anchors'.")

    binary_mask = dilate_mask(binary_mask, dilation_px=dilation_px)

    try:
        result = inpaint_with_strategy(image_rgb=image_rgb, mask=binary_mask)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inpainting failed: {exc}") from exc

    json_outputs = [
        {
            "id": item["id"],
            "label": item["label"],
            "image": png_data_url(item["image"]),
        }
        for item in result["outputs"]
    ]
    crop_bbox = result["crop_bbox"]
    payload = {
        "strategy": result["strategy"],
        "width": width,
        "height": height,
        "mask": png_data_url(binary_mask),
        "mask_overlay": png_data_url(mask_overlay(image_rgb, binary_mask)),
        "crop_bbox": (
            {"x1": crop_bbox[0], "y1": crop_bbox[1], "x2": crop_bbox[2], "y2": crop_bbox[3]}
            if crop_bbox is not None
            else None
        ),
        "outputs": json_outputs,
    }

    if response_format.lower().strip() in {"base64", "json"}:
        return JSONResponse(payload)

    first_output = result["outputs"][0]
    return StreamingResponse(
        io.BytesIO(encode_png_bytes(first_output["image"])),
        media_type="image/png",
        headers={
            "X-Inpaint-Strategy": result["strategy"],
            "X-Inpaint-Outputs": str(len(result["outputs"])),
        },
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
