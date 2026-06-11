from __future__ import annotations

import base64
import io
import json
import math
import os
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect

try:
    from model import LaMaGenerator
except Exception as exc:
    LaMaGenerator = None
    MODEL_IMPORT_ERROR = exc
else:
    MODEL_IMPORT_ERROR = None

try:
    from lasso_core import LassoCore
except Exception as exc:
    LassoCore = None
    LASSO_IMPORT_ERROR = exc
else:
    LASSO_IMPORT_ERROR = None


ROOT_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT_DIR / "results" / "checkpoints" / "best_lama_generator_only.pth"
DEVICE = torch.device("cpu")

app = FastAPI(title="Object Removal via Image Inpainting API")

cors_origins = os.getenv(
    "CORS_ALLOW_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in cors_origins.split(",") if origin.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


sessions: dict[str, Any] = {}


class InitSessionResponse(BaseModel):
    session_id: str
    width: int
    height: int


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


def encode_png_bytes(image_rgb: np.ndarray) -> bytes:
    image_bgr = cv2.cvtColor(np.clip(image_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".png", image_bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode result image.")
    return buffer.tobytes()


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


def clamp_points(points: Iterable[dict[str, int]], width: int, height: int) -> list[dict[str, int]]:
    clamped = []
    for point in points:
        clamped.append(
            {
                "x": max(0, min(int(point["x"]), width - 1)),
                "y": max(0, min(int(point["y"]), height - 1)),
            }
        )
    return clamped


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
        raise HTTPException(
            status_code=500,
            detail=f"Module 1 mask generation failed: {exc}",
        ) from exc


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
        raise HTTPException(
            status_code=500,
            detail=f"Module 1 Intelligent Scissors failed: {exc}",
        ) from exc


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

    try:
        return LaMaGenerator(input_channels=4, output_channels=3, num_ffc_blocks=9)
    except TypeError:
        return LaMaGenerator()


@lru_cache(maxsize=1)
def get_lama_model() -> torch.nn.Module:
    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    model = instantiate_lama().to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device("cpu"))
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


def mask_bbox(mask: np.ndarray, margin_ratio: float = 0.5) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise HTTPException(status_code=400, detail="The generated mask is empty.")

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1

    box_w = x2 - x1
    box_h = y2 - y1
    margin_x = int(round(box_w * margin_ratio))
    margin_y = int(round(box_h * margin_ratio))

    height, width = mask.shape[:2]
    return (
        max(0, x1 - margin_x),
        max(0, y1 - margin_y),
        min(width, x2 + margin_x),
        min(height, y2 + margin_y),
    )


def resize_for_cpu(
    crop_rgb: np.ndarray,
    crop_mask: np.ndarray,
    max_crop_side: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_crop_side <= 0:
        return crop_rgb, crop_mask

    height, width = crop_rgb.shape[:2]
    if max(height, width) <= max_crop_side:
        return crop_rgb, crop_mask

    scale = max_crop_side / max(height, width)
    new_w = max(8, int(round(width * scale)))
    new_h = max(8, int(round(height * scale)))

    resized_rgb = cv2.resize(crop_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(crop_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
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

    image_padded = cv2.copyMakeBorder(
        image_rgb,
        0,
        pad_h,
        0,
        pad_w,
        borderType=cv2.BORDER_REFLECT_101,
    )
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
def run_lama_on_crop(
    crop_rgb: np.ndarray,
    crop_mask: np.ndarray,
    max_crop_side: int,
) -> np.ndarray:
    original_h, original_w = crop_rgb.shape[:2]
    infer_rgb, infer_mask = resize_for_cpu(crop_rgb, crop_mask, max_crop_side)
    infer_rgb, infer_mask, unpadded_size = pad_to_multiple_of_8(infer_rgb, infer_mask)

    mask_01 = (infer_mask > 127).astype(np.float32)
    image_01 = infer_rgb.astype(np.float32) / 255.0
    image_norm = image_01 * 2.0 - 1.0
    masked_norm = image_norm * (1.0 - mask_01[..., None])

    input_np = np.concatenate(
        [masked_norm.transpose(2, 0, 1), mask_01[None, :, :]],
        axis=0,
    )
    input_tensor = torch.from_numpy(input_np).unsqueeze(0).to(DEVICE, dtype=torch.float32)

    model = get_lama_model()
    generated_batch = first_tensor_output(model(input_tensor))
    generated = generated_batch[0].detach().cpu()
    generated_01 = (generated * 0.5 + 0.5).clamp(0.0, 1.0)
    generated_rgb = (
        generated_01.permute(1, 2, 0).numpy() * 255.0
    ).round().astype(np.uint8)

    unpadded_h, unpadded_w = unpadded_size
    generated_rgb = generated_rgb[:unpadded_h, :unpadded_w]

    if (unpadded_h, unpadded_w) != (original_h, original_w):
        generated_rgb = cv2.resize(
            generated_rgb,
            (original_w, original_h),
            interpolation=cv2.INTER_LINEAR,
        )

    repaired_crop = crop_rgb.copy()
    mask_bool = crop_mask > 127
    repaired_crop[mask_bool] = generated_rgb[mask_bool]
    return repaired_crop


def inpaint_full_image(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    margin_ratio: float = 0.5,
    max_crop_side: int = 1024,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    binary_mask = ((mask > 127).astype(np.uint8) * 255)
    x1, y1, x2, y2 = mask_bbox(binary_mask, margin_ratio=margin_ratio)

    crop_rgb = image_rgb[y1:y2, x1:x2].copy()
    crop_mask = binary_mask[y1:y2, x1:x2].copy()
    repaired_crop = run_lama_on_crop(crop_rgb, crop_mask, max_crop_side=max_crop_side)

    result = image_rgb.copy()
    result[y1:y2, x1:x2] = repaired_crop
    return result, (x1, y1, x2, y2)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "checkpoint_exists": CHECKPOINT_PATH.exists(),
        "model_import_ok": LaMaGenerator is not None,
        "lasso_import_ok": LassoCore is not None,
    }


@app.post("/api/inpaint")
async def inpaint_endpoint(
    image: UploadFile = File(...),
    mask: UploadFile | None = File(None),
    points: str | None = Form(None),
    selection_mode: str = Form("path"),
    margin_ratio: float = Form(0.5),
    max_crop_side: int = Form(1024),
    response_format: str = Form("image"),
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
            raise HTTPException(
                status_code=400,
                detail="Send either a binary mask file or points JSON.",
            )

        mode = selection_mode.lower().strip()
        if mode in {"anchor", "anchors", "intelligent_scissors"}:
            binary_mask = build_mask_from_anchors(image_bytes, image_rgb, parsed_points)
        elif mode in {"path", "polygon", "points", "contour"}:
            binary_mask = build_mask_from_path(image_bytes, image_rgb, parsed_points)
        else:
            raise HTTPException(
                status_code=400,
                detail="selection_mode must be 'path' or 'anchors'.",
            )

    if margin_ratio < 0:
        raise HTTPException(status_code=400, detail="margin_ratio must be >= 0.")

    try:
        result_rgb, bbox = inpaint_full_image(
            image_rgb=image_rgb,
            mask=binary_mask,
            margin_ratio=margin_ratio,
            max_crop_side=max_crop_side,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inpainting failed: {exc}") from exc

    png_bytes = encode_png_bytes(result_rgb)

    if response_format.lower().strip() in {"base64", "json"}:
        image_b64 = base64.b64encode(png_bytes).decode("utf-8")
        return JSONResponse(
            {
                "image": f"data:image/png;base64,{image_b64}",
                "bbox": {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]},
            }
        )

    return StreamingResponse(
        io.BytesIO(png_bytes),
        media_type="image/png",
        headers={
            "X-Inpaint-BBox": json.dumps(
                {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]}
            )
        },
    )


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    if LassoCore is None:
        raise HTTPException(
            status_code=500,
            detail=f"Could not import LassoCore: {LASSO_IMPORT_ERROR}",
        )

    contents = await file.read()
    session_id = str(uuid.uuid4())

    core = LassoCore()
    try:
        width, height = core.apply_image(contents)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    sessions[session_id] = core
    return InitSessionResponse(session_id=session_id, width=width, height=height)


@app.websocket("/ws/lasso/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()

    if session_id not in sessions:
        await websocket.close(code=1008)
        return

    core = sessions[session_id]

    try:
        while True:
            data = json.loads(await websocket.receive_text())
            action = data.get("type")

            if action == "anchor":
                point = [int(data["x"]), int(data["y"])]
                core.build_map(point)
                await websocket.send_json({"type": "map_built", "x": point[0], "y": point[1]})

            elif action == "move":
                start_pt = [int(data["start_x"]), int(data["start_y"])]
                end_pt = [int(data["x"]), int(data["y"])]
                contour_points = core.get_smart_contour(start_pt, end_pt)
                await websocket.send_json({"type": "contour", "points": contour_points})

            elif action == "close":
                raw_points = data.get("points", [])
                points = [normalize_point(point) for point in raw_points]
                if len(points) > 2:
                    binary_mask = core.generate_mask(points)
                    ok, buffer = cv2.imencode(".png", binary_mask)
                    if not ok:
                        await websocket.send_json({"type": "error", "message": "Mask encoding failed."})
                        continue
                    image_b64 = base64.b64encode(buffer).decode("utf-8")
                    await websocket.send_json(
                        {
                            "type": "mask_generated",
                            "image": f"data:image/png;base64,{image_b64}",
                        }
                    )

    except WebSocketDisconnect:
        sessions.pop(session_id, None)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
