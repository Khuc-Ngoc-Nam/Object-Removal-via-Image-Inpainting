from __future__ import annotations

import math
import sys
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
import torch

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from model import LaMaGenerator


CHECKPOINT_PATH = APP_DIR / "results" / "checkpoints" / "best_lama_generator_only.pth"
DEVICE = torch.device("cpu")

SCISSORS_SESSIONS: dict[str, "ScissorsSession"] = {}


def to_rgb_uint8(image: Any) -> np.ndarray:
    if image is None:
        raise gr.Error("Upload an image first.")

    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    elif arr.ndim != 3 or arr.shape[2] != 3:
        raise gr.Error("Expected an RGB or RGBA image.")

    if arr.dtype == np.uint8:
        return arr.copy()

    arr = np.nan_to_num(arr.astype(np.float32))
    if arr.max() <= 1.0:
        arr *= 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def rgba_alpha_mask(layer: Any, target_shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(layer)
    if arr.ndim == 2:
        alpha = arr
    elif arr.ndim == 3 and arr.shape[2] == 4:
        alpha = arr[:, :, 3]
    elif arr.ndim == 3:
        alpha = (np.max(arr[:, :, :3], axis=2) > 8).astype(np.uint8) * 255
    else:
        return np.zeros(target_shape, dtype=np.uint8)

    if alpha.shape[:2] != target_shape:
        alpha = cv2.resize(alpha, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

    return ((alpha > 8).astype(np.uint8) * 255)


def extract_editor_image_and_mask(editor_value: Any) -> tuple[np.ndarray, np.ndarray]:
    if editor_value is None:
        raise gr.Error("Upload an image and paint the object to remove.")

    if not isinstance(editor_value, dict):
        image_rgb = to_rgb_uint8(editor_value)
        raise gr.Error("Paint a mask layer over the object before running inpainting.")

    background = editor_value.get("background")
    composite = editor_value.get("composite")
    layers = editor_value.get("layers") or []

    if background is None:
        background = composite
    image_rgb = to_rgb_uint8(background)
    height, width = image_rgb.shape[:2]

    mask = np.zeros((height, width), dtype=np.uint8)
    for layer in layers:
        mask = np.maximum(mask, rgba_alpha_mask(layer, (height, width)))

    if not np.any(mask) and composite is not None and background is not None:
        composite_rgb = to_rgb_uint8(composite)
        if composite_rgb.shape[:2] != (height, width):
            composite_rgb = cv2.resize(composite_rgb, (width, height), interpolation=cv2.INTER_AREA)
        diff = np.max(np.abs(composite_rgb.astype(np.int16) - image_rgb.astype(np.int16)), axis=2)
        mask = ((diff > 12).astype(np.uint8) * 255)

    if not np.any(mask):
        raise gr.Error("The mask is empty. Paint over the object first.")

    return image_rgb, mask


def dilate_mask(mask: np.ndarray, dilation_px: int) -> np.ndarray:
    mask = ((mask > 127).astype(np.uint8) * 255)
    if dilation_px <= 0:
        return mask
    kernel_size = int(dilation_px) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=1)


def mask_preview(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image_rgb.copy()
    red = np.zeros_like(overlay)
    red[:, :, 0] = 255
    alpha = ((mask > 0).astype(np.float32) * 0.45)[:, :, None]
    return np.clip(overlay * (1.0 - alpha) + red * alpha, 0, 255).astype(np.uint8)


def checkpoint_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint is not a state_dict dictionary.")

    for key in ("generator_state_dict", "state_dict", "model_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def strip_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = ("module.", "_orig_mod.", "generator.")
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        clean_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
                    changed = True
        cleaned[clean_key] = value
    return cleaned


def load_checkpoint_cpu(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


@lru_cache(maxsize=1)
def load_lama_model() -> torch.nn.Module:
    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    model = LaMaGenerator(input_channels=4, output_channels=3, num_ffc_blocks=9).to(DEVICE)
    checkpoint = load_checkpoint_cpu(CHECKPOINT_PATH)
    state_dict = strip_prefixes(checkpoint_state_dict(checkpoint))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def first_tensor_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (tuple, list)):
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


def expanded_bbox(mask: np.ndarray, margin_ratio: float = 0.5) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise gr.Error("The mask is empty.")

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


def ceil_to_multiple(value: int, multiple: int = 8, minimum: int = 16) -> int:
    return max(minimum, int(math.ceil(value / multiple) * multiple))


def resize_for_cpu(
    crop_rgb: np.ndarray,
    crop_mask: np.ndarray,
    max_side: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_side <= 0 or max(crop_rgb.shape[:2]) <= max_side:
        return crop_rgb, crop_mask

    height, width = crop_rgb.shape[:2]
    scale = max_side / max(height, width)
    new_w = ceil_to_multiple(int(round(width * scale)))
    new_h = ceil_to_multiple(int(round(height * scale)))

    resized_rgb = cv2.resize(crop_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(crop_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return resized_rgb, resized_mask


def pad_to_multiple_of_8(
    image_rgb: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    height, width = image_rgb.shape[:2]
    target_h = ceil_to_multiple(height)
    target_w = ceil_to_multiple(width)
    pad_h = target_h - height
    pad_w = target_w - width

    if pad_h == 0 and pad_w == 0:
        return image_rgb, mask, (height, width)

    padded_rgb = cv2.copyMakeBorder(
        image_rgb,
        0,
        pad_h,
        0,
        pad_w,
        borderType=cv2.BORDER_REFLECT_101,
    )
    padded_mask = cv2.copyMakeBorder(
        mask,
        0,
        pad_h,
        0,
        pad_w,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )
    return padded_rgb, padded_mask, (height, width)


@torch.inference_mode()
def inpaint_image(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    context_margin: float,
    max_crop_side: int,
) -> np.ndarray:
    image_rgb = to_rgb_uint8(image_rgb)
    mask = ((mask > 127).astype(np.uint8) * 255)

    x1, y1, x2, y2 = expanded_bbox(mask, margin_ratio=context_margin)
    original_crop = image_rgb[y1:y2, x1:x2].copy()
    original_mask = mask[y1:y2, x1:x2].copy()
    crop_h, crop_w = original_crop.shape[:2]

    infer_crop, infer_mask = resize_for_cpu(original_crop, original_mask, max_crop_side)
    infer_crop, infer_mask, unpadded_size = pad_to_multiple_of_8(infer_crop, infer_mask)

    mask_01 = (infer_mask > 127).astype(np.float32)
    image_01 = infer_crop.astype(np.float32) / 255.0
    image_norm = image_01 * 2.0 - 1.0
    masked_norm = image_norm * (1.0 - mask_01[:, :, None])

    input_np = np.concatenate(
        [masked_norm.transpose(2, 0, 1), mask_01[None, :, :]],
        axis=0,
    )
    input_tensor = torch.from_numpy(input_np).unsqueeze(0).to(DEVICE, dtype=torch.float32)

    model = load_lama_model()
    generated_batch = first_tensor_output(model(input_tensor))
    generated = generated_batch[0].detach().cpu()
    generated_01 = (generated * 0.5 + 0.5).clamp(0.0, 1.0)
    generated_rgb = (generated_01.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)

    unpadded_h, unpadded_w = unpadded_size
    generated_rgb = generated_rgb[:unpadded_h, :unpadded_w]

    if generated_rgb.shape[:2] != (crop_h, crop_w):
        generated_rgb = cv2.resize(generated_rgb, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

    mask_float = (original_mask > 127).astype(np.float32)[:, :, None]
    repaired_crop = (
        original_crop.astype(np.float32) * (1.0 - mask_float)
        + generated_rgb.astype(np.float32) * mask_float
    )
    repaired_crop = np.clip(repaired_crop, 0, 255).astype(np.uint8)

    result = image_rgb.copy()
    result[y1:y2, x1:x2] = repaired_crop
    return result


def create_scissors() -> Any:
    if hasattr(cv2, "segmentation_IntelligentScissorsMB"):
        scissors = cv2.segmentation_IntelligentScissorsMB()
    elif hasattr(cv2, "segmentation") and hasattr(cv2.segmentation, "IntelligentScissorsMB_create"):
        scissors = cv2.segmentation.IntelligentScissorsMB_create()
    else:
        raise gr.Error("OpenCV IntelligentScissorsMB is unavailable. Install opencv-contrib-python.")

    scissors.setEdgeFeatureCannyParameters(threshold1=50, threshold2=100)
    scissors.setWeights(0.3, 0.3, 0.4)
    return scissors


def clamp_point(x: int, y: int, width: int, height: int) -> tuple[int, int]:
    return max(0, min(int(x), width - 1)), max(0, min(int(y), height - 1))


def line_contour(start: tuple[int, int], end: tuple[int, int]) -> np.ndarray:
    dist = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
    if dist <= 0:
        return np.array([[end]], dtype=np.int32).reshape(-1, 1, 2)

    xs = np.linspace(start[0], end[0], dist + 1, dtype=np.int32)
    ys = np.linspace(start[1], end[1], dist + 1, dtype=np.int32)
    return np.column_stack((xs, ys)).reshape(-1, 1, 2)


@dataclass
class ScissorsSession:
    image_rgb: np.ndarray
    target_max_side: int = 768
    scissors: Any = field(init=False)
    proc_image_bgr: np.ndarray = field(init=False)
    height: int = field(init=False)
    width: int = field(init=False)
    proc_h: int = field(init=False)
    proc_w: int = field(init=False)
    scale_x: float = field(init=False)
    scale_y: float = field(init=False)
    anchors: list[tuple[int, int]] = field(default_factory=list)
    segments: list[list[tuple[int, int]]] = field(default_factory=list)
    mask: np.ndarray | None = None
    closed: bool = False

    def __post_init__(self) -> None:
        self.image_rgb = to_rgb_uint8(self.image_rgb)
        self.height, self.width = self.image_rgb.shape[:2]
        scale = min(1.0, self.target_max_side / max(self.height, self.width))
        self.proc_w = max(1, int(round(self.width * scale)))
        self.proc_h = max(1, int(round(self.height * scale)))
        self.scale_x = self.proc_w / self.width
        self.scale_y = self.proc_h / self.height
        self.proc_image_bgr = cv2.resize(
            cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2BGR),
            (self.proc_w, self.proc_h),
            interpolation=cv2.INTER_AREA,
        )
        self.reset_scissors()

    def reset_scissors(self) -> None:
        self.scissors = create_scissors()
        self.scissors.applyImage(self.proc_image_bgr)

    def reset_selection(self) -> None:
        self.anchors.clear()
        self.segments.clear()
        self.mask = None
        self.closed = False
        self.reset_scissors()

    def replay(self, anchors: list[tuple[int, int]]) -> None:
        self.reset_selection()
        for point in anchors:
            self.add_anchor(*point)

    def undo(self) -> None:
        if self.anchors:
            self.replay(self.anchors[:-1])

    def to_proc(self, point: tuple[int, int]) -> tuple[int, int]:
        x = int(round(point[0] * self.scale_x))
        y = int(round(point[1] * self.scale_y))
        return clamp_point(x, y, self.proc_w, self.proc_h)

    def to_original_points(self, contour: np.ndarray) -> list[tuple[int, int]]:
        points = []
        for x, y in contour.reshape(-1, 2):
            points.append(clamp_point(int(round(x / self.scale_x)), int(round(y / self.scale_y)), self.width, self.height))
        return points

    def build_map(self, point: tuple[int, int]) -> None:
        self.scissors.buildMap(self.to_proc(point))

    def contour_between(self, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
        proc_start = self.to_proc(start)
        proc_end = self.to_proc(end)
        same_boundary = (
            (proc_start[0] == 0 and proc_end[0] == 0)
            or (proc_start[0] == self.proc_w - 1 and proc_end[0] == self.proc_w - 1)
            or (proc_start[1] == 0 and proc_end[1] == 0)
            or (proc_start[1] == self.proc_h - 1 and proc_end[1] == self.proc_h - 1)
        )

        if same_boundary:
            contour = line_contour(proc_start, proc_end)
        else:
            contour = self.scissors.getContour(proc_end)
            if contour is None or len(contour) == 0:
                contour = line_contour(proc_start, proc_end)

        return self.to_original_points(contour)

    def add_anchor(self, x: int, y: int) -> None:
        point = clamp_point(x, y, self.width, self.height)
        if self.closed:
            self.reset_selection()

        if not self.anchors:
            self.anchors.append(point)
            self.build_map(point)
            return

        previous = self.anchors[-1]
        segment = self.contour_between(previous, point)
        if segment:
            self.segments.append(segment)
        self.anchors.append(point)
        self.build_map(point)
        self.mask = None

    def close(self, dilation_px: int = 3) -> np.ndarray:
        if len(self.anchors) < 3:
            raise gr.Error("Add at least 3 anchors before closing the mask.")

        if not self.closed:
            segment = self.contour_between(self.anchors[-1], self.anchors[0])
            if segment:
                self.segments.append(segment)
            self.closed = True

        points = [point for segment in self.segments for point in segment]
        if len(points) < 3:
            points = self.anchors

        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        contour = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [contour], 255)
        self.mask = dilate_mask(mask, dilation_px)
        return self.mask

    def overlay(self) -> np.ndarray:
        overlay = self.image_rgb.copy()
        for segment in self.segments:
            if len(segment) >= 2:
                pts = np.array(segment, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(overlay, [pts], False, (0, 255, 0), 2)
        for idx, point in enumerate(self.anchors):
            color = (0, 128, 255) if idx == 0 else (255, 0, 0)
            cv2.circle(overlay, point, 5, color, -1)
        if self.anchors and not self.closed:
            cv2.circle(overlay, self.anchors[0], 12, (0, 128, 255), 2)
        if self.mask is not None:
            overlay = mask_preview(overlay, self.mask)
        return overlay


def get_session(session_id: str | None) -> ScissorsSession:
    if not session_id or session_id not in SCISSORS_SESSIONS:
        raise gr.Error("Upload an image in the Intelligent Scissors tab first.")
    return SCISSORS_SESSIONS[session_id]


def scissors_upload(image: Any):
    image_rgb = to_rgb_uint8(image)
    session_id = str(uuid.uuid4())
    session = ScissorsSession(image_rgb)
    SCISSORS_SESSIONS[session_id] = session
    return session_id, session.overlay(), None, "Click anchors around the object boundary."


def scissors_add_anchor(session_id: str | None, evt: gr.SelectData):
    session = get_session(session_id)
    if evt.index is None:
        raise gr.Error("Click on the image to add an anchor.")

    x, y = evt.index[:2]
    session.add_anchor(int(round(x)), int(round(y)))
    message = f"{len(session.anchors)} anchor(s) added."
    if len(session.anchors) >= 3:
        message += " Press Close Mask when ready."
    return session.overlay(), None, message


def scissors_undo(session_id: str | None):
    session = get_session(session_id)
    session.undo()
    return session.overlay(), None, f"Undo complete. {len(session.anchors)} anchor(s) remain."


def scissors_reset(session_id: str | None):
    session = get_session(session_id)
    session.reset_selection()
    return session.overlay(), None, "Selection reset."


def scissors_close(session_id: str | None, dilation_px: int):
    session = get_session(session_id)
    mask = session.close(dilation_px)
    return session.overlay(), mask_preview(session.image_rgb, mask), "Mask closed. Run inpainting now."


def run_from_editor(editor_value: Any, dilation_px: int, context_margin: float, max_crop_side: int):
    image_rgb, mask = extract_editor_image_and_mask(editor_value)
    mask = dilate_mask(mask, dilation_px)
    result = inpaint_image(image_rgb, mask, context_margin, max_crop_side)
    return result, mask_preview(image_rgb, mask), "Done."


def run_from_scissors(session_id: str | None, dilation_px: int, context_margin: float, max_crop_side: int):
    session = get_session(session_id)
    if session.mask is None:
        session.close(dilation_px)
    result = inpaint_image(session.image_rgb, session.mask, context_margin, max_crop_side)
    return result, mask_preview(session.image_rgb, session.mask), "Done."


CSS = """
.app-shell {max-width: 1280px; margin: 0 auto;}
.status-box {font-size: 0.95rem;}
"""


with gr.Blocks(title="Object Removal via LaMa") as demo:
    gr.Markdown(
        """
        # Object Removal via Image Inpainting
        Paint a mask directly, or use Intelligent Scissors anchors, then run CPU LaMa on an expanded crop.
        """,
        elem_classes=["app-shell"],
    )

    with gr.Row(elem_classes=["app-shell"]):
        with gr.Column(scale=7):
            with gr.Tabs():
                with gr.Tab("Brush mask"):
                    editor = gr.ImageEditor(
                        label="Upload image and paint the object mask",
                        type="numpy",
                        image_mode="RGBA",
                        sources=["upload", "clipboard"],
                        brush=gr.Brush(default_size=28, colors=["#ffffff"], default_color="#ffffff", color_mode="fixed"),
                        eraser=gr.Eraser(default_size=28),
                        layers=gr.LayerOptions(allow_additional_layers=False, layers=["Mask"]),
                        transforms=["crop", "resize"],
                        canvas_size=(900, 650),
                        height=620,
                    )
                    run_editor_btn = gr.Button("Run Inpainting From Brush Mask", variant="primary")

                with gr.Tab("Intelligent Scissors"):
                    scissors_state = gr.State(None)
                    scissors_input = gr.Image(
                        label="Upload image for Intelligent Scissors",
                        type="numpy",
                        image_mode="RGB",
                        sources=["upload", "clipboard"],
                        height=240,
                    )
                    scissors_canvas = gr.Image(
                        label="Click boundary anchors here",
                        type="numpy",
                        image_mode="RGB",
                        interactive=True,
                        height=460,
                    )
                    with gr.Row():
                        undo_btn = gr.Button("Undo Anchor")
                        close_btn = gr.Button("Close Mask", variant="primary")
                        reset_btn = gr.Button("Reset")
                    run_scissors_btn = gr.Button("Run Inpainting From Scissors Mask", variant="primary")

        with gr.Column(scale=5):
            result_output = gr.Image(label="Output image", type="numpy", image_mode="RGB", height=500)
            mask_output = gr.Image(label="Mask preview", type="numpy", image_mode="RGB", height=300)
            status = gr.Markdown("Ready.", elem_classes=["status-box"])

            with gr.Accordion("Inference settings", open=True):
                dilation = gr.Slider(
                    label="Mask dilation",
                    minimum=0,
                    maximum=30,
                    value=3,
                    step=1,
                )
                context_margin = gr.Slider(
                    label="Context margin around mask bbox",
                    minimum=0.0,
                    maximum=1.5,
                    value=0.5,
                    step=0.05,
                )
                max_crop_side = gr.Slider(
                    label="Max crop side for CPU inference",
                    minimum=256,
                    maximum=1536,
                    value=768,
                    step=64,
                )

    run_editor_btn.click(
        run_from_editor,
        inputs=[editor, dilation, context_margin, max_crop_side],
        outputs=[result_output, mask_output, status],
    )

    scissors_input.change(
        scissors_upload,
        inputs=scissors_input,
        outputs=[scissors_state, scissors_canvas, mask_output, status],
    )
    scissors_canvas.select(
        scissors_add_anchor,
        inputs=scissors_state,
        outputs=[scissors_canvas, mask_output, status],
        trigger_mode="multiple",
    )
    undo_btn.click(
        scissors_undo,
        inputs=scissors_state,
        outputs=[scissors_canvas, mask_output, status],
    )
    reset_btn.click(
        scissors_reset,
        inputs=scissors_state,
        outputs=[scissors_canvas, mask_output, status],
    )
    close_btn.click(
        scissors_close,
        inputs=[scissors_state, dilation],
        outputs=[scissors_canvas, mask_output, status],
    )
    run_scissors_btn.click(
        run_from_scissors,
        inputs=[scissors_state, dilation, context_margin, max_crop_side],
        outputs=[result_output, mask_output, status],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        theme=gr.themes.Soft(),
        css=CSS,
    )
