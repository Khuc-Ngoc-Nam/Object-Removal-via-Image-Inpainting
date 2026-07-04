"use client";

/* eslint-disable @next/next/no-img-element */

import React, { useCallback, useEffect, useRef, useState } from "react";

type Mode = "brush" | "scissors" | "grabcut" | "sam2";
type BrushAction = "paint" | "erase";
type GrabcutTool = "rect" | "fg" | "bg";
type Point = { x: number; y: number };
type Dimensions = { width: number; height: number };
type Rect = { x: number; y: number; width: number; height: number };
type OutputImage = { id: string; label: string; image: string };
type PreviewImage = { src: string; label: string; downloadName?: string };
type SegmentResponse = {
  mask: string;
  overlay: string;
  width: number;
  height: number;
  score?: number;
};
type InpaintResponse = {
  strategy: string;
  width: number;
  height: number;
  mask: string;
  mask_overlay: string;
  outputs: OutputImage[];
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const MODES: Array<{ id: Mode; label: string }> = [
  { id: "brush", label: "Brush mask" },
  { id: "scissors", label: "IntelligentScissors" },
  { id: "grabcut", label: "Grabcut" },
  { id: "sam2", label: "SAM2" },
];
const SNAP_MARGIN = 15;

function isInpaintResponse(payload: unknown): payload is InpaintResponse {
  return (
    typeof payload === "object" &&
    payload !== null &&
    "outputs" in payload &&
    Array.isArray((payload as InpaintResponse).outputs)
  );
}

function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return "Unexpected error.";
}

function downloadDataUrl(src: string, filename: string) {
  const link = document.createElement("a");
  link.href = src;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
}

export default function ObjectRemovalStudio() {
  const [mode, setMode] = useState<Mode>("brush");
  const [brushAction, setBrushAction] = useState<BrushAction>("paint");
  const [grabcutTool, setGrabcutTool] = useState<GrabcutTool>("rect");
  const [grabcutRect, setGrabcutRect] = useState<Rect | null>(null);
  const [brushSize, setBrushSize] = useState(28);
  const [file, setFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageObj, setImageObj] = useState<HTMLImageElement | null>(null);
  const [dimensions, setDimensions] = useState<Dimensions | null>(null);
  const [maskPreviewUrl, setMaskPreviewUrl] = useState<string | null>(null);
  const [outputs, setOutputs] = useState<OutputImage[]>([]);
  const [anchors, setAnchors] = useState<Point[]>([]);
  const [samPoint, setSamPoint] = useState<Point | null>(null);
  const [isPainting, setIsPainting] = useState(false);
  const [isBusy, setIsBusy] = useState(false);
  const [status, setStatus] = useState("Ready.");
  const [previewImage, setPreviewImage] = useState<PreviewImage | null>(null);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement>(null);
  const grabcutFgCanvasRef = useRef<HTMLCanvasElement>(null);
  const grabcutBgCanvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const lastBrushPointRef = useRef<Point | null>(null);
  const grabcutDragStartRef = useRef<Point | null>(null);

  const createBinaryPreviewUrl = useCallback((sourceCanvas: HTMLCanvasElement | null) => {
    if (!sourceCanvas) {
      return null;
    }

    const maskCtx = sourceCanvas.getContext("2d", { willReadFrequently: true });
    if (!maskCtx) {
      return null;
    }

    const preview = document.createElement("canvas");
    preview.width = sourceCanvas.width;
    preview.height = sourceCanvas.height;
    const previewCtx = preview.getContext("2d");
    if (!previewCtx) {
      return null;
    }

    const src = maskCtx.getImageData(0, 0, sourceCanvas.width, sourceCanvas.height);
    const dst = previewCtx.createImageData(sourceCanvas.width, sourceCanvas.height);
    for (let i = 0; i < src.data.length; i += 4) {
      const alpha = src.data[i + 3];
      const value = alpha > 0 ? 255 : 0;
      dst.data[i] = value;
      dst.data[i + 1] = value;
      dst.data[i + 2] = value;
      dst.data[i + 3] = 255;
    }

    previewCtx.putImageData(dst, 0, 0);
    return preview.toDataURL("image/png");
  }, []);

  const createMaskPreviewUrl = useCallback(() => createBinaryPreviewUrl(maskCanvasRef.current), [createBinaryPreviewUrl]);

  const canvasHasPaint = useCallback((sourceCanvas: HTMLCanvasElement | null) => {
    if (!sourceCanvas) {
      return false;
    }

    const ctx = sourceCanvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) {
      return false;
    }

    const pixels = ctx.getImageData(0, 0, sourceCanvas.width, sourceCanvas.height).data;
    for (let i = 3; i < pixels.length; i += 4) {
      if (pixels[i] > 0) {
        return true;
      }
    }
    return false;
  }, []);

  const clearCanvas = useCallback((sourceCanvas: HTMLCanvasElement | null) => {
    if (!sourceCanvas) {
      return;
    }
    const ctx = sourceCanvas.getContext("2d");
    ctx?.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
  }, []);

  const normalizeRect = useCallback((start: Point, end: Point): Rect => {
    return {
      x: Math.min(start.x, end.x),
      y: Math.min(start.y, end.y),
      width: Math.max(1, Math.abs(end.x - start.x)),
      height: Math.max(1, Math.abs(end.y - start.y)),
    };
  }, []);

  const snapToEdge = useCallback(
    (point: Point): Point => {
      if (!dimensions) {
        return point;
      }

      let x = point.x;
      let y = point.y;
      if (x <= SNAP_MARGIN) {
        x = 0;
      } else if (x >= dimensions.width - 1 - SNAP_MARGIN) {
        x = dimensions.width - 1;
      }

      if (y <= SNAP_MARGIN) {
        y = 0;
      } else if (y >= dimensions.height - 1 - SNAP_MARGIN) {
        y = dimensions.height - 1;
      }

      return { x, y };
    },
    [dimensions],
  );

  const redrawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageObj || !dimensions) {
      return;
    }

    if (canvas.width !== dimensions.width || canvas.height !== dimensions.height) {
      canvas.width = dimensions.width;
      canvas.height = dimensions.height;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }

    ctx.clearRect(0, 0, dimensions.width, dimensions.height);
    ctx.drawImage(imageObj, 0, 0, dimensions.width, dimensions.height);

    const maskCanvas = maskCanvasRef.current;
    if (maskCanvas && maskCanvas.width > 0 && maskCanvas.height > 0) {
      const overlay = document.createElement("canvas");
      overlay.width = dimensions.width;
      overlay.height = dimensions.height;
      const overlayCtx = overlay.getContext("2d");
      if (overlayCtx) {
        overlayCtx.fillStyle = "rgb(255, 0, 0)";
        overlayCtx.fillRect(0, 0, dimensions.width, dimensions.height);
        overlayCtx.globalCompositeOperation = "destination-in";
        overlayCtx.drawImage(maskCanvas, 0, 0, dimensions.width, dimensions.height);
        ctx.save();
        ctx.globalAlpha = 0.45;
        ctx.drawImage(overlay, 0, 0);
        ctx.restore();
      }
    }

    if (mode === "scissors" && anchors.length > 0) {
      ctx.save();
      ctx.strokeStyle = "#00a651";
      ctx.lineWidth = Math.max(2, Math.round(Math.max(dimensions.width, dimensions.height) / 500));
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(anchors[0].x, anchors[0].y);
      anchors.slice(1).forEach((point) => ctx.lineTo(point.x, point.y));
      ctx.stroke();

      anchors.forEach((point, index) => {
        ctx.beginPath();
        ctx.fillStyle = index === 0 ? "#ff8a00" : "#006dff";
        ctx.arc(point.x, point.y, 5, 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.restore();
    }

    if (mode === "grabcut") {
      const drawPaintOverlay = (sourceCanvas: HTMLCanvasElement | null, color: string, alpha: number) => {
        if (!sourceCanvas || sourceCanvas.width === 0 || sourceCanvas.height === 0) {
          return;
        }
        const overlay = document.createElement("canvas");
        overlay.width = dimensions.width;
        overlay.height = dimensions.height;
        const overlayCtx = overlay.getContext("2d");
        if (!overlayCtx) {
          return;
        }
        overlayCtx.fillStyle = color;
        overlayCtx.fillRect(0, 0, dimensions.width, dimensions.height);
        overlayCtx.globalCompositeOperation = "destination-in";
        overlayCtx.drawImage(sourceCanvas, 0, 0, dimensions.width, dimensions.height);
        ctx.save();
        ctx.globalAlpha = alpha;
        ctx.drawImage(overlay, 0, 0);
        ctx.restore();
      };

      if (grabcutRect) {
        ctx.save();
        ctx.strokeStyle = "#006dff";
        ctx.lineWidth = Math.max(2, Math.round(Math.max(dimensions.width, dimensions.height) / 420));
        ctx.setLineDash([8, 6]);
        ctx.strokeRect(grabcutRect.x, grabcutRect.y, grabcutRect.width, grabcutRect.height);
        ctx.restore();
      }

      drawPaintOverlay(grabcutFgCanvasRef.current, "#00a651", 0.55);
      drawPaintOverlay(grabcutBgCanvasRef.current, "#0b6bcb", 0.48);
    }

    if (mode === "sam2" && samPoint) {
      ctx.save();
      ctx.strokeStyle = "#0a0a0a";
      ctx.fillStyle = "#ffffff";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(samPoint.x, samPoint.y, 8, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }
  }, [anchors, dimensions, grabcutRect, imageObj, mode, samPoint]);

  const syncMaskPreview = useCallback(() => {
    const nextPreviewUrl = createMaskPreviewUrl();
    setMaskPreviewUrl(nextPreviewUrl);
    redrawCanvas();
  }, [createMaskPreviewUrl, redrawCanvas]);

  const clearMaskCanvas = useCallback(() => {
    const maskCanvas = maskCanvasRef.current;
    if (!maskCanvas) {
      return;
    }
    clearCanvas(maskCanvas);
    setMaskPreviewUrl(null);
    redrawCanvas();
  }, [clearCanvas, redrawCanvas]);

  const resetMaskState = useCallback(() => {
    setAnchors([]);
    setSamPoint(null);
    setGrabcutRect(null);
    setGrabcutTool("rect");
    setOutputs([]);
    clearCanvas(grabcutFgCanvasRef.current);
    clearCanvas(grabcutBgCanvasRef.current);
    clearMaskCanvas();
  }, [clearCanvas, clearMaskCanvas]);

  useEffect(() => {
    redrawCanvas();
  }, [redrawCanvas]);

  useEffect(() => {
    return () => {
      if (imageUrl) {
        URL.revokeObjectURL(imageUrl);
      }
    };
  }, [imageUrl]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setPreviewImage(null);
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  useEffect(() => {
    if (!dimensions) {
      return;
    }

    const canvas = canvasRef.current;
    const maskCanvas = maskCanvasRef.current;
    if (canvas) {
      canvas.width = dimensions.width;
      canvas.height = dimensions.height;
    }
    if (maskCanvas) {
      maskCanvas.width = dimensions.width;
      maskCanvas.height = dimensions.height;
      const ctx = maskCanvas.getContext("2d");
      ctx?.clearRect(0, 0, dimensions.width, dimensions.height);
    }
    for (const extraCanvas of [grabcutFgCanvasRef.current, grabcutBgCanvasRef.current]) {
      if (extraCanvas) {
        extraCanvas.width = dimensions.width;
        extraCanvas.height = dimensions.height;
        clearCanvas(extraCanvas);
      }
    }

    redrawCanvas();
  }, [clearCanvas, dimensions, redrawCanvas]);

  const getCanvasPoint = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement>): Point | null => {
      const canvas = canvasRef.current;
      if (!canvas || !dimensions) {
        return null;
      }

      const rect = canvas.getBoundingClientRect();
      const scaleX = dimensions.width / rect.width;
      const scaleY = dimensions.height / rect.height;
      const x = Math.round((event.clientX - rect.left) * scaleX);
      const y = Math.round((event.clientY - rect.top) * scaleY);

      return {
        x: Math.max(0, Math.min(x, dimensions.width - 1)),
        y: Math.max(0, Math.min(y, dimensions.height - 1)),
      };
    },
    [dimensions],
  );

  const drawBrush = useCallback(
    (point: Point, previous: Point | null) => {
      const maskCanvas = maskCanvasRef.current;
      if (!maskCanvas) {
        return;
      }

      const ctx = maskCanvas.getContext("2d");
      if (!ctx) {
        return;
      }

      ctx.save();
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.lineWidth = brushSize;
      ctx.globalCompositeOperation = brushAction === "erase" ? "destination-out" : "source-over";
      ctx.strokeStyle = "rgba(255, 255, 255, 1)";
      ctx.fillStyle = "rgba(255, 255, 255, 1)";

      if (previous) {
        ctx.beginPath();
        ctx.moveTo(previous.x, previous.y);
        ctx.lineTo(point.x, point.y);
        ctx.stroke();
      } else {
        ctx.beginPath();
        ctx.arc(point.x, point.y, brushSize / 2, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();

      syncMaskPreview();
    },
    [brushAction, brushSize, syncMaskPreview],
  );

  const drawGrabcutStroke = useCallback(
    (point: Point, previous: Point | null) => {
      const targetCanvas = grabcutTool === "fg" ? grabcutFgCanvasRef.current : grabcutBgCanvasRef.current;
      if (!targetCanvas) {
        return;
      }

      const ctx = targetCanvas.getContext("2d");
      if (!ctx) {
        return;
      }

      ctx.save();
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.lineWidth = brushSize;
      ctx.globalCompositeOperation = "source-over";
      ctx.strokeStyle = "rgba(255, 255, 255, 1)";
      ctx.fillStyle = "rgba(255, 255, 255, 1)";

      if (previous) {
        ctx.beginPath();
        ctx.moveTo(previous.x, previous.y);
        ctx.lineTo(point.x, point.y);
        ctx.stroke();
      } else {
        ctx.beginPath();
        ctx.arc(point.x, point.y, brushSize / 2, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
      redrawCanvas();
    },
    [brushSize, grabcutTool, redrawCanvas],
  );

  const hasMask = useCallback(() => {
    const maskCanvas = maskCanvasRef.current;
    if (!maskCanvas) {
      return false;
    }

    const ctx = maskCanvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) {
      return false;
    }

    const pixels = ctx.getImageData(0, 0, maskCanvas.width, maskCanvas.height).data;
    for (let i = 3; i < pixels.length; i += 4) {
      if (pixels[i] > 0) {
        return true;
      }
    }
    return false;
  }, []);

  const getMaskBlob = useCallback(async () => {
    const previewUrl = createMaskPreviewUrl();
    if (!previewUrl) {
      throw new Error("Mask is empty.");
    }

    const response = await fetch(previewUrl);
    return await response.blob();
  }, [createMaskPreviewUrl]);

  const getBinaryCanvasBlob = useCallback(
    async (sourceCanvas: HTMLCanvasElement | null) => {
      const previewUrl = createBinaryPreviewUrl(sourceCanvas);
      if (!previewUrl) {
        return null;
      }

      const response = await fetch(previewUrl);
      return await response.blob();
    },
    [createBinaryPreviewUrl],
  );

  const applyMaskDataUrl = useCallback(
    async (maskDataUrl: string) => {
      const maskCanvas = maskCanvasRef.current;
      if (!maskCanvas || !dimensions) {
        return;
      }

      const img = new Image();
      img.src = maskDataUrl;
      await img.decode();

      const ctx = maskCanvas.getContext("2d", { willReadFrequently: true });
      if (!ctx) {
        return;
      }

      ctx.clearRect(0, 0, dimensions.width, dimensions.height);
      ctx.drawImage(img, 0, 0, dimensions.width, dimensions.height);

      const imageData = ctx.getImageData(0, 0, dimensions.width, dimensions.height);
      for (let i = 0; i < imageData.data.length; i += 4) {
        const value = Math.max(imageData.data[i], imageData.data[i + 1], imageData.data[i + 2]);
        const alpha = value > 127 ? 255 : 0;
        imageData.data[i] = 255;
        imageData.data[i + 1] = 255;
        imageData.data[i + 2] = 255;
        imageData.data[i + 3] = alpha;
      }
      ctx.putImageData(imageData, 0, 0);
      syncMaskPreview();
    },
    [dimensions, syncMaskPreview],
  );

  const handleUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selected = event.target.files?.[0];
    if (!selected) {
      return;
    }

    const url = URL.createObjectURL(selected);
    const img = new Image();
    img.onload = () => {
      setImageObj(img);
      setDimensions({ width: img.naturalWidth, height: img.naturalHeight });
      setStatus(`${img.naturalWidth}x${img.naturalHeight}`);
    };
    img.src = url;

    setFile(selected);
    setImageUrl(url);
    setMode("brush");
    setOutputs([]);
    setAnchors([]);
    setSamPoint(null);
    setGrabcutRect(null);
    setGrabcutTool("rect");
    setMaskPreviewUrl(null);
  };

  const handleModeChange = (nextMode: Mode) => {
    setMode(nextMode);
    setIsPainting(false);
    lastBrushPointRef.current = null;
    resetMaskState();
    setStatus(file ? "Ready." : "Ready.");
  };

  const runScissorsSegmentation = async () => {
    if (!file || anchors.length < 3) {
      setStatus("Need at least 3 anchors.");
      return;
    }

    setIsBusy(true);
    setStatus("Segmenting with Intelligent Scissors...");
    try {
      const formData = new FormData();
      formData.append("image", file);
      formData.append("points", JSON.stringify(anchors));
      formData.append("dilation_px", "3");

      const response = await fetch(`${API_BASE}/api/segment/scissors`, {
        method: "POST",
        body: formData,
      });
      const payload = (await response.json()) as SegmentResponse | { detail?: string };
      if (!response.ok) {
        throw new Error("detail" in payload ? payload.detail : "Scissors segmentation failed.");
      }

      await applyMaskDataUrl((payload as SegmentResponse).mask);
      setStatus("Mask ready.");
    } catch (error) {
      setStatus(getErrorMessage(error));
    } finally {
      setIsBusy(false);
    }
  };

  const runGrabcutSegmentation = async () => {
    if (!file || !grabcutRect) {
      setStatus("Draw a GrabCut rectangle first.");
      return;
    }
    if (grabcutRect.width < 2 || grabcutRect.height < 2) {
      setStatus("GrabCut rectangle is too small.");
      return;
    }

    setIsBusy(true);
    setStatus("Segmenting with GrabCut...");
    try {
      const formData = new FormData();
      formData.append("image", file);
      formData.append("rect", JSON.stringify(grabcutRect));
      formData.append("num_iters", "5");
      formData.append("max_side", "1024");

      if (canvasHasPaint(grabcutFgCanvasRef.current)) {
        const fgBlob = await getBinaryCanvasBlob(grabcutFgCanvasRef.current);
        if (fgBlob) {
          formData.append("fg_mask", fgBlob, "grabcut-fg.png");
        }
      }
      if (canvasHasPaint(grabcutBgCanvasRef.current)) {
        const bgBlob = await getBinaryCanvasBlob(grabcutBgCanvasRef.current);
        if (bgBlob) {
          formData.append("bg_mask", bgBlob, "grabcut-bg.png");
        }
      }

      const response = await fetch(`${API_BASE}/api/segment/grabcut`, {
        method: "POST",
        body: formData,
      });
      const payload = (await response.json()) as SegmentResponse | { detail?: string };
      if (!response.ok) {
        throw new Error("detail" in payload ? payload.detail : "GrabCut segmentation failed.");
      }

      await applyMaskDataUrl((payload as SegmentResponse).mask);
      setStatus("Mask ready.");
    } catch (error) {
      setStatus(getErrorMessage(error));
    } finally {
      setIsBusy(false);
    }
  };

  const runSam2Segmentation = async (point: Point) => {
    if (!file) {
      return;
    }

    setIsBusy(true);
    setSamPoint(point);
    setStatus("Segmenting with SAM2...");
    try {
      const formData = new FormData();
      formData.append("image", file);
      formData.append("x", String(point.x));
      formData.append("y", String(point.y));
      formData.append("dilation_px", "3");

      const response = await fetch(`${API_BASE}/api/segment/sam2`, {
        method: "POST",
        body: formData,
      });
      const payload = (await response.json()) as SegmentResponse | { detail?: string };
      if (!response.ok) {
        throw new Error("detail" in payload ? payload.detail : "SAM2 segmentation failed.");
      }

      const segment = payload as SegmentResponse;
      await applyMaskDataUrl(segment.mask);
      setStatus(segment.score !== undefined ? `Mask ready. score=${segment.score.toFixed(3)}` : "Mask ready.");
    } catch (error) {
      setStatus(getErrorMessage(error));
    } finally {
      setIsBusy(false);
    }
  };

  const runInpainting = async () => {
    if (!file) {
      setStatus("Upload an image first.");
      return;
    }
    if (!hasMask()) {
      setStatus("Mask is empty.");
      return;
    }

    setIsBusy(true);
    setOutputs([]);
    setStatus("Inpainting...");
    try {
      const maskBlob = await getMaskBlob();
      const formData = new FormData();
      formData.append("image", file);
      formData.append("mask", maskBlob, "mask.png");
      formData.append("dilation_px", "0");
      formData.append("response_format", "json");

      const response = await fetch(`${API_BASE}/api/inpaint`, {
        method: "POST",
        body: formData,
      });
      const payload = (await response.json()) as unknown;
      if (!response.ok || !isInpaintResponse(payload)) {
        const detail =
          typeof payload === "object" && payload !== null && "detail" in payload
            ? String((payload as { detail: unknown }).detail)
            : "Inpainting failed.";
        throw new Error(detail);
      }

      await applyMaskDataUrl(payload.mask);
      setOutputs(payload.outputs);
      setStatus(payload.outputs.length > 1 ? "Done. Returned 2 images." : "Done.");
    } catch (error) {
      setStatus(getErrorMessage(error));
    } finally {
      setIsBusy(false);
    }
  };

  const handleCanvasMouseDown = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!file || !dimensions || isBusy) {
      return;
    }

    const point = getCanvasPoint(event);
    if (!point) {
      return;
    }

    if (mode === "brush") {
      setOutputs([]);
      setIsPainting(true);
      lastBrushPointRef.current = point;
      drawBrush(point, null);
      return;
    }

    if (mode === "scissors") {
      const snappedPoint = snapToEdge(point);
      setOutputs([]);
      clearMaskCanvas();
      setAnchors((previous) => [...previous, snappedPoint]);
      setStatus(`${anchors.length + 1} anchors.`);
      return;
    }

    if (mode === "grabcut") {
      setOutputs([]);
      clearMaskCanvas();
      if (grabcutTool === "rect") {
        grabcutDragStartRef.current = point;
        setGrabcutRect({ x: point.x, y: point.y, width: 1, height: 1 });
        setIsPainting(true);
        return;
      }

      setIsPainting(true);
      lastBrushPointRef.current = point;
      drawGrabcutStroke(point, null);
      return;
    }

    void runSam2Segmentation(point);
  };

  const handleCanvasMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isPainting || !file || isBusy) {
      return;
    }

    const point = getCanvasPoint(event);
    if (!point) {
      return;
    }

    if (mode === "brush") {
      drawBrush(point, lastBrushPointRef.current);
      lastBrushPointRef.current = point;
      return;
    }

    if (mode === "grabcut") {
      if (grabcutTool === "rect" && grabcutDragStartRef.current) {
        setGrabcutRect(normalizeRect(grabcutDragStartRef.current, point));
        return;
      }

      if (grabcutTool !== "rect") {
        drawGrabcutStroke(point, lastBrushPointRef.current);
        lastBrushPointRef.current = point;
      }
    }
  };

  const stopPainting = () => {
    setIsPainting(false);
    lastBrushPointRef.current = null;
    grabcutDragStartRef.current = null;
  };

  const undoAnchor = () => {
    setAnchors((previous) => previous.slice(0, -1));
    clearMaskCanvas();
    setOutputs([]);
    setStatus("Ready.");
  };

  const clearCurrentMask = () => {
    resetMaskState();
    setStatus("Ready.");
  };

  return (
    <div className="studio-shell">
      <section className="workspace-panel">
        <nav className="mode-tabs" aria-label="Segmentation modes">
          {MODES.map((item, index) => (
            <React.Fragment key={item.id}>
              <button
                type="button"
                className={item.id === mode ? "mode-tab active" : "mode-tab"}
                onClick={() => handleModeChange(item.id)}
              >
                {item.label}
              </button>
              {index < MODES.length - 1 ? <span className="mode-separator">|</span> : null}
            </React.Fragment>
          ))}
        </nav>

        <div className="image-stage">
          {!imageObj ? (
            <button type="button" className="upload-button" onClick={() => fileInputRef.current?.click()}>
              <span className="upload-icon" aria-hidden="true">
                &#8593;
              </span>
              <span>Upload Image</span>
              <small>Require min(height/width) = 256 or 512</small>
            </button>
          ) : (
            <canvas
              ref={canvasRef}
              className={`main-canvas ${mode}`}
              onMouseDown={handleCanvasMouseDown}
              onMouseMove={handleCanvasMouseMove}
              onMouseUp={stopPainting}
              onMouseLeave={stopPainting}
              onContextMenu={(event) => event.preventDefault()}
            />
          )}
          <input ref={fileInputRef} type="file" accept="image/*" onChange={handleUpload} hidden />
          <canvas ref={maskCanvasRef} hidden />
          <canvas ref={grabcutFgCanvasRef} hidden />
          <canvas ref={grabcutBgCanvasRef} hidden />
        </div>

        <div className="tool-strip">
          <button type="button" onClick={() => fileInputRef.current?.click()}>
            Upload
          </button>

          {mode === "brush" ? (
            <>
              <div className="segmented-control" aria-label="Brush action">
                <button
                  type="button"
                  className={brushAction === "paint" ? "selected" : ""}
                  onClick={() => setBrushAction("paint")}
                >
                  Paint
                </button>
                <button
                  type="button"
                  className={brushAction === "erase" ? "selected" : ""}
                  onClick={() => setBrushAction("erase")}
                >
                  Erase
                </button>
              </div>
              <label className="range-control">
                Size
                <input
                  type="range"
                  min="4"
                  max="96"
                  value={brushSize}
                  onChange={(event) => setBrushSize(Number(event.target.value))}
                />
              </label>
            </>
          ) : null}

          {mode === "scissors" ? (
            <>
              <button type="button" onClick={undoAnchor} disabled={anchors.length === 0 || isBusy}>
                Undo
              </button>
              <button type="button" onClick={runScissorsSegmentation} disabled={anchors.length < 3 || isBusy}>
                Close Mask
              </button>
            </>
          ) : null}

          {mode === "grabcut" ? (
            <>
              <div className="segmented-control" aria-label="GrabCut tool">
                <button
                  type="button"
                  className={grabcutTool === "rect" ? "selected" : ""}
                  onClick={() => setGrabcutTool("rect")}
                >
                  Rect
                </button>
                <button
                  type="button"
                  className={grabcutTool === "fg" ? "selected" : ""}
                  onClick={() => setGrabcutTool("fg")}
                >
                  FG
                </button>
                <button
                  type="button"
                  className={grabcutTool === "bg" ? "selected" : ""}
                  onClick={() => setGrabcutTool("bg")}
                >
                  BG
                </button>
              </div>
              {grabcutTool !== "rect" ? (
                <label className="range-control">
                  Size
                  <input
                    type="range"
                    min="4"
                    max="96"
                    value={brushSize}
                    onChange={(event) => setBrushSize(Number(event.target.value))}
                  />
                </label>
              ) : null}
              <button type="button" onClick={runGrabcutSegmentation} disabled={!grabcutRect || isBusy}>
                Run GrabCut
              </button>
            </>
          ) : null}

          <button type="button" onClick={clearCurrentMask} disabled={!file || isBusy}>
            Clear Mask
          </button>
          <button type="button" className="run-button" onClick={runInpainting} disabled={!file || isBusy}>
            Inpaint
          </button>
          <span className="status-line">{isBusy ? "Working..." : status}</span>
        </div>
      </section>

      <aside className="result-panel">
        <div className="preview-section">
          <h2>Mask preview</h2>
          <div className="preview-box mask-box">
            {maskPreviewUrl ? (
              <button
                type="button"
                className="preview-image-button"
                onClick={() => setPreviewImage({ src: maskPreviewUrl, label: "Mask preview", downloadName: "mask.png" })}
                aria-label="Open mask preview"
              >
                <img src={maskPreviewUrl} alt="Mask preview" />
              </button>
            ) : null}
          </div>
        </div>

        <div className="preview-section output-section">
          <h2>Output image</h2>
          <div className={outputs.length > 1 ? "output-grid two" : "output-grid"}>
            {outputs.length > 0
              ? outputs.map((item) => (
                  <figure key={item.id} className="output-item">
                    <button
                      type="button"
                      className="output-image-button"
                      onClick={() =>
                        setPreviewImage({
                          src: item.image,
                          label: item.label,
                          downloadName: `${item.id}.png`,
                        })
                      }
                      aria-label={`Open ${item.label}`}
                    >
                      <img src={item.image} alt={item.label} />
                    </button>
                    <figcaption>
                      <span>{item.label}</span>
                      <button type="button" onClick={() => downloadDataUrl(item.image, `${item.id}.png`)}>
                        Download
                      </button>
                    </figcaption>
                  </figure>
                ))
              : null}
          </div>
        </div>
      </aside>

      {previewImage ? (
        <div className="image-viewer" role="dialog" aria-modal="true" aria-label={previewImage.label}>
          <div className="viewer-toolbar">
            <span>{previewImage.label}</span>
            <div className="viewer-actions">
              {previewImage.downloadName ? (
                <button type="button" onClick={() => downloadDataUrl(previewImage.src, previewImage.downloadName ?? "image.png")}>
                  Download
                </button>
              ) : null}
              <button type="button" onClick={() => setPreviewImage(null)} aria-label="Close preview">
                X
              </button>
            </div>
          </div>
          <button type="button" className="viewer-backdrop" onClick={() => setPreviewImage(null)} aria-label="Close preview">
            <img src={previewImage.src} alt={previewImage.label} />
          </button>
        </div>
      ) : null}
    </div>
  );
}
