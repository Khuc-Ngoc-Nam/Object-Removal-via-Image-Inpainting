# Object Removal via Image Inpainting

## How to Run Locally

This project runs as two local processes:

- **FastAPI backend** on `http://127.0.0.1:8000`
- **Next.js frontend** on `http://localhost:3000`

Open two terminals from the repository root.

### 1. Start the Backend

PowerShell on Windows:

```powershell
cd "D:\path\to\Object-Removal-via-Image-Inpainting"

# Optional only if PowerShell blocks venv activation scripts
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

# Use the existing project virtual environment if present
.\inpainting\Scripts\python.exe -m pip install -r requirements.txt

# Required by the current backend inference path
.\inpainting\Scripts\python.exe -m pip install numpy opencv-contrib-python torch torchvision

# Start FastAPI
.\inpainting\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

If you do not already have `inpainting/`, create it first:

```powershell
python -m venv inpainting
.\inpainting\Scripts\python.exe -m pip install --upgrade pip
.\inpainting\Scripts\python.exe -m pip install -r requirements.txt
.\inpainting\Scripts\python.exe -m pip install numpy opencv-contrib-python torch torchvision
```

Verify the backend:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

### 2. Start the Frontend

In a second terminal:

```powershell
cd "D:\path\to\Object-Removal-via-Image-Inpainting\frontend"
npm install
npm run dev
```

Open:

```text
http://localhost:3000
```

### 3. Required Model Files for Inpainting

The lasso UI can run with only the Intelligent Scissors backend. Full LaMa inpainting additionally requires:

```text
model.py
results/checkpoints/best_lama_generator_only.pth
```

The backend expects the checkpoint here:

```text
results/checkpoints/best_lama_generator_only.pth
```

The model is loaded on CPU with `torch.device("cpu")` and `map_location=torch.device("cpu")`.

## Current Application Status

The repository currently contains a full-stack Magnetic Lasso MVP:

- `main.py` starts the FastAPI backend.
- `lasso_core.py` wraps OpenCV Intelligent Scissors.
- `frontend/src/components/LassoTool.tsx` provides the canvas-based lasso UI.
- The frontend uploads an image to the backend, opens a WebSocket, receives snapped contours, and receives a generated binary mask.

The backend also exposes an `/api/inpaint` endpoint for the LaMa inpainting pipeline. The current frontend button shown after mask generation is still a placeholder alert, so wiring that button to `/api/inpaint` is the next integration step for the React UI.

## Project Structure

```text
Object-Removal-via-Image-Inpainting/
├── main.py                         # FastAPI API, lasso WebSocket, LaMa inpainting endpoint
├── lasso_core.py                   # OpenCV Intelligent Scissors wrapper
├── model.py                        # LaMaGenerator architecture, required for inpainting
├── requirements.txt                # Minimal FastAPI backend dependencies
├── module1/
│   └── scissors.py                 # Original OpenCV Intelligent Scissors experiment
├── results/
│   └── checkpoints/
│       └── best_lama_generator_only.pth
├── frontend/
│   ├── package.json
│   └── src/
│       ├── app/page.tsx
│       └── components/LassoTool.tsx
└── inpainting/                     # Local Python virtual environment, not source code
```

## Backend Overview

### `POST /upload`

Used by the Next.js frontend to initialize an Intelligent Scissors session.

Request:

- `multipart/form-data`
- `file`: uploaded image

Response:

```json
{
  "session_id": "uuid",
  "width": 1280,
  "height": 720
}
```

### `WS /ws/lasso/{session_id}`

Real-time Magnetic Lasso endpoint.

Client messages:

```json
{ "type": "anchor", "x": 100, "y": 120 }
```

```json
{
  "type": "move",
  "start_x": 100,
  "start_y": 120,
  "x": 160,
  "y": 180
}
```

```json
{
  "type": "close",
  "points": [{ "x": 100, "y": 120 }, { "x": 160, "y": 180 }]
}
```

Server messages include:

- `map_built`
- `contour`
- `mask_generated`

### `POST /api/inpaint`

Runs object removal using the image plus either a frontend mask or lasso point data.

Request format:

- `multipart/form-data`
- `image`: original image file
- optional `mask`: binary mask image, white pixels are removed
- optional `points`: JSON array of points if no mask file is sent
- optional `selection_mode`: `path`, `polygon`, `contour`, `anchors`, or `intelligent_scissors`
- optional `margin_ratio`: defaults to `0.5`
- optional `max_crop_side`: defaults to `1024`
- optional `response_format`: `image` for PNG stream, or `base64`/`json`

Example frontend call:

```ts
const form = new FormData();
form.append("image", imageFile);
form.append("mask", maskBlob, "mask.png");
form.append("response_format", "base64");

const res = await fetch("http://127.0.0.1:8000/api/inpaint", {
  method: "POST",
  body: form,
});

const data = await res.json();
const outputDataUrl = data.image;
```

Point-based request:

```ts
const form = new FormData();
form.append("image", imageFile);
form.append(
  "points",
  JSON.stringify([
    { x: 100, y: 120 },
    { x: 240, y: 135 },
    { x: 220, y: 260 }
  ])
);
form.append("selection_mode", "anchors");
form.append("response_format", "base64");
```

## Inpainting Pipeline

The backend follows this CPU-friendly data path:

1. Build or decode a binary object mask.
2. Find the mask bounding box.
3. Expand the bounding box by a configurable context margin, default `50%`.
4. Crop the image and mask to the expanded box.
5. Resize if needed for CPU performance.
6. Pad the crop to dimensions divisible by `8`.
7. Convert the image to `[-1, 1]` and the mask to `[0, 1]`.
8. Concatenate into `[1, 4, H, W]`.
9. Run `LaMaGenerator` on CPU.
10. Convert output back to image pixels.
11. Composite generated pixels only inside the mask.
12. Paste the repaired crop back into the original full-resolution image.

## Frontend Overview

The Next.js app lives in `frontend/`.

Important files:

- `frontend/src/app/page.tsx` renders the lasso tool.
- `frontend/src/components/LassoTool.tsx` contains upload, canvas drawing, WebSocket handling, and mask preview state.

Current frontend flow:

1. Upload an image.
2. Backend creates a lasso session via `POST /upload`.
3. Frontend opens `ws://localhost:8000/ws/lasso/{session_id}`.
4. User clicks anchors around an object.
5. Backend returns snapped contours from Intelligent Scissors.
6. User closes the loop.
7. Backend returns a base64 PNG mask.

The inpainting API is ready on the backend, but the React button still needs to send the original image and mask to `/api/inpaint`.

## Environment Variables

### `CORS_ALLOW_ORIGINS`

Optional comma-separated allowlist for the FastAPI CORS middleware.

Default:

```text
http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001
```

Example:

```powershell
$env:CORS_ALLOW_ORIGINS="http://localhost:3000,http://127.0.0.1:3000"
```

## Troubleshooting

### `localhost refused to connect`

The backend or frontend is not running. Check:

```text
http://127.0.0.1:8000/health
http://localhost:3000
```

### `[WinError 10013]` or port already in use

Stop the process using the port.

```powershell
Get-NetTCPConnection -LocalPort 8000
Stop-Process -Id <OwningProcess>
```

For Gradio experiments on port `7860`:

```powershell
Get-NetTCPConnection -LocalPort 7860
Stop-Process -Id <OwningProcess>
```

### `No module named model`

Make sure `model.py` is in the repository root, next to `main.py`, then restart the backend.

### `cv2` has no `segmentation_IntelligentScissorsMB`

Install OpenCV contrib, not only the base OpenCV package:

```powershell
.\inpainting\Scripts\python.exe -m pip install opencv-contrib-python
```

### CPU inference is slow

LaMa runs on CPU in this project. Reduce `max_crop_side` in `/api/inpaint` requests for faster inference at the cost of detail.

## Development Notes

- Do not commit `inpainting/`, `.venv/`, `node_modules/`, generated masks, or local outputs.
- Keep large checkpoints out of normal source commits unless intentionally using Git LFS or release assets.
- The backend is designed to run on CPU for local reproducibility.
- The current frontend is an MVP lasso client; final inpainting UX requires wiring the generated mask to `/api/inpaint`.
