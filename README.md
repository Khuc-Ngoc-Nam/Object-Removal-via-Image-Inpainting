# How to run

This repository contains the local web app for object removal via interactive segmentation and LaMa-Fourier inpainting.

The app has two processes:

- Backend: FastAPI at `http://127.0.0.1:8000`
- Frontend: Next.js at `http://127.0.0.1:3000`

## 0. Prerequisites

Use Git clone, not GitHub `Download ZIP`. GitHub ZIP files do not include Git LFS weights correctly.

Install:

- Python 3.10 or newer
- Node.js 20 or newer
- Git LFS

Clone and pull LFS files:

```powershell
git clone https://github.com/Khuc-Ngoc-Nam/Object-Removal-via-Image-Inpainting.git
cd Object-Removal-via-Image-Inpainting
git lfs install
git lfs pull
```

## 1. Required folder structure

Before running the app, make sure the model weights exist in these exact paths:

```text
Object-Removal-via-Image-Inpainting/
|-- main.py
|-- model.py
|-- lasso_core.py
|-- requirements.txt
|-- module1/
|   |-- grabcut.py
|   |-- sam2_click.py
|   `-- checkpoint_3.pt
|-- results/
|   `-- checkpoints/
|       `-- best_lama_generator_only.pth
`-- frontend/
    |-- package.json
    `-- src/
        |-- app/
        `-- components/
```

Required weights:

```text
module1/checkpoint_3.pt
results/checkpoints/best_lama_generator_only.pth
```

Check weight sizes on Windows PowerShell:

```powershell
Get-Item .\module1\checkpoint_3.pt, .\results\checkpoints\best_lama_generator_only.pth |
  Select-Object Name, @{Name="MB";Expression={[math]::Round($_.Length/1MB,2)}}
```

Expected sizes are about `470 MB` for `checkpoint_3.pt` and `161 MB` for `best_lama_generator_only.pth`. If the files are only a few KB, run `git lfs pull` again.

## 2. Install backend

Run from the repository root:

```powershell
python -m venv inpainting
.\inpainting\Scripts\python.exe -m pip install --upgrade pip
.\inpainting\Scripts\python.exe -m pip install -r requirements.txt
```

## 3. Run backend

Run from the repository root:

```powershell
.\inpainting\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Check backend:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

## 4. Install frontend

Open a second terminal:

```powershell
cd frontend
npm install
```

## 5. Run frontend

Run from the `frontend` folder:

```powershell
$env:NEXT_PUBLIC_API_BASE_URL="http://127.0.0.1:8000"
npm run dev -- --hostname 127.0.0.1 --port 3000
```

Open:

```text
http://127.0.0.1:3000
```

## 6. Quick checks

Backend health must return `"status":"ok"`:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/health -UseBasicParsing
```

The response should include:

```text
"lama_checkpoint_exists":true
"sam2_checkpoint_exists":true
"sam2_package_available":true
```

If port `3000` or `8000` is already used, stop the old process or choose another port.

## 7. Use the web app

1. Upload an image.
2. Choose one segmentation mode: `Brush mask`, `IntelligentScissors`, `Grabcut`, or `SAM2`.
3. Create or refine the mask.
4. Check `Mask preview`.
5. Click `Inpaint`.
6. Click an output image to view it full size or download it.

Note: the upload UI shows `Require min(height/width) = 256 or 512`.
