# How to run

This repository contains the local web app for object removal via interactive segmentation and LaMa-Fourier inpainting.

The app has two processes:

- Backend: FastAPI at `http://127.0.0.1:8000`
- Frontend: Next.js at `http://127.0.0.1:3000`

## 1. Required folder structure

Before running the app, place the model weights in these exact paths:

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

The weights are stored with Git LFS. After cloning the repository, install Git LFS and pull the weight files:

```powershell
git lfs install
git lfs pull
```

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

## 6. Use the web app

1. Upload an image.
2. Choose one segmentation mode: `Brush mask`, `IntelligentScissors`, `Grabcut`, or `SAM2`.
3. Create or refine the mask.
4. Check `Mask preview`.
5. Click `Inpaint`.
6. Click an output image to view it full size or download it.

Note: the upload UI shows `Require min(height/width) = 256 or 512`.
