# How to run

This project is a local web app for object removal via interactive segmentation and LaMa-Fourier inpainting.

It has two processes:

- Backend: FastAPI at `http://127.0.0.1:8000`
- Frontend: Next.js at `http://127.0.0.1:3000`

## 1. Run from the submitted ZIP file

If you receive this project as a `.zip` file from Microsoft Teams:

1. Extract the ZIP file.
2. Open PowerShell.
3. Go to the extracted project folder.

Example:

```powershell
cd "D:\Object-Removal-via-Image-Inpainting"
```

Do not run commands inside the ZIP preview window. Extract the ZIP first.

## 2. Required folder structure

The extracted folder must contain these files:

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
|       |-- best_lama_generator_only.pth
|       `-- lama_full_checkpoint_latest.pth
`-- frontend/
    |-- package.json
    |-- package-lock.json
    `-- src/
        |-- app/
        `-- components/
```

Required weights for running the web app:

```text
module1/checkpoint_3.pt
results/checkpoints/best_lama_generator_only.pth
```

`results/checkpoints/lama_full_checkpoint_latest.pth` is included for completeness, but the web app uses `best_lama_generator_only.pth` for inference.

Check the weight files:

```powershell
Get-Item .\module1\checkpoint_3.pt, .\results\checkpoints\best_lama_generator_only.pth, .\results\checkpoints\lama_full_checkpoint_latest.pth |
  Select-Object Name, @{Name="MB";Expression={[math]::Round($_.Length/1MB,2)}}
```

Expected sizes:

```text
checkpoint_3.pt                    about 470 MB
best_lama_generator_only.pth        about 161 MB
lama_full_checkpoint_latest.pth     about 514 MB
```

If a weight file is only a few KB, it is not the real model file.

## 3. Install backend

Run from the project root:

```powershell
python -m venv inpainting
.\inpainting\Scripts\python.exe -m pip install --upgrade pip
.\inpainting\Scripts\python.exe -m pip install -r requirements.txt
```

This step needs internet access because `requirements.txt` installs the SAM2 package from GitHub.

## 4. Run backend

Run from the project root:

```powershell
.\inpainting\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Keep this terminal open.

Check backend in a browser:

```text
http://127.0.0.1:8000/health
```

The response should include:

```text
"status":"ok"
"lama_checkpoint_exists":true
"sam2_checkpoint_exists":true
"sam2_package_available":true
```

## 5. Install frontend

Open a second PowerShell terminal:

```powershell
cd "D:\Object-Removal-via-Image-Inpainting\frontend"
npm install
```

Replace the path with your extracted project path.

## 6. Run frontend

Run from the `frontend` folder:

```powershell
$env:NEXT_PUBLIC_API_BASE_URL="http://127.0.0.1:8000"
npm run dev -- --hostname 127.0.0.1 --port 3000
```

Open:

```text
http://127.0.0.1:3000
```

## 7. Use the web app

1. Upload an image.
2. Choose one segmentation mode: `Brush mask`, `IntelligentScissors`, `Grabcut`, or `SAM2`.
3. Create or refine the mask.
4. Check `Mask preview`.
5. Click `Inpaint`.
6. Click an output image to view it full size or download it.

Note: the upload UI shows `Require min(height/width) = 256 or 512`.

## Troubleshooting

If the frontend shows `Failed to fetch`, the backend is usually not running. Start the backend at `http://127.0.0.1:8000` and reload the frontend.

If port `8000` or `3000` is already used, stop the old process or choose another port.

If SAM2 is slow on the first click, wait a few seconds. The backend loads the SAM2 checkpoint on the first request.

If `/health` shows `"sam2_package_available":false`, reinstall backend dependencies:

```powershell
.\inpainting\Scripts\python.exe -m pip install -r requirements.txt
```

## If using GitHub instead of the submitted ZIP

GitHub uses Git LFS for the weight files. Do not use GitHub `Download ZIP`, because it may not include the real LFS weight files.

Use:

```powershell
git clone https://github.com/Khuc-Ngoc-Nam/Object-Removal-via-Image-Inpainting.git
cd Object-Removal-via-Image-Inpainting
git lfs install
git lfs pull
```
