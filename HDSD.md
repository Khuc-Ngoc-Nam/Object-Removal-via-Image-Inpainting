# HDSD - Huong Dan Su Dung Localhost 8000

Tai lieu nay huong dan cach chay ung dung Object Removal via Image Inpainting tren may local va truy cap:

```text
http://localhost:8000
```

Ung dung o port `8000` la Gradio all-in-one app trong file `app.py`. App nay gom giao dien upload anh, ve mask hoac dung Intelligent Scissors, sau do chay LaMa Fourier Inpainting bang checkpoint trong `results/checkpoints/`.

## 1. Yeu Cau Moi Truong

Can cai dat:

- Python 3.10 tro len
- Git
- Git LFS, vi file checkpoint `.pth` duoc quan ly bang Git LFS
- Trinh duyet web hien dai, vi du Chrome hoac Edge

Neu chua co Git LFS:

```powershell
git lfs install
```

## 2. Lay Source Code Va Model Weight

Clone repo:

```powershell
git clone https://github.com/Khuc-Ngoc-Nam/Object-Removal-via-Image-Inpainting.git
cd Object-Removal-via-Image-Inpainting
```

Tai file checkpoint tu Git LFS:

```powershell
git lfs pull
```

Kiem tra file model weight ton tai:

```powershell
Test-Path .\results\checkpoints\best_lama_generator_only.pth
```

Neu lenh tren tra ve `True` la dung.

## 3. Tao Moi Truong Python

Neu da co san thu muc `inpainting/` tren may, co the dung lai. Neu chua co, tao moi:

```powershell
python -m venv inpainting
```

Nang cap pip:

```powershell
.\inpainting\Scripts\python.exe -m pip install --upgrade pip
```

Cai cac dependency co ban:

```powershell
.\inpainting\Scripts\python.exe -m pip install -r requirements.txt
```

Cai cac dependency can cho Gradio, OpenCV Intelligent Scissors va LaMa model:

```powershell
.\inpainting\Scripts\python.exe -m pip install gradio numpy pillow opencv-contrib-python
```

Cai PyTorch CPU:

```powershell
.\inpainting\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Luu y: `opencv-contrib-python` la bat buoc vi Intelligent Scissors nam trong OpenCV contrib.

## 4. Chay Ung Dung O Port 8000

Dam bao dang o thu muc goc cua project:

```powershell
cd "D:\Kì này\Computer Vision\Object-Removal-via-Image-Inpainting"
```

Chay Gradio app:

```powershell
.\inpainting\Scripts\python.exe app.py
```

Neu chay thanh cong, terminal se hien dia chi tuong tu:

```text
Running on local URL:  http://127.0.0.1:8000
```

Mo trinh duyet:

```text
http://localhost:8000
```

Hoac:

```text
http://127.0.0.1:8000
```

## 5. Cach Su Dung Tren Giao Dien

Ung dung co 2 cach tao mask.

### Cach 1: Brush Mask

1. Mo tab `Brush mask`.
2. Upload anh can xoa vat the.
3. Dung brush mau trang to len vat the can xoa.
4. Chinh `Mask dilation`, `Context margin`, `Max crop side` neu can.
5. Bam `Run Inpainting From Brush Mask`.
6. Anh ket qua se hien o panel `Output image`.

### Cach 2: Intelligent Scissors

1. Mo tab `Intelligent Scissors`.
2. Upload anh.
3. Click cac diem neo quanh bien vat the.
4. Khi da bao quanh vat the, bam `Close Mask`.
5. Kiem tra `Mask preview`.
6. Bam `Run Inpainting From Scissors Mask`.
7. Anh ket qua se hien o panel `Output image`.

## 6. Pipeline Xu Ly Ben Trong

Khi bam nut inpainting, app thuc hien:

1. Lay anh goc va mask.
2. Tim bounding box cua mask.
3. Mo rong bounding box voi margin mac dinh `50%`.
4. Crop anh va mask theo bounding box da mo rong.
5. Resize neu crop qua lon de chay CPU nhe hon.
6. Pad kich thuoc crop ve boi so cua `8`.
7. Chuyen anh sang tensor trong khoang `[-1, 1]`.
8. Chuyen mask sang tensor trong khoang `[0, 1]`.
9. Noi anh va mask thanh tensor `[1, 4, H, W]`.
10. Chay `LaMaGenerator` tu `model.py` tren CPU.
11. Composite pixel sinh ra chi trong vung mask.
12. Paste crop da sua ve dung toa do tren anh goc.

## 7. Cach Dung Server

Neu terminal dang chay app, bam:

```text
Ctrl + C
```

Neu app dang chay nen hoac port 8000 bi chiem, xem process:

```powershell
Get-NetTCPConnection -LocalPort 8000
```

Dung process:

```powershell
Stop-Process -Id <OwningProcess>
```

Sau do chay lai:

```powershell
.\inpainting\Scripts\python.exe app.py
```

## 8. Loi Thuong Gap

### 8.1. `localhost refused to connect`

Nguyen nhan: app chua chay hoac da bi tat.

Cach sua:

```powershell
.\inpainting\Scripts\python.exe app.py
```

Sau do mo lai:

```text
http://localhost:8000
```

### 8.2. Port 8000 da bi chiem

Kiem tra:

```powershell
Get-NetTCPConnection -LocalPort 8000
```

Dung process dang chiem port:

```powershell
Stop-Process -Id <OwningProcess>
```

### 8.3. `No module named model`

Kiem tra `model.py` co nam cung cap voi `app.py` khong:

```powershell
Test-Path .\model.py
```

Neu tra ve `False`, source code chua day du.

### 8.4. Khong thay checkpoint

Kiem tra:

```powershell
Test-Path .\results\checkpoints\best_lama_generator_only.pth
```

Neu tra ve `False`, chay:

```powershell
git lfs pull
```

### 8.5. Loi OpenCV Intelligent Scissors

Neu gap loi lien quan `segmentation_IntelligentScissorsMB`, cai lai OpenCV contrib:

```powershell
.\inpainting\Scripts\python.exe -m pip uninstall opencv-python opencv-contrib-python -y
.\inpainting\Scripts\python.exe -m pip install opencv-contrib-python
```

### 8.6. Chay rat cham

Model dang chay CPU. Co the giam `Max crop side` tren giao dien, vi crop nho hon se inference nhanh hon.

## 9. Ghi Chu Ve Frontend Next.js

Thu muc `frontend/` van co Next.js UI MVP cho Magnetic Lasso. Tuy nhien ung dung chinh co model deep learning va chay truc tiep o:

```text
http://localhost:8000
```

Port `3000` chi danh cho Next.js MVP, khong phai giao dien chinh cua Gradio LaMa app.
