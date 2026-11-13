## Setup (First time)

### 1. Load modules
```
module load miniforge3
# CUDA Toolkit (bao gồm nvcc) là bắt buộc để build các CUDA submodules ở bước 5.
module load cuda/12.2.0
```

### 2. Tạo environment
```
mamba create -n instantsplatPP python=3.10.13 cmake=3.14.0 -y
mamba activate instantsplatPP
```

### 3. PyTorch + CUDA
```
mamba install pytorch==2.1.2 torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Phải in ra PyTorch 2.1.2 và CUDA 12.1 trước khi tiếp tục.
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.version.cuda)"
```

> Nếu CUDA in ra `13.x` (hoặc PyTorch không phải `2.1.2`), môi trường đã không khớp
> với cấu hình project. Cài lại bước 3 với `--force-reinstall` trước khi build các extension:
> `mamba install --force-reinstall pytorch==2.1.2 torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y`

### 4. Python dependencies
```
cd /fs04/scratch2/ml23/htran0002/InstantSplatPP-Original
pip install -r requirements.txt
pip uninstall -y numpy opencv-python
pip install --no-cache-dir "numpy<2" "opencv-python<4.12"

# setuptools >= 81 đã bỏ pkg_resources; một số CUDA extension vẫn cần module này.
python -m pip install --upgrade "setuptools<81" wheel
```

### 5. Build CUDA submodules
```
# Xác nhận CUDA Toolkit compiler đã có trong terminal hiện tại.
which nvcc
nvcc --version

# PyTorch cần biết chính xác CUDA Toolkit dùng để compile extension.
export CUDA_HOME=/apps/cuda/12.2.0
export PATH="$CUDA_HOME/bin:$PATH"

pip install -v --no-build-isolation ./submodules/simple-knn
pip install -v --no-build-isolation ./submodules/diff-gaussian-rasterization
pip install -v --no-build-isolation ./submodules/fused-ssim
```

> `pytorch-cuda=12.1` là CUDA runtime; nó không thay thế CUDA Toolkit/nvcc.
> CUDA Toolkit 12.2 ở trên tương thích để build extension cho PyTorch CUDA 12.1.

### 6. (Optional) RoPE CUDA kernels cho CroCo v2
```
cd croco/models/curope/
python setup.py build_ext --inplace
```
cd /fs04/scratch2/ml23/htran0002/InstantSplatPP-Original

## Chạy inference mỗi session

### 1. Load module + activate env
```
module load miniforge3
module load cuda/12.2.0
mamba activate instantsplatPP
export CUDA_HOME=/apps/cuda/12.2.0
```

### 2. Chạy inference
```
bash scripts/run_all_mast3r.sh
```
hoặc
```
bash scripts/run_all_vggt.sh
```
