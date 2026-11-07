# Recorded execution environments

## Complete technology stack

| Component | Version / detail |
|---|---|
| Hardware | NVIDIA Jetson Orin Nano Developer Kit (8 GB), JetPack 6 / L4T R36.3 (CUDA 12.2, cuDNN 8.9, TensorRT 8.6.2) |
| Main environment (`.venv`) | Python 3.11, OpenCV 5.0.0, NumPy 2.4.6, pytest. Used for the pipeline, viewer server and tests. Its PyTorch 2.15 is a CUDA 13.2 build and cannot use this JetPack's GPU |
| System `python3` (JetPack) | Python 3.10.12, OpenCV 4.8.0, NumPy 1.26.4, **PyTorch 2.1.0 with CUDA 12.2 (GPU works)**, `jetson-inference` with TensorRT running NVIDIA's pretrained Pose-ResNet18-Hand |
| GPU environment (`.venv-jetson`) | A venv that inherits the system packages above unchanged, plus `pytest` 9.2.0 and `psutil` 7.2.2 installed into the venv only. Used for TensorRT inference, the CPU/GPU benchmark and the CUDA parity tests |
| Video tools | FFmpeg / ffprobe 4.4.2 |
| Viewer | Static HTML/CSS/JavaScript (no frameworks or CDNs) served by Python's `http.server` with added Range support |

### Which Python environment to use (on the Jetson)

| Task | Environment | Why |
|---|---|---|
| Classical measurement pipeline, viewer server, tests | `.venv` | The pipeline is OpenCV/NumPy on the CPU and needs no CUDA |
| AI-assisted strategy (TensorRT hand pose) | `.venv-jetson` | Needs JetPack's `jetson_inference`/TensorRT, which only the system Python provides |
| CPU vs GPU benchmark, CUDA parity tests | `.venv-jetson` | Needs a PyTorch that can use this JetPack's GPU: the system PyTorch 2.1 / CUDA 12.2 |

The `.venv` PyTorch (2.15, built for CUDA 13.2) cannot initialise CUDA on JetPack 6's CUDA 12.2 driver, which is why there are two environments. Neither environment's packages were replaced.

Set up `.venv-jetson` (once, about a minute; no system changes):

```bash
cd ~/Desktop/CV-Challenge-
# JetPack's Python lacks ensurepip (python3.10-venv), so create the venv without its own pip and use
# the system pip from inside it; installs go into the venv only.
python3 -m venv --system-site-packages --without-pip .venv-jetson
.venv-jetson/bin/python -m pip install pytest psutil
.venv-jetson/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 2.1.0 True
```


All commands run from the repository root. These are the recorded environments, not a request
to upgrade CUDA, replace packages or rebuild the cached TensorRT engine. Core dependencies are
in `requirements.txt`; `requirements-gpu.txt` is optional for the PyTorch benchmark. The
hand-assisted strategy uses JetPack's existing `jetson_inference` and `jetson_utils` instead.
The inspection-only YOLO script additionally requires `ultralytics` and supplied weights; it is
not part of either production strategy and was not run in the recorded evaluation.
