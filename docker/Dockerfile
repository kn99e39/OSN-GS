# CUDA 11.8 is compatible with the server's NVIDIA 525 driver and supports
# both installed Ampere GPUs (RTX 3090 / RTX 3080 Ti, compute capability 8.6).
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST=8.6 \
    MAX_JOBS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip python3.10-venv \
    build-essential ninja-build git ca-certificates \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip setuptools wheel \
    && python3.10 -m pip install --index-url https://download.pytorch.org/whl/cu118 \
        torch==2.1.2+cu118 \
    && python3.10 -m pip install \
        numpy==1.26.4 \
        'pillow>=10.0,<12' \
        'jupyterlab>=4.0,<5' \
        ipykernel

WORKDIR /workspace
COPY . /workspace

# Compile the vendored CUDA rasterizer while the image is built. The source is
# also mounted at runtime, while this installed copy keeps the extension import
# fast and deterministic for the normal training path.
RUN python3.10 -m pip install --no-build-isolation \
    /workspace/osn_gs/render/vendor/diff_gaussian_rasterization

CMD ["bash"]
