FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        build-essential \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
        python3-pip \
        python-is-python3 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Pin pip (matches conda spec)
RUN python -m pip install --upgrade 'pip==24.0'

# PyTorch + CUDA 11.8 (replaces the pytorch / nvidia conda channels)
RUN pip install --index-url https://download.pytorch.org/whl/cu118 \
        torch torchvision torchaudio

# Python deps from the conda yaml's pip section
RUN pip install \
        "setuptools<70.0.0" \
        mlflow==2.22.2 \
        monai \
        einops \
        nibabel \
        itk \
        SimpleITK \
        numpy \
        pandas \
        scipy \
        scikit-image \
        matplotlib \
        tqdm \
        psutil \
        nvidia-ml-py \
        omegaconf

WORKDIR /workspace
CMD ["/bin/bash"]