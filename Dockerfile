FROM python:3.13-slim

WORKDIR /app

# libgomp1: required by PyTorch CPU kernels (OpenMP).
# g++/build-essential + ninja: let BoTorch JIT-compile its fused qLogEHVI C++
# kernel on first use (~3x faster hypervolume acqf than the pure-Python fallback).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (~200 MB vs ~2 GB for CUDA)
RUN pip install --no-cache-dir \
    torch==2.13.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install BoTorch and remaining ML deps (after torch so they link correctly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY vam_space.py optimizer_core.py main.py ./

ENV PORT=8080

# 1 worker, 8 threads — per-user threading.Lock serialises same-user requests;
# different users run concurrently across threads.
# timeout 300s: GP fitting + mixed acqf optimisation can take a while on CPU.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 300 main:app
