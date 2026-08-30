FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

LABEL org.opencontainers.image.title="iMED NVS Endo-4DGS submission" \
      org.opencontainers.image.description="Source-only Endoscope-2 training with normalized soft-overlap weighting" \
      org.opencontainers.image.version="final-softoverlap-seed1"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX" \
    MAX_JOBS=4 \
    INPUT_DIR=/input \
    OUTPUT_DIR=/output \
    IMED_SEED=1 \
    IMED_CONFIG=/app/arguments/imed_extent10_smooth002_softoverlap_iter2500.py \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        ninja-build \
        python3.8 \
        python3.8-dev \
        python3-pip \
    && ln -sf /usr/bin/python3.8 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.8 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade \
        pip==24.0 setuptools==69.5.1 wheel==0.43.0

RUN python -m pip install --no-cache-dir \
        torch==2.0.0+cu118 \
        torchvision==0.15.1+cu118 \
        --index-url https://download.pytorch.org/whl/cu118

COPY imed_nvs_submission/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

COPY submodules/diff-gaussian-rasterization-depth /tmp/diff-gaussian-rasterization-depth
COPY submodules/simple-knn /tmp/simple-knn
RUN python -m pip install --no-cache-dir --no-build-isolation \
        /tmp/diff-gaussian-rasterization-depth \
    && python -m pip install --no-cache-dir --no-build-isolation /tmp/simple-knn \
    && rm -rf /tmp/diff-gaussian-rasterization-depth /tmp/simple-knn \
    && python -c "from diff_gaussian_rasterization_depth import GaussianRasterizer; from simple_knn._C import distCUDA2; print('CUDA extensions import OK')"

WORKDIR /app
COPY . /app

RUN python -m compileall -q \
        /app/arguments \
        /app/gaussian_renderer \
        /app/imed_nvs_submission \
        /app/scene \
        /app/utils \
        /app/train.py \
    && python -c "from imed_nvs_submission.predict import new_view; assert callable(new_view)" \
    && test -f "${IMED_CONFIG}"

ENTRYPOINT ["python", "/app/imed_nvs_submission/predict.py"]
