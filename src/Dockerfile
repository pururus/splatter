ARG CUDA_BASE_IMAGE="nvidia/cuda:12.8.0-devel-ubuntu22.04"
FROM ${CUDA_BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    git \
    ca-certificates \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    libglm-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir -U pip setuptools wheel -i https://mirrors.aliyun.com/pypi/simple/

WORKDIR /app

ARG PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
ARG TORCH_VERSION="2.7.0"
ARG TORCHVISION_VERSION="0.22.0"
ARG TORCHAUDIO_VERSION="2.7.0"
RUN python3 -m pip install --no-cache-dir \
    --index-url ${PYTORCH_INDEX_URL} \
    torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} -i https://mirrors.aliyun.com/pypi/simple/

COPY docker/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

ARG TORCH_CUDA_ARCH_LIST="8.9;12.0"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

ARG PYTORCH3D_REF="V0.7.8"
RUN FORCE_CUDA=1 python3 -m pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@${PYTORCH3D_REF}"

ARG SIMPLE_KNN_REF="86710c2d4b46680c02301765dd79e465819c8f19"
RUN python3 -m pip install --no-cache-dir --no-build-isolation \
    "git+https://gitlab.inria.fr/bkerbl/simple-knn@${SIMPLE_KNN_REF}" \
     -i https://mirrors.aliyun.com/pypi/simple/

COPY . /app/src

RUN python3 -m pip install --no-cache-dir --no-build-isolation /app/src/submodules/dptr -i https://mirrors.aliyun.com/pypi/simple/

ENV PYTHONPATH=/app/src

