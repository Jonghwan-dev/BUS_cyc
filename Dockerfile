# Base Image (CUDA 12.1.1 + Ubuntu 20.04)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu20.04

# 필수 환경 설정
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
WORKDIR /workspace

# 시스템 패키지 설치
RUN apt update && apt install -y \
    curl bzip2 git wget libgl1 build-essential python3-pip

# Miniconda 설치
RUN curl -sLo miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# mamba + conda-lock 설치
RUN conda install -y -n base -c conda-forge mamba conda-lock

# conda-lock 기반 환경 생성
COPY conda-lock.yml .
RUN conda-lock install conda-lock.yml --name cyclenv --mamba && \
    mamba clean --all -y

# PyTorch Nightly (CUDA 12.8) 설치 (pip만 가능)
RUN /opt/conda/envs/cyclenv/bin/pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# pip 패키지 설치
COPY requirements.txt .
RUN /opt/conda/envs/cyclenv/bin/pip install -r requirements.txt

# 코드 복사
COPY . .

# 환경 활성화 및 실행
SHELL ["conda", "run", "-n", "cyclenv", "/bin/bash", "-c"]
CMD python -u check_computing.py