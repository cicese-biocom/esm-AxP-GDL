FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04

# Metadatos de la imagen
LABEL authors="Greneter"

# Install base utilities and Python 3.7
RUN apt-get update \
    && apt-get install -y  \
    build-essential \
    wget \
    libopenblas-dev \
    git \
    gcc \
    python3.7 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget -P /tmp \
    "https://repo.anaconda.com/miniconda/Miniconda3-py37_23.1.0-1-Linux-x86_64.sh" \
    && bash /tmp/Miniconda3-py37_23.1.0-1-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniconda3-py37_23.1.0-1-Linux-x86_64.sh
ENV PATH /opt/conda/bin:$PATH

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# Establecer el directorio de trabajo
WORKDIR /opt/project

# Instalar las dependencias de Python

# requirements.txt
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# CUDA 11.3
RUN conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch

# hhsuite
RUN conda install -c conda-forge -c bioconda hhsuite==3.3.0

# esm2
RUN python3 -m pip install --no-cache-dir fair-esm && \
    python3 -m pip install --no-cache-dir fair-esm[esmfold]

RUN python3 -m pip install --no-cache-dir 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
RUN python3 -m pip install --no-cache-dir 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'

RUN python3 -m pip install --no-cache-dir torch-sparse==0.6.15 torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html

RUN wget https://anaconda.org/pyg/pytorch-scatter/2.0.9/download/linux-64/pytorch-scatter-2.0.9-py37_torch_1.12.0_cu113.tar.bz2
RUN conda install pytorch-scatter-2.0.9-py37_torch_1.12.0_cu113.tar.bz2


RUN wget -P /tmp \
    "https://anaconda.org/pyg/pytorch-scatter/2.0.9/download/linux-64/pytorch-scatter-2.0.9-py37_torch_1.12.0_cu113.tar.bz2" \
    && conda install pytorch-scatter-2.0.9-py37_torch_1.12.0_cu113.tar.bz2 \
    && rm /tmp/pytorch-scatter-2.0.9-py37_torch_1.12.0_cu113.tar.bz2

RUN conda env export> /opt/project/environment.yaml

# Comando predeterminado
CMD ["/bin/bash"]