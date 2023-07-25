FROM python:3.7

# Metadatos de la imagen
LABEL authors="Greneter"

# Install base utilities
RUN apt-get update \
    && apt-get install -y  \
    build-essential \
    wget \
    libopenblas-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py37_23.1.0-1-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# Establecer el directorio de trabajo
WORKDIR /app

# Instalar las dependencias de Python

# requirements.txt
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# PyTorch
RUN python3 -m pip install --no-cache-dir torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 && \
    python3 -m pip install --no-cache-dir torch-scatter==2.0.9 torch-sparse==0.6.15 torch-cluster==1.6.0 torch-spline-conv==1.2.1 torch-geometric==1.7.2

# hhsuite
RUN conda install -c conda-forge -c bioconda hhsuite==3.3.0

# esm2
RUN python3 -m pip install --no-cache-dir fair-esm

# Comando predeterminado
CMD ["/bin/bash"]