FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04

# Metadatos de la imagen
LABEL version="1.0" maintainer="Greneter Cordoves Delgado <grenetercordovesdelgado@gmail.com>" description="esm-AxP-GDL framework environment"

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

# Set up the working directory
WORKDIR /opt/project

# Install requirements.txt
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Install PyTorch 1.12.0+cu113
RUN conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch

# Install esm2 and esmfold
RUN python3 -m pip install --no-cache-dir fair-esm && \
    python3 -m pip install --no-cache-dir fair-esm[esmfold] && \
    python3 -m pip install --no-cache-dir 'dllogger @ git+https://github.com/NVIDIA/dllogger.git' && \
    python3 -m pip install --no-cache-dir 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'

# Install PyTorch Geometric (torch-cluster, torch-sparse, torch-geometric and torch-scatter)
RUN python3 -m pip install --no-cache-dir torch-sparse==0.6.15 torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html

COPY misc/linux-64_pytorch-scatter-2.0.9-py37_torch_1.12.0_cu113.tar.bz2 /opt/project/
RUN conda install -y /opt/project/linux-64_pytorch-scatter-2.0.9-py37_torch_1.12.0_cu113.tar.bz2 \
    && rm /opt/project/linux-64_pytorch-scatter-2.0.9-py37_torch_1.12.0_cu113.tar.bz2



# Command
CMD ["/bin/bash"]