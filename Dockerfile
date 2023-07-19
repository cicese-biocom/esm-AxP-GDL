FROM python:3.7

RUN apt-get update && apt-get -y upgrade \
  && apt-get install -y --no-install-recommends \
    git \
    wget \
    g++ \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \

ENV PATH="/root/miniconda3/bin:${PATH}"

ARG PATH="/root/miniconda3/bin:${PATH}"

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py37_23.1.0-1-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py37_23.1.0-1-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py37_23.1.0-1-Linux-x86_64.sh \
    && echo "Running $(conda --version)" && \
    conda init bash && \
    . /root/.bashrc && \
    conda update conda && \
    conda create -n python-app && \
    conda activate python-app && \
    conda install python=3.7 pip && \
    echo 'print("Hello World!")' > python-app.py \

RUN echo 'conda activate python-app \n\
alias python-app="python python-app.py"' >> /root/.bashrc

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar el archivo requirements.txt
COPY requirements.txt .

# Instalar dependencias de Python
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# PyTorch
RUN python3 -m pip install --no-cache-dir torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 && \
    python3 -m pip install --no-cache-dir torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric==1.7.2 -f https://data.pyg.org/whl/torch-1.11.0+cu113.html

# hhblits
conda install -c conda-forge -c bioconda hhsuite==3.3.0

# Comando predeterminado
CMD ["/bin/bash"]
