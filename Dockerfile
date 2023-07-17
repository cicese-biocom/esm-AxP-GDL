FROM python:3.7

# Metadatos de la imagen
LABEL authors="Greneter"

# Instalar las dependencias de desarrollo
RUN apt-get update && apt-get install -y \
    libopenblas-dev \
    build-essential \
    git

# Establecer el directorio de trabajo
WORKDIR /app

# Instalar las dependencias de Python

# requirements.txt
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# PyTorch
RUN python3 -m pip install --no-cache-dir torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 && \
    python3 -m pip install --no-cache-dir torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric==1.7.2 -f https://data.pyg.org/whl/torch-1.11.0+cu113.html

# Comando predeterminado
CMD ["/bin/bash"]
