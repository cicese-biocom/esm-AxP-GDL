FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04

LABEL maintainer="Greneter Cordoves Delgado <grenetercordovesdelgado@gmail.com>" description="esm-AxP-GDL framework environment"

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

COPY environment.yml .
RUN conda env update --file environment.yml --name base && \
    conda init bash

# Set up the working directory
WORKDIR /opt/project

# Command
CMD ["/bin/bash"]