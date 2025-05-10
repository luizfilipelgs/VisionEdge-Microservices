FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Define diretório de trabalho
WORKDIR /app

# Instala Python e dependências básicas
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3-pip \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Define aliases
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Copia o projeto
COPY . .

# Instala dependências Python
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Comando de inicialização
CMD ["python", "app.py"]
