# Imagem base leve e compatível
FROM python:3.11-slim

# Define diretório de trabalho
WORKDIR /app

# Copia tudo para dentro do container
COPY . .

# Instala dependências de sistema necessárias
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Instala dependências Python
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Comando de inicialização
CMD ["python", "app.py"]
