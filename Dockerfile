# Escolhe uma imagem base
FROM python:3.12-slim-bookworm

# Define o diretório de trabalho
WORKDIR /app

# Copia os arquivos de requirements
COPY requirements.txt /app/requirements.txt

# Instala as dependências
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o código
COPY . /app

# Comando default (sobreposto pelo Docker Compose)
CMD ["fastapi", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
