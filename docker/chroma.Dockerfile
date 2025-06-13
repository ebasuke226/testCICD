FROM python:3.9-slim

WORKDIR /app

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ChromaDBとその依存関係をインストール
RUN pip install --no-cache-dir \
    chromadb==0.4.22 \
    uvicorn==0.27.1 \
    numpy==1.24.4

# ポートを公開
EXPOSE 8000

# ChromaDBを起動
CMD ["uvicorn", "chromadb.app:app", "--host", "0.0.0.0", "--port", "8000"] 