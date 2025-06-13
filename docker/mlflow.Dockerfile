FROM python:3.9-slim

WORKDIR /app

# 必要なパッケージをインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# MLflowとその依存関係をインストール
RUN pip install --no-cache-dir \
    mlflow==2.9.1 \
    psycopg2-binary==2.9.9 \
    boto3==1.34.69

# ポートの公開
EXPOSE 5000

# 起動コマンド
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"] 