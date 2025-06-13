FROM python:3.9-slim

WORKDIR /app

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# MLflowとその依存関係をインストール
RUN pip install --no-cache-dir \
    mlflow==2.9.1 \
    psycopg2-binary==2.9.9 \
    boto3==1.34.69 \
    scikit-learn==1.4.1

# ポートを公開
EXPOSE 5001

# MLflow Model Servingを起動
CMD ["mlflow", "models", "serve", "--model-uri", "models:/Stock_Chart_Classification_Model/Production", "--host", "0.0.0.0", "--port", "5001"] 