FROM python:3.11-slim

# 作業ディレクトリを /app に変更
WORKDIR /app

# 必要なライブラリをインストール
RUN apt-get install -y gnupg ca-certificates && \
    apt-get update --allow-unauthenticated || true && \
    apt-get install -y --no-install-recommends --allow-unauthenticated \
        gcc \
        python3.11-dev \
        libpq-dev \
        git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 環境変数を設定（FastAPI の app モジュールを認識させる）
ENV PYTHONPATH=/app

# 依存関係をインストール
COPY requirements/fastapi.txt /tmp/
RUN pip install --no-cache-dir --use-deprecated=legacy-resolver --timeout 1000 -r /tmp/fastapi.txt --no-deps && \
    pip install --no-cache-dir --timeout 1000 \
        frozendict \
        peewee \
        soupsieve \
        platformdirs \
        multitasking \
        lxml \
        appdirs

# FastAPI のソースコードをコピー
COPY fastapi/app /app/app

# ポートを公開
EXPOSE 8000

# FastAPIのDockerfile (fastapi.Dockerfile)
COPY docker/client.json /app/client.json

# 環境変数として設定
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/client.json"

# エントリーポイントを修正（app.main:app に変更）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "debug"]
