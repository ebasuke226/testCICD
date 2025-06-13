# ベースイメージにMinicondaを使用
FROM continuumio/miniconda3

# 必要なライブラリをインストール
RUN apt-get update && apt-get install -y \
    libssl-dev \
    libpq-dev \
    gcc \
    python3-dev \
    curl \
    postgresql-client \
    && apt-get clean

# conda-forgeチャンネルを追加
RUN conda config --add channels conda-forge

# Python環境とJupyter Notebookをインストール (MLflowは pip でインストール)
RUN conda create -n mlflow_jupyter_env python=3.11 jupyterlab scikit-learn pandas numpy matplotlib seaborn

# MLflowを conda 環境にインストール
RUN conda run -n mlflow_jupyter_env pip install mlflow psycopg2-binary gunicorn && \
    echo 'eval "$(conda shell.bash hook)"' >> ~/.bashrc && \
    echo 'conda activate mlflow_jupyter_env' >> ~/.bashrc && \
    echo 'export PATH=$CONDA_PREFIX/bin:$PATH' >> ~/.bashrc

# 作業ディレクトリを設定
WORKDIR /app/mlflow-tracking

# 環境変数を設定（PostgreSQL のパスワードを Docker の環境変数から取得）
ENV POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
RUN apt-get update && apt-get install -y git

# start.shをコピーして実行権限を付与
COPY mlflow-tracking /app/mlflow-tracking
RUN chmod -R 777 /app/mlflow-tracking
RUN chmod +x /app/mlflow-tracking/start.sh

# Conda環境をアクティベートして、MLflowサーバーを起動
CMD ["bash", "-c", "conda run -n mlflow_jupyter_env /app/mlflow-tracking/start.sh"]
