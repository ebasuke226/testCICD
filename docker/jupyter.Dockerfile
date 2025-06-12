FROM jupyter/scipy-notebook

USER root

# 必要なライブラリをインストール
COPY requirements/jupyter.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/jupyter.txt

# MySQLクライアントをインストール
RUN apt-get update --allow-releaseinfo-change && \
    apt-get install -y default-mysql-client && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

USER $NB_UID
