FROM jupyter/scipy-notebook

USER root

# 必要なライブラリをインストール
COPY requirements/jupyter.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/jupyter.txt

# MySQLクライアントをインストール
RUN apt-get update --allow-releaseinfo-change && \
    apt-get install -y default-mysql-client && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# カスタムエントリポイントスクリプトを作成
COPY docker/entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

# Jupyterのユーザーディレクトリのパーミッションを設定
RUN mkdir -p /home/jovyan/.local/share && \
    chown -R jovyan:users /home/jovyan/.local && \
    chmod -R 755 /home/jovyan/.local

USER $NB_UID

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["start-notebook.sh", "--NotebookApp.token=''"]
