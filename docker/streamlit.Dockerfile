# Streamlit の公式イメージをベースに構築
FROM python:3.9

# 作業ディレクトリの設定
WORKDIR /app

# 必要なパッケージのインストール
COPY requirements/streamlit.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションファイルのコピー
COPY streamlit/app.py /app/app.py

# Streamlitの実行
CMD ["streamlit", "run", "/app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
