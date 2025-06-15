FROM python:3.11-slim

WORKDIR /app

# Pythonパスを明示的に設定して、appモジュールをimport可能にする
ENV PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["pytest", "--cov=app", "--cov-report=term-missing"]
