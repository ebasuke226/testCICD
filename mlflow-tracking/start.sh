#!/bin/bash

# Git 連携を無効化
export MLFLOW_DISABLE_ENV_MANAGER=true
export GIT_PYTHON_GIT_EXECUTABLE=""
export GIT_PYTHON_REFRESH=quiet

# conda環境をアクティベート
conda activate mlflow_jupyter_env
export PATH=$CONDA_PREFIX/bin:$PATH

# デバッグログを有効化
export MLFLOW_TRACKING_DEBUG=1

echo "Starting MLFlow database migration..."
# データベースのマイグレーションを実行
mlflow db upgrade postgresql://mlflow:${POSTGRES_PASSWORD}@postgres/mlflow

echo "Starting MLFlow server..."
# MLflow Tracking Server を起動（ポートを 5003 に変更）
mlflow server \
    --backend-store-uri postgresql://mlflow:${POSTGRES_PASSWORD}@postgres/mlflow \
    --default-artifact-root /app/mlflow-tracking/artifacts \
    --host 0.0.0.0 \
    --port 5003 \
    --workers 1
