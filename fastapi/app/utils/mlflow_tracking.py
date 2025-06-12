import mlflow
import os
import subprocess

# 📌 **MLflow の設定**
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
MLFLOW_EXPERIMENT_NAME = "LLM_Tracking_"
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# 🚀 FastAPIコンテナでは `/app` がプロジェクトのルート
PROJECT_ROOT = "/app"

# 📌 **Git の情報を取得**
def get_git_commit_hash():
    """現在の Git のコミットハッシュを取得"""
    try:
#        repo = git.Repo(PROJECT_ROOT, search_parent_directories=True)
#        return repo.head.object.hexsha
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
    except Exception:
        return "unknown"

def get_git_branch():
    """現在の Git のブランチ名を取得"""
    try:
        return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode("utf-8")
#        repo = git.Repo(PROJECT_ROOT, search_parent_directories=True)
#        return repo.active_branch.name
    except Exception:
        return "unknown"

def track_llm_response(prompt: str, response: str, model_name="gemini-1.5-flash"):
    """ LLM のリクエストとレスポンスを MLflow に記録 """
    with mlflow.start_run(nested=True):  # ✅ 変更: nested=True を統一
        # 🔹 Git の情報を MLflow に記録
        git_commit = get_git_commit_hash()
        git_branch = get_git_branch()
        
        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("git_branch", git_branch)
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("task", "llm_analysis")
        # 例: llm_handler.py の generate_llm_response 内
        snippet_length = 100  # 適宜調整
        snippet = prompt[:snippet_length] + ("..." if len(prompt) > snippet_length else "")

        # ✅ `log_param()` も使用して Parameters に表示させる
        mlflow.log_param("git_commit", git_commit)
        mlflow.log_param("git_branch", git_branch)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("task", "llm_analysis")
        mlflow.log_param("prompt_snippet", snippet)  # Parameters タブ用（短い抜粋）

        mlflow.log_text(prompt, "prompt.txt")
        mlflow.log_text(response, "response.txt")

        mlflow.end_run()
