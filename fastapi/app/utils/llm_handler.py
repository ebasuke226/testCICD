import os
import google.generativeai as genai
import mlflow
# もし Git のコミットハッシュも記録したいなら
import subprocess

from .mlflow_tracking import track_llm_response

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def generate_llm_response(
    prompt: str,
    model_name: str = "gemini-1.5-flash",
    prompt_template_version: str = "v1.0",
    user_id: str = "anonymous"
):
    """
    LLM を呼び出してレスポンスを取得し、MLflow に各種パラメータ/メタ情報を記録する。
    引数:
        prompt: 実行時に組み立てた最終プロンプト
        model_name: 使用するGeminiのモデル名 (例: "gemini-1.5-flash")
        prompt_template_version: テンプレートのバージョン (Gitハッシュやv1.0など任意の文字列)
        user_id: ユーザIDやセッションIDなど、呼び出し主体を識別できる情報
    """

    # --- (1) Gitのコミットハッシュを取得（任意） ---
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        commit_hash = "unknown_commit"

    # --- (2) モデル呼び出し ---
    gemini_model = genai.GenerativeModel(model_name)

    # ここで、prompt とは別にシステムプロンプトを組み立てる場合もあるが、例では単純化
    response = gemini_model.generate_content(prompt)  # レスポンス本体

    # Gemini API の返却オブジェクトにトークン使用量やコストが含まれていれば、ここで取得
    #  (現状 Google Generative AI Python SDK ではトークン数の取得ができない場合が多いので仮コード)
    # usage_info = response["usage"] if "usage" in response else {}

    # --- (3) MLflow へ詳細ログを記録 ---
#    with mlflow.start_run():
#        mlflow.log_param("llm_model_name", model_name)
#        mlflow.log_param("prompt_template_version", prompt_template_version)
#        mlflow.log_param("user_id", user_id)
#        mlflow.log_param("git_commit_hash", commit_hash)

        # 例: llm_handler.py の generate_llm_response 内
#        snippet_length = 100  # 適宜調整
#        snippet = prompt[:snippet_length] + ("..." if len(prompt) > snippet_length else "")

#        mlflow.log_param("prompt_snippet", snippet)  # Parameters タブ用（短い抜粋）

#        mlflow.log_text(prompt, "prompt.txt")        # Artifacts タブ用（全文）

        # この prompt（最終的に LLM に渡したテキスト）を .txt ファイルとしてArtifact保存
#        mlflow.log_text(prompt, "prompt.txt")
        # LLMの返答を .txt として Artifact 保存
#        mlflow.log_text(response.text, "response.txt")

        # もし usage_info が取れるならログしておく
        # mlflow.log_metric("prompt_tokens", usage_info.get("prompt_tokens", 0))
        # mlflow.log_metric("completion_tokens", usage_info.get("completion_tokens", 0))
        # mlflow.log_metric("total_tokens", usage_info.get("total_tokens", 0))

    # 既存のトラッキングロジックを呼び出し（必要に応じて修正・除去）
    track_llm_response(prompt, response.text, model_name)

    return response.text
