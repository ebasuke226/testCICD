import logging
from app.agents.short_term_analysis import run_short_term_analysis
from app.agents.short_term_analysis_v3 import run_short_term_analysis_v3
from fastapi import FastAPI, Request, Query
from pydantic import BaseModel  # PydanticのBaseModelをインポート

# デバッグログを有効化
logging.basicConfig(level=logging.DEBUG)

# リクエストデータのスキーマを定義
class StockAnalysisRequest(BaseModel):
    stock_code: str

app = FastAPI()

@app.get("/")
async def root():
    logging.debug("FastAPIアプリが起動しました。")
    return {"message": "FastAPI is running"}

from fastapi import Request

@app.post("/short_term_analysis")
async def short_term_analysis(request: Request, payload: StockAnalysisRequest):
    logging.debug(f"受信したリクエスト: {await request.json()}")
    
    result = run_short_term_analysis(payload.stock_code)
    return result

@app.post("/short_term_analysis_v3")
async def short_term_analysis_v3(request: Request, payload: StockAnalysisRequest):
    logging.debug(f"受信したリクエスト: {await request.json()}")
    
    result = run_short_term_analysis_v3(payload.stock_code)
    return result

# ✅ RAG（類似検索 & LLM での回答生成）エンドポイント
@app.get("/rag")
def query_rag(question: str = Query(..., title="質問内容")):
    """ ChromaDB から類似データを検索し、Gemini に投げる """

    logging.info(f"RAGリクエスト受信: {question}")

    # 1️⃣ 質問の Embedding を取得
    embedding = openai.Embedding.create(
        input=[question], model="text-embedding-ada-002"
    )["data"][0]["embedding"]

    # 2️⃣ ChromaDB で類似ドキュメントを検索
    results = collection.query(query_embeddings=[embedding], n_results=3)
    retrieved_texts = [doc["document"] for doc in results["documents"]]

    if not retrieved_texts:
        return {"answer": "関連する情報が見つかりませんでした。"}

    # 3️⃣ LLM (Gemini) へのプロンプト作成
    context = "\n".join(retrieved_texts)
    prompt = f"{context}\n\nQ: {question}\nA:"

    # 4️⃣ Gemini で回答を生成
    response = openai.ChatCompletion.create(
        model="gemini-1.5-flash", messages=[{"role": "user", "content": prompt}]
    )

    answer = response["choices"][0]["message"]["content"]

    # 5️⃣ MLflow に記録
    with mlflow.start_run():
        mlflow.log_param("question", question)
        mlflow.log_text(answer, "response.txt")
        mlflow.end_run()

    return {"answer": answer}
