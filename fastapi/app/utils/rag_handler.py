from app.utils.llm_handler import generate_llm_response
import chromadb
import google.generativeai as genai
import os

# 環境変数からGemini APIキー取得
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
client = chromadb.HttpClient(host="chromadb", port=8000)

# コレクションの設定
collection_settings = {
    "name": "daily_docs",
    "metadata": {"hnsw:space": "cosine"},
    "embedding_function": None  # カスタム埋め込み関数を使用するため
}

collections = {
    "daily": client.get_or_create_collection(
        name="daily_docs",
        metadata={"hnsw:space": "cosine"}
    ),
    "weekly": client.get_or_create_collection(
        name="weekly_docs",
        metadata={"hnsw:space": "cosine"}
    ),
    "monthly": client.get_or_create_collection(
        name="monthly_docs",
        metadata={"hnsw:space": "cosine"}
    ),
}

EMBEDDING_MODEL = "models/text-embedding-004"

def get_gemini_embedding(text):
    try:
        response = genai.embed_content(model=EMBEDDING_MODEL, content=text)
        return response["embedding"]
    except Exception as e:
        print(f"❌ 埋め込みエラー: {str(e)}")
        return None

def retrieve_relevant_info(query: str, top_k=3):
    """RAGでChromaDBから関連情報を取得"""
    query_embedding = get_gemini_embedding(query)
    if query_embedding is None:
        return "🔍 クエリ埋め込み取得失敗。"

    results = collections["daily"].query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    if not results.get("documents"):
        return "🔍 関連情報なし"

    retrieved_docs = []
    for i in range(len(results["documents"][0])):
        retrieved_docs.append(f"{results['documents'][0][i]}\n")

    return "\n---\n".join(retrieved_docs)
