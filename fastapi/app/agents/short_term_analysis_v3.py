import os
import json
import requests
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from app.utils.stock_data import get_stock_technical_data, get_stock_news
from app.utils.llm_handler import generate_llm_response
from app.utils.rag_handler import retrieve_relevant_info
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import feedparser

def fetch_additional_context_from_RAG(state: Dict[str, Any]) -> Dict[str, Any]:
    stock_code = state.get("stock_code")
    model_prediction = state.get("model_prediction", "推論結果なし")

    query = f"{stock_code} {model_prediction}"
    relevant_info = retrieve_relevant_info(query=query, top_k=3)

    prompt = f"""
    あなたは金融市場に詳しい投資アナリストです。
    以下の追加情報を読み、短期投資に影響するポイントをまとめてください。

    ### 【追加情報】
    {relevant_info}

    ### 【出力フォーマット】
    - 【追加分析結果】簡潔な要約
    - 【リスク・注意点】短期投資におけるリスク要因を記載
    """

    additional_summary = generate_llm_response(
        prompt,
        model_name="gemini-1.5-flash",
        prompt_template_version="v2.3",  # バージョンなど指定
        user_id="agent1"
    )

    state["additional_summary"] = additional_summary
    return state

# デバッグ用関数
def debug_print(title, data):
    print(f"\n=== {title} ===")
    print(data)

# 🔹 Google News RSS からニュース取得
def google_news_search(query: str):
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    
    articles = []
    for entry in feed.entries[:10]:  # 最新の3件を取得
        articles.append(f"{entry.title}: {entry.link}")
    
    return "\n".join(articles) if articles else "追加ニュースが見つかりませんでした。"


def summarize_technical_analysis(state: Dict[str, Any]) -> Dict[str, Any]:
    print(f"\n📌 {state['stock_code']} のテクニカル分析開始...")
    stock_data = state.get("technical_data", pd.DataFrame())
    if stock_data is None or stock_data.empty:
        print(f"⚠️ 株価データが取得できませんでした: {state['stock_code']}")
        return {**state, "technical_summary": "データが取得できませんでした"}
    
    latest_data = stock_data.iloc[-1]
    prompt = f"""
    あなたは投資アナリストです。
    以下のテクニカル指標に基づき、この銘柄の短期投資判断を要約してください。

    ### 【テクニカル指標】
    - **5日移動平均線**: {latest_data['SMA_5']:.2f}
    - **10日移動平均線**: {latest_data['SMA_10']:.2f}
    - **20日移動平均線**: {latest_data['SMA_20']:.2f}
    - **RSI (相対力指数)**: {latest_data['RSI']:.2f}
    - **MACD**: {latest_data['MACD']:.2f}
    - **ATR (ボラティリティ指標)**: {latest_data['ATR']:.2f}

    ### 【タスク】
    1. 短期のトレンドを評価してください（上昇・下落・横ばい）。
    2. 指標の組み合わせから、エントリーポイントの推奨をしてください。

    ### 【出力フォーマット】
    - 【短期投資判断】上昇傾向 / 下落傾向 / 横ばい
    - 【理由】簡潔に説明
    - 【リスク要因】変動要因を記載
    """
    technical_summary = generate_llm_response(
        prompt,
        model_name="gemini-1.5-flash",
        prompt_template_version="v2.3",  # バージョンなど指定
        user_id="agent1"
    )

    print("=== テクニカル分析結果 ===")
    print(technical_summary)
    return {**state, "technical_summary": technical_summary}

# --------------------------------------------------
# 【ニュース不足時】に google_news_search を利用して追加情報取得
def react_based_news_analysis(state: Dict[str, Any]) -> Dict[str, Any]:
    ticker = state["stock_code"]
    news_summary = state.get("news_summary", "")
    if "データが取得できませんでした" in news_summary or len(news_summary) < 100:
        print(f"⚠️ {ticker} のニュース情報が不足しています。追加取得を試みます。")
        additional_news = google_news_search(ticker)
        prompt = f"""
        以下のニュース要約は情報が不足している可能性があります:
        
        【現在のニュース要約】:
        {news_summary}

        【追加のニュース情報】:
        {additional_news}

        追加情報を考慮して、最終的なニュースセンチメントを再評価してください。

        ### 【出力フォーマット】
        - 【センチメント】ポジティブ / ネガティブ / 中立
        - 【理由】簡潔に説明
        - 【リスク要因】考慮すべきポイント
        """
        updated_news = generate_llm_response(
        prompt,
        model_name="gemini-1.5-flash",
        prompt_template_version="v2.3",  # バージョンなど指定
        user_id="agent1"
    )

        return {**state, "news_summary": updated_news}
    return state

def summarize_news(state: Dict[str, Any]) -> Dict[str, Any]:
    # get_stock_news() によりニュース原文を取得
    ticker = state["stock_code"]
    combined_news = get_stock_news(ticker)
    prompt = f"""
    あなたは投資アナリストです。
    以下の企業ニュースに基づき、市場のセンチメントを要約してください。

    ### 【ニュース概要】
    {combined_news}

    ### 【タスク】
    1. ポジティブなニュース、ネガティブなニュースを分類してください。
    2. これらのニュースが短期的な株価変動に与える影響を分析してください。

    ### 【出力フォーマット】
    - 【センチメント】ポジティブ / ネガティブ / 中立
    - 【理由】簡潔に説明
    - 【リスク要因】考慮すべき要素
    """
    news_summary = generate_llm_response(
        prompt,
        model_name="gemini-1.5-flash",
        prompt_template_version="v2.3",
        user_id="agent1"
    )

    print("=== ニュース分析結果 ===")
    print(news_summary)
    return {**state, "news_summary": news_summary}

def final_investment_evaluation(state: Dict[str, Any]) -> Dict[str, Any]:
    technical_summary = state.get("technical_summary", "テクニカル分析情報なし")
    news_summary = state.get("news_summary", "ニュース情報なし")
    model_pred = state.get("model_prediction", "モデル予測なし")
    additional_context = state.get("additional_summary", "追加情報なし")

    prompt = f"""
    あなたは経験豊富な投資アナリストです。
    以下の情報をもとに、短期投資評価を行ってください。

    ### 【テクニカル分析結果】
    {technical_summary}

    ### 【ニュースセンチメント分析結果】
    {news_summary}

    ### 【モデル予測結果】
    {model_pred}

    ### 【追加の参考情報（RAG）】
    {additional_context}

    ### 【タスク】
    1. 短期投資評価を10段階スコア（1: 非常に悪い 〜 10: 非常に良い）で提示してください。
    2. その理由と短期リスク要因を簡潔に記述してください。

    ### 【出力フォーマット】
    - 【短期投資評価】スコア
    - 【理由】
    - 【リスク要因】
    """
    final_eval = generate_llm_response(
        prompt,
        model_name="gemini-1.5-flash",
        prompt_template_version="v2.3",  # バージョンなど指定
        user_id="agent1"
    )

    print("=== 最終投資評価 ===")
    print(final_eval)
    return {**state, "final_evaluation": final_eval}

def reflect_on_evaluation(state: Dict[str, Any]) -> Dict[str, Any]:
    # Reflectionエージェントで最終評価結果の再検討を行う
    original_eval = state.get("final_evaluation", "")
    prompt = f"""
    あなたは熟練の投資アナリストです。
    以下は先ほどの短期投資評価です。これを踏まえ、さらに検討して、評価の妥当性や補足すべきリスク要因があれば再評価してください。

    ### 【元の短期投資評価】
    {original_eval}

    ### 【タスク】
    1. 元の評価に基づいて、評価の改善点があれば具体的に指摘してください。
    2. 必要であれば、評価を修正し、再評価の結果を10段階スコアで示してください。
    3. 補足の理由やリスク要因も記載してください。

    ### 【出力フォーマット】
    - 【再評価短期投資評価】スコア（1〜10）

    - 【評価理由】詳細な説明

    - 【補足リスク要因】考慮すべきポイント

    **※出力は、読みやすいように適切な改行を入れてください。**

    """
    reflection = generate_llm_response(
        prompt,
        model_name="gemini-1.5-flash",
        prompt_template_version="v2.3",  # バージョンなど指定
        user_id="agent1"
    )

    print("=== 再評価（Reflection） ===")
    print(reflection)
    return {**state, "final_evaluation": reflection}

############################
# ① MLflow Tracking/Models/Registry の設定（既存部分）
############################

# MLflow の Tracking URI と Artifact URI を Tracking 用コンテナの設定に合わせる
os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow-tracking:5003"
#os.environ["MLFLOW_ARTIFACT_URI"] = "models:/Stock_Chart_Classification_Model/Production"
os.environ["MLFLOW_ARTIFACT_URI"] = "/app/mlflow-tracking/artifacts"
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

print("MLFLOW_TRACKING_URI:", os.getenv("MLFLOW_TRACKING_URI"))
print("MLFLOW_ARTIFACT_URI:", os.getenv("MLFLOW_ARTIFACT_URI"))

client = mlflow.tracking.MlflowClient()
experiment_name = "Stock_Chart_Classification_3"
experiment = client.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = client.create_experiment(
        name=experiment_name,
        artifact_location=os.getenv("MLFLOW_ARTIFACT_URI")
    )
    print(f"新しい Experiment '{experiment_name}' を作成しました。")
else:
    experiment_id = experiment.experiment_id
    print(f"既存の Experiment '{experiment_name}' を使用します。")
mlflow.set_experiment(experiment_name)

############################
# ③ 新たに追加するエージェント：MLflow にサービング中のモデルによる推論
############################

# 数値ラベルとカテゴリ名の対応を辞書として定義
category_mapping = {
    0: "これから上昇？", 1: "しっかり?", 2: "そろそろ天井?", 3: "まだ上昇?",
    4: "まだ下落?", 5: "もみ合い?", 6: "リバウンド?", 7: "上昇?",
    8: "上昇ストップ？", 9: "上昇一服?", 10: "上昇基調?", 11: "下げとまった？",
    12: "下げ渋る?", 13: "下押す?", 14: "下落?", 15: "下落ストップ？",
    16: "下落基調?", 17: "売り？", 18: "弱含み?", 19: "強含み?",
    20: "急上昇?", 21: "急落?", 22: "戻ってくる？", 23: "戻らない？",
    24: "行って来い?"
}

def predict_stock_category(processed_df: pd.DataFrame) -> str:
    """
    processed_df は各ティッカーの特徴量を含む DataFrame（例：カラムは
    ["Ticker", "Category", "all_騰落率", "avg_Volume", "Day1_騰落率", ..., "Day22_騰落率"]）であると仮定。
    モデルには Ticker, Category 以外の数値特徴量を入力すると想定。
    """
    # 推論に必要な特徴量だけ抽出（例：Ticker, Categoryは除く）
    feature_columns = [col for col in processed_df.columns if col not in ["Ticker","Category"]]
#    if "Category" in processed_df.columns:
#        processed_df["Category"] = pd.to_numeric(processed_df["Category"], errors="coerce").fillna(1)
    input_df = processed_df[feature_columns]
#    print("学習時の特徴量:", model.metadata.get_input_schema())
    print("現在の入力データの特徴量:", input_df.columns.tolist())

    # **MLflow モデルのロード**
    model_uri = "models:/Stock_Chart_Classification_Model/Production"  # 適宜変更
    model = None  # 初期化
    try:
        print(f"🔄 MLflow からモデルをロード中: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        print("✅ モデルロード成功")

        # **モデルのロードが成功した後に特徴量を表示**
        print("学習時の特徴量:", model.metadata.get_input_schema())

    except Exception as e:
        print("❌ モデルロードエラー:", e)
        raise Exception(f"Model loading failed: {str(e)}")

    # **モデルのロードが成功した場合のみ予測を行う**
    if model is None:
        raise ValueError("❌ モデルがロードされていません。MLflow から正しく取得できているか確認してください。")

    # **モデルの推論**
    try:
        predictions = model.predict(input_df)
        # **予測結果を数値ラベルから日本語ラベルに変換**
        predictions_mapped = [category_mapping.get(int(pred), "未知のカテゴリ") for pred in predictions]
        processed_df["Prediction"] = predictions_mapped
        print("✅ モデル推論成功:", predictions_mapped)
        return predictions_mapped[0]  # 🔄 データフレームを返さず、リストのみ返す

#        predictions = model.predict(input_df)
#        processed_df["Prediction"] = predictions
#        print("✅ モデル推論成功:", predictions)
#        return processed_df
    except Exception as e:
        print("❌ モデル推論エラー:", e)
        raise Exception(f"Model inference failed: {str(e)}")

def model_inference_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph のパイプライン内のノードとして、state に
    "processed_data"（pandas DataFrame）が含まれている前提で、モデル推論を実施する。
    推論結果は state["model_prediction"] に格納する。
    """
    processed_data = state.get("processed_data")
    if processed_data is None or processed_data.empty:
        print("予測対象の processed_data が存在しません。")
        state["model_prediction"] = None
        return state

    try:
        prediction_df = predict_stock_category(processed_data)
        state["model_prediction"] = prediction_df
        print("✅ 推論結果を取得しました。")
    except Exception as e:
        state["model_prediction"] = str(e)
        print("推論エラー:", e)
    return state

# --------------------------------------------------
# 【processed_data】生成ノード（technical_dataから単純コピー例）
def create_processed_data(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    1. Yahoo Financeから株価データを取得しDataFrame化する（get_stock_technical_data() を利用）
    2. DataFrameから特徴量を生成する
       - 騰落率(Return)の計算
       - 全体の騰落率 (all_騰落率)
       - 平均出来高 (avg_Volume)
       - 各日（Day1_～Day22_騰落率）の騰落率
       ※ Category は state に "category" キーがあれば利用し、なければ "Unknown"
    3. 生成した特徴量の1行 DataFrame を元に、MLflow のモデル（StockCategoryModel の Production ステージ）をローカルロードして予測を実行する
    4. 予測結果（例：チャート情報など）を state["model_prediction"] に格納して返す
    """
    import pandas as pd
    ticker = state.get("stock_code")
    if not ticker:
        print("Error: 'stock_code' が state に存在しません。")
        state["processed_data"] = pd.DataFrame()
        state["model_prediction"] = None
        return state

    # --- 1. データの取得 ---
    stock_data = get_stock_technical_data(ticker)
    if stock_data is None or stock_data.empty:
        print(f"株価データが取得できませんでした: {ticker}")
        state["processed_data"] = pd.DataFrame()
        state["model_prediction"] = None
        return state

    # --- 2. データの変換（特徴量生成） ---
    # 騰落率の計算
    stock_data['Return'] = stock_data['Close'].pct_change()
    # 1行目は NaN となるため、以降の値をリスト化（例：Day1～DayN の騰落率）
    return_rates = stock_data['Return'].iloc[1:].tolist()
    # 全体の騰落率：初値から最終値までの変化率
    all_return_rate = (stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0]) - 1

    # 特徴量の生成
    processed_data = pd.DataFrame({
        'Ticker': [ticker],
        'Category': [state.get('category', 'Unknown')],
        'all_騰落率': [all_return_rate],
        'avg_Volume': [stock_data['Volume'].mean()]
    })

    # 各日の騰落率を追加
    for i, rate in enumerate(return_rates[:22], 1):  # 最大22日分
        processed_data[f'Day{i}_騰落率'] = [rate]

    state["processed_data"] = processed_data
    return state

# LangGraph のワークフロー構築
graph = StateGraph(Dict[str, Any])
graph.add_node("fetch_technical_data", lambda state: {**state, "technical_data": get_stock_technical_data(state["stock_code"])})
graph.add_node("summarize_technical_analysis", summarize_technical_analysis)
graph.add_node("summarize_news", summarize_news)
graph.add_node("react_based_news_analysis", react_based_news_analysis)
graph.add_node("create_processed_data", create_processed_data)
graph.add_node("model_inference", model_inference_node)
graph.add_node("fetch_additional_context_from_RAG", fetch_additional_context_from_RAG)
graph.add_node("final_investment_evaluation", final_investment_evaluation)
graph.add_node("reflect_on_evaluation", reflect_on_evaluation)

graph.set_entry_point("fetch_technical_data")
graph.add_edge("fetch_technical_data", "summarize_technical_analysis")
graph.add_edge("summarize_technical_analysis", "summarize_news")
graph.add_edge("summarize_news", "react_based_news_analysis")
graph.add_edge("react_based_news_analysis", "create_processed_data")
graph.add_edge("create_processed_data", "model_inference")
graph.add_edge("model_inference", "fetch_additional_context_from_RAG")
graph.add_edge("fetch_additional_context_from_RAG", "final_investment_evaluation")
graph.add_edge("final_investment_evaluation", "reflect_on_evaluation")
graph.add_edge("reflect_on_evaluation", END)


# コンパイルしてエージェントを生成
short_term_agent = graph.compile()

def run_short_term_analysis_v3(stock_code: str):
    initial_state = {"stock_code": stock_code}
    result = short_term_agent.invoke(initial_state)
    # 最終評価（LLM による評価）とともに、モデル推論結果も取得
    final_eval = result.get("final_evaluation", "評価が取得できませんでした")
    model_pred = result.get("model_prediction", "推論結果が取得できませんでした")
    return final_eval, model_pred
