import os
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from app.utils.stock_data import get_stock_technical_data, get_stock_news
from app.utils.llm_handler import generate_llm_response
import pandas as pd

# デバッグ用関数
def debug_print(title, data):
    print(f"\n=== {title} ===")
    print(data)

# 🔹 Google News RSS からニュース取得
def google_news_search(query: str):
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    
    articles = []
    for entry in feed.entries[:3]:  # 最新の3件を取得
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
    technical_summary = generate_llm_response(prompt)
    print("=== テクニカル分析結果 ===")
    print(technical_summary)
    return {**state, "technical_summary": technical_summary}

# 🔹 ニュース不足時に Google News RSS から追加情報を取得
def react_based_news_analysis(state: Dict[str, Any]) -> Dict[str, Any]:
    ticker = state["stock_code"]
    news_summary = state["news_summary"]

    # ニュースが不足しているかをチェック（簡易的に「ニュースが少ない場合」を想定）
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
        updated_news = generate_llm_response(prompt)
        return {**state, "news_summary": updated_news}
    
    return state  # 十分なニュースがあればそのまま


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
    news_summary = generate_llm_response(prompt)
    print("=== ニュース分析結果 ===")
    print(news_summary)
    return {**state, "news_summary": news_summary}

def final_investment_evaluation(state: Dict[str, Any]) -> Dict[str, Any]:
    technical_summary = state.get("technical_summary", "")
    news_summary = state.get("news_summary", "")
    prompt = f"""
    あなたは経験豊富な投資アナリストです。
    以下のテクニカル分析とニュースセンチメントを統合し、短期投資評価を行ってください.

    ### 【テクニカル分析結果】
    {technical_summary}

    ### 【ニュースセンチメント分析結果】
    {news_summary}

    ### 【タスク】
    1. これらの情報をもとに、投資評価を10段階スコア（1: 非常に悪い 〜 10: 非常に良い）で出してください.
    2. 短期のリスク要因をリストアップしてください.

    ### 【出力フォーマット】
    - 【短期投資評価】スコア（1〜10）
    - 【理由】簡潔に説明
    - 【リスク要因】考慮すべきポイント
    """
    final_eval = generate_llm_response(prompt)
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
    - 【再評価理由】詳細な説明
    - 【補足リスク要因】考慮すべきポイント
    """
    reflection = generate_llm_response(prompt)
    print("=== 再評価（Reflection） ===")
    print(reflection)
    return {**state, "final_evaluation": reflection}

# 🔹 LangGraph のワークフロー構築
graph = StateGraph(Dict[str, Any])
graph.add_node("fetch_technical_data", lambda state: {**state, "technical_data": get_stock_technical_data(state["stock_code"])})
graph.add_node("summarize_technical_analysis", summarize_news)
graph.add_node("analyze_news", summarize_news)
graph.add_node("react_based_news_analysis", react_based_news_analysis)
graph.add_node("final_investment_evaluation", final_investment_evaluation)
graph.add_node("reflect_on_evaluation", reflect_on_evaluation)

graph.set_entry_point("fetch_technical_data")
graph.add_edge("fetch_technical_data", "summarize_technical_analysis")
graph.add_edge("summarize_technical_analysis", "analyze_news")
graph.add_edge("analyze_news", "react_based_news_analysis")
graph.add_edge("react_based_news_analysis", "final_investment_evaluation")
graph.add_edge("final_investment_evaluation", "reflect_on_evaluation")
graph.add_edge("reflect_on_evaluation", END)

short_term_agent = graph.compile()

def run_short_term_analysis(stock_code: str):
    initial_state = {"stock_code": stock_code}
    result = short_term_agent.invoke(initial_state)
    return result.get("final_evaluation", "データが取得できませんでした")
