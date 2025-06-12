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

# エージェントのワークフロー構築
from langgraph.graph import StateGraph, END

graph = StateGraph(Dict[str, Any])
graph.add_node("fetch_technical_data", lambda state: {**state, "technical_data": get_stock_technical_data(state["stock_code"])})
graph.add_node("summarize_technical_analysis", summarize_technical_analysis)
graph.add_node("analyze_news", summarize_news)
graph.add_node("final_investment_evaluation", final_investment_evaluation)

graph.set_entry_point("fetch_technical_data")
graph.add_edge("fetch_technical_data", "summarize_technical_analysis")
graph.add_edge("summarize_technical_analysis", "analyze_news")
graph.add_edge("analyze_news", "final_investment_evaluation")
graph.add_edge("final_investment_evaluation", END)

short_term_agent = graph.compile()

def run_short_term_analysis(stock_code: str):
    initial_state = {"stock_code": stock_code}
    result = short_term_agent.invoke(initial_state)
    return result.get("final_evaluation", "データが取得できませんでした")
