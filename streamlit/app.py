import streamlit as st
import requests
import os

# FastAPI のエンドポイント
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://fastapi:8000")

st.title("📈 投資判断エージェント")

# ユーザーが入力する銘柄コード
stock_code = st.text_input("銘柄コードを入力（例: 6501.T）", "6501.T")

# 分析実行ボタン_短期投資判断
if st.button("短期投資判断を実行"):
    st.info(f"🔍 {stock_code} の短期投資分析を実行中...")

    # FastAPI のエンドポイントを呼び出し
    response = requests.post(f"{FASTAPI_URL}/short_term_analysis_v3", json={"stock_code": stock_code})

    if response.status_code == 200:
        result = response.json()
        st.success("✅ 分析完了！")

        # 結果の表示
        st.subheader("📊 短期投資判断結果")
        st.markdown(result)

    else:
        st.error("❌ 分析に失敗しました。サーバーログを確認してください。")

# 分析実行ボタン_短期投資判断_ReAct（Reasoning + Acting） _Self-Reflective追加
#if st.button("短期投資判断を実行_AgentVerUP"):
#    st.info(f"🔍 {stock_code} の短期投資分析を実行中...")
#
#    # FastAPI のエンドポイントを呼び出し
#    response = requests.post(f"{FASTAPI_URL}/short_term_analysis", json={"stock_code": stock_code})
#
#    if response.status_code == 200:
#        result = response.json()
#        st.success("✅ 分析完了！")#
#
#        # 結果の表示
#        st.subheader("📊 短期投資判断結果")
#        st.markdown(result)
#        st.markdown(result["technical_summary"])
#
#    else:
#        st.error("❌ 分析に失敗しました。サーバーログを確認してください。")
