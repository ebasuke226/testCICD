import yfinance as yf
import ta
import pandas as pd
import requests
from typing import Optional
import logging

# ロガーの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# セッションを作成し、User-Agentを設定
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Referer': 'https://finance.yahoo.com',
    'Origin': 'https://finance.yahoo.com'
})

# セッションのクッキーをクリア
session.cookies.clear()

def get_stock_technical_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """
    Yahoo Financeから株価データを取得し、テクニカル指標を計算する

    Args:
        ticker (str): 銘柄コード（例: "6501.T"）
        period (str): 取得期間（デフォルト: "6mo"）

    Returns:
        pd.DataFrame: テクニカル指標が追加されたDataFrame
    """
    try:
        stock = yf.Ticker(ticker, session=session)
        hist = stock.history(period=period)

        if hist.empty:
            logger.warning(f"⚠️ 株価データが取得できませんでした: {ticker}")
            return pd.DataFrame()

        # テクニカル指標の計算
        hist['SMA_5'] = ta.trend.sma_indicator(hist['Close'], window=5)
        hist['SMA_10'] = ta.trend.sma_indicator(hist['Close'], window=10)
        hist['SMA_20'] = ta.trend.sma_indicator(hist['Close'], window=20)
        hist['RSI'] = ta.momentum.RSIIndicator(hist['Close']).rsi()
        macd = ta.trend.MACD(hist['Close'])
        hist['MACD'] = macd.macd()
        hist['ATR'] = ta.volatility.AverageTrueRange(
            high=hist['High'], low=hist['Low'], close=hist['Close']
        ).average_true_range()

        logger.info(f"✅ 株価データ取得成功: {ticker}")
        return hist.tail(30)

    except Exception as e:
        logger.error(f"❌ 株価データ取得エラー: {e}")
        return pd.DataFrame()

def get_stock_news(ticker: str) -> str:
    """
    Yahoo Financeからニュースを取得する

    Args:
        ticker (str): 銘柄コード（例: "6501.T"）

    Returns:
        str: ニュースの要約
    """
    try:
        stock = yf.Ticker(ticker, session=session)
        news = stock.news

        if not news:
            return "⚠️ ニュースデータが取得できませんでした"

        # ニュースをフォーマットして返す
        formatted_news = "\n".join([
            f"- {item['title']}: {item.get('summary', '要約なし')}"
            for item in news[:5]  # 最新5件のニュースを取得
        ])

        logger.info(f"✅ ニュース取得成功: {ticker}")
        return formatted_news

    except Exception as e:
        logger.error(f"❌ ニュース取得エラー: {e}")
        return "⚠️ ニュースデータの取得に失敗しました" 