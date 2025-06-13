import yfinance as yf
import ta
import pandas as pd
import requests
from typing import Optional
import logging
import time
from functools import lru_cache
import os
from datetime import datetime, timedelta

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

# キャッシュディレクトリの設定
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_path(ticker: str, data_type: str) -> str:
    """キャッシュファイルのパスを生成"""
    return os.path.join(CACHE_DIR, f"{ticker}_{data_type}.csv")

def is_cache_valid(cache_path: str, max_age_hours: int = 1) -> bool:
    """キャッシュが有効かどうかをチェック"""
    if not os.path.exists(cache_path):
        return False
    cache_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
    return datetime.now() - cache_time < timedelta(hours=max_age_hours)

def save_to_cache(data: pd.DataFrame, cache_path: str):
    """データをキャッシュに保存"""
    data.to_csv(cache_path)

def load_from_cache(cache_path: str) -> pd.DataFrame:
    """キャッシュからデータを読み込み"""
    return pd.read_csv(cache_path, index_col=0, parse_dates=True)

def retry_with_backoff(func, max_retries=3, initial_delay=1):
    """リトライロジックを実装したデコレータ"""
    def wrapper(*args, **kwargs):
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
    return wrapper

@retry_with_backoff
def get_stock_technical_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """
    Yahoo Financeから株価データを取得し、テクニカル指標を計算する
    キャッシュとリトライロジックを実装

    Args:
        ticker (str): 銘柄コード（例: "6501.T"）
        period (str): 取得期間（デフォルト: "6mo"）

    Returns:
        pd.DataFrame: テクニカル指標が追加されたDataFrame
    """
    cache_path = get_cache_path(ticker, "technical")
    
    # キャッシュが有効な場合はキャッシュから読み込み
    if is_cache_valid(cache_path):
        logger.info(f"✅ キャッシュから株価データを読み込み: {ticker}")
        return load_from_cache(cache_path)

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

        # データをキャッシュに保存
        save_to_cache(hist, cache_path)
        
        logger.info(f"✅ 株価データ取得成功: {ticker}")
        return hist.tail(30)

    except Exception as e:
        logger.error(f"❌ 株価データ取得エラー: {e}")
        return pd.DataFrame()

@retry_with_backoff
def get_stock_news(ticker: str) -> str:
    """
    Yahoo Financeからニュースを取得する
    キャッシュとリトライロジックを実装

    Args:
        ticker (str): 銘柄コード（例: "6501.T"）

    Returns:
        str: ニュースの要約
    """
    cache_path = get_cache_path(ticker, "news")
    
    # キャッシュが有効な場合はキャッシュから読み込み
    if is_cache_valid(cache_path, max_age_hours=4):  # ニュースは4時間キャッシュ
        logger.info(f"✅ キャッシュからニュースを読み込み: {ticker}")
        return load_from_cache(pd.DataFrame({'news': [pd.read_csv(cache_path).iloc[0, 0]]})).iloc[0, 0]

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

        # ニュースをキャッシュに保存
        pd.DataFrame({'news': [formatted_news]}).to_csv(cache_path)
        
        logger.info(f"✅ ニュース取得成功: {ticker}")
        return formatted_news

    except Exception as e:
        logger.error(f"❌ ニュース取得エラー: {e}")
        return "⚠️ ニュースデータの取得に失敗しました" 