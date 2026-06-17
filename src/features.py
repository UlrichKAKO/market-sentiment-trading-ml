import numpy as np
import pandas as pd


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_market_features(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])

    g = df.groupby("ticker")

    df["return"] = g["close"].transform(lambda x: np.log(x / x.shift(1)))
    df["return_lag_1"] = g["return"].shift(1)
    df["return_lag_2"] = g["return"].shift(2)
    df["volatility_5d"] = g["return"].transform(lambda x: x.rolling(5).std())
    df["volatility_20d"] = g["return"].transform(lambda x: x.rolling(20).std())
    df["momentum_5d"] = g["close"].transform(lambda x: x / x.shift(5) - 1)
    df["volume_change"] = g["volume"].pct_change()
    df["rsi_14"] = g["close"].transform(compute_rsi)

    df["target"] = (g["return"].shift(-1) > 0).astype(int)

    return df


def merge_market_and_sentiment(market: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    df = market.copy()
    df["date"] = pd.to_datetime(df["date"])

    sentiment = sentiment.copy()
    sentiment["date"] = pd.to_datetime(sentiment["date"])

    if "ticker" in sentiment.columns:
        df = df.merge(sentiment, on=["date", "ticker"], how="left")
    else:
        df = df.merge(sentiment, on="date", how="left")

    cols = ["sentiment_mean", "sentiment_std", "sentiment_count"]
    df[cols] = df[cols].fillna(0)
    df = df.sort_values(["ticker", "date"])
    g = df.groupby("ticker")
    df["sentiment_lag_1"] = g["sentiment_mean"].shift(1)
    df["sentiment_rolling_3d"] = g["sentiment_mean"].transform(lambda x: x.rolling(3).mean())
    df["sentiment_rolling_7d"] = g["sentiment_mean"].transform(lambda x: x.rolling(7).mean())
    return df.dropna().reset_index(drop=True)
