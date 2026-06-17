import os
import pandas as pd

from src.config import (
    TICKERS, START_DATE, END_DATE, TRANSACTION_COST, TRAIN_SIZE,
    MARKET_FEATURES, SENTIMENT_FEATURES, EURONEXT_DATA_DIRS,
)
from src.data_loader import load_multiple_prices
from src.euronext_loader import build_euronext_microstructure_from_dirs
from src.sentiment import score_news_sentiment, aggregate_daily_sentiment
from src.features import build_market_features, merge_market_and_sentiment
from src.models import time_split, get_models, evaluate_classifier
from src.backtest import run_backtest, financial_metrics


def load_or_build_prices():
    prices_path = "data/processed/prices.csv"
    if os.path.exists(prices_path):
        return pd.read_csv(prices_path, parse_dates=["date"])

    prices = load_multiple_prices(TICKERS, START_DATE, END_DATE)
    prices.to_csv(prices_path, index=False)
    return prices


def load_or_score_news():
    scored_path = "data/processed/news_scored_finbert.csv"
    if os.path.exists(scored_path):
        cached = pd.read_csv(scored_path, parse_dates=["date"])
        start = pd.to_datetime(START_DATE)
        end = pd.to_datetime(END_DATE)
        overlaps_study_period = cached["date"].between(start, end).any()
        if overlaps_study_period:
            return cached

    news = pd.read_csv("data/raw/news.csv", parse_dates=["date"])
    news_scored = score_news_sentiment(news)
    news_scored.to_csv(scored_path, index=False)
    return news_scored


def main():
    os.makedirs("data/processed", exist_ok=True)

    euronext_dirs = [path for path in EURONEXT_DATA_DIRS if os.path.exists(path)]
    if euronext_dirs:
        euronext = build_euronext_microstructure_from_dirs(euronext_dirs)
        if not euronext.empty:
            euronext.to_csv("data/processed/euronext_microstructure.csv", index=False)

    prices = load_or_build_prices()
    news_scored = load_or_score_news()

    daily_sentiment = aggregate_daily_sentiment(news_scored)
    daily_sentiment.to_csv("data/processed/daily_sentiment_finbert.csv", index=False)

    market = build_market_features(prices)
    data = merge_market_and_sentiment(market, daily_sentiment)
    data.to_csv("data/processed/model_dataset.csv", index=False)

    experiments = {
        "market_only": MARKET_FEATURES,
        "market_plus_sentiment": MARKET_FEATURES + SENTIMENT_FEATURES,
    }
    results = []

    for experiment_name, features in experiments.items():
        for ticker, ticker_data in data.groupby("ticker"):
            X_train, X_test, y_train, y_test, dates, returns = time_split(ticker_data, features, train_size=TRAIN_SIZE)
            if X_train.empty or X_test.empty or y_train.nunique() < 2:
                print(f"Jeu ignoré: {ticker} / {experiment_name} (données insuffisantes)")
                continue

            for model_name, model in get_models().items():
                model.fit(X_train, y_train)
                clf_metrics = evaluate_classifier(model, X_test, y_test)
                predictions = model.predict(X_test)
                bt = run_backtest(dates, returns, predictions, TRANSACTION_COST)
                fin_metrics = financial_metrics(bt)
                results.append({
                    "ticker": ticker,
                    "experiment": experiment_name,
                    "model": model_name,
                    **clf_metrics,
                    **fin_metrics,
                })
                bt.to_csv(f"data/processed/backtest_{ticker}_{experiment_name}_{model_name}.csv", index=False)

    results_df = pd.DataFrame(results)
    results_df.to_csv("data/processed/results.csv", index=False)
    print(results_df)


if __name__ == "__main__":
    main()
