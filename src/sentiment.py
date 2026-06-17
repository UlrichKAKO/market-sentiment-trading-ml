import pandas as pd
from tqdm import tqdm


def load_sentiment_model():
    from transformers import pipeline

    return pipeline("sentiment-analysis", model="ProsusAI/finbert")


def score_news_with_vader(news: pd.DataFrame) -> pd.DataFrame:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    news = news.copy()
    news["sentiment_score"] = news["text"].astype(str).apply(
        lambda text: analyzer.polarity_scores(text)["compound"]
    )
    news["sentiment_label"] = pd.cut(
        news["sentiment_score"],
        bins=[-1.01, -0.05, 0.05, 1.01],
        labels=["negative", "neutral", "positive"],
    ).astype(str)
    news["sentiment_model"] = "vader"
    return news


def score_news_sentiment(news: pd.DataFrame) -> pd.DataFrame:
    try:
        model = load_sentiment_model()
    except Exception as exc:
        print(f"FinBERT indisponible ({exc}). Utilisation du fallback VADER.")
        return score_news_with_vader(news)

    scores = []
    labels = []

    for text in tqdm(news["text"].tolist()):
        result = model(text[:512])[0]
        labels.append(result["label"])
        scores.append(result["score"])

    news = news.copy()
    news["sentiment_label"] = labels
    news["sentiment_score_raw"] = scores

    mapping = {"positive": 1, "neutral": 0, "negative": -1}
    news["sentiment_score"] = (
        news["sentiment_label"].str.lower().map(mapping) * news["sentiment_score_raw"]
    )
    news["sentiment_model"] = "finbert"
    return news


def aggregate_daily_sentiment(news: pd.DataFrame) -> pd.DataFrame:
    daily = (
        news.groupby("date")
        .agg(
            sentiment_mean=("sentiment_score", "mean"),
            sentiment_std=("sentiment_score", "std"),
            sentiment_count=("sentiment_score", "count"),
        )
        .reset_index()
    )
    daily["sentiment_std"] = daily["sentiment_std"].fillna(0)
    return daily
