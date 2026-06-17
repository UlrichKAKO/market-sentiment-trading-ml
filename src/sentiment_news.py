import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Chargement du modèle (une seule fois)
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def score_news_sentiment(news: pd.DataFrame, text_col="text") -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()

    df = news.copy()
    df[text_col] = df[text_col].astype(str)

    df["sentiment_score"] = df[text_col].apply(
        lambda x: analyzer.polarity_scores(x)["compound"]
    )

    df["sentiment_label"] = pd.cut(
        df["sentiment_score"],
        bins=[-1.01, -0.05, 0.05, 1.01],
        labels=["negative", "neutral", "positive"]
    )

    return df


def aggregate_daily_sentiment(news: pd.DataFrame) -> pd.DataFrame:
    df = news.copy()
    df["date"] = pd.to_datetime(df["date"])

    daily = (
        df.groupby("date")
        .agg(
            news_count=("text", "count"),
            sentiment_mean=("sentiment_score", "mean"),
            sentiment_std=("sentiment_score", "std"),
            sentiment_positive_ratio=("sentiment_label", lambda x: (x == "positive").mean()),
            sentiment_negative_ratio=("sentiment_label", lambda x: (x == "negative").mean()),
        )
        .reset_index()
    )

    daily["sentiment_std"] = daily["sentiment_std"].fillna(0)

    return daily

    from transformers import pipeline

def finbert_score(text):
    try:
        result = finbert(text[:512])[0]

        if result["label"] == "positive":
            return result["score"]
        elif result["label"] == "negative":
            return -result["score"]
        else:
            return 0
    except:
        return 0


        from transformers import pipeline

# Chargement du modèle (lazy load)
def load_finbert():
    return pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        device=-1
    )


def finbert_score(text, model=None):
    if model is None:
        model = load_finbert()

    try:
        result = model(str(text)[:512])[0]

        label = result["label"].lower()
        score = result["score"]

        if label == "positive":
            return score
        elif label == "negative":
            return -score
        else:
            return 0
    except:
        return 0