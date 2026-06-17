import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import EURONEXT_DATA_DIRS, EURONEXT_FEATURES, EURONEXT_SENTIMENT_FEATURES, RANDOM_STATE
from src.euronext_loader import build_euronext_microstructure_from_dirs


MICROSTRUCTURE_PATH = "data/processed/euronext_microstructure.csv"
MAPPING_PATH = "data/processed/euronext_instrument_mapping.csv"
NEWS_PATH = "data/raw/euronext_news.csv"
SENTIMENT_PATH = "data/processed/euronext_daily_sentiment.csv"
DATASET_PATH = "data/processed/euronext_prediction_dataset.csv"
RESULTS_PATH = "data/processed/euronext_prediction_results.csv"
PREDICTIONS_PATH = "data/processed/euronext_predictions.csv"
STATUS_PATH = "data/processed/euronext_prediction_status.csv"
BACKTEST_PATH = "data/processed/euronext_backtest.csv"
FEATURE_IMPORTANCE_PATH = "data/processed/euronext_feature_importance.csv"


def load_or_build_microstructure() -> pd.DataFrame:
    if os.path.exists(MICROSTRUCTURE_PATH):
        return pd.read_csv(MICROSTRUCTURE_PATH)

    microstructure = build_euronext_microstructure_from_dirs(EURONEXT_DATA_DIRS)
    os.makedirs("data/processed", exist_ok=True)
    microstructure.to_csv(MICROSTRUCTURE_PATH, index=False)
    return microstructure


def build_instrument_mapping(microstructure: pd.DataFrame) -> pd.DataFrame:
    cols = ["isin", "symbol", "name", "market", "country", "currency", "instrument_id"]
    mapping = (
        microstructure[[col for col in cols if col in microstructure.columns]]
        .drop_duplicates(subset=["isin"])
        .sort_values("isin")
        .reset_index(drop=True)
    )
    mapping.to_csv(MAPPING_PATH, index=False)
    return mapping


def load_euronext_news() -> pd.DataFrame:
    if not os.path.exists(NEWS_PATH):
        return pd.DataFrame(columns=["date", "text"])

    news = pd.read_csv(NEWS_PATH)
    if "date" not in news.columns or "text" not in news.columns:
        raise ValueError("data/raw/euronext_news.csv doit contenir au minimum les colonnes date,text")

    news["date"] = pd.to_datetime(news["date"], errors="coerce")
    news = news.dropna(subset=["date", "text"]).copy()
    news["date"] = news["date"].dt.normalize()
    return news


def score_news_with_vader(news: pd.DataFrame) -> pd.DataFrame:
    if news.empty:
        return news.copy()

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    scored = news.copy()
    scored["sentiment_score"] = scored["text"].astype(str).apply(
        lambda text: analyzer.polarity_scores(text)["compound"]
    )
    scored["sentiment_label"] = pd.cut(
        scored["sentiment_score"],
        bins=[-1.01, -0.05, 0.05, 1.01],
        labels=["negative", "neutral", "positive"],
    ).astype(str)
    return scored


def attach_news_to_instruments(news: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    if news.empty:
        return pd.DataFrame(columns=["date", "isin", "text", "sentiment_score", "sentiment_label"])

    news = score_news_with_vader(news)
    if "isin" in news.columns:
        attached = news.dropna(subset=["isin"]).copy()
        attached["isin"] = attached["isin"].astype(str)
        return attached

    rows = []
    candidates = mapping[["isin", "symbol", "name"]].dropna(subset=["isin"]).copy()
    candidates["symbol"] = candidates["symbol"].fillna("").astype(str)
    candidates["name"] = candidates["name"].fillna("").astype(str)

    for _, item in news.iterrows():
        text = str(item["text"]).lower()
        for _, instrument in candidates.iterrows():
            symbol = instrument["symbol"].lower()
            name = instrument["name"].lower()
            symbol_match = len(symbol) >= 3 and f" {symbol} " in f" {text} "
            name_match = len(name) >= 5 and name in text
            if symbol_match or name_match:
                row = item.to_dict()
                row["isin"] = instrument["isin"]
                rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["date", "isin", "text", "sentiment_score", "sentiment_label"])
    return pd.DataFrame(rows)


def build_daily_sentiment(microstructure: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    mapping = build_instrument_mapping(microstructure)
    raw_news = load_euronext_news()
    attached_news = attach_news_to_instruments(raw_news, mapping)

    if attached_news.empty:
        sentiment = pd.DataFrame(
            columns=[
                "date",
                "isin",
                "sentiment_mean",
                "sentiment_std",
                "sentiment_count",
                "sentiment_positive_ratio",
                "sentiment_negative_ratio",
            ]
        )
    else:
        sentiment = (
            attached_news.groupby(["date", "isin"], as_index=False)
            .agg(
                sentiment_mean=("sentiment_score", "mean"),
                sentiment_std=("sentiment_score", "std"),
                sentiment_count=("sentiment_score", "count"),
                sentiment_positive_ratio=("sentiment_label", lambda x: (x == "positive").mean()),
                sentiment_negative_ratio=("sentiment_label", lambda x: (x == "negative").mean()),
            )
        )
        sentiment["sentiment_std"] = sentiment["sentiment_std"].fillna(0)

    sentiment.to_csv(SENTIMENT_PATH, index=False)
    info = {
        "news_rows": len(raw_news),
        "matched_news_rows": len(attached_news),
        "sentiment_instruments": sentiment["isin"].nunique() if not sentiment.empty else 0,
    }
    return sentiment, info


def prepare_euronext_prediction_dataset(microstructure: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    df = microstructure.copy()
    df["date"] = pd.to_datetime(df["date"])
    sentiment, sentiment_info = build_daily_sentiment(df)
    if not sentiment.empty:
        sentiment["date"] = pd.to_datetime(sentiment["date"])
        df = df.merge(sentiment, on=["date", "isin"], how="left")

    numeric_cols = [
        "n_trades",
        "traded_volume",
        "traded_value",
        "vwap",
        "open_price",
        "close_price",
        "high_low_spread",
        "intraday_return",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["log_n_trades"] = np.log1p(df["n_trades"])
    df["log_traded_volume"] = np.log1p(df["traded_volume"])
    df["log_traded_value"] = np.log1p(df["traded_value"])
    df["vwap_to_open"] = df["vwap"] / df["open_price"] - 1
    df["close_to_vwap"] = df["close_price"] / df["vwap"] - 1
    for col in EURONEXT_SENTIMENT_FEATURES:
        if col not in df.columns:
            df[col] = 0
    df[EURONEXT_SENTIMENT_FEATURES] = df[EURONEXT_SENTIMENT_FEATURES].fillna(0)
    sentiment_info["dataset_sentiment_rows"] = int((df["sentiment_count"] > 0).sum())
    sentiment_info["dataset_sentiment_instruments"] = int(df.loc[df["sentiment_count"] > 0, "isin"].nunique())

    n_dates = df["date"].nunique()
    if n_dates >= 3:
        df = df.sort_values(["isin", "date"])
        df["target_return"] = df.groupby("isin")["close_price"].shift(-1) / df["close_price"] - 1
        mode = "next_day_direction"
    else:
        df["target_return"] = df["intraday_return"]
        mode = "same_day_cross_section_exploratory"

    df["target"] = (df["target_return"] > 0).astype(int)
    df["prediction_mode"] = mode
    df.attrs["sentiment_info"] = sentiment_info
    df = df.dropna(subset=EURONEXT_FEATURES + EURONEXT_SENTIMENT_FEATURES + ["target_return", "target"]).reset_index(drop=True)
    return df, mode


def get_euronext_models():
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }


def split_euronext_data(dataset: pd.DataFrame, mode: str):
    if mode == "next_day_direction":
        dates = sorted(dataset["date"].unique())
        split_idx = max(1, int(len(dates) * 0.7))
        train_dates = dates[:split_idx]
        test_dates = dates[split_idx:]
        train = dataset[dataset["date"].isin(train_dates)]
        test = dataset[dataset["date"].isin(test_dates)]
    else:
        stratify = dataset["target"] if dataset["target"].nunique() > 1 else None
        train, test = train_test_split(
            dataset,
            test_size=0.3,
            random_state=RANDOM_STATE,
            stratify=stratify,
        )

    return train, test


def evaluate_model(model, X_test, y_test):
    pred = model.predict(X_test)
    auc = None
    if hasattr(model, "predict_proba") and y_test.nunique() > 1:
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    return {
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "auc": auc,
    }


def extract_feature_importance(model, model_name: str, experiment: str, feature_cols: list[str]) -> pd.DataFrame:
    fitted_model = model.named_steps["model"] if hasattr(model, "named_steps") else model
    if hasattr(fitted_model, "feature_importances_"):
        values = fitted_model.feature_importances_
    elif hasattr(fitted_model, "coef_"):
        values = np.abs(fitted_model.coef_[0])
    else:
        return pd.DataFrame()

    importance = pd.DataFrame(
        {
            "experiment": experiment,
            "model": model_name,
            "feature": feature_cols,
            "importance": values,
        }
    )
    total = importance["importance"].sum()
    if total > 0:
        importance["importance"] = importance["importance"] / total
    return importance.sort_values("importance", ascending=False)


def build_euronext_backtest(predictions: pd.DataFrame, top_n: int = 20, transaction_cost: float = 0.001) -> pd.DataFrame:
    rows = []
    if predictions.empty:
        return pd.DataFrame()

    for (experiment, model_name, date), group in predictions.groupby(["experiment", "model", "date"]):
        group = group.dropna(subset=["probability_positive", "target_return"])
        if group.empty:
            continue

        n_select = min(top_n, max(1, len(group) // 10))
        ranked = group.sort_values("probability_positive", ascending=False)
        long_leg = ranked.head(n_select)
        short_leg = ranked.tail(n_select)

        long_return = long_leg["target_return"].mean()
        short_leg_return = short_leg["target_return"].mean()
        benchmark_return = group["target_return"].mean()
        long_net = long_return - transaction_cost
        long_short_net = ((long_return - short_leg_return) / 2) - (2 * transaction_cost)

        rows.append(
            {
                "date": date,
                "experiment": experiment,
                "model": model_name,
                "n_selected": n_select,
                "long_return": long_return,
                "long_return_net": long_net,
                "long_short_return_net": long_short_net,
                "benchmark_return": benchmark_return,
            }
        )

    backtest = pd.DataFrame(rows).sort_values(["experiment", "model", "date"])
    if backtest.empty:
        return backtest

    backtest["long_equity"] = backtest.groupby(["experiment", "model"])["long_return_net"].transform(lambda x: (1 + x).cumprod())
    backtest["long_short_equity"] = backtest.groupby(["experiment", "model"])["long_short_return_net"].transform(lambda x: (1 + x).cumprod())
    backtest["benchmark_equity"] = backtest.groupby(["experiment", "model"])["benchmark_return"].transform(lambda x: (1 + x).cumprod())
    return backtest


def run_euronext_prediction_pipeline():
    os.makedirs("data/processed", exist_ok=True)
    microstructure = load_or_build_microstructure()
    dataset, mode = prepare_euronext_prediction_dataset(microstructure)
    sentiment_info = dataset.attrs.get("sentiment_info", {})
    dataset.to_csv(DATASET_PATH, index=False)

    status = {
        "mode": mode,
        "n_rows": len(dataset),
        "n_dates": dataset["date"].nunique(),
        "n_instruments": dataset["isin"].nunique(),
        "target_positive_rate": dataset["target"].mean(),
        "news_rows": sentiment_info.get("news_rows", 0),
        "matched_news_rows": sentiment_info.get("matched_news_rows", 0),
        "sentiment_instruments": sentiment_info.get("sentiment_instruments", 0),
        "dataset_sentiment_rows": sentiment_info.get("dataset_sentiment_rows", 0),
        "dataset_sentiment_instruments": sentiment_info.get("dataset_sentiment_instruments", 0),
        "note": (
            "Vraie prediction temporelle du prochain jour."
            if mode == "next_day_direction"
            else "Mode exploratoire: une seule date Euronext disponible, classification cross-section du rendement intrajournalier."
        ),
    }
    pd.DataFrame([status]).to_csv(STATUS_PATH, index=False)

    if dataset.empty or dataset["target"].nunique() < 2:
        pd.DataFrame().to_csv(RESULTS_PATH, index=False)
        pd.DataFrame().to_csv(PREDICTIONS_PATH, index=False)
        pd.DataFrame().to_csv(BACKTEST_PATH, index=False)
        pd.DataFrame().to_csv(FEATURE_IMPORTANCE_PATH, index=False)
        return pd.DataFrame(), pd.DataFrame(), status

    train, test = split_euronext_data(dataset, mode)
    y_train = train["target"]
    y_test = test["target"]

    experiments = {
        "microstructure_only": EURONEXT_FEATURES,
        "microstructure_plus_sentiment": EURONEXT_FEATURES + EURONEXT_SENTIMENT_FEATURES,
    }
    results = []
    predictions = []
    importances = []
    for experiment, feature_cols in experiments.items():
        X_train = train[feature_cols]
        X_test = test[feature_cols]

        for model_name, model in get_euronext_models().items():
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)
            pred = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else pred

            results.append(
                {
                    "pipeline": "euronext_prediction",
                    "mode": mode,
                    "experiment": experiment,
                    "model": model_name,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    **metrics,
                }
            )

            importance = extract_feature_importance(model, model_name, experiment, feature_cols)
            if not importance.empty:
                importances.append(importance)

            pred_df = test[
                [
                    "date",
                    "isin",
                    "symbol",
                    "name",
                    "target_return",
                    "target",
                    "traded_value",
                    "n_trades",
                ]
            ].copy()
            pred_df["experiment"] = experiment
            pred_df["model"] = model_name
            pred_df["prediction"] = pred
            pred_df["probability_positive"] = proba
            predictions.append(pred_df)

    results_df = pd.DataFrame(results).sort_values("balanced_accuracy", ascending=False)
    predictions_df = pd.concat(predictions, ignore_index=True)
    feature_importance_df = pd.concat(importances, ignore_index=True) if importances else pd.DataFrame()
    backtest_df = build_euronext_backtest(predictions_df)

    results_df.to_csv(RESULTS_PATH, index=False)
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)
    feature_importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    backtest_df.to_csv(BACKTEST_PATH, index=False)
    return results_df, predictions_df, status


def main():
    results, _, status = run_euronext_prediction_pipeline()
    print(pd.DataFrame([status]))
    print(results)


if __name__ == "__main__":
    main()
