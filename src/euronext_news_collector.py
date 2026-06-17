import argparse
import os
from datetime import datetime

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup


MAPPING_PATH = "data/processed/euronext_instrument_mapping.csv"
MICROSTRUCTURE_PATH = "data/processed/euronext_microstructure.csv"
OUTPUT_PATH = "data/raw/euronext_news.csv"

YAHOO_SUFFIX_BY_MARKET = {
    "XPAR": ".PA",
    "XAMS": ".AS",
    "XBRU": ".BR",
    "XLIS": ".LS",
    "XMIL": ".MI",
    "XETR": ".DE",
    "XMAD": ".MC",
}


def load_candidate_instruments(limit: int) -> pd.DataFrame:
    if not os.path.exists(MAPPING_PATH):
        raise FileNotFoundError("Lance d'abord `python -m src.euronext_prediction` pour créer le mapping Euronext.")

    mapping = pd.read_csv(MAPPING_PATH)
    if os.path.exists(MICROSTRUCTURE_PATH):
        micro = pd.read_csv(MICROSTRUCTURE_PATH)
        liquidity = (
            micro.groupby("isin", as_index=False)
            .agg(traded_value=("traded_value", "sum"), n_trades=("n_trades", "sum"))
            .sort_values("traded_value", ascending=False)
        )
        mapping = mapping.merge(liquidity, on="isin", how="left")
        mapping = mapping.sort_values("traded_value", ascending=False)

    mapping = mapping.dropna(subset=["isin", "symbol"]).copy()
    mapping["market"] = mapping["market"].fillna("")
    mapping["yahoo_ticker"] = mapping.apply(
        lambda row: f"{row['symbol']}{YAHOO_SUFFIX_BY_MARKET.get(row['market'], '')}",
        axis=1,
    )
    mapping = mapping[mapping["yahoo_ticker"].str.contains(r"\.", regex=True)]
    return mapping.head(limit).reset_index(drop=True)


def _parse_yahoo_item(item):
    content = item.get("content", {}) if isinstance(item, dict) else {}
    title = item.get("title") or content.get("title") or ""
    summary = item.get("summary") or content.get("summary") or ""
    provider = item.get("publisher") or content.get("provider", {}).get("displayName") or "Yahoo Finance"
    url = item.get("link") or content.get("canonicalUrl", {}).get("url") or content.get("clickThroughUrl", {}).get("url") or ""

    publish_time = item.get("providerPublishTime")
    if publish_time:
        date = pd.to_datetime(publish_time, unit="s", errors="coerce")
    else:
        date = pd.to_datetime(content.get("pubDate") or content.get("displayTime"), errors="coerce")

    text = f"{title}. {summary}".strip()
    return date, provider, url, text


def collect_yahoo_news(instruments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, instrument in instruments.iterrows():
        ticker = instrument["yahoo_ticker"]
        try:
            items = yf.Ticker(ticker).news or []
        except Exception as exc:
            print(f"Yahoo ignoré pour {ticker}: {exc}")
            continue

        for item in items:
            date, provider, url, text = _parse_yahoo_item(item)
            if pd.isna(date) or not text:
                continue

            if getattr(date, "tzinfo", None) is not None:
                date = date.tz_convert(None)

            rows.append(
                {
                    "date": date.date().isoformat(),
                    "isin": instrument["isin"],
                    "symbol": instrument["symbol"],
                    "company": instrument.get("name", ""),
                    "source": provider,
                    "url": url,
                    "text": text,
                }
            )

    return pd.DataFrame(rows)


def collect_boursorama_general_news() -> pd.DataFrame:
    url = "https://www.boursorama.com/bourse/actualites/"
    try:
        response = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
    except Exception as exc:
        print(f"Boursorama ignoré: {exc}")
        return pd.DataFrame()

    soup = BeautifulSoup(response.text, "html.parser")
    rows = []
    today = datetime.today().date().isoformat()
    for item in soup.find_all(["article", "li", "a"]):
        text = item.get_text(" ", strip=True)
        if len(text) < 80:
            continue
        rows.append(
            {
                "date": today,
                "isin": "",
                "symbol": "",
                "company": "",
                "source": "Boursorama",
                "url": item.get("href", ""),
                "text": text,
            }
        )

    return pd.DataFrame(rows).drop_duplicates(subset=["text"])


def collect_euronext_news(limit: int = 80, include_boursorama: bool = False) -> pd.DataFrame:
    instruments = load_candidate_instruments(limit)
    yahoo = collect_yahoo_news(instruments)
    frames = [yahoo]

    if include_boursorama:
        frames.append(collect_boursorama_general_news())

    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["date", "isin", "symbol", "company", "source", "url", "text"])

    news = pd.concat(frames, ignore_index=True)
    news = news.drop_duplicates(subset=["date", "isin", "text"]).sort_values(["date", "isin"])
    return news


def main():
    parser = argparse.ArgumentParser(description="Collecte des news Euronext depuis Yahoo Finance et Boursorama.")
    parser.add_argument("--limit", type=int, default=80, help="Nombre d'instruments Euronext liquides à interroger sur Yahoo.")
    parser.add_argument("--include-boursorama", action="store_true", help="Ajoute une collecte générale Boursorama, plus bruitée.")
    args = parser.parse_args()

    os.makedirs("data/raw", exist_ok=True)
    news = collect_euronext_news(limit=args.limit, include_boursorama=args.include_boursorama)
    news.to_csv(OUTPUT_PATH, index=False)
    print(f"News écrites: {OUTPUT_PATH}")
    print(f"Lignes: {len(news)}")
    if not news.empty:
        print(news[["date", "isin", "symbol", "source", "text"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
