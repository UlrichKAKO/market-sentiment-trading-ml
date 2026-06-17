import requests
import pandas as pd
import yfinance as yf
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime


def load_boursorama_news(url="https://www.boursorama.com/bourse/actualites/finances/"):
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    articles = []
    for item in soup.find_all(["article", "li"]):
        text = item.get_text(" ", strip=True)

        if len(text) > 80:
            articles.append({
                "date": datetime.today().date(),
                "source": "boursorama",
                "text": text
            })

    return pd.DataFrame(articles).drop_duplicates()


def load_pappers_company(api_token, siren):
    url = "https://api.pappers.fr/v2/entreprise"
    params = {
        "api_token": api_token,
        "siren": siren
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    return {
        "siren": siren,
        "nom_entreprise": data.get("nom_entreprise"),
        "forme_juridique": data.get("forme_juridique"),
        "date_creation": data.get("date_creation"),
        "capital": data.get("capital"),
        "chiffre_affaires": data.get("chiffre_affaires"),
        "resultat": data.get("resultat"),
    }


    TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"]

def tag_ticker(text):
    text = text.lower()
    for t in TICKERS:
        if t.lower() in text:
            return t
    return "OTHER"

def load_yahoo_news(tickers):
    rows = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            news_items = stock.news or []

            for item in news_items:
                publish_time = item.get("providerPublishTime")

                if publish_time is None:
                    continue

                date = pd.to_datetime(
                    publish_time,
                    unit="s",
                    errors="coerce"
                )

                if pd.isna(date):
                    continue

                title = item.get("title")

                if title is None:
                    continue

                rows.append({
                    "ticker": ticker,
                    "date": date.date(),
                    "source": item.get("publisher"),
                    "title": title,
                    "text": title,
                    "link": item.get("link"),
                })

        except Exception as e:
            print(ticker, e)

    return pd.DataFrame(rows)