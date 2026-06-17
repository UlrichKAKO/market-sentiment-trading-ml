TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"]
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
TRANSACTION_COST = 0.001
TRAIN_SIZE = 0.7
RANDOM_STATE = 42
EURONEXT_DATA_DIRS = [
    (
        "/Users/ulrichfossokako/Desktop/DOCS/usb /Cours et entretiens/"
        "MASTER FINANCE/GFN260 Hybride Machine Learning/20240102"
    ),
    (
        "/Users/ulrichfossokako/Desktop/DOCS/usb /Cours et entretiens/"
        "MASTER FINANCE/GFN260 Hybride Machine Learning/20240103"
    ),
    (
        "/Users/ulrichfossokako/Desktop/DOCS/usb /Cours et entretiens/"
        "MASTER FINANCE/GFN260 Hybride Machine Learning/20240104"
    ),
    (
        "/Users/ulrichfossokako/Desktop/DOCS/usb /Cours et entretiens/"
        "MASTER FINANCE/GFN260 Hybride Machine Learning/20240105"
    ),
]
EURONEXT_DATA_DIR = EURONEXT_DATA_DIRS[-1]

MARKET_FEATURES = [
    "return_lag_1",
    "return_lag_2",
    "volatility_5d",
    "volatility_20d",
    "momentum_5d",
    "volume_change",
    "rsi_14",
]

SENTIMENT_FEATURES = [
    "sentiment_mean",
    "sentiment_std",
    "sentiment_count",
    "sentiment_lag_1",
    "sentiment_rolling_3d",
    "sentiment_rolling_7d",
]

EURONEXT_FEATURES = [
    "log_n_trades",
    "log_traded_volume",
    "log_traded_value",
    "vwap_to_open",
    "close_to_vwap",
    "high_low_spread",
]

EURONEXT_SENTIMENT_FEATURES = [
    "sentiment_mean",
    "sentiment_std",
    "sentiment_count",
    "sentiment_positive_ratio",
    "sentiment_negative_ratio",
]
