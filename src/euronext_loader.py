import csv
from pathlib import Path

import pandas as pd


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def load_standing_data(data_dir) -> pd.DataFrame:
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("StandingData_*.csv"))
    if not files:
        return pd.DataFrame()

    rows = []
    with files[0].open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.reader(handle):
            if len(row) < 57:
                continue
            rows.append(
                {
                    "instrument_id": row[3],
                    "name": row[6],
                    "short_name": row[7],
                    "isin": row[8],
                    "market": row[28],
                    "country": row[30],
                    "symbol": row[31] or row[56],
                    "currency": row[34] or row[49],
                    "reference_price": _to_float(row[39]),
                }
            )

    return pd.DataFrame(rows).drop_duplicates(subset=["instrument_id", "isin"])


def load_full_trade_information(data_dir) -> pd.DataFrame:
    data_dir = Path(data_dir)
    rows = []

    for path in sorted(data_dir.glob("*/FullTradeInformation_*.csv")):
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            for row in csv.reader(handle):
                if len(row) < 14:
                    continue

                price = _to_float(row[12])
                quantity = _to_int(row[13])
                if price is None or quantity is None:
                    continue

                rows.append(
                    {
                        "sequence_number": _to_int(row[0]),
                        "instrument_id": row[5],
                        "trade_time": pd.to_datetime(row[6], errors="coerce", utc=True),
                        "isin": row[10],
                        "trade_id": row[11],
                        "price": price,
                        "quantity": quantity,
                        "currency": row[15],
                    }
                )

    trades = pd.DataFrame(rows)
    if trades.empty:
        return trades

    trades = trades.dropna(subset=["trade_time"])
    trades["date"] = trades["trade_time"].dt.date
    trades["traded_value"] = trades["price"] * trades["quantity"]
    return trades


def build_euronext_microstructure(data_dir) -> pd.DataFrame:
    data_dir = Path(data_dir)
    aggregates = {}

    for path in sorted(data_dir.glob("*/FullTradeInformation_*.csv")):
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            for row in csv.reader(handle):
                if len(row) < 14:
                    continue

                price = _to_float(row[12])
                quantity = _to_int(row[13])
                trade_time = row[6]
                if price is None or quantity is None or not trade_time:
                    continue

                date = trade_time[:10]
                key = (date, row[10], row[5])
                traded_value = price * quantity

                if key not in aggregates:
                    aggregates[key] = {
                        "date": date,
                        "isin": row[10],
                        "instrument_id": row[5],
                        "n_trades": 0,
                        "traded_volume": 0,
                        "traded_value": 0.0,
                        "open_price": price,
                        "high_price": price,
                        "low_price": price,
                        "close_price": price,
                        "first_trade_time": trade_time,
                        "last_trade_time": trade_time,
                    }

                agg = aggregates[key]
                agg["n_trades"] += 1
                agg["traded_volume"] += quantity
                agg["traded_value"] += traded_value
                agg["high_price"] = max(agg["high_price"], price)
                agg["low_price"] = min(agg["low_price"], price)

                if trade_time < agg["first_trade_time"]:
                    agg["first_trade_time"] = trade_time
                    agg["open_price"] = price
                if trade_time > agg["last_trade_time"]:
                    agg["last_trade_time"] = trade_time
                    agg["close_price"] = price

    if not aggregates:
        return pd.DataFrame()

    standing = load_standing_data(data_dir)
    daily = pd.DataFrame(aggregates.values())
    daily["vwap"] = daily["traded_value"] / daily["traded_volume"].replace(0, pd.NA)
    daily["intraday_return"] = daily["close_price"] / daily["open_price"] - 1
    daily["high_low_spread"] = daily["high_price"] / daily["low_price"] - 1

    if not standing.empty:
        daily = daily.merge(
            standing,
            on=["instrument_id", "isin"],
            how="left",
            suffixes=("", "_standing"),
        )

    ordered_cols = [
        "date",
        "isin",
        "symbol",
        "name",
        "market",
        "country",
        "currency",
        "instrument_id",
        "n_trades",
        "traded_volume",
        "traded_value",
        "vwap",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "intraday_return",
        "high_low_spread",
        "first_trade_time",
        "last_trade_time",
    ]
    existing_cols = [col for col in ordered_cols if col in daily.columns]
    return daily[existing_cols].sort_values("traded_value", ascending=False)


def build_euronext_microstructure_from_dirs(data_dirs) -> pd.DataFrame:
    frames = []
    for data_dir in data_dirs:
        path = Path(data_dir)
        if not path.exists():
            print(f"Dossier Euronext introuvable: {path}")
            continue

        frame = build_euronext_microstructure(path)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True)
    return data.sort_values(["date", "isin"]).reset_index(drop=True)
