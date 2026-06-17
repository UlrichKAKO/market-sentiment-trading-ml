import yfinance as yf
import pandas as pd


def load_prices(ticker, start, end):

    data = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True
    )

    # Fix MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    data.columns = [
        str(col).lower().strip()
        for col in data.columns
    ]

    # Rename first column if needed
    if "date" not in data.columns:
        data.rename(
            columns={data.columns[0]: "date"},
            inplace=True
        )

    data["date"] = pd.to_datetime(data["date"]).dt.date

    data["ticker"] = ticker

    return data


def load_multiple_prices(tickers, start, end):

    dfs = []

    for ticker in tickers:

        try:
            df = load_prices(
                ticker,
                start,
                end
            )

            dfs.append(df)

        except Exception as e:

            print(ticker, e)

    return pd.concat(
        dfs,
        ignore_index=True
    )


def load_macro_market_data(start, end):

    macro_tickers = {
        "vix": "^VIX",
        "sp500": "^GSPC",
        "nasdaq": "^IXIC",
        "dowjones": "^DJI",
        "treasury_10y": "^TNX",
        "treasury_5y": "^FVX",
    }

    all_data = []

    for name, ticker in macro_tickers.items():

        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True
        )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        df.columns = [
            str(c).lower().strip()
            for c in df.columns
        ]

        if "date" not in df.columns:
            df.rename(
                columns={df.columns[0]: "date"},
                inplace=True
            )

        df["date"] = pd.to_datetime(
            df["date"]
        ).dt.date

        df = df[["date", "close"]]

        df = df.rename(columns={
            "close": name
        })

        all_data.append(df)

    macro = all_data[0]

    for df in all_data[1:]:

        macro = macro.merge(
            df,
            on="date",
            how="outer"
        )

    return macro.sort_values("date")