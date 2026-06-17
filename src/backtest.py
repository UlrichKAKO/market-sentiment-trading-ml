import numpy as np
import pandas as pd


def run_backtest(dates, returns, predictions, transaction_cost=0.001) -> pd.DataFrame:
    bt = pd.DataFrame({
        "date": dates.values,
        "market_return": returns.values,
        "prediction": predictions,
    })
    bt["position"] = bt["prediction"].replace({0: -1, 1: 1})
    bt["trade"] = bt["position"].diff().abs()
    bt.loc[bt.index[0], "trade"] = abs(bt.loc[bt.index[0], "position"])
    bt["strategy_return_gross"] = bt["position"].shift(1) * bt["market_return"]
    bt["cost"] = bt["trade"] * transaction_cost
    bt["strategy_return_net"] = bt["strategy_return_gross"] - bt["cost"]
    bt = bt.dropna()
    bt["market_equity"] = np.exp(bt["market_return"].cumsum())
    bt["strategy_equity"] = np.exp(bt["strategy_return_net"].cumsum())
    return bt


def sharpe_ratio(returns, periods_per_year=252):
    if returns.std() == 0:
        return 0
    return np.sqrt(periods_per_year) * returns.mean() / returns.std()


def max_drawdown(equity_curve):
    rolling_max = equity_curve.cummax()
    drawdown = equity_curve / rolling_max - 1
    return drawdown.min()


def financial_metrics(bt):
    market_total_return = bt["market_equity"].iloc[-1] - 1
    strategy_total_return = bt["strategy_equity"].iloc[-1] - 1
    return {
        "market_total_return": market_total_return,
        "total_return": strategy_total_return,
        "excess_return": strategy_total_return - market_total_return,
        "sharpe": sharpe_ratio(bt["strategy_return_net"]),
        "max_drawdown": max_drawdown(bt["strategy_equity"]),
        "hit_ratio": (bt["strategy_return_net"] > 0).mean(),
        "turnover": bt["trade"].mean(),
        "n_trades": int((bt["trade"] > 0).sum()),
    }
