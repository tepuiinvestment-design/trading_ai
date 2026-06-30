import numpy as np
import pandas as pd

def compute_sharpe(returns, risk_free_rate=0.0):
    returns = np.array(returns)
    if len(returns) == 0 or np.std(returns) == 0:
        return float("nan")
    excess_returns = returns - risk_free_rate
    return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)

def compute_max_drawdown(equity_curve):
    equity = np.array(equity_curve)
    if len(equity) == 0:
        return float("nan")
    peaks = np.maximum.accumulate(equity)
    drawdowns = (equity - peaks) / peaks
    return drawdowns.min()

def compute_volatility(returns):
    returns = np.array(returns)
    if len(returns) == 0:
        return float("nan")
    return np.std(returns) * np.sqrt(252)

def compute_cagr(equity_curve):
    equity_curve = pd.Series(equity_curve)
    if len(equity_curve) == 0:
        return float("nan")
    start = equity_curve.iloc[0]
    end = equity_curve.iloc[-1]
    years = len(equity_curve) / 252
    if start <= 0 or years <= 0:
        return float("nan")
    return (end / start) ** (1 / years) - 1

def compute_win_rate(trade_returns):
    trade_returns = np.array(trade_returns)
    if len(trade_returns) == 0:
        return float("nan")
    return np.mean(trade_returns > 0)

def compute_trade_count(trade_returns):
    return len(trade_returns)
