from src.metrics import (
    compute_sharpe,
    compute_max_drawdown,
    compute_volatility,
    compute_cagr,
    compute_win_rate,
    compute_trade_count
)

def performance_report(df, trade_log, daily_returns, excess_cumulative,
                       annualized_excess_return, correlation_to_spy, beta_to_spy):
    print("Final equity:", df["Equity"].iloc[-1])
    print("Sharpe ratio:", compute_sharpe(daily_returns))
    print("Max drawdown:", compute_max_drawdown(df["Equity"]))
    print("Volatility:", compute_volatility(daily_returns))
    print("CAGR:", compute_cagr(df["Equity"]))
    print("Win rate:", compute_win_rate(trade_log["pnl"].tolist()))
    print("Trade count:", compute_trade_count(trade_log["pnl"].tolist()))
    print("Annualized Excess Return (Alpha):", annualized_excess_return)
    print("Final Excess Return vs SPY:", excess_cumulative.iloc[-1] - 1)
    print("Correlation vs SPY:", correlation_to_spy)
    print("Beta vs SPY:", beta_to_spy)
