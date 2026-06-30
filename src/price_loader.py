import pandas as pd
import os

def load_price_data(symbol="SPY"):
    """
    Offline price loader.
    Loads historical data from data/raw/<symbol>.csv
    Guaranteed: no yfinance, no network, no curl-cffi.
    """
    try:
        base_path = os.path.join("data", "raw")
        csv_path = os.path.join(base_path, f"{symbol}.csv")
        df = pd.read_csv(csv_path)
        return df
    except Exception:
        return None

if __name__ == "__main__":
    df = load_price_data("SPY")
    print(df.head() if df is not None else "No data loaded.")
