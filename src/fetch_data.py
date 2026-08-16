import yfinance as yf
import pandas as pd


TICKERS = [
    "RELIANCE.NS",   # Energy / Conglomerate
    "TCS.NS",        # IT
    "HDFCBANK.NS",   # Banking
    "INFY.NS",       # IT
    "ICICIBANK.NS",  # Banking
    "HINDUNILVR.NS", # FMCG
    "ITC.NS",        # FMCG
    "KOTAKBANK.NS",  # Banking
    "LT.NS",         # Infrastructure
    "SBIN.NS",       # Banking (PSU)
    "BHARTIARTL.NS", # Telecom
    "MARUTI.NS",     # Auto
]

BENCHMARK = "^NSEI"

START_DATE = "2019-01-01"
END_DATE = "2024-12-31"


def fetch_prices(tickers=TICKERS, start=START_DATE, end=END_DATE):
    
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    data = data.dropna(how="all")
    return data


def fetch_benchmark(start=START_DATE, end=END_DATE):
    
    bm = yf.download(BENCHMARK, start=start, end=end, auto_adjust=True)["Close"]
    return bm


if __name__ == "__main__":
    prices = fetch_prices()
    prices.to_csv("../data/nifty_prices.csv")
    print(f"Saved {prices.shape[0]} days x {prices.shape[1]} stocks to data/nifty_prices.csv")

    bm = fetch_benchmark()
    bm.to_csv("../data/nifty_index.csv")
    print("Saved benchmark index to data/nifty_index.csv")
