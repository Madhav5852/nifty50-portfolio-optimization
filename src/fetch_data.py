"""
Step 1: Data collection.

Pulls historical daily adjusted close prices for a basket of Nifty50 stocks
using yfinance. Run this on a machine with internet access — it will not
run inside a sandboxed/offline environment.

Usage:
    python fetch_data.py
"""

import yfinance as yf
import pandas as pd

# 12 liquid, well-known Nifty50 constituents across different sectors.
# Yahoo Finance uses the ".NS" suffix for NSE-listed tickers.
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

# Benchmark index for later comparison
BENCHMARK = "^NSEI"  # Nifty 50 index

START_DATE = "2019-01-01"
END_DATE = "2024-12-31"


def fetch_prices(tickers=TICKERS, start=START_DATE, end=END_DATE):
    """Download adjusted close prices for all tickers into one DataFrame."""
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    data = data.dropna(how="all")
    return data


def fetch_benchmark(start=START_DATE, end=END_DATE):
    """Download the Nifty50 index itself, for benchmarking later."""
    bm = yf.download(BENCHMARK, start=start, end=end, auto_adjust=True)["Close"]
    return bm


if __name__ == "__main__":
    prices = fetch_prices()
    prices.to_csv("../data/nifty_prices.csv")
    print(f"Saved {prices.shape[0]} days x {prices.shape[1]} stocks to data/nifty_prices.csv")

    bm = fetch_benchmark()
    bm.to_csv("../data/nifty_index.csv")
    print("Saved benchmark index to data/nifty_index.csv")
