import argparse
import pandas as pd
import yfinance as yf
import numpy as np

def backtest(symbol, fast, slow):
    data = yf.download(symbol, period="5y", auto_adjust=True, progress=False)
    if data.empty:
        raise RuntimeError("No market data returned")
    close = data["Close"].dropna()
    fast_ma, slow_ma = close.rolling(fast).mean(), close.rolling(slow).mean()
    position = (fast_ma > slow_ma).astype(int).shift(1).fillna(0)
    returns = close.pct_change().fillna(0)
    strategy = position * returns
    equity = (1 + strategy).cumprod()
    drawdown = equity / equity.cummax() - 1
    sharpe = np.sqrt(252) * strategy.mean() / (strategy.std() + 1e-12)
    print(f"Symbol: {symbol}")
    print(f"Total return: {equity.iloc[-1]-1:.2%}")
    print(f"Sharpe ratio: {sharpe:.2f}")
    print(f"Maximum drawdown: {drawdown.min():.2%}")
    equity.plot(title=f"{symbol} Moving-Average Backtest", figsize=(12, 5)).figure.tight_layout()
    import matplotlib.pyplot as plt
    plt.show()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("symbol", nargs="?", default="AAPL")
    p.add_argument("--fast", type=int, default=20)
    p.add_argument("--slow", type=int, default=50)
    a = p.parse_args()
    backtest(a.symbol, a.fast, a.slow)
