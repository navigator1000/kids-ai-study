#!/usr/bin/env python3
"""BTC / Gold のリアルタイム・テクニカル指標を取得して表示する。

データ元:
  BTC  : data-api.binance.vision (BTCUSDT)
  Gold : query1.finance.yahoo.com (GC=F 金先物 COMEX)
使い方: python3 project/tools/market_ta.py
"""
import json
import urllib.request

import numpy as np
import pandas as pd

UA = {"User-Agent": "Mozilla/5.0"}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=20) as r:
        return json.load(r)


def binance(symbol, interval, limit=500):
    d = get(f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}")
    df = pd.DataFrame(d).iloc[:, :6]
    df.columns = ["t", "open", "high", "low", "close", "volume"]
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    return df.set_index("t").astype(float)


def yahoo(symbol, rng, interval):
    d = get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={rng}&interval={interval}")
    r = d["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    df = pd.DataFrame({k: q[k] for k in ["open", "high", "low", "close", "volume"]},
                      index=pd.to_datetime(r["timestamp"], unit="s", utc=True))
    return df.dropna()


def yahoo_4h(symbol):
    df = yahoo(symbol, "6mo", "1h")
    return df.resample("4h").agg({"open": "first", "high": "max", "low": "min",
                                  "close": "last", "volume": "sum"}).dropna()


def indicators(df):
    c, h, l = df["close"], df["high"], df["low"]
    out = {}
    for n in (20, 50, 200):
        out[f"SMA{n}"] = c.rolling(n).mean().iloc[-1] if len(c) >= n else np.nan
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn)
    out["RSI14"] = rsi.iloc[-1]
    out["RSI14_prev5"] = rsi.iloc[-6]
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    out["MACD"], out["MACD_signal"] = macd.iloc[-1], sig.iloc[-1]
    out["MACD_hist"], out["MACD_hist_prev"] = (macd - sig).iloc[-1], (macd - sig).iloc[-2]
    m, s = c.rolling(20).mean(), c.rolling(20).std()
    out["BB_upper"], out["BB_lower"] = (m + 2 * s).iloc[-1], (m - 2 * s).iloc[-1]
    out["BB_width%"] = (4 * s / m * 100).iloc[-1]
    out["BB_width%_min120"] = (4 * s / m * 100).iloc[-120:].min()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    out["ATR14"] = tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
    # 直近のスイング高値・安値（前後5本で極値）
    w = 5
    highs = [h.iloc[i] for i in range(w, len(h) - w) if h.iloc[i] == h.iloc[i - w:i + w + 1].max()]
    lows = [l.iloc[i] for i in range(w, len(l) - w) if l.iloc[i] == l.iloc[i - w:i + w + 1].min()]
    out["swing_highs"] = [round(x, 1) for x in highs[-4:]]
    out["swing_lows"] = [round(x, 1) for x in lows[-4:]]
    out["high_60"], out["low_60"] = h.iloc[-60:].max(), l.iloc[-60:].min()
    out["chg_5bars%"] = (c.iloc[-1] / c.iloc[-6] - 1) * 100
    out["chg_20bars%"] = (c.iloc[-1] / c.iloc[-21] - 1) * 100
    out["last"] = c.iloc[-1]
    out["last_time"] = str(df.index[-1])
    return out


def show(name, tf, df):
    o = indicators(df)
    print(f"\n=== {name} [{tf}] {o['last_time']} ===")
    for k, v in o.items():
        if k == "last_time":
            continue
        print(f"  {k:18s} {v:,.2f}" if isinstance(v, (float, np.floating)) else f"  {k:18s} {v}")


if __name__ == "__main__":
    show("BTCUSDT", "1D", binance("BTCUSDT", "1d"))
    show("BTCUSDT", "4H", binance("BTCUSDT", "4h"))
    show("BTCUSDT", "1W", binance("BTCUSDT", "1w", 300))
    show("Gold GC=F", "1D", yahoo("GC=F", "2y", "1d"))
    show("Gold GC=F", "4H", yahoo_4h("GC=F"))
    show("Gold GC=F", "1W", yahoo("GC=F", "10y", "1wk"))
