#!/usr/bin/env python3
"""価格データ取得（日足）。

BTC / ETH : data-api.binance.vision（Binance公開データ、日足は日本時間9:00に確定）
Gold      : query1.finance.yahoo.com（GC=F 金先物）
未確定の足（進行中の足）は取り除いて返す。
"""
import json
import os
import time
import urllib.request
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

UA = {"User-Agent": "Mozilla/5.0"}


def _get(url):
    for k in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return json.load(r)
        except Exception:
            if k == 2:
                raise
            time.sleep(2)


def binance_daily(symbol):
    rows, end = [], None
    while True:
        url = f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval=1d&limit=1000"
        if end:
            url += f"&endTime={end}"
        d = _get(url)
        rows = d + rows
        if len(d) < 1000:
            break
        end = d[0][0] - 1
    df = pd.DataFrame(rows).iloc[:, [0, 1, 2, 3, 4, 6]]
    df.columns = ["t", "open", "high", "low", "close", "close_time"]
    df = df.drop_duplicates("t")
    now_ms = time.time() * 1000
    df = df[df["close_time"].astype(float) < now_ms]          # 確定足のみ
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    return df.set_index("t")[["open", "high", "low", "close"]].astype(float)


def gold_daily():
    return yahoo_daily("GC=F")


def yahoo_daily(symbol):
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
           "?period1=946684800&period2=4102444800&interval=1d")
    r = _get(url)["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    df = pd.DataFrame({k: q[k] for k in ["open", "high", "low", "close"]},
                      index=pd.to_datetime(r["timestamp"], unit="s")).dropna()
    # 足の日付＝取引日。先物は米東部18時に翌取引日が始まるため、
    # 現在の取引日の足（進行中）を取り除く。
    df.index = (df.index + pd.Timedelta(hours=2)).normalize()
    now_et = datetime.now(ZoneInfo("America/New_York"))
    current_trade_date = pd.Timestamp((now_et + timedelta(hours=6)).date())
    df = df[df.index < current_trade_date]
    return df[~df.index.duplicated(keep="last")]


def load_all(cache_dir=None):
    out = {}
    for name, fn in (("BTC", lambda: binance_daily("BTCUSDT")),
                     ("ETH", lambda: binance_daily("ETHUSDT")),
                     ("Gold", gold_daily)):
        path = os.path.join(cache_dir, f"{name}.csv") if cache_dir else None
        if path and os.path.exists(path) and time.time() - os.path.getmtime(path) < 6 * 3600:
            out[name] = pd.read_csv(path, index_col=0, parse_dates=True)
            continue
        df = fn()
        if path:
            os.makedirs(cache_dir, exist_ok=True)
            df.to_csv(path)
        out[name] = df
    return out
