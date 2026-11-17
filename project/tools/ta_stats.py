#!/usr/bin/env python3
"""テクニカル手法の簡易統計検証（BTC / ETH / Gold 日足）。

1. フィボナッチ: 押し・戻りの深さがフィボ比率(0.382/0.5/0.618/0.786)付近に
   集中するか。比率をずらした「ダミー水準」と比較する。
2. フィボ水準での反発率: 水準にタッチ後、先に+N%反発するか-N%割れるか。
   フィボ水準とダミー水準で比較。
3. トレンドフォロー: 終値 > SMA(n) の時だけ保有（翌日から）vs 買い持ち。
4. RSI<30 / >70 の後の20日リターン vs 全期間平均。
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from market_ta import binance, yahoo  # noqa: E402

FIB = [0.382, 0.5, 0.618, 0.786]
DUMMY = [0.33, 0.44, 0.56, 0.70, 0.84]


def binance_all(sym):
    frames, end = [], None
    import json, urllib.request
    while True:
        url = f"https://data-api.binance.vision/api/v3/klines?symbol={sym}&interval=1d&limit=1000"
        if end:
            url += f"&endTime={end}"
        d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "x"})))
        if not d:
            break
        frames = d + frames
        if len(d) < 1000:
            break
        end = d[0][0] - 1
    df = pd.DataFrame(frames).iloc[:, :5]
    df.columns = ["t", "open", "high", "low", "close"]
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    return df.set_index("t").astype(float).drop_duplicates()


def zigzag(df, th):
    """終値ベースではなく高値/安値で、th(比率)以上の反転でピボット確定。"""
    h, l = df["high"].values, df["low"].values
    piv = []  # (idx, price, type)
    trend, ext_i, ext_p = 0, 0, l[0]
    hi_i, hi_p, lo_i, lo_p = 0, h[0], 0, l[0]
    for i in range(1, len(df)):
        if h[i] > hi_p:
            hi_i, hi_p = i, h[i]
        if l[i] < lo_p:
            lo_i, lo_p = i, l[i]
        if trend >= 0 and l[i] < hi_p * (1 - th) and hi_i > (piv[-1][0] if piv else -1):
            piv.append((hi_i, hi_p, "H")); trend = -1; lo_i, lo_p = i, l[i]
        elif trend <= 0 and h[i] > lo_p * (1 + th) and lo_i > (piv[-1][0] if piv else -1):
            piv.append((lo_i, lo_p, "L")); trend = 1; hi_i, hi_p = i, h[i]
    return piv


def retrace_ratios(piv):
    r = []
    for a, b, c in zip(piv, piv[1:], piv[2:]):
        leg = abs(b[1] - a[1])
        if leg > 0:
            r.append(abs(c[1] - b[1]) / leg)
    return np.array(r)


def near_share(r, levels, tol=0.03):
    r = r[(r > 0.25) & (r < 0.9)]
    return np.mean([any(abs(x - lv) <= tol for lv in levels) for x in r]) if len(r) else np.nan, len(r)


def bounce_test(df, piv, levels, n=0.03):
    """各スイングA→Bの後、価格が水準に初タッチ→先に+n反発(勝)か-n割れ(負)か。"""
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    win = tot = 0
    for (ai, ap, at), (bi, bp, bt), nxt in zip(piv, piv[1:], piv[2:] + [(len(df) - 1, None, None)]):
        up = bt == "H"
        end = nxt[0] if nxt[1] is not None else len(df) - 1
        for lv in levels:
            px = bp - (bp - ap) * lv
            for i in range(bi + 1, min(end + 60, len(df))):
                touched = l[i] <= px if up else h[i] >= px
                if touched:
                    for j in range(i + 1, min(i + 60, len(df))):
                        if up:
                            if h[j] >= px * (1 + n): win += 1; tot += 1; break
                            if l[j] <= px * (1 - n): tot += 1; break
                        else:
                            if l[j] <= px * (1 - n): win += 1; tot += 1; break
                            if h[j] >= px * (1 + n): tot += 1; break
                    break
    return win / tot if tot else np.nan, tot


def gold_daily():
    import json, urllib.request
    u = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F?period1=946684800&period2=4102444800&interval=1d"
    d = json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})))["chart"]["result"][0]
    q = d["indicators"]["quote"][0]
    return pd.DataFrame({k: q[k] for k in ["open", "high", "low", "close"]},
                        index=pd.to_datetime(d["timestamp"], unit="s")).dropna()


ANN = 365


def stats(ret):
    ret = ret.dropna()
    eq = (1 + ret).cumprod()
    yrs = len(ret) / ANN
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1).min()
    sharpe = ret.mean() / ret.std() * np.sqrt(ANN) if ret.std() else np.nan
    return f"年率{cagr*100:6.1f}%  最大DD{dd*100:6.1f}%  シャープ{sharpe:5.2f}"


def trend_test(df, n):
    c = df["close"]
    r = c.pct_change()
    sig = (c > c.rolling(n).mean()).shift(1).fillna(False)
    fee = sig.astype(int).diff().abs().fillna(0) * 0.001
    k = sig.index >= c.index[n]
    return r[k], (r * sig - fee)[k], sig.mean()


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


if __name__ == "__main__":
    data = {
        "BTC": (binance_all("BTCUSDT"), 0.15, 0.03),
        "ETH": (binance_all("ETHUSDT"), 0.20, 0.04),
        "Gold": (gold_daily(), 0.04, 0.01),
    }
    for name, (df, th, n) in data.items():
        print(f"\n##### {name}  {df.index[0].date()}〜{df.index[-1].date()}  ({len(df)}日)")
        ANN = 252 if name == "Gold" else 365
        globals()["ANN"] = ANN
        rs_all = []
        for t in (th / 2, th * 0.75, th):
            rs_all.append(retrace_ratios(zigzag(df, t)))
        r = np.concatenate(rs_all)
        piv = zigzag(df, th / 2)
        fs, cnt = near_share(r, FIB)
        ds, _ = near_share(r, DUMMY)
        print(f"[1] スイング{cnt}本: 押し戻り比率が±0.03以内に入る割合  フィボ{fs*100:.0f}%  ダミー{ds*100:.0f}%"
              f"  (一様分布なら フィボ{4*0.06/0.65*100:.0f}% / ダミー{5*0.06/0.65*100:.0f}%)")
        print(f"    押し戻り比率の中央値 {np.median(r):.2f}  四分位 {np.percentile(r,25):.2f}〜{np.percentile(r,75):.2f}")
        fw, ft = bounce_test(df, piv, FIB, n)
        dw, dt = bounce_test(df, piv, DUMMY, n)
        print(f"[2] 水準タッチ後に先に±{n*100:.0f}%動いた時の反発率  フィボ{fw*100:.0f}% (n={ft})  ダミー{dw*100:.0f}% (n={dt})")
        for sma in (50, 100, 200):
            bh, tf, expo = trend_test(df, sma)
            print(f"[3] SMA{sma:3d}上のみ保有: {stats(tf)}  (保有率{expo*100:.0f}%)")
        print(f"    買い持ち        : {stats(bh)}")
        c = df["close"]
        fwd = c.shift(-20) / c - 1
        rs = rsi(c)
        print(f"[4] 20日後リターン  全期間平均{fwd.mean()*100:5.1f}% (上昇確率{(fwd>0).mean()*100:.0f}%)"
              f" | RSI<30後 {fwd[rs<30].mean()*100:5.1f}% ({(fwd[rs<30]>0).mean()*100:.0f}%, n={(rs<30).sum()})"
              f" | RSI>70後 {fwd[rs>70].mean()*100:5.1f}% ({(fwd[rs>70]>0).mean()*100:.0f}%, n={(rs>70).sum()})")
