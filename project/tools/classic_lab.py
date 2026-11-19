#!/usr/bin/env python3
"""定番ロジックの比較（日足・現物で全額を入れる／抜く想定、2018年〜）。

  ① 移動平均クロス（5/20・20/50・50/200）  ② MACD(12,26,9)がシグナルより上
  ③ ボリンジャーバンド(20,2)上抜けで買い・中央線割れで売り
  ④ タートル（20日高値で買い・10日安値で売り／55日・20日）
  ⑤ 一目均衡表（雲の上かつ転換線＞基準線で買い・基準線割れで売り）  ⑥ パラボリックSAR(0.02, 0.2)
  ⑦ 強い銘柄に乗り換え（毎週月曜、8銘柄の過去28日の上昇率上位2つ。BTCが50日平均より下なら現金）
  ⑧ アノマリー（曜日・月。BTC-USD 2014年〜）
売買は終値で判定して次の日から反映、コストは往復0.15%（片道0.075%）、スワップなし（現物）。
比較の基準は「ずっと持つ（ガチホ）」。
使い方: python3 project/tools/classic_lab.py <キャッシュ（BTC.csv, ETH.csv, SOL.csv など日足）>
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
from marketdata import yahoo_daily  # noqa: E402

START, SPLIT = pd.Timestamp("2018-01-01"), pd.Timestamp("2023-01-01")
COST = 0.0015
NO_TRADES = pd.DataFrame({"t": pd.to_datetime(pd.Series([], dtype="datetime64[ns]")), "r": pd.Series([], dtype=float)})
ALTS = ("SOL", "XRP", "BNB", "ADA", "LTC", "DOGE")


def load(cache, a):
    df = pd.read_csv(os.path.join(cache, f"{a}.csv"), index_col=0, parse_dates=True)
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    return df[~df.index.duplicated()]


# ---------------------------------------------------------------- 指標とポジション（1＝持つ、0＝持たない）
def sma(x, n):
    return pd.Series(x).rolling(n).mean().to_numpy()


def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def state(enter, exit_):
    """入る条件・出る条件から保有状態を作る。"""
    pos, on = np.zeros(len(enter)), False
    for i in range(len(enter)):
        if not on and enter[i]:
            on = True
        elif on and exit_[i]:
            on = False
        pos[i] = on
    return pos


def psar(h, l, af0=0.02, step=0.02, amax=0.2):
    n = len(h)
    sar, up, ep, af = np.zeros(n), True, h[0], af0
    sar[0] = l[0]
    for i in range(1, n):
        prev = sar[i - 1] + af * (ep - sar[i - 1])
        if up:
            prev = min(prev, l[i - 1], l[i - 2] if i > 1 else l[i - 1])
            if l[i] < prev:
                up, sar[i], ep, af = False, ep, l[i], af0
                continue
            if h[i] > ep:
                ep, af = h[i], min(af + step, amax)
        else:
            prev = max(prev, h[i - 1], h[i - 2] if i > 1 else h[i - 1])
            if h[i] > prev:
                up, sar[i], ep, af = True, ep, h[i], af0
                continue
            if l[i] < ep:
                ep, af = l[i], min(af + step, amax)
        sar[i] = prev
    return sar


def positions(df):
    o, h, l, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    H, L = pd.Series(h), pd.Series(l)
    P = {}
    for f, s in ((5, 20), (20, 50), (50, 200)):
        P[f"①移動平均クロス {f}/{s}日"] = (sma(c, f) > sma(c, s)).astype(float)
    macd = ema(c, 12) - ema(c, 26)
    P["②MACD(12,26,9)"] = (macd > ema(macd, 9)).astype(float)
    mid, sd = sma(c, 20), pd.Series(c).rolling(20).std().to_numpy()
    P["③ボリンジャー上抜け"] = state(c > mid + 2 * sd, c < mid)
    for a, b in ((20, 10), (55, 20)):
        P[f"④タートル {a}日高値/{b}日安値"] = state(c > H.rolling(a).max().shift(1).to_numpy(), c < L.rolling(b).min().shift(1).to_numpy())
    tenkan = ((H.rolling(9).max() + L.rolling(9).min()) / 2).to_numpy()
    kijun = ((H.rolling(26).max() + L.rolling(26).min()) / 2).to_numpy()
    span_a = pd.Series((tenkan + kijun) / 2).shift(26).to_numpy()
    span_b = ((H.rolling(52).max() + L.rolling(52).min()) / 2).shift(26).to_numpy()
    P["⑤一目均衡表"] = state((c > np.fmax(span_a, span_b)) & (tenkan > kijun), c < kijun)
    P["⑥パラボリックSAR"] = (c > psar(h, l)).astype(float)
    P["（参考）ずっと持つ"] = np.ones(len(c))
    return P


# ---------------------------------------------------------------- 損益の計算
def backtest(df, pos):
    c = df.close.to_numpy(float)
    ret = np.r_[0.0, c[1:] / c[:-1] - 1]
    p = np.nan_to_num(pos)
    held = np.r_[0.0, p[:-1]]
    turn = np.abs(np.diff(np.r_[0.0, p]))
    daily = held * ret - np.r_[0.0, turn[:-1]] * COST / 2
    trades, i = [], 0
    while i < len(p):
        if p[i] == 1 and (i == 0 or p[i - 1] == 0):
            j = i
            while j + 1 < len(p) and p[j + 1] == 1:
                j += 1
            k = min(j + 1, len(c) - 1)
            trades.append((df.index[i], c[k] / c[i] - 1 - COST))
            i = j + 1
        else:
            i += 1
    return pd.Series(daily, index=df.index), pd.DataFrame(trades, columns=["t", "r"])


def metrics(daily, trades, since=START):
    d = daily[daily.index >= since]
    eq = (1 + d).cumprod()
    yrs = (d.index[-1] - d.index[0]).days / 365.25
    yearly = (1 + d).groupby(d.index.year).prod() - 1
    t = trades[trades.t >= since].r.to_numpy()
    win = t[t > 0]
    loss = -t[t < 0]
    return dict(cagr=eq.iloc[-1] ** (1 / yrs) - 1, dd=(1 - eq / eq.cummax()).max(), yearly=yearly,
                neg=[int(y) for y, v in yearly.items() if v < 0], n_yr=len(t) / yrs, win=(t > 0).mean() if len(t) else np.nan,
                payoff=win.mean() / loss.mean() if len(win) and len(loss) else np.nan,
                pf=win.sum() / loss.sum() if len(loss) else np.inf, expo=(d != 0).mean())


def classic(cache):
    for a in ("BTC", "ETH"):
        df = load(cache, a)
        print(f"\n===== {a}（2018年〜、現物、往復0.15%）")
        print(f"{'ロジック':24s} 年率   後半年率 最大落込 マイナスの年                 年回数 勝率  損益比  PF")
        for name, pos in positions(df).items():
            daily, trades = backtest(df, pos)
            m, mt = metrics(daily, trades), metrics(daily, trades, SPLIT)
            neg = ",".join(str(y)[2:] for y in m["neg"]) or "なし"
            print(f"{name:24s} {m['cagr']*100:+5.0f}% {mt['cagr']*100:+6.0f}% {m['dd']*100:6.0f}%  {neg:28s}"
                  f" {m['n_yr']:5.1f} {m['win']*100:4.0f}% {m['payoff']:5.2f} {m['pf']:5.2f}")


def rotation(cache):
    names = ("BTC", "ETH") + ALTS
    closes = pd.concat({a: load(cache, a).close for a in names}, axis=1).sort_index()
    closes = closes[closes.index >= pd.Timestamp("2017-10-01")]
    rets = closes.pct_change()
    mom = closes / closes.shift(28) - 1
    btc_ok = closes["BTC"] > closes["BTC"].rolling(50).mean()
    print("\n===== ⑦強い銘柄に乗り換え（8銘柄、毎週月曜に入れ替え、往復0.15%）")
    for use_filter in (False, True):
        w = pd.DataFrame(0.0, index=closes.index, columns=names)
        cur = pd.Series(0.0, index=names)
        for t in closes.index:
            if t.weekday() == 0:
                m = mom.loc[t].dropna()
                m = m[m > 0].sort_values(ascending=False).head(2)
                cur = pd.Series(0.0, index=names)
                if len(m) and (not use_filter or btc_ok.loc[t]):
                    cur[m.index] = 1.0 / len(m)
            w.loc[t] = cur
        held = w.shift(1).fillna(0)
        turn = w.diff().abs().sum(axis=1).shift(1).fillna(0)
        daily = (held * rets.fillna(0)).sum(axis=1) - turn * COST / 2
        m, mt = metrics(daily, NO_TRADES), metrics(daily, NO_TRADES, SPLIT)
        print(f"  上位2銘柄{'＋BTCが50日平均より上の時だけ' if use_filter else ''}: 年率{m['cagr']*100:+.0f}%（後半{mt['cagr']*100:+.0f}%）"
              f" 最大の落ち込み{m['dd']*100:.0f}% マイナスの年 {m['neg'] or 'なし'}")
    eqw = rets.fillna(0).mean(axis=1)
    m = metrics(eqw, NO_TRADES)
    print(f"  （参考）8銘柄を均等にずっと持つ: 年率{m['cagr']*100:+.0f}% 最大の落ち込み{m['dd']*100:.0f}% マイナスの年 {m['neg']}")


def anomalies(cache):
    path = os.path.join(cache, "BTC-USD_yahoo.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        df = yahoo_daily("BTC-USD")
        df.to_csv(path)
    c = df.close
    print(f"\n===== ⑧アノマリー（BTC-USD {c.index[0]:%Y-%m}〜）")
    r = c.pct_change().dropna()
    first, second = r[r.index < "2020-07-01"], r[r.index >= "2020-07-01"]
    days = ["月", "火", "水", "木", "金", "土", "日"]
    print("  曜日別の平均（1日の値動き）  前半(〜2020年6月) ／ 後半(2020年7月〜)")
    for k in range(7):
        a, b = first[first.index.weekday == k], second[second.index.weekday == k]
        print(f"   {days[k]}曜: {a.mean()*100:+.2f}%（上昇{(a > 0).mean()*100:.0f}%） ／ {b.mean()*100:+.2f}%（上昇{(b > 0).mean()*100:.0f}%）")
    mon = c.resample("ME").last().pct_change().dropna()
    mon = mon[mon.index.year <= 2025]
    print("  月別（2014〜2025年）: 平均 ／ 上がった年の割合 ／ 年ごとの騰落")
    for mth in range(1, 13):
        x = mon[mon.index.month == mth]
        seq = "".join("＋" if v > 0 else "－" for v in x)
        print(f"   {mth:2d}月: {x.mean()*100:+5.1f}% ／ {(x > 0).mean()*100:3.0f}%（{len(x)}年中{(x > 0).sum()}回） {seq}")


if __name__ == "__main__":
    cache = sys.argv[1]
    classic(cache)
    rotation(cache)
    anomalies(cache)
