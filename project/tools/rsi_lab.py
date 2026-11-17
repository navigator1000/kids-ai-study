#!/usr/bin/env python3
"""RSIの使い方別の検証（日足、BTC / ETH / Gold）。

RSIの代表的な使い方と、採用ルール（20日高値ブレイク）・ランダム買いを同じ条件で比べる。
  RSI逆張り買い  : RSI(14)が30未満で買い（売られすぎ）
  RSI逆張り売り  : RSI(14)が70超で売り（買われすぎ）
  RSI順張り買い  : RSI(14)が70超で買い（勢いが強い）
  押し目買い(RSI2): 200日線より上で RSI(2)が10未満なら買い（上昇中の短い押し目）
  押し目買い(RSI14): 200日線より上で RSI(14)が40未満なら買い
  20日高値ブレイク: 採用ルール（200日線より上で20日高値更新）
  ランダム買い    : 毎日買う（比較の基準）
出口は「損切りATR×s・利確R倍・最大H日」と「RSIが戻ったら決済（逆張り系のみ）」を総当たりし、
学習期間で選んだ出口を検証期間で確かめる。
コスト: 往復 BTC/ETH 0.2%、金 0.05%。スワップ: Vantageの実際の率
（買い BTC年25%・ETH年35%・金年6.5%、売り BTC/ETH 0%・金は年2.9%受け取り）。
ルール選定に使っていない11銘柄では、選んだ設定をそのまま全期間で検証する（スワップ抜き）。
使い方: python3 project/tools/rsi_lab.py <キャッシュ用ディレクトリ>
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
import shortterm_lab as T  # noqa: E402
from marketdata import load_all  # noqa: E402

COST = {"BTC": 0.002, "ETH": 0.002, "Gold": 0.0005}
SPLIT = {"BTC": "2023-01-01", "ETH": "2023-01-01", "Gold": "2016-01-01"}
SWAP_LONG = {"BTC": 25.35, "ETH": 35.49, "Gold": 6.53}
SWAP_SHORT = {"BTC": 0.0, "ETH": 0.0, "Gold": -2.9}
UNSEEN = ["SOL", "XRP", "BNB", "ADA", "LTC", "DOGE", "Silver", "S&P500", "Nikkei", "USDJPY", "Crude"]
CRYPTO = {"BTC", "ETH", "SOL", "XRP", "BNB", "ADA", "LTC", "DOGE"}


def rsi(c, n):
    d = np.diff(c, prepend=np.nan)
    up = pd.Series(np.clip(d, 0, None)).ewm(alpha=1 / n, adjust=False).mean()
    dn = pd.Series(np.clip(-d, 0, None)).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + up / dn)).to_numpy()


def families(P):
    c = P["c"]
    r14, r2 = rsi(c, 14), rsi(c, 2)
    up = c > pd.Series(c).rolling(200).mean().to_numpy()
    hh = pd.Series(P["h"]).rolling(20).max().shift(1).to_numpy()
    return {   # 名前: (向き, 入る条件, RSIの戻りで決済する条件)
        "RSI逆張り買い(<30)": (1, r14 < 30, r14 > 50),
        "RSI逆張り売り(>70)": (-1, r14 > 70, r14 < 50),
        "RSI順張り買い(>70)": (1, r14 > 70, None),
        "押し目買い(RSI2<10)": (1, up & (r2 < 10), r2 > 70),
        "押し目買い(RSI14<40)": (1, up & (r14 < 40), r14 > 55),
        "20日高値ブレイク": (1, up & (c > hh), None),
        "ランダム買い": (1, np.ones(len(c), bool), None),
    }


def run(P, sig, side, s, tR, H, cost, swap_pct, exit_sig=None):
    o, h, l, c, atr, t = P["o"], P["h"], P["l"], P["c"], P["atr"], P["t"]
    N, free, rows = len(c), 0, []
    BIG = 10**9
    for i in np.flatnonzero(sig[:-1]):
        if i < free or i < 200:
            continue
        j = i + 1
        fill, risk = o[j], s * atr[i]
        if not risk > 0:
            continue
        stop = fill - side * risk
        tgt = fill + side * tR * risk if tR else None
        end = min(j + H, N)
        adv = l[j:end] if side == 1 else h[j:end]
        fav = h[j:end] if side == 1 else l[j:end]
        hs = (adv - stop) * side <= 0
        a = hs.argmax() if hs.any() else BIG
        b = BIG
        if tgt is not None:
            ht = (fav - tgt) * side >= 0
            b = ht.argmax() if ht.any() else BIG
        e = BIG
        if exit_sig is not None:
            es = exit_sig[j:end]
            e = es.argmax() if es.any() else BIG
        first = min(a, b, e)
        if first == BIG:
            if end - j < H:
                break
            k, ex, add = end - 1, c[end - 1], 1.0
        elif a == first:                                   # 同じ足なら損切りを優先（保守的）
            k = j + a
            ex = o[k] if k > j and (o[k] - stop) * side <= 0 else stop
            add = 0.5
        elif b == first:                                   # 利確は足の途中、RSI決済は終値
            k = j + b
            ex = o[k] if k > j and (o[k] - tgt) * side >= 0 else tgt
            add = 0.5
        else:
            k, ex, add = j + e, c[j + e], 1.0
        nights = (t[k] - t[j]) / np.timedelta64(1, "D") + add
        R = side * (ex - fill) / risk - cost * fill / risk - swap_pct / 36500 * nights * fill / risk
        rows.append((t[j], R))
        free = k
    return pd.DataFrame(rows, columns=["t", "R"])


EXITS = ([(s, tR, H, False) for s, tR, H in itertools.product((1, 2), (1, 2, 4), (5, 10, 20, 40))]
         + [(s, None, H, True) for s, H in itertools.product((2, 3), (5, 10, 20))])


def exit_label(s, tR, H, ind):
    return f"損切りATR×{s}・{'RSIが戻ったら決済' if ind else f'利確{tR}R'}・最大{H}日"


def main(cache):
    data = load_all(cache)
    Ps = {a: T.prep(df) for a, df in data.items()}
    Fs = {a: families(P) for a, P in Ps.items()}
    rows = []
    for fam in Fs["BTC"]:
        for s, tR, H, ind in EXITS:
            side, _, ex_sig = Fs["BTC"][fam]
            if ind and ex_sig is None:
                continue
            for a, P in Ps.items():
                side, sig, ex_sig = Fs[a][fam]
                swap = SWAP_LONG[a] if side == 1 else SWAP_SHORT[a]
                tr = run(P, sig, side, s, tR, H, COST[a], swap, ex_sig if ind else None)
                sp = pd.Timestamp(SPLIT[a])
                tin, tout = tr[tr.t < sp], tr[tr.t >= sp]
                yrs = (P["t"][-1] - sp.to_datetime64()) / np.timedelta64(365, "D")
                rows.append(dict(fam=fam, s=s, tR=tR, H=H, ind=ind, asset=a,
                                 in_n=len(tin), in_exp=tin.R.mean(), out_n=len(tout), out_exp=tout.R.mean(),
                                 out_win=(tout.R > 0).mean(), out_yr=tout.R.sum() / yrs))
    df = pd.DataFrame(rows)
    g = df.groupby(["fam", "s", "tR", "H", "ind"], dropna=False).agg(
        in_exp=("in_exp", "mean"), in_n=("in_n", "min"), out_exp=("out_exp", "mean"),
        out_min=("out_exp", "min"), out_yr=("out_yr", "sum")).reset_index()
    print("===== BTC・ETH・金（コスト・スワップ込み）：学習期間で選んだ出口の、検証期間の成績 =====")
    best = {}
    for fam in Fs["BTC"]:
        x = g[(g.fam == fam) & (g.in_n >= 15)].sort_values("in_exp", ascending=False)
        if not len(x):
            continue
        b = x.iloc[0]
        best[fam] = b
        per = df[(df.fam == fam) & (df.s == b.s) & (df.H == b.H) & (df.ind == b.ind)
                 & ((df.tR == b.tR) | (df.tR.isna() & pd.isna(b.tR)))]
        pa = " ".join(f"{r.asset}{r.out_exp:+.2f}" for r in per.itertuples())
        print(f"{fam:18s} 出口[{exit_label(b.s, b.tR, int(b.H), b.ind)}] 学習{b.in_exp:+.2f}R → 検証{b.out_exp:+.2f}R"
              f"（{pa}）年{b.out_yr:+.1f}R ｜ 出口{len(x)}通り中 検証プラス{(x.out_exp > 0).mean()*100:.0f}%")

    print("\n===== ルール選定に使っていない11銘柄（上で選んだ設定のまま、全期間・コスト込み・スワップ抜き） =====")
    for fam, b in best.items():
        res = []
        for a in UNSEEN:
            path = os.path.join(cache, f"{a}.csv")
            if not os.path.exists(path):
                continue
            d = pd.read_csv(path, index_col=0, parse_dates=True)
            d = d[(d[["open", "high", "low", "close"]] > 0).all(axis=1)]
            P = T.prep(d)
            side, sig, ex_sig = families(P)[fam]
            tr = run(P, sig, side, b.s, b.tR, int(b.H), 0.002 if a in CRYPTO else 0.0005, 0.0,
                     ex_sig if b.ind else None)
            if len(tr) >= 10:
                res.append(tr.R.mean())
        res = np.array(res)
        print(f"{fam:18s} プラス {int((res > 0).sum())}/{len(res)}銘柄 ｜ 平均{res.mean():+.2f}R")


if __name__ == "__main__":
    main(sys.argv[1])
