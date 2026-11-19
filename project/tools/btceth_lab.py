#!/usr/bin/env python3
"""BTC・ETHに絞った詳しい検証：MACD・一目均衡表・ボリンジャーの設定ずらし、前半/後半、大勝ちを除いた成績、組み合わせ。

・損益は毎日の含み損益まで入れて計算する（含み損込みの落ち込み）。
・1回の量: 入る時点のATR1本分の値動き＝資金の0.5%（単利）。
・売買: 終値で判定→翌日の始値で売買（値幅ブレイクは当日中に逆指値で買い、その日の終値で決済）。
・コスト: 往復 BTC/ETH 0.05%、買いのスワップ年 BTC25%・ETH35%（Vantage並み）。「現物」はスワップなし。
・条件: なし／BTCが200日平均の上／トレンドがはっきり（効率性レシオ≥0.2）……入る判定の日の値で判定。
使い方: python3 project/tools/btceth_lab.py <キャッシュ（BTC.csv, ETH.csv）>
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

START, SPLIT = pd.Timestamp("2018-01-01"), pd.Timestamp("2023-01-01")
RISK = 0.5
COINS = {"BTC": (0.0005, 25.0), "ETH": (0.0005, 35.0)}


def load(cache, a):
    df = pd.read_csv(os.path.join(cache, f"{a}.csv"), index_col=0, parse_dates=True)
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    return df[~df.index.duplicated()].sort_index()


def ind(df, btc):
    o, h, l, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - l, abs(h - pc), abs(l - pc)]), axis=0)
    cs = pd.Series(c)
    er = ((cs - cs.shift(20)).abs() / cs.diff().abs().rolling(20).sum()).to_numpy()
    b = btc.close.reindex(df.index)
    return dict(o=o, h=h, l=l, c=c, idx=df.index, atr=pd.Series(tr).rolling(14).mean().to_numpy(), er=er,
                btc200=(b > b.rolling(200).mean()).to_numpy())


def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def state(enter, exit_):
    pos, on = np.zeros(len(enter)), False
    for i in range(len(enter)):
        if not on and enter[i]:
            on = True
        elif on and exit_[i]:
            on = False
        pos[i] = on
    return pos


def macd_pos(I, f=12, s=26, g=9):
    m = ema(I["c"], f) - ema(I["c"], s)
    return (m > ema(m, g)).astype(float)


def ichimoku_pos(I, t=9, k=26, b=52):
    H, L, c = pd.Series(I["h"]), pd.Series(I["l"]), I["c"]
    tenkan = ((H.rolling(t).max() + L.rolling(t).min()) / 2).to_numpy()
    kijun = ((H.rolling(k).max() + L.rolling(k).min()) / 2).to_numpy()
    span_a = pd.Series((tenkan + kijun) / 2).shift(k).to_numpy()
    span_b = ((H.rolling(b).max() + L.rolling(b).min()) / 2).shift(k).to_numpy()
    return state((c > np.fmax(span_a, span_b)) & (tenkan > kijun), c < kijun)


def bb_pos(I, n=20, k=2.0):
    c = pd.Series(I["c"])
    mid, sd = c.rolling(n).mean(), c.rolling(n).std()
    return state((c > mid + k * sd).to_numpy(), (c < mid).to_numpy())


def gate(I, filt):
    if filt == "BTC200日上":
        return I["btc200"]
    if filt == "トレンド明確":
        return I["er"] >= 0.2
    return np.ones(len(I["c"]), bool)


def pos_daily(I, pos, filt, cost, swap_pct):
    """保有状態→毎日の損益（R）と取引の一覧。入る判定の日に条件を満たさない取引は見送る。"""
    o, c, atr = I["o"], I["c"], I["atr"]
    ok = gate(I, filt)
    sw = swap_pct / 36500
    N, daily, trades = len(c), np.zeros(len(c)), []
    p = np.nan_to_num(pos)
    i = 0
    while i < N - 1:
        if p[i] == 1 and (i == 0 or p[i - 1] == 0):
            j = i
            while j + 1 < N and p[j + 1] == 1:
                j += 1
            if ok[i] and atr[i] > 0:
                # 終値で「持たない」に変わった（j+1）のを確認して、翌日（j+2）の始値で決済
                d, e, u = i + 1, (j + 2 if j + 2 < N else None), 1.0 / atr[i]
                last = e if e is not None else N
                seg = np.zeros(N)
                seg[d] = (c[d] - o[d]) * u if (e is None or e > d) else 0
                if last - 1 > d:
                    seg[d + 1:last] = (c[d + 1:last] - c[d:last - 1]) * u
                if e is not None:
                    seg[e] = (o[e] - c[e - 1]) * u
                    seg[e] -= cost / 2 * o[e] * u
                seg[d] -= cost / 2 * o[d] * u
                seg[d:last] -= sw * c[d:last] * u
                daily += seg
                if e is not None:
                    trades.append((I["idx"][d], I["idx"][e], seg.sum()))
            i = j + 1
        else:
            i += 1
    return daily, pd.DataFrame(trades, columns=["t", "exit_t", "R"])


def vbo_daily(I, filt, cost, swap_pct, k=0.8, n=5):
    o, h, l, c = I["o"], I["h"], I["l"], I["c"]
    atr = np.r_[np.nan, I["atr"][:-1]]
    rng = np.r_[np.nan, (h - l)[:-1]]
    ma = np.r_[np.nan, pd.Series(c).rolling(n).mean().to_numpy()[:-1]]
    okf = np.r_[False, gate(I, filt)[:-1]]
    trig = o + k * rng
    hit = (h >= trig) & (o > ma) & okf & (atr > 0) & (rng > 0)
    R = np.where(hit, (c - trig) / atr - cost * trig / atr - swap_pct / 36500 * trig / atr, 0.0)
    R = np.nan_to_num(R)
    t = I["idx"][hit]
    return R, pd.DataFrame({"t": t, "exit_t": t, "R": R[hit]})


def run(data, kind, filt="なし", params=None, spot=False):
    """BTC+ETHの毎日の損益（R、日付でそろえて合計）と取引の一覧。"""
    days, trs = [], []
    for a, I in data.items():
        cost, sw = COINS[a]
        sw = 0.0 if spot else sw
        if kind == "値幅ブレイク":
            d, t = vbo_daily(I, filt, cost, sw)
        else:
            pos = {"MACD": macd_pos, "一目均衡表": ichimoku_pos, "ボリンジャー": bb_pos}[kind](I, *(params or ()))
            d, t = pos_daily(I, pos, filt, cost, sw)
        days.append(pd.Series(d, index=I["idx"]))
        trs.append(t)
    daily = pd.concat(days, axis=1).fillna(0).sum(axis=1)
    return daily[daily.index >= START], pd.concat(trs).sort_values("exit_t").pipe(lambda x: x[x.t >= START])


def stats(daily, trades, since=START, until=None):
    d = daily[(daily.index >= since) & ((daily.index < until) if until is not None else True)] * RISK
    tr = trades[(trades.t >= since) & ((trades.t < until) if until is not None else True)]
    eq = d.cumsum()
    yrs = max((d.index[-1] - d.index[0]).days / 365.25, 0.5)
    yearly = d.groupby(d.index.year).sum()
    R = tr.R.to_numpy()
    real = np.cumsum(R) * RISK
    top = np.sort(R)[-5:].sum() / R.sum() if R.sum() > 0 else np.nan
    return dict(annual=d.sum() / yrs, dd=(eq.cummax() - eq).max(), dd_real=(np.maximum.accumulate(np.r_[0, real])[1:] - real).max(),
                yearly=yearly, neg=[int(y) for y, v in yearly.items() if v < 0], worst=yearly.min(), n_yr=len(R) / yrs,
                win=(R > 0).mean() if len(R) else np.nan, pf=R[R > 0].sum() / -R[R < 0].sum() if (R < 0).any() else np.inf,
                top5=top, no_top3=(np.sort(R)[:-3].sum() * RISK / yrs) if len(R) > 3 else np.nan)


def line(name, s):
    neg = ",".join(str(v)[2:] for v in s["neg"]) or "なし"
    return (f"{name:28s} 年{s['n_yr']:4.0f}回 勝率{s['win']*100:3.0f}% PF{s['pf']:5.2f} 年平均{s['annual']:+5.0f}%"
            f" 落込(含み込み){s['dd']:4.0f}%（決済だけ{s['dd_real']:3.0f}%） 最悪の年{s['worst']:+4.0f}% マイナスの年 {neg}"
            f" ｜ 上位5回{s['top5']*100:3.0f}% 上位3回を除く年{s['no_top3']:+.0f}%")


def main(cache):
    btc = load(cache, "BTC")
    data = {a: ind(load(cache, a), btc) for a in COINS}

    print("==== (1) 設定をずらした場合（BTC+ETH、条件なし）前半PF→後半PF ／ 全期間の年平均・含み込みの落ち込み・マイナスの年")
    for kind, grid in (("MACD", list(itertools.product((8, 12, 16), (21, 26, 35), (5, 9, 13)))),
                       ("一目均衡表", [(7, 22, 44), (9, 26, 52), (10, 30, 60), (12, 36, 72), (20, 60, 120)]),
                       ("ボリンジャー", list(itertools.product((15, 20, 30), (1.5, 2.0, 2.5))))):
        res = []
        for p in grid:
            d, t = run(data, kind, "なし", p)
            a, b, f = stats(d, t, START, SPLIT), stats(d, t, SPLIT), stats(d, t)
            res.append((p, a["pf"], b["pf"], f))
        both = sum(1 for r in res if r[1] > 1 and r[2] > 1)
        nonneg = sum(1 for r in res if not r[3]["neg"])
        print(f"\n-- {kind}: {len(res)}通り ｜ 前半・後半ともPF>1: {both}通り ｜ マイナスの年なし: {nonneg}通り"
              f" ｜ 年平均の中央値 {np.median([r[3]['annual'] for r in res]):+.0f}% ｜ 含み込みの落ち込みの中央値 {np.median([r[3]['dd'] for r in res]):.0f}%")
        show = res if len(res) <= 9 else [r for r in res if r[0] in ((12, 26, 9), (8, 21, 5), (16, 35, 13), (8, 35, 9), (16, 21, 9))]
        for p, pa, pb, f in show:
            neg = ",".join(str(v)[2:] for v in f["neg"]) or "なし"
            print(f"   設定{p}: 前半PF{pa:.2f}→後半PF{pb:.2f} ｜ 年平均{f['annual']:+.0f}% 落込{f['dd']:.0f}% マイナスの年 {neg}")

    print("\n==== (2) 標準設定の詳細（条件別）")
    for kind, p in (("値幅ブレイク", None), ("MACD", (12, 26, 9)), ("一目均衡表", (9, 26, 52)), ("ボリンジャー", (20, 2.0))):
        for filt in ("なし", "BTC200日上", "トレンド明確"):
            d, t = run(data, kind, filt, p)
            print(line(f"{kind}・{filt}", stats(d, t)))

    print("\n==== (3) 組み合わせ（BTC+ETH、それぞれ1回の量はATR1本＝0.5%、同時に運用）")
    combos = {
        "値幅ブレイク(トレンド明確)+MACD": [("値幅ブレイク", "トレンド明確", None), ("MACD", "なし", (12, 26, 9))],
        "値幅ブレイク(トレンド明確)+一目(BTC200)": [("値幅ブレイク", "トレンド明確", None), ("一目均衡表", "BTC200日上", (9, 26, 52))],
        "値幅+MACD+一目": [("値幅ブレイク", "トレンド明確", None), ("MACD", "なし", (12, 26, 9)), ("一目均衡表", "BTC200日上", (9, 26, 52))],
        "値幅+MACD+一目+ボリンジャー": [("値幅ブレイク", "トレンド明確", None), ("MACD", "なし", (12, 26, 9)),
                                ("一目均衡表", "BTC200日上", (9, 26, 52)), ("ボリンジャー", "なし", (20, 2.0))],
    }
    parts = {}
    for name, items in combos.items():
        ds, ts = [], []
        for kind, filt, p in items:
            key = (kind, filt, p)
            if key not in parts:
                parts[key] = run(data, kind, filt, p)
            ds.append(parts[key][0])
            ts.append(parts[key][1])
        d = pd.concat(ds, axis=1).fillna(0).sum(axis=1)
        t = pd.concat(ts).sort_values("exit_t")
        s = stats(d, t)
        print(line(name, s))
        print("      年別: " + " ".join(f"{y}:{v:+.0f}%" for y, v in s["yearly"].items())
              + f" ｜ 前半PF{stats(d, t, START, SPLIT)['pf']:.2f}→後半PF{stats(d, t, SPLIT)['pf']:.2f}")
    yearly = pd.DataFrame({k[0] + "・" + k[1]: (v[0] * RISK).groupby(v[0].index.year).sum() for k, v in parts.items()})
    print("\n   年ごとの損益（％）の動き方の似かた（相関）:")
    print(yearly.corr().round(2).to_string())
    print("\n==== (4) 現物（スワップなし）の場合")
    for kind, filt, p in (("MACD", "なし", (12, 26, 9)), ("一目均衡表", "BTC200日上", (9, 26, 52))):
        d, t = run(data, kind, filt, p, spot=True)
        print(line(f"{kind}・{filt}・現物", stats(d, t)))


if __name__ == "__main__":
    main(sys.argv[1])
