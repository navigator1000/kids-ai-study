#!/usr/bin/env python3
"""短期（1日以内）ルールの探索ラボ：BTC 1時間足で探し、前半で選んで後半とETHで確かめる。

型1 値幅ブレイク（ボラティリティ・ブレイクアウト）
    当日の始値（UTC 0時）＋前日の値幅×k を上抜けたら買い（売りは下抜け）。
    条件: 始値が日足n本平均より上（売りは下）。損切り: 約定価格からATR×s。
    決済: UTC20時（日本時間朝5時、スワップがつく前）か UTC24時（1晩分のスワップ）。
型2 時間帯: UTC a時から w時間だけ持つ（条件: 日足n本平均より上／下）。
コスト: スプレッド往復 0.05%（基本）／0.1%（厳しめ）。スワップ: 買いのみ年25%（Vantage表示）。
成績は「1回の損失の目安＝1R＝初期資金の1%（単利）」で円に直して表示する。
使い方: python3 project/tools/intraday_lab.py <キャッシュ用ディレクトリ（BTC_1h.csv, ETH_1h.csv）> [detail]
  detail: 値幅ブレイクの設定ずらし・年別・他の11銘柄での確認
  portfolio: BTCとETHの両方で取引した場合（損切りの有無・コスト別）
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

TRAIN_END, START = "2023-01-01", "2018-01-01"
SWAP_LONG = 25.0 / 365 / 100           # 1晩あたり（建玉比）


def day_matrix(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df[~df.index.duplicated()]
    full = pd.date_range(df.index[0].normalize() + pd.Timedelta(days=1), df.index[-1].normalize() - pd.Timedelta(hours=1), freq="1h")
    df = df.reindex(full).ffill()
    days = df.index.normalize().unique()
    M = {k: df[k].to_numpy().reshape(len(days), 24) for k in ("open", "high", "low", "close")}
    dO, dH, dL, dC = M["open"][:, 0], M["high"].max(1), M["low"].min(1), M["close"][:, -1]
    pc = np.r_[np.nan, dC[:-1]]
    tr = np.nanmax(np.vstack([dH - dL, abs(dH - pc), abs(dL - pc)]), axis=0)
    atr = pd.Series(tr).rolling(14).mean().shift(1).to_numpy()          # 前日までのATR
    rng = np.r_[np.nan, (dH - dL)[:-1]]                                  # 前日の値幅
    sma = {n: pd.Series(dC).rolling(n).mean().shift(1).to_numpy() for n in (3, 5, 10, 20, 50, 200)}
    return dict(days=days, M=M, dO=dO, atr=atr, rng=rng, sma=sma)


def vbo(D, side, k, n, s, exit_h, cost):
    O, H, L, C = D["M"]["open"], D["M"]["high"], D["M"]["low"], D["M"]["close"]
    dO, atr, rng = D["dO"], D["atr"], D["rng"]
    ok = ~np.isnan(atr) & ~np.isnan(rng)
    if n:
        ok &= (dO - D["sma"][n]) * side > 0
    trig = dO + side * k * rng
    hours = np.arange(24)
    hit = ((H >= trig[:, None]) if side == 1 else (L <= trig[:, None])) & (hours[None, :] < exit_h - 1)
    has = hit.any(1) & ok
    h0 = hit.argmax(1)
    d_idx = np.flatnonzero(has)
    h0 = h0[d_idx]
    fill = O[d_idx, h0]
    fill = np.maximum(fill, trig[d_idx]) if side == 1 else np.minimum(fill, trig[d_idx])
    unit = (s if s < 99 else 1.0) * atr[d_idx]
    ex = C[d_idx, exit_h - 1].copy()
    if s < 99:
        stop = fill - side * unit
        after = hours[None, :] >= h0[:, None]
        before_exit = hours[None, :] < exit_h
        hs = ((L[d_idx] <= stop[:, None]) if side == 1 else (H[d_idx] >= stop[:, None])) & after & before_exit
        stopped = hs.any(1)
        first = hs.argmax(1)
        gap_px = O[d_idx, first]
        is_gap = (first > h0) & (((gap_px <= stop) if side == 1 else (gap_px >= stop)))
        px_stop = np.where(is_gap, gap_px, stop)
        ex = np.where(stopped, px_stop, ex)
    R = side * (ex - fill) / unit - cost * fill / unit
    if side == 1 and exit_h == 24:
        R -= SWAP_LONG * fill / unit
    return pd.DataFrame({"t": D["days"][d_idx], "R": R})


def window(D, side, a, w, n, cost):
    O, C = D["M"]["open"].ravel(), D["M"]["close"].ravel()
    nd = len(D["days"])
    i = np.arange(nd) * 24 + a
    j = i + w - 1
    keep = j < len(C)
    i, j = i[keep], j[keep]
    d = i // 24
    ok = ~np.isnan(D["atr"][d])
    if n:
        ok &= (D["dO"][d] - D["sma"][n][d]) * side > 0
    i, j, d = i[ok], j[ok], d[ok]
    unit = D["atr"][d]
    R = side * (C[j] - O[i]) / unit - cost * O[i] / unit
    crosses = (a <= 21 < a + w) | (a + w > 24 + 21)                     # UTC21時（スワップ付与）をまたぐ
    if side == 1 and crosses:
        R -= SWAP_LONG * O[i] / unit
    return pd.DataFrame({"t": D["days"][d], "R": R})


def stats(tr, years=None):
    R = tr.R.to_numpy()
    if len(R) < 30:
        return None
    g, l = R[R > 0].sum(), -R[R < 0].sum()
    eq = 1_000_000 + 10_000 * np.cumsum(R)
    peak = np.maximum.accumulate(np.r_[1_000_000, eq])[1:]
    mx = cur = 0
    for r in R:
        cur = cur + 1 if r <= 0 else 0
        mx = max(mx, cur)
    yrs = years or max((tr.t.iloc[-1] - tr.t.iloc[0]).days / 365.25, 0.5)
    return dict(n=len(R), per_yr=len(R) / yrs, win=(R > 0).mean(), pf=g / l if l else np.inf, exp=R.mean(),
                t=R.mean() / R.std(ddof=1) * np.sqrt(len(R)), payoff=(R[R > 0].mean() / -R[R < 0].mean()) if l else np.inf,
                net=10_000 * R.sum(), dd=((peak - eq) / peak).max(), maxl=mx)


def split(tr):
    tr = tr[tr.t >= pd.Timestamp(START)]
    return tr[tr.t < pd.Timestamp(TRAIN_END)], tr[tr.t >= pd.Timestamp(TRAIN_END)]


def search(B, E, cost):
    rows = []
    for side, k, n, s, eh in itertools.product((1, -1), (0.2, 0.3, 0.4, 0.5, 0.6, 0.8), (0, 3, 5, 10, 20, 50, 200), (0.5, 1.0, 1.5, 99), (20, 24)):
        a, b = split(vbo(B, side, k, n, s, eh, cost))
        sa, sb, se = stats(a), stats(b), stats(vbo(E, side, k, n, s, eh, cost))
        if sa and sb and se:
            rows.append(dict(fam="値幅ブレイク", side=side, p=(k, n, s, eh), tr=sa, te=sb, eth=se))
    for side, a0, w, n in itertools.product((1, -1), range(24), (4, 8, 12), (0, 20, 50, 200)):
        a, b = split(window(B, side, a0, w, n, cost))
        sa, sb, se = stats(a), stats(b), stats(window(E, side, a0, w, n, cost))
        if sa and sb and se:
            rows.append(dict(fam="時間帯", side=side, p=(a0, w, n), tr=sa, te=sb, eth=se))
    return rows


def label(r):
    if r["fam"] == "値幅ブレイク":
        k, n, s, eh = r["p"]
        return (f"値幅ブレイク{'買い' if r['side'] == 1 else '売り'} k={k} 平均{n or 'なし'} "
                f"損切り{'なし' if s == 99 else f'ATR×{s}'} 決済UTC{eh}時")
    a0, w, n = r["p"]
    return f"時間帯{'買い' if r['side'] == 1 else '売り'} UTC{a0}時から{w}時間 平均{n or 'なし'}"


def line(tag, s):
    return (f"{tag}: 年{s['per_yr']:.0f}回 勝率{s['win']*100:.0f}% PF{s['pf']:.2f} 平均{s['exp']:+.3f}R "
            f"損益比{s['payoff']:.2f} 最大DD{s['dd']*100:.1f}% 最大{s['maxl']}連敗")


if __name__ == "__main__" and len(sys.argv) == 2:
    cache = sys.argv[1]
    B, E = day_matrix(os.path.join(cache, "BTC_1h.csv")), day_matrix(os.path.join(cache, "ETH_1h.csv"))
    for cost in (0.0005, 0.001):
        rows = search(B, E, cost)
        print(f"\n######## スプレッド往復{cost*100:.2f}% ｜ 試した数 {len(rows)}通り")
        for fam in ("値幅ブレイク", "時間帯"):
            x = [r for r in rows if r["fam"] == fam and r["tr"]["per_yr"] >= 50]
            x.sort(key=lambda r: r["tr"]["t"], reverse=True)          # 前半の確からしさ（t値）で選ぶ
            pos_te = np.mean([r["te"]["exp"] > 0 for r in x]) * 100
            print(f"\n== {fam}（年50回以上）{len(x)}通り ｜ 後半プラスの割合 {pos_te:.0f}%")
            for r in x[:3]:
                print(f"* {label(r)}")
                print("   " + line("前半2018-22", r["tr"]))
                print("   " + line("後半2023- ", r["te"]))
                print("   " + line("ETH全期間 ", r["eth"]))
        best_all = max(rows, key=lambda r: min(r["tr"]["pf"], r["te"]["pf"]) if r["tr"]["per_yr"] >= 50 else 0)
        print(f"\n（参考）前半・後半の両方を見て一番良いもの: {label(best_all)} 前半PF{best_all['tr']['pf']:.2f} 後半PF{best_all['te']['pf']:.2f}")


# ---------------------------------------------------------------- 値幅ブレイク（買い・損切りなし・UTC24時決済）の確認
def vbo_daily(df, k, n, cost, swap_night):
    """日足だけで計算できる版（損切りなし・その日の終値で決済）。他の銘柄の確認用。"""
    o, h, l, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - l, abs(h - pc), abs(l - pc)]), axis=0)
    atr = pd.Series(tr).rolling(14).mean().shift(1).to_numpy()
    rng = np.r_[np.nan, (h - l)[:-1]]
    trig = o + k * rng
    ok = (h >= trig) & ~np.isnan(atr) & ~np.isnan(rng)
    if n:
        ok &= o > pd.Series(c).rolling(n).mean().shift(1).to_numpy()
    fill = np.maximum(o, trig)
    R = (c - fill) / atr - cost * fill / atr - swap_night * fill / atr
    return pd.DataFrame({"t": df.index[ok], "R": R[ok]})


def detail(cache):
    B, E = day_matrix(os.path.join(cache, "BTC_1h.csv")), day_matrix(os.path.join(cache, "ETH_1h.csv"))
    print("== 設定を少しずらした場合（買い・損切りなし・UTC24時決済、スプレッド往復0.05%）PF 前半/後半/ETH")
    for n in (0, 5, 20, 200):
        cells = []
        for k in (0.4, 0.5, 0.6, 0.7, 0.8, 1.0):
            a, b = split(vbo(B, 1, k, n, 99, 24, 0.0005))
            e = vbo(E, 1, k, n, 99, 24, 0.0005)
            sa, sb, se = stats(a), stats(b), stats(e)
            cells.append(f"k{k}: {sa['pf']:.2f}/{sb['pf']:.2f}/{se['pf']:.2f}")
        print(f"  平均{n or 'なし':>3}: " + "  ".join(cells))
    tr = vbo(B, 1, 0.8, 0, 99, 24, 0.0005)
    tr = tr[tr.t >= pd.Timestamp(START)]
    print("\n== BTC 年別（k=0.8・平均なし、1R＝1万円の単利、100万円スタート）")
    for y, g in tr.groupby(tr.t.dt.year):
        s = stats(g, years=1.0)
        if s:
            print(f"  {y}年: {s['n']:3d}回 勝率{s['win']*100:.0f}% PF{s['pf']:.2f} 損益{s['net']:+,.0f}円")
    s = stats(tr)
    print(f"  合計: {s['n']}回 勝率{s['win']*100:.1f}% PF{s['pf']:.2f} 損益比{s['payoff']:.2f} 純利益{s['net']:+,.0f}円"
          f" 最大DD{s['dd']*100:.1f}% 最大{s['maxl']}連敗 ｜ 1回の最大損失 {tr.R.min():.1f}R（{tr.R.min()*10000:,.0f}円）")
    print("\n== ルール選定に使っていない11銘柄（k=0.8・平均なし・日足で計算、コスト込み・スワップ1晩）")
    res = []
    for a in ("SOL", "XRP", "BNB", "ADA", "LTC", "DOGE", "Silver", "S&P500", "Nikkei", "USDJPY", "Crude"):
        p = os.path.join(cache, f"{a}.csv")
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
        crypto = a in ("SOL", "XRP", "BNB", "ADA", "LTC", "DOGE")
        t = vbo_daily(df, 0.8, 0, 0.0005 if crypto else 0.0002, SWAP_LONG if crypto else 0.05 / 365)
        s = stats(t)
        res.append(s["pf"])
        print(f"  {a:7s} 年{s['per_yr']:.0f}回 勝率{s['win']*100:.0f}% PF{s['pf']:.2f} 平均{s['exp']:+.3f}R")
    print(f"  → PFが1を超えた銘柄 {sum(p > 1 for p in res)}/{len(res)}")


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "detail":
    detail(sys.argv[1])


def portfolio(cache):
    """BTCとETHの両方で値幅ブレイク買い（UTC24時決済）を行った場合。1R＝初期資金100万円の1%（単利）。"""
    B, E = day_matrix(os.path.join(cache, "BTC_1h.csv")), day_matrix(os.path.join(cache, "ETH_1h.csv"))
    for n, s, cost in ((0, 99, 0.0005), (0, 3.0, 0.0005), (0, 2.0, 0.0005), (5, 99, 0.0005), (5, 3.0, 0.0005), (5, 3.0, 0.001)):
        tb, te = vbo(B, 1, 0.8, n, s, 24, cost), vbo(E, 1, 0.8, n, s, 24, cost)
        tr = pd.concat([tb, te]).sort_values("t")
        tr = tr[tr.t >= pd.Timestamp(START)]
        a, b = tr[tr.t < pd.Timestamp(TRAIN_END)], tr[tr.t >= pd.Timestamp(TRAIN_END)]
        st, sa, sb = stats(tr), stats(a), stats(b)
        yearly = tr.groupby(tr.t.dt.year).R.sum()
        print(f"\n== k=0.8 平均{n or 'なし'} 損切り{'なし' if s == 99 else f'ATR×{s}'} スプレッド往復{cost*100:.2f}%")
        print(f"   全期間: {st['n']}回（年{st['per_yr']:.0f}回） 勝率{st['win']*100:.1f}% PF{st['pf']:.2f} 損益比{st['payoff']:.2f}"
              f" 純利益{st['net']:+,.0f}円 最大DD{st['dd']*100:.1f}% 最大{st['maxl']}連敗 1回の最大損失{tr.R.min():.1f}R")
        print(f"   前半PF{sa['pf']:.2f} ／ 後半PF{sb['pf']:.2f} ｜ マイナスの年: {[int(y) for y, v in yearly.items() if v < 0] or 'なし'}")


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "portfolio":
    portfolio(sys.argv[1])
