#!/usr/bin/env python3
"""負けが続く条件の分析と、負けにくい条件の検証（暗号資産8銘柄・日足・2018年〜）。

対象スタイル
  VBO : ③値幅ブレイク（k=0.8・5日平均の上・当日の終値で決済）……回数多め・短期
  MA  : ⑤10日平均の上で保有（翌日の始値で売買・損切りATR×3）……回数少なめ・トレンドに乗る
相場の物差し（入る日の前日までの情報）
  BTCの200日平均の上下／その銘柄の200日平均の上下／効率性レシオ（20日間の正味の値動き÷値動きの合計＝トレンドの明確さ）
  値動きの荒さ（ATR%の過去1年での位置）／市場の広がり（8銘柄のうち50日平均より上の割合）
手順: (1) 物差しごとの成績 (2) 最悪の負け期間とその時の相場 (3) 前半(2018-22)で選んだ「取引しない条件」を後半で確認
      (4) 銘柄数・条件・量の調整を変えた組み合わせの一覧
損益: 1R＝ATR1本分＝資金の0.5%（単利）。コスト: 往復 BTC/ETH 0.05%・アルト 0.15%、買いのスワップ年25〜35%。
使い方: python3 project/tools/regime_lab.py <キャッシュ（日足）>
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
from winrate_lab import vbo_daily_tp  # noqa: E402

START, SPLIT = pd.Timestamp("2018-01-01"), pd.Timestamp("2023-01-01")
COINS = {"BTC": (0.0005, 25), "ETH": (0.0005, 35), "SOL": (0.0015, 30), "XRP": (0.0015, 30),
         "BNB": (0.0015, 30), "ADA": (0.0015, 30), "LTC": (0.0015, 30), "DOGE": (0.0015, 30)}
RISK = 0.5


def load(cache, a):
    df = pd.read_csv(os.path.join(cache, f"{a}.csv"), index_col=0, parse_dates=True)
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    return df[~df.index.duplicated()].sort_index()


def atr_of(df):
    h, l, c = df.high, df.low, df.close
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean()


def ma_hold(df, cost, swap, n=10, s=3.0):
    o, h, l, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    atr = atr_of(df).to_numpy()
    sma = pd.Series(c).rolling(n).mean().to_numpy()
    rows, i, N = [], 0, len(c)
    while i < N - 1:
        if not (c[i] > sma[i] and atr[i] > 0):
            i += 1
            continue
        d = i + 1
        entry, unit = o[d], atr[i]
        stop, j, ex = entry - s * unit, d, None
        while j < N:
            if j > d and o[j] <= stop:
                ex = o[j]
            elif l[j] <= stop:
                ex = stop
            elif c[j] < sma[j]:
                if j + 1 < N:
                    ex, j = o[j + 1], j + 1
                break
            if ex is not None:
                break
            j += 1
        if ex is None:
            break
        nights = j - d
        rows.append((df.index[d], df.index[j], (ex - entry) / unit - cost * entry / unit - swap * max(nights, 1) * entry / unit, nights))
        i = j
    return pd.DataFrame(rows, columns=["t", "exit_t", "R", "nights"])


def vbo(df, cost, swap, k=0.8, n=5):
    x = vbo_daily_tp(df, k, n, None, cost, swap)
    x["exit_t"], x["nights"] = x.t, 1
    return x


def features(data):
    btc = data["BTC"].close
    above50 = pd.concat({a: (df.close > df.close.rolling(50).mean()).where(df.close.rolling(50).count() == 50)
                         for a, df in data.items()}, axis=1)
    breadth = above50.mean(axis=1)
    F = {}
    for a, df in data.items():
        c = df.close
        er = (c - c.shift(20)).abs() / c.diff().abs().rolling(20).sum()
        atrp = atr_of(df) / c
        volpct = atrp.rolling(365, min_periods=120).rank(pct=True)
        f = pd.DataFrame({"btc_bull": (btc > btc.rolling(200).mean()).reindex(c.index),
                          "coin_bull": c > c.rolling(200).mean(), "er": er, "vol": volpct,
                          "breadth": breadth.reindex(c.index)})
        F[a] = f.shift(1)                                  # 入る日の前日までの情報だけを使う
    return F


def trades(style, data, F, coins=None):
    parts = []
    for a in coins or COINS:
        cost, sw = COINS[a]
        x = (vbo if style == "VBO" else ma_hold)(data[a], cost, sw / 36500)
        x = x.join(F[a], on="t")
        x["coin"] = a
        parts.append(x)
    x = pd.concat(parts).sort_values("exit_t")
    return x[x.t >= START].reset_index(drop=True)


def pf(R):
    R = np.asarray(R)
    return R[R > 0].sum() / -R[R < 0].sum() if (R < 0).any() else np.inf


def buckets(x):
    defs = {"BTCが200日平均の": [("上", x.btc_bull == True), ("下", x.btc_bull == False)],  # noqa: E712
            "銘柄が200日平均の": [("上", x.coin_bull == True), ("下", x.coin_bull == False)],  # noqa: E712
            "トレンドの明確さ（効率性レシオ）": [("低い<0.15", x.er < 0.15), ("中0.15-0.35", (x.er >= 0.15) & (x.er < 0.35)), ("高い≥0.35", x.er >= 0.35)],
            "値動きの荒さ（過去1年での位置）": [("静か<30%", x.vol < 0.3), ("普通", (x.vol >= 0.3) & (x.vol < 0.7)), ("荒い≥70%", x.vol >= 0.7)],
            "市場の広がり（50日平均より上の割合）": [("<25%", x.breadth < 0.25), ("25-75%", (x.breadth >= 0.25) & (x.breadth < 0.75)), ("≥75%", x.breadth >= 0.75)]}
    for name, items in defs.items():
        cells = []
        for lab, m in items:
            R = x.R[m]
            cells.append(f"{lab}: {len(R)}回 PF{pf(R):.2f} 平均{R.mean():+.2f}R")
        print(f"   {name}  " + " ／ ".join(cells))


def worst_periods(x, data, top=3):
    eq = x.R.cumsum().to_numpy()
    dates = x.exit_t.to_numpy()
    btc = data["BTC"].close
    used, out = np.zeros(len(eq), bool), []
    for _ in range(top):
        peak = np.maximum.accumulate(np.where(used, -np.inf, eq))
        dd = np.where(used, 0, peak - eq)
        j = int(dd.argmax())
        if dd[j] <= 0:
            break
        i = int(np.flatnonzero((eq[: j + 1] == peak[j]) & ~used[: j + 1])[-1])
        used[i : j + 1] = True
        p0, p1 = pd.Timestamp(dates[i]), pd.Timestamp(dates[j])
        seg = x.iloc[i + 1 : j + 1]
        b0, b1 = btc.asof(p0), btc.asof(p1)
        out.append(f"{p0:%Y-%m-%d}〜{p1:%Y-%m-%d}（{(p1 - p0).days}日・{len(seg)}回）{-dd[j]*RISK:+.0f}% ｜ その間のBTC {(b1/b0-1)*100:+.0f}%"
                   f"・BTCが200日平均より下の割合{(seg.btc_bull == False).mean()*100:.0f}%・平均の効率性レシオ{seg.er.mean():.2f}"  # noqa: E712
                   f"・市場の広がり{seg.breadth.mean()*100:.0f}%")
    return out


FILTERS = {
    "なし": lambda x: np.ones(len(x), bool),
    "BTCが200日平均の上": lambda x: x.btc_bull == True,  # noqa: E712
    "銘柄が200日平均の上": lambda x: x.coin_bull == True,  # noqa: E712
    "広がり≥50%": lambda x: x.breadth >= 0.5,
    "効率性レシオ≥0.2": lambda x: x.er >= 0.2,
    "BTC200日の上＋広がり≥50%": lambda x: (x.btc_bull == True) & (x.breadth >= 0.5),  # noqa: E712
}


def summary(x, size=None):
    R = x.R.to_numpy() * (size if size is not None else 1.0)
    pct = R * RISK
    yrs = (x.exit_t.max() - START).days / 365.25
    yearly = pd.Series(pct).groupby(x.exit_t.dt.year.to_numpy()).sum()
    eq = np.cumsum(pct)
    mx = cur = 0
    for r in R:
        cur = cur + 1 if r <= 0 else 0
        mx = max(mx, cur)
    return dict(n_yr=len(R) / yrs, annual=pct.sum() / yrs, dd=(np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max(),
                neg=[int(y) for y, v in yearly.items() if v < 0], worst=yearly.min(), maxl=mx, pf=pf(R))


def main(cache):
    data = {a: load(cache, a) for a in COINS}
    F = features(data)
    for style in ("VBO", "MA"):
        x = trades(style, data, F)
        lab = "③値幅ブレイク" if style == "VBO" else "⑤10日平均の上で保有"
        print(f"\n################ {lab}（8銘柄）")
        print(" (1) 相場の物差しごとの成績（前半2018-22年）")
        buckets(x[x.t < SPLIT])
        print(" (2) 最悪の負け期間（全期間）")
        for s in worst_periods(x, data):
            print("   " + s)
        print(" (3) 取引しない条件（前半で比較 → 後半で確認）")
        for name, f in FILTERS.items():
            m = f(x).fillna(False).to_numpy() if hasattr(f(x), "fillna") else f(x)
            a, b = x[m & (x.t < SPLIT).to_numpy()], x[m & (x.t >= SPLIT).to_numpy()]
            sa, sb, sf = summary(a), summary(b), summary(x[m])
            print(f"   {name:18s} 前半PF{sa['pf']:.2f}（年{sa['n_yr']:.0f}回）→ 後半PF{sb['pf']:.2f}"
                  f" ｜ 全期間 年{sf['annual']:+.0f}% 最大の落ち込み{sf['dd']:.0f}% マイナスの年{sf['neg'] or 'なし'}")
    print("\n################ (4) 回数と落ち込みの組み合わせ（全期間）")
    print(f"   {'スタイル':10s} {'銘柄':6s} {'条件':16s} {'量の調整':14s} 年回数  年平均 最大落込 最悪の年 最大連敗 マイナスの年")
    for style, coins, fname, sizing in itertools.product(("VBO", "MA"), (("BTC",), ("BTC", "ETH"), tuple(COINS)),
                                                        ("なし", "BTCが200日平均の上"), ("一定", "同じ日は分ける")):
        x = trades(style, data, F, coins)
        m = FILTERS[fname](x)
        m = m.fillna(False).to_numpy() if hasattr(m, "fillna") else m
        y = x[m].reset_index(drop=True)
        size = None
        if sizing == "同じ日は分ける":
            size = 1.0 / y.groupby("t").t.transform("count").to_numpy()
        s = summary(y, size)
        cn = {1: "BTC", 2: "BTC+ETH", 8: "8銘柄"}[len(coins)]
        print(f"   {('値幅ブレイク' if style == 'VBO' else '10日平均保有'):10s} {cn:6s} {fname:16s} {sizing:14s} {s['n_yr']:5.0f} {s['annual']:+6.0f}% {s['dd']:6.0f}%"
              f" {s['worst']:+6.0f}% {s['maxl']:6d}  {','.join(str(v)[2:] for v in s['neg']) or 'なし'}")


if __name__ == "__main__":
    main(sys.argv[1])
