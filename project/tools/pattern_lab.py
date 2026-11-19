#!/usr/bin/env python3
"""1日1回（日本時間12:30）の操作で運用できるルールの比較（BTC 30分足で探し、前半で選んで後半とETHで確かめる）。

1日の区切り＝日本時間12:30（UTC 3:30）。どのルールも 12:30 に注文を出し、翌日以降の 12:30 に決済する。
  値幅ブレイク   : 12:30の価格＋前日の値幅×k に逆指値の買い（翌12:30決済、損切りATR×3）【逆指値が必要】
  前日高値ブレイク: 前日の高値に逆指値の買い。前日の値幅が小さい日（直近7日で最小＝NR7）に絞る型も【逆指値が必要】
  前日の勢い     : 前の24時間の上昇が ATR×θ を超えたら12:30に成行買い、翌12:30決済【成行だけでできる】
  平均線の上で保有: 12:30の価格がn日平均より上なら保有、下回ったら決済（数日〜数か月）【成行だけでできる】
  毎日買う（比較用）: 毎日12:30に買って翌12:30に決済
コスト: Vantage（往復0.05%＋買いのスワップ年25%）／SBI VCトレード現物（往復0.15%＝テイカー0.05%×2＋すべり0.05%、スワップなし）
成績は 1R＝ATR（14日）1本分の値動き＝初期資金の1%（単利）で表示。
使い方: python3 project/tools/pattern_lab.py <キャッシュ（BTC_30m.csv, ETH_30m.csv）> [30m|1h] [区切りの日本時間 例 12:30] [ma]
  ma を付けると「平均線の上で保有」の詳しい確認（日数・年別・連敗・他の銘柄）
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
from intraday_lab import stats  # noqa: E402

START, TRAIN_END = "2018-01-01", "2023-01-01"
OFFSET = pd.Timedelta(hours=3, minutes=30)              # UTC 3:30 = 日本時間 12:30
BARS = 48                                               # 30分足 × 48 = 24時間
COSTS = {"Vantage": (0.0005, 25.0 / 365 / 100), "SBI現物": (0.0015, 0.0)}


def load(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df[~df.index.duplicated()]
    df.index = df.index - OFFSET
    step = "30min" if BARS == 48 else "1h"
    full = pd.date_range(df.index[0].normalize() + pd.Timedelta(days=1), df.index[-1].normalize() - pd.Timedelta(step), freq=step)
    df = df.reindex(full).ffill()
    nd = len(full) // BARS
    M = {k: df[k].to_numpy()[: nd * BARS].reshape(nd, BARS) for k in ("open", "high", "low", "close")}
    days = full[::BARS][:nd] + OFFSET                     # 各日の開始時刻（UTC）
    dO, dH, dL, dC = M["open"][:, 0], M["high"].max(1), M["low"].min(1), M["close"][:, -1]
    pc = np.r_[np.nan, dC[:-1]]
    tr = np.nanmax(np.vstack([dH - dL, abs(dH - pc), abs(dL - pc)]), axis=0)
    atr = pd.Series(tr).rolling(14).mean().shift(1).to_numpy()
    rng = dH - dL
    prev_rng = np.r_[np.nan, rng[:-1]]
    nr7 = np.r_[False, (pd.Series(rng).rolling(7).min() == pd.Series(rng)).to_numpy()[:-1]]
    sma = {n: pd.Series(dC).rolling(n).mean().shift(1).to_numpy() for n in (5, 10, 20, 50, 100, 200)}
    mom = np.r_[np.nan, (dC - pc)[:-1]]                   # 前の24時間の値動き
    return dict(days=days, M=M, dO=dO, dH=dH, dC=dC, atr=atr, prev_rng=prev_rng,
                prev_high=np.r_[np.nan, dH[:-1]], nr7=nr7, sma=sma, mom=mom)


def stop_entry(D, trig, ok, cost, swap, stop_atr=3.0):
    """逆指値の買い（その日のうちに trig に届いたら約定）→ 損切りATR×3 か 翌12:30で決済。"""
    O, H, L, C = D["M"]["open"], D["M"]["high"], D["M"]["low"], D["M"]["close"]
    atr = D["atr"]
    ok = ok & ~np.isnan(atr) & ~np.isnan(trig) & (trig > O[:, 0])
    hit = H >= trig[:, None]
    has = hit.any(1) & ok
    d = np.flatnonzero(has)
    h0 = hit.argmax(1)[d]
    fill = np.maximum(O[d, h0], trig[d])
    unit = stop_atr * atr[d]
    stop = fill - unit
    bars = np.arange(BARS)
    hs = (L[d] <= stop[:, None]) & (bars[None, :] >= h0[:, None])
    first = hs.argmax(1)
    gap = O[d, first]
    px_stop = np.where((first > h0) & (gap <= stop), gap, stop)
    ex = np.where(hs.any(1), px_stop, C[d, -1])
    R = (ex - fill) / atr[d] - cost * fill / atr[d] - swap * fill / atr[d]
    return pd.DataFrame({"t": D["days"][d], "R": R})


def open_entry(D, ok, cost, swap):
    """12:30に成行で買い、翌12:30に決済（損切りなし）。"""
    O, C, atr = D["M"]["open"][:, 0], D["M"]["close"][:, -1], D["atr"]
    d = np.flatnonzero(ok & ~np.isnan(atr))
    R = (C[d] - O[d]) / atr[d] - cost * O[d] / atr[d] - swap * O[d] / atr[d]
    return pd.DataFrame({"t": D["days"][d], "R": R})


def hold_above(D, n, cost, swap):
    """12:30の価格がn日平均より上なら保有を続ける。連続した保有期間を1回の取引とする。"""
    O, C, atr, sma = D["M"]["open"][:, 0], D["M"]["close"][:, -1], D["atr"], D["sma"][n]
    on = (O > sma) & ~np.isnan(sma) & ~np.isnan(atr)
    rows, i, N = [], 0, len(O)
    while i < N:
        if not on[i]:
            i += 1
            continue
        j = i
        while j + 1 < N and on[j + 1]:
            j += 1
        nights = j - i + 1
        R = (C[j] - O[i]) / atr[i] - cost * O[i] / atr[i] - swap * nights * O[i] / atr[i]
        rows.append((D["days"][i], R))
        i = j + 1
    return pd.DataFrame(rows, columns=["t", "R"])


def families(D, cost, swap):
    O, atr = D["dO"], D["atr"]
    yield "毎日買う（比較用）", "", open_entry(D, np.ones(len(O), bool), cost, swap)
    for k, n in itertools.product((0.5, 0.6, 0.7, 0.8, 1.0), (0, 5, 10, 20)):
        ok = np.ones(len(O), bool) if not n else O > D["sma"][n]
        yield "値幅ブレイク", f"k={k} 平均{n or 'なし'}", stop_entry(D, O + k * D["prev_rng"], ok, cost, swap)
    for filt, n in itertools.product(("なし", "NR7"), (0, 5, 20)):
        ok = np.ones(len(O), bool) if filt == "なし" else D["nr7"]
        if n:
            ok = ok & (O > D["sma"][n])
        yield "前日高値ブレイク", f"絞り込み{filt} 平均{n or 'なし'}", stop_entry(D, D["prev_high"], ok, cost, swap)
    for th, n in itertools.product((0.0, 0.5, 1.0), (0, 5, 20)):
        ok = D["mom"] > th * atr
        if n:
            ok = ok & (O > D["sma"][n])
        yield "前日の勢い", f"上昇>ATR×{th} 平均{n or 'なし'}", open_entry(D, ok, cost, swap)
    for n in (5, 10, 20, 50, 100, 200):
        yield "平均線の上で保有", f"{n}日平均", hold_above(D, n, cost, swap)


def split(tr):
    tr = tr[tr.t >= pd.Timestamp(START)]
    return tr[tr.t < pd.Timestamp(TRAIN_END)], tr[tr.t >= pd.Timestamp(TRAIN_END)]


def main(cache, bar="30m", jst="12:30"):
    global OFFSET, BARS
    hh, mm = (int(x) for x in jst.split(":"))
    OFFSET = pd.Timedelta(hours=(hh - 9) % 24, minutes=mm)
    BARS = 48 if bar == "30m" else 24
    print(f"=== 区切り 日本時間{jst}（{bar}足）")
    B, E = load(os.path.join(cache, f"BTC_{bar}.csv")), load(os.path.join(cache, f"ETH_{bar}.csv"))
    for venue, (cost, swap) in COSTS.items():
        rows = {}
        for (fam, p, trb), (_, _, tre) in zip(families(B, cost, swap), families(E, cost, swap)):
            a, b = split(trb)
            sa, sb, se = stats(a), stats(b), stats(tre[tre.t >= pd.Timestamp(START)])
            if sa and sb and se:
                rows.setdefault(fam, []).append((p, sa, sb, se))
        print(f"\n######## {venue}（往復{cost*100:.2f}%{'＋スワップ' if swap else '・スワップなし'}）")
        for fam, xs in rows.items():
            xs.sort(key=lambda x: x[1]["t"], reverse=True)       # 前半の t値で選ぶ
            p, sa, sb, se = xs[0]
            pos = np.mean([x[2]["exp"] > 0 and x[3]["exp"] > 0 for x in xs]) * 100
            print(f"{fam:10s} 前半で選んだ設定[{p}] PF 前半{sa['pf']:.2f}／後半{sb['pf']:.2f}／ETH{se['pf']:.2f}"
                  f" ｜ 後半 年{sb['per_yr']:.0f}回 勝率{sb['win']*100:.0f}% 平均{sb['exp']:+.3f}R 最大DD{sb['dd']*100:.1f}%"
                  f" ｜ {len(xs)}通り中 後半とETHの両方プラス {pos:.0f}%")


if __name__ == "__main__" and "ma" not in sys.argv[2:]:
    main(sys.argv[1], *(sys.argv[2:4]))


def ma_detail(cache, bar="1h", jst="12:00"):
    """「平均線の上で保有」の確認：日数を変えた場合・年別・連敗・他の銘柄（日足）。"""
    global OFFSET, BARS
    hh, mm = (int(x) for x in jst.split(":"))
    OFFSET = pd.Timedelta(hours=(hh - 9) % 24, minutes=mm)
    BARS = 48 if bar == "30m" else 24
    B, E = load(os.path.join(cache, f"BTC_{bar}.csv")), load(os.path.join(cache, f"ETH_{bar}.csv"))
    print(f"=== 平均線の上で保有（区切り 日本時間{jst}）PF 前半/後半/ETH ｜ 後半の年回数・勝率・最大連敗")
    for venue, (cost, swap) in COSTS.items():
        for n in (5, 10, 20, 50, 100, 200):
            a, b = split(hold_above(B, n, cost, swap))
            e = hold_above(E, n, cost, swap)
            e = e[e.t >= pd.Timestamp(START)]
            f = lambda x: f"{stats(x)['pf']:.2f}" if stats(x) else "－"   # noqa: E731
            sb = stats(b)
            extra = f" ｜ 年{sb['per_yr']:.0f}回 勝率{sb['win']*100:.0f}% 最大{sb['maxl']}連敗" if sb else ""
            print(f"  {venue:7s} {n:3d}日平均: {f(a)}/{f(b)}/{f(e)}{extra}")
    tr = hold_above(B, 10, *COSTS["SBI現物"])
    tr = tr[tr.t >= pd.Timestamp(START)]
    print("\n=== BTC 10日平均・SBI現物 年別（1R＝1万円の単利）")
    for y, g in tr.groupby(tr.t.dt.year):
        R = g.R.to_numpy()
        pf = R[R > 0].sum() / -R[R < 0].sum() if (R < 0).any() else np.inf
        print(f"  {y}年: {len(R):2d}回 勝率{(R > 0).mean()*100:3.0f}% PF{pf:5.2f} 損益{R.sum()*10000:+9,.0f}円 最大の勝ち{R.max():+.1f}R")
    s = stats(tr)
    print(f"  合計: {s['n']}回 勝率{s['win']*100:.0f}% PF{s['pf']:.2f} 純利益{s['net']:+,.0f}円 最大DD{s['dd']*100:.1f}% 最大{s['maxl']}連敗")
    print("\n=== ルール選定に使っていない銘柄（日足の終値で判定、10日・20日平均、往復0.15%・スワップなし）PF")
    for a in ("SOL", "XRP", "BNB", "ADA", "LTC", "DOGE", "Silver", "S&P500", "Nikkei", "USDJPY", "Crude"):
        df = pd.read_csv(os.path.join(cache, f"{a}.csv"), index_col=0, parse_dates=True)
        c = df.close.to_numpy(float)
        pc = np.r_[np.nan, c[:-1]]
        tr_ = np.nanmax(np.vstack([df.high - df.low, abs(df.high - pc), abs(df.low - pc)]), axis=0)
        atr = pd.Series(tr_).rolling(14).mean().to_numpy()
        out = []
        for n in (10, 20):
            sma = pd.Series(c).rolling(n).mean().to_numpy()
            on = c > sma
            R, i = [], 0
            while i < len(c) - 1:
                if not on[i] or np.isnan(atr[i]):
                    i += 1
                    continue
                j = i
                while j + 1 < len(c) - 1 and on[j + 1]:
                    j += 1
                R.append((c[j + 1] - c[i]) / atr[i] - 0.0015 * c[i] / atr[i])
                i = j + 1
            R = np.array(R)
            out.append(R[R > 0].sum() / -R[R < 0].sum())
        print(f"  {a:7s} 10日 {out[0]:.2f} ／ 20日 {out[1]:.2f}")


if __name__ == "__main__" and "ma" in sys.argv[2:]:
    args = [x for x in sys.argv[2:] if x != "ma"]
    ma_detail(sys.argv[1], *args)
