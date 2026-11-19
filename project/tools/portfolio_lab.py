#!/usr/bin/env python3
"""複数のロジックを1つの固まり（ポートフォリオ）として運用した場合の検証（BTC・ETH中心、日足、2018年〜）。

部品（それぞれ「入る時点のATR1本分の値動き＝1R」、毎日の含み損益まで計算）
  ①値幅ブレイク買い  ②値幅ブレイク売り（始値−前日値幅×0.8を割ったら売り、当日決済）
  ③MACD買い（MACD＞シグナルの間）  ④MACD売り（MACD＜シグナルの間）
  ⑤もみ合いの逆張り（効率性レシオ<0.15の時だけ、RSI(2)<10で買い／>90で売り、RSIが50に戻るか3日で決済、損切りATR×2）
  ⑥ETH/BTCの強弱（比率が20日平均の上ならETH買い＋BTC売り、下なら逆。方向に関係なし）
  ⑦金の値幅ブレイク（買い・売り、おまけ）
コスト（Vantage想定）: 往復 BTC/ETH 0.05%・金 0.03%。スワップ 年: BTC買い25%・ETH買い35%・売り0%、金買い6.5%・金売り−2.9%（受け取り）。
重みづけ: 「同じ量」（1R＝資金の0.5%）／「ばらつきをそろえる」（前半2018-22年の値動きの大きさで割る）。
評価: 年平均・含み込みの最大の落ち込み・マイナスの年・月ごとの勝率・最大の連続マイナス月・前の高値を回復できなかった最長日数。
使い方: python3 project/tools/portfolio_lab.py <キャッシュ（BTC.csv, ETH.csv, Gold.csv）> [extra]
  extra: 単独でプラスの部品だけを配分を変えて組み合わせる
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
from btceth_lab import ind, load, macd_pos  # noqa: E402
from rsi_lab import rsi  # noqa: E402

START, SPLIT = pd.Timestamp("2018-01-01"), pd.Timestamp("2023-01-01")
RISK = 0.5
SPEC = {"BTC": (0.0005, 25.35, 0.0), "ETH": (0.0005, 35.49, 0.0), "Gold": (0.0003, 6.53, -2.9)}   # コスト, 買いスワップ, 売りスワップ（年%）


def vbo_side(I, side, cost, sw, k=0.8, n=5):
    o, h, l, c = I["o"], I["h"], I["l"], I["c"]
    atr = np.r_[np.nan, I["atr"][:-1]]
    rng = np.r_[np.nan, (h - l)[:-1]]
    ma = np.r_[np.nan, pd.Series(c).rolling(n).mean().to_numpy()[:-1]]
    trig = o + side * k * rng
    hit = ((h >= trig) & (o > ma)) if side == 1 else ((l <= trig) & (o < ma))
    hit &= (atr > 0) & (rng > 0)
    R = np.where(hit, side * (c - trig) / atr - cost * trig / atr - sw / 36500 * trig / atr, 0.0)
    return pd.Series(np.nan_to_num(R), index=I["idx"])


def pos_side(I, pos, side, cost, sw):
    """保有状態（終値で判定）→ 翌日の始値で入り、「持たない」に変わった終値の翌日の始値で決済。毎日の含み損益（R）。"""
    o, c, atr = I["o"], I["c"], I["atr"]
    N, daily = len(c), np.zeros(len(c))
    p = np.nan_to_num(pos)
    i = 0
    while i < N - 1:
        if p[i] == 1 and (i == 0 or p[i - 1] == 0):
            j = i
            while j + 1 < N and p[j + 1] == 1:
                j += 1
            if atr[i] > 0:
                d, e, u = i + 1, (j + 2 if j + 2 < N else None), 1.0 / atr[i]
                last = e if e is not None else N
                seg = np.zeros(N)
                seg[d] = side * (c[d] - o[d]) * u
                if last - 1 > d:
                    seg[d + 1:last] = side * (c[d + 1:last] - c[d:last - 1]) * u
                if e is not None:
                    seg[e] = side * (o[e] - c[e - 1]) * u - cost / 2 * o[e] * u
                seg[d] -= cost / 2 * o[d] * u
                seg[d:last] -= sw / 36500 * c[d:last] * u
                daily += seg
            i = j + 1
        else:
            i += 1
    return pd.Series(daily, index=I["idx"])


def mr_range(I, cost, sw_long, sw_short, er_max=0.15, lo=10, hi=90, maxd=3, s=2.0):
    o, h, l, c, atr, er = I["o"], I["h"], I["l"], I["c"], I["atr"], I["er"]
    r2 = rsi(c, 2)
    N, daily, i = len(c), np.zeros(len(c)), 0
    while i < N - 2:
        if not (er[i] < er_max and atr[i] > 0 and (r2[i] < lo or r2[i] > hi)):
            i += 1
            continue
        side = 1 if r2[i] < lo else -1
        d, u = i + 1, 1.0 / atr[i]
        entry, stop = o[d], o[d] - side * s * atr[i]
        j, ex, k = d, None, d
        while j < N:
            adverse = l[j] <= stop if side == 1 else h[j] >= stop
            if adverse:
                ex, k = (stop if j == d else (min(o[j], stop) if side == 1 else max(o[j], stop))), j
                break
            back = r2[j] > 50 if side == 1 else r2[j] < 50
            if (back or j - d + 1 >= maxd) and j + 1 < N:
                ex, k = o[j + 1], j + 1
                break
            j += 1
        if ex is None:
            break
        nights = max(k - d, 1)
        sw = sw_long if side == 1 else sw_short
        daily[k] += side * (ex - entry) * u - cost * entry * u - sw / 36500 * nights * entry * u
        i = k
    return pd.Series(daily, index=I["idx"])


def pair(IE, IB, n=20):
    """ETH/BTC 比率の強弱。1R＝比率のATR相当の値動き。"""
    e = pd.Series(IE["c"], index=IE["idx"])
    b = pd.Series(IB["c"], index=IB["idx"])
    df = pd.concat({"e": e, "b": b}, axis=1).dropna()
    ratio = df.e / df.b
    sign = np.sign(ratio - ratio.rolling(n).mean()).shift(1).fillna(0)       # 前日の終値で判定
    re, rb = df.e.pct_change().fillna(0), df.b.pct_change().fillna(0)
    spread = re - rb
    unit = spread.abs().rolling(20).mean().shift(1) * 1.25                   # 比率の1日の平均的な動き（％）
    flip = sign.diff().abs().fillna(0)
    long_swap = np.where(sign > 0, SPEC["ETH"][1], np.where(sign < 0, SPEC["BTC"][1], 0)) / 36500
    R = (sign * spread - flip * 2 * SPEC["ETH"][0] - long_swap) / unit
    return R.replace([np.inf, -np.inf], 0).fillna(0)


def build(cache):
    btc = load(cache, "BTC")
    I = {a: ind(load(cache, a), btc) for a in ("BTC", "ETH")}
    gold = load(cache, "Gold")
    G = ind(gold, btc.reindex(gold.index).ffill().bfill())
    comp = {}
    for nm, fn in (("①値幅ブレイク買い", lambda a: vbo_side(I[a], 1, SPEC[a][0], SPEC[a][1])),
                   ("②値幅ブレイク売り", lambda a: vbo_side(I[a], -1, SPEC[a][0], SPEC[a][2])),
                   ("③MACD買い", lambda a: pos_side(I[a], macd_pos(I[a]), 1, SPEC[a][0], SPEC[a][1])),
                   ("④MACD売り", lambda a: pos_side(I[a], 1 - macd_pos(I[a]), -1, SPEC[a][0], SPEC[a][2])),
                   ("⑤もみ合いの逆張り", lambda a: mr_range(I[a], SPEC[a][0], SPEC[a][1], SPEC[a][2]))):
        comp[nm] = fn("BTC").add(fn("ETH"), fill_value=0)
    comp["⑥ETH/BTCの強弱"] = pair(I["ETH"], I["BTC"])
    comp["⑦金の値幅ブレイク"] = vbo_side(G, 1, *SPEC["Gold"][:2]).add(vbo_side(G, -1, SPEC["Gold"][0], SPEC["Gold"][2]), fill_value=0)
    df = pd.DataFrame(comp).fillna(0)
    return df[df.index >= START] * RISK                                      # 資金に対する％（1R＝0.5%）


def metrics(x):
    eq = x.cumsum()
    yrs = (x.index[-1] - x.index[0]).days / 365.25
    yearly = x.groupby(x.index.year).sum()
    monthly = x.resample("ME").sum()
    neg_run = run = 0
    for v in monthly:
        run = run + 1 if v < 0 else 0
        neg_run = max(neg_run, run)
    peak = eq.cummax()
    under = (eq < peak - 1e-9).astype(int)
    longest = cur = 0
    for v in under:
        cur = cur + 1 if v else 0
        longest = max(longest, cur)
    return dict(annual=x.sum() / yrs, dd=(peak - eq).max(), neg=[int(y) for y, v in yearly.items() if v < 0], worst=yearly.min(),
                pos_m=(monthly > 0).mean(), neg_run=neg_run, longest=longest, sd_m=monthly.std(), yearly=yearly)


def main(cache):
    X = build(cache)
    tr, te = X[X.index < SPLIT], X[X.index >= SPLIT]
    print("==== 部品ごとの成績（1R＝資金の0.5%、BTC+ETHの合計）")
    print(f"{'部品':16s} 年平均 最大落込 最悪の年 プラスの月 最長連続マイナス月 回復できない最長日数 マイナスの年")
    for c in X.columns:
        m = metrics(X[c])
        neg = ",".join(str(v)[2:] for v in m["neg"]) or "なし"
        print(f"{c:16s} {m['annual']:+5.1f}% {m['dd']:6.1f}% {m['worst']:+6.1f}% {m['pos_m']*100:6.0f}% {m['neg_run']:10d}か月 {m['longest']:12d}日  {neg}")
    print("\n==== 月ごとの損益の似かた（相関、全期間）")
    print(X.resample("ME").sum().corr().round(2).to_string())

    sets = {
        "A ①だけ": ["①値幅ブレイク買い"],
        "B ①＋②（短期の買いと売り）": ["①値幅ブレイク買い", "②値幅ブレイク売り"],
        "C ③＋④（トレンドの買いと売り）": ["③MACD買い", "④MACD売り"],
        "D ①②③④": ["①値幅ブレイク買い", "②値幅ブレイク売り", "③MACD買い", "④MACD売り"],
        "E ①②＋⑥": ["①値幅ブレイク買い", "②値幅ブレイク売り", "⑥ETH/BTCの強弱"],
        "F ①②④⑥": ["①値幅ブレイク買い", "②値幅ブレイク売り", "④MACD売り", "⑥ETH/BTCの強弱"],
        "G ①②③④⑥": ["①値幅ブレイク買い", "②値幅ブレイク売り", "③MACD買い", "④MACD売り", "⑥ETH/BTCの強弱"],
        "H ①〜⑥ 全部": ["①値幅ブレイク買い", "②値幅ブレイク売り", "③MACD買い", "④MACD売り", "⑤もみ合いの逆張り", "⑥ETH/BTCの強弱"],
        "I G＋⑦金": ["①値幅ブレイク買い", "②値幅ブレイク売り", "③MACD買い", "④MACD売り", "⑥ETH/BTCの強弱", "⑦金の値幅ブレイク"],
    }
    vol = tr.std()
    print("\n==== 組み合わせ（全期間）※年平均と落ち込みは1R＝0.5%の場合。量を2倍にすればどちらも2倍")
    print(f"{'組み合わせ':22s} {'重み':10s} 年平均 最大落込 効率 マイナスの年 最悪の年 プラスの月 連続マイナス月 回復最長 ｜ 前半→後半の年平均")
    best = []
    for (name, cols), wmode in itertools.product(sets.items(), ("同じ量", "ばらつき均等")):
        w = pd.Series(1.0, index=cols) if wmode == "同じ量" else (1 / vol[cols]) / (1 / vol[cols]).mean()
        x = (X[cols] * w).sum(axis=1)
        m, ma, mb = metrics(x), metrics(x[x.index < SPLIT]), metrics(x[x.index >= SPLIT])
        neg = ",".join(str(v)[2:] for v in m["neg"]) or "なし"
        eff = m["annual"] / m["dd"] if m["dd"] > 0 else np.inf
        best.append((eff, name, wmode, m))
        print(f"{name:22s} {wmode:10s} {m['annual']:+5.1f}% {m['dd']:5.1f}% {eff:4.1f} {neg:12s} {m['worst']:+5.1f}% {m['pos_m']*100:5.0f}% {m['neg_run']:6d}か月 {m['longest']:5d}日"
              f" ｜ {ma['annual']:+.1f}%→{mb['annual']:+.1f}%")
    print("\n==== 効率の高い組み合わせ上位3つの年別（％）")
    for eff, name, wmode, m in sorted(best, key=lambda r: r[0], reverse=True)[:3]:
        print(f"  {name}（{wmode}）: " + " ".join(f"{y}:{v:+.1f}" for y, v in m["yearly"].items()))


if __name__ == "__main__" and len(sys.argv) == 2:
    main(sys.argv[1])


def extra(cache):
    """単独でプラスの部品だけを、配分を変えて組み合わせる。"""
    from btceth_lab import ichimoku_pos
    X = build(cache)
    btc = load(cache, "BTC")
    I = {a: ind(load(cache, a), btc) for a in ("BTC", "ETH")}
    ichi = None
    for a in ("BTC", "ETH"):
        pos = ichimoku_pos(I[a]) * I[a]["btc200"]          # BTCが200日平均の上の時だけ入る
        s = pos_side(I[a], pos, 1, SPEC[a][0], SPEC[a][1])
        ichi = s if ichi is None else ichi.add(s, fill_value=0)
    X["⑧一目均衡表買い(BTC200)"] = (ichi[ichi.index >= START] * RISK).reindex(X.index).fillna(0)
    m8 = metrics(X["⑧一目均衡表買い(BTC200)"])
    print(f"⑧一目均衡表買い(BTC200日上): 年{m8['annual']:+.1f}% 落込{m8['dd']:.1f}% マイナスの年{m8['neg'] or 'なし'} プラスの月{m8['pos_m']*100:.0f}%")
    plans = {
        "①1.0": {"①値幅ブレイク買い": 1.0},
        "①1.0＋②0.5": {"①値幅ブレイク買い": 1.0, "②値幅ブレイク売り": 0.5},
        "①1.0＋③0.5": {"①値幅ブレイク買い": 1.0, "③MACD買い": 0.5},
        "①1.0＋②0.5＋③0.5": {"①値幅ブレイク買い": 1.0, "②値幅ブレイク売り": 0.5, "③MACD買い": 0.5},
        "①1.0＋⑧0.5": {"①値幅ブレイク買い": 1.0, "⑧一目均衡表買い(BTC200)": 0.5},
        "①1.0＋②0.5＋⑧0.5": {"①値幅ブレイク買い": 1.0, "②値幅ブレイク売り": 0.5, "⑧一目均衡表買い(BTC200)": 0.5},
        "①1.0＋②0.5＋③0.25＋⑧0.25": {"①値幅ブレイク買い": 1.0, "②値幅ブレイク売り": 0.5, "③MACD買い": 0.25, "⑧一目均衡表買い(BTC200)": 0.25},
    }
    print(f"\n{'配分（①の量を1とした比）':28s} 年平均 最大落込 効率 マイナスの年 最悪の年 プラスの月 連続マイナス月 回復最長 ｜ 前半→後半")
    for name, w in plans.items():
        x = sum(X[c] * v for c, v in w.items())
        m, ma, mb = metrics(x), metrics(x[x.index < SPLIT]), metrics(x[x.index >= SPLIT])
        neg = ",".join(str(v)[2:] for v in m["neg"]) or "なし"
        print(f"{name:28s} {m['annual']:+5.1f}% {m['dd']:5.1f}% {m['annual']/m['dd']:4.1f} {neg:10s} {m['worst']:+5.1f}% {m['pos_m']*100:5.0f}% {m['neg_run']:5d}か月 {m['longest']:5d}日"
              f" ｜ {ma['annual']:+.1f}%→{mb['annual']:+.1f}%")
        print("      年別: " + " ".join(f"{y}:{v:+.1f}" for y, v in m["yearly"].items()))


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "extra":
    extra(sys.argv[1])
