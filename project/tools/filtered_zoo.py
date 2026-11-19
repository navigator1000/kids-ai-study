#!/usr/bin/env python3
"""「負けにくい条件」を色々なロジックに当てはめた比較（暗号資産8銘柄・日足・2018年〜・買いのみ）。

条件: なし／トレンドがはっきり（効率性レシオ≥0.2）／その銘柄が200日平均の上／BTCが200日平均の上（入る日の前日の値）
ロジックは15種類（1日で決済・数日・長め）。どれもこれまでの設定か一般的な設定のままで、ここでは探し直さない。
損益: 1R＝ATR1本分＝資金の0.5%（単利）。コスト: 往復 BTC/ETH 0.05%・アルト 0.15%、買いのスワップ年25〜35%（Vantage並み）。
使い方: python3 project/tools/filtered_zoo.py <キャッシュ（日足）> [check]
  check: 良すぎる結果の確認（BTC+ETHだけ／アルトだけ／上位5回の取引の割合）
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
import classic_lab as CL  # noqa: E402
import strategy_lab as S  # noqa: E402
from regime_lab import COINS, SPLIT, START, features, load, ma_hold, pf, summary  # noqa: E402
from rsi_lab import rsi  # noqa: E402
from style_lab import vbo_multiday  # noqa: E402
from winrate_lab import vbo_daily_tp  # noqa: E402


def base(df):
    o, h, l, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - l, abs(h - pc), abs(l - pc)]), axis=0)
    atr = pd.Series(tr).rolling(14).mean().to_numpy()          # その日の終値までのATR
    return o, h, l, c, atr


def one_day(df, cost, swap, kind):
    """前日高値ブレイク／前日の勢い（当日の終値で決済）。"""
    o, h, l, c, atr = base(df)
    a = np.r_[np.nan, atr[:-1]]
    if kind == "prevhigh":
        trig = np.r_[np.nan, h[:-1]]
        ok = (o < trig) & (h >= trig) & (a > 0)
        entry = trig
    else:
        mom = np.r_[np.nan, (c - np.r_[np.nan, c[:-1]])[:-1]]
        ok = (mom > 0.5 * a) & (a > 0)
        entry = o
    R = (c - entry) / a - cost * entry / a - swap * entry / a
    return pd.DataFrame({"t": df.index[ok], "exit_t": df.index[ok], "R": R[ok], "nights": 1})


def from_positions(df, pos, cost, swap):
    """保有状態（終値で判定）→ 翌日の始値で売買した取引に直す。"""
    o, h, l, c, atr = base(df)
    p = np.nan_to_num(pos)
    rows, i, N = [], 0, len(c)
    while i < N - 1:
        if p[i] == 1 and (i == 0 or p[i - 1] == 0) and atr[i] > 0:
            j = i
            while j + 1 < N and p[j + 1] == 1:
                j += 1
            if j + 2 >= N:
                break
            d, e = i + 1, j + 2       # 終値で「持たない」に変わった（j+1）のを確認して、翌日（j+2）の始値で決済
            nights = max(e - d, 1)
            rows.append((df.index[d], df.index[e], (o[e] - o[d]) / atr[i] - cost * o[d] / atr[i] - swap * nights * o[d] / atr[i], nights))
            i = j + 1
        else:
            i += 1
    return pd.DataFrame(rows, columns=["t", "exit_t", "R", "nights"])


def rsi2_dip(df, cost, swap, s=3.0, maxd=10):
    o, h, l, c, atr = base(df)
    up = c > pd.Series(c).rolling(200).mean().to_numpy()
    r2 = rsi(c, 2)
    rows, i, N = [], 0, len(c)
    while i < N - 1:
        if not (up[i] and r2[i] < 10 and atr[i] > 0):
            i += 1
            continue
        d, entry, unit = i + 1, o[i + 1], atr[i]
        stop, j, ex = entry - s * unit, d, None
        while j < N:
            if l[j] <= stop:
                ex = min(o[j], stop) if j > d else stop
                break
            if r2[j] > 70 or j - d + 1 >= maxd:
                if j + 1 < N:
                    ex, j = o[j + 1], j + 1
                break
            j += 1
        if ex is None:
            break
        nights = max(j - d, 1)
        rows.append((df.index[d], df.index[j], (ex - entry) / unit - cost * entry / unit - swap * nights * entry / unit, nights))
        i = j
    return pd.DataFrame(rows, columns=["t", "exit_t", "R", "nights"])


def breakout40(df, cost, swap):
    S.HOLD = 40
    rows = []
    for t in S.run(S.prep(df), 1, 200, "bo", 2.0, 4.0, cost):
        if t["open"]:
            continue
        nights = (t["exit_t"] - t["t"]).days + 1
        unit = t["risk"] / 2.0
        rows.append((t["t"], t["exit_t"], t["R"] * 2.0 - swap * nights * t["fill"] / unit, nights))
    return pd.DataFrame(rows, columns=["t", "exit_t", "R", "nights"])


def vbo1(df, cost, swap, k, n, tp=None):
    x = vbo_daily_tp(df, k, n, tp, cost, swap)
    x["exit_t"], x["nights"] = x.t, 1
    return x


def classic_pos(df, name):
    return CL.positions(df)[name]


LOGICS = {
    "1日｜値幅ブレイク k0.8・5日平均": lambda df, c, s: vbo1(df, c, s, 0.8, 5),
    "1日｜値幅ブレイク k0.5（回数多め）": lambda df, c, s: vbo1(df, c, s, 0.5, 0),
    "1日｜値幅ブレイク＋早め利確": lambda df, c, s: vbo1(df, c, s, 0.8, 5, 0.5),
    "1日｜前日高値ブレイク": lambda df, c, s: one_day(df, c, s, "prevhigh"),
    "1日｜前日の勢い": lambda df, c, s: one_day(df, c, s, "mom"),
    "数日｜値幅ブレイク＋数日保有": lambda df, c, s: vbo_multiday(df, c, s),
    "数日｜RSI2押し目買い": lambda df, c, s: rsi2_dip(df, c, s),
    "数日｜ボリンジャー上抜け": lambda df, c, s: from_positions(df, classic_pos(df, "③ボリンジャー上抜け"), c, s),
    "数日｜パラボリックSAR": lambda df, c, s: from_positions(df, classic_pos(df, "⑥パラボリックSAR"), c, s),
    "長め｜10日平均の上で保有": lambda df, c, s: ma_hold(df, c, s),
    "長め｜移動平均クロス20/50": lambda df, c, s: from_positions(df, classic_pos(df, "①移動平均クロス 20/50日"), c, s),
    "長め｜MACD": lambda df, c, s: from_positions(df, classic_pos(df, "②MACD(12,26,9)"), c, s),
    "長め｜タートル20/10": lambda df, c, s: from_positions(df, classic_pos(df, "④タートル 20日高値/10日安値"), c, s),
    "長め｜一目均衡表": lambda df, c, s: from_positions(df, classic_pos(df, "⑤一目均衡表"), c, s),
    "長め｜20日高値＋最大40日": breakout40,
}
FILTERS = {"なし": None, "トレンド明確": lambda x: x.er >= 0.2, "銘柄200日上": lambda x: x.coin_bull == True,  # noqa: E712
           "BTC200日上": lambda x: x.btc_bull == True}  # noqa: E712


def main(cache):
    data = {a: load(cache, a) for a in COINS}
    F = features(data)
    rows = []
    for lname, fn in LOGICS.items():
        parts = []
        for a, df in data.items():
            cost, sw = COINS[a]
            x = fn(df, cost, sw / 36500).join(F[a], on="t")
            parts.append(x)
        x = pd.concat(parts).sort_values("exit_t")
        x = x[x.t >= START].reset_index(drop=True)
        for fname, f in FILTERS.items():
            y = x if f is None else x[f(x).fillna(False).to_numpy() if hasattr(f(x), "fillna") else f(x)]
            y = y.reset_index(drop=True)
            if len(y) < 30:
                continue
            s = summary(y)
            te = y[y.t >= SPLIT]
            R = y.R.to_numpy()
            rows.append(dict(logic=lname, filt=fname, per_day=s["n_yr"] / 365, win=(R > 0).mean(),
                             payoff=R[R > 0].mean() / -R[R < 0].mean(), annual=s["annual"], dd=s["dd"], neg=s["neg"],
                             worst=s["worst"], maxl=s["maxl"], pf_te=pf(te.R) if len(te) > 10 else np.nan,
                             eff=s["annual"] / s["dd"] if s["dd"] > 0 else np.inf))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cache, "filtered_zoo.csv"), index=False)
    print(f"{'ロジック':24s} {'条件':8s} 1日回数 勝率 損益比  年平均 最大落込 効率 後半PF マイナスの年")
    for r in df.itertuples():
        neg = ",".join(str(v)[2:] for v in r.neg) or "なし"
        print(f"{r.logic:24s} {r.filt:8s} {r.per_day:6.2f} {r.win*100:4.0f}% {r.payoff:5.2f} {r.annual:+6.0f}% {r.dd:6.0f}% {r.eff:4.1f} {r.pf_te:5.2f}  {neg}")
    ok = df[df.neg.apply(len) == 0].sort_values("eff", ascending=False)
    print(f"\nマイナスの年がない組み合わせ: {len(ok)}/{len(df)}通り（効率＝年平均÷最大の落ち込み の高い順）")
    for r in ok.itertuples():
        print(f"  {r.logic:24s} {r.filt:8s} 1日{r.per_day:.2f}回 勝率{r.win*100:.0f}% 年{r.annual:+.0f}% 落込{r.dd:.0f}% 効率{r.eff:.1f} 最悪の年{r.worst:+.0f}% 後半PF{r.pf_te:.2f}")


if __name__ == "__main__" and len(sys.argv) == 2:
    main(sys.argv[1])


def check(cache):
    """良すぎる結果の確認：BTC+ETHだけ／アルト6銘柄だけ／上位5回の取引が利益に占める割合。"""
    data = {a: load(cache, a) for a in COINS}
    F = features(data)
    picks = [("長め｜一目均衡表", "BTC200日上"), ("数日｜ボリンジャー上抜け", "なし"), ("長め｜MACD", "なし"),
             ("数日｜パラボリックSAR", "なし"), ("長め｜タートル20/10", "なし"), ("1日｜値幅ブレイク k0.8・5日平均", "トレンド明確")]
    groups = {"BTC+ETH": ("BTC", "ETH"), "アルト6銘柄": ("SOL", "XRP", "BNB", "ADA", "LTC", "DOGE")}
    print(f"{'ロジック':24s} {'条件':8s} {'銘柄':8s} 年平均 最大落込 マイナスの年        後半PF 上位5回の割合")
    for lname, fname in picks:
        for gname, coins in groups.items():
            parts = []
            for a in coins:
                cost, sw = COINS[a]
                parts.append(LOGICS[lname](data[a], cost, sw / 36500).join(F[a], on="t").assign(coin=a))
            x = pd.concat(parts).sort_values("exit_t")
            x = x[x.t >= START].reset_index(drop=True)
            f = FILTERS[fname]
            if f is not None:
                x = x[f(x).fillna(False).to_numpy()].reset_index(drop=True)
            s = summary(x)
            te = x[x.t >= SPLIT]
            top5 = np.sort(x.R.to_numpy())[-5:].sum() / x.R.sum() if x.R.sum() > 0 else np.nan
            neg = ",".join(str(v)[2:] for v in s["neg"]) or "なし"
            print(f"{lname:24s} {fname:8s} {gname:8s} {s['annual']:+6.0f}% {s['dd']:6.0f}%  {neg:18s} {pf(te.R):5.2f} {top5*100:6.0f}%")


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "check":
    check(sys.argv[1])
