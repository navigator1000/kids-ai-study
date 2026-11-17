#!/usr/bin/env python3
"""ルール検証ラボ（日足）。

トレンドフィルター × エントリー方法 × 損切り幅 × 利確倍率 を総当たりで過去検証し、
前半（学習期間）だけで設定を選び、後半（検証期間）で通用するかを確かめる。
設定は3銘柄共通（銘柄ごとの最適化はしない＝過剰最適化を避ける）。

売買の流れ（すべて日足の確定後に判定）:
  シグナル : 終値がSMA(n)より上（買い）／下（売り）。n=0はフィルターなし
  エントリー: 0 = 翌日の始値で成行
             0.5 / 1.0 = 終値からATR×0.5 / ×1.0 押した価格に指値（翌日1日だけ有効）
             "bo" = 終値が直近20日高値（売りは安値）を更新したら翌日始値で成行
  損切り   : エントリー価格 − ATR×s
  利確     : エントリー価格 + 損切り幅×tR
  時間切れ : 20日で決着しなければ終値で手仕舞い
  コスト   : 往復 BTC/ETH 0.2%、Gold 0.05%
同じ足で損切りと利確の両方に届いた場合は損切りとみなす（保守的）。
"""
import itertools
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from marketdata import load_all  # noqa: E402

HOLD = 20
COST = {"BTC": 0.002, "ETH": 0.002, "Gold": 0.0005}
SPLIT = {"BTC": "2023-01-01", "ETH": "2023-01-01", "Gold": "2016-01-01"}
SMAS = (0, 50, 100, 200)
ENTRIES = (0, 0.5, 1.0, "bo")
STOPS = (1.0, 1.5, 2.0, 3.0)
TARGETS = (1.0, 1.5, 2.0, 3.0)


def prep(df):
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
    sma = {n: pd.Series(c).rolling(n).mean().to_numpy() for n in SMAS if n}
    hh = pd.Series(h).rolling(20).max().shift(1).to_numpy()
    ll = pd.Series(l).rolling(20).min().shift(1).to_numpy()
    return dict(o=o, h=h, l=l, c=c, atr=atr, sma=sma, hh=hh, ll=ll, t=df.index)


def signal(P, i, side, n, entry):
    c = P["c"][i]
    if n:
        m = P["sma"][n][i]
        if np.isnan(m) or (c - m) * side <= 0:
            return False
    if entry == "bo":
        ref = P["hh"][i] if side == 1 else P["ll"][i]
        if np.isnan(ref) or (c - ref) * side <= 0:
            return False
    return True


def order(P, i, side, entry, s, tR):
    """終値iの時点で出す注文。entry=0/'bo'は成行のためNone（翌日始値で確定）。"""
    A = P["atr"][i]
    if entry in (0, "bo"):
        return None, s * A, A
    L = P["c"][i] - side * entry * A
    return L, s * A, A


def run(P, side, n, entry, s, tR, cost, i0=200):
    o, h, l, c = P["o"], P["h"], P["l"], P["c"]
    adv, fav = (l, h) if side == 1 else (h, l)
    N, trades, i = len(c), [], i0
    while i < N - 1:
        if not signal(P, i, side, n, entry):
            i += 1
            continue
        j = i + 1
        L, risk, _ = order(P, i, side, entry, s, tR)
        if L is None:
            L = fill = o[j]
            at_open = True
        else:
            if (adv[j] - L) * side > 0:          # 指値に届かず
                i += 1
                continue
            fill = min(o[j], L) if side == 1 else max(o[j], L)
            at_open = (o[j] - L) * side <= 0
        stop, tgt = L - side * risk, L + side * tR * risk
        hs = lambda x: (x - stop) * side <= 0      # noqa: E731
        ht = lambda x: (x - tgt) * side >= 0       # noqa: E731
        ex, k, why = None, j, ""
        if hs(fill):
            ex, why = fill, "損切り"
        elif hs(adv[j]):
            ex, why = stop, "損切り"
        elif at_open and ht(fav[j]):
            ex, why = tgt, "利確"
        while ex is None:
            if k - j + 1 >= HOLD:
                ex, why = c[k], "時間切れ"
                break
            k += 1
            if k >= N:
                break
            if hs(o[k]) or ht(o[k]):
                ex, why = o[k], "損切り" if hs(o[k]) else "利確"
            elif hs(adv[k]):
                ex, why = stop, "損切り"
            elif ht(fav[k]):
                ex, why = tgt, "利確"
        info = dict(t=P["t"][j], fill=fill, stop=stop, tgt=tgt, risk=risk)
        if ex is None:
            trades.append(dict(**info, R=np.nan, open=True, bars=N - j))
            break
        R = side * (ex - fill) / risk - cost * fill / risk
        trades.append(dict(**info, R=R, open=False, bars=k - j + 1, why=why, exit_t=P["t"][k], ex=ex))
        i = k
    return trades


def summarize(R):
    R = np.asarray([r for r in R if not np.isnan(r)])
    n = len(R)
    if n < 2:
        return dict(n=n, win=np.nan, exp=np.nan, t=np.nan, pf=np.nan, maxl=0)
    mx = cur = 0
    for r in R:
        cur = cur + 1 if r <= 0 else 0
        mx = max(mx, cur)
    sd = R.std(ddof=1)
    return dict(n=n, win=(R > 0).mean(), exp=R.mean(), t=R.mean() / sd * np.sqrt(n) if sd else np.nan,
                pf=R[R > 0].sum() / -R[R < 0].sum() if (R < 0).any() else np.inf, maxl=mx)


def evaluate(data):
    Ps = {k: prep(v) for k, v in data.items()}
    rows = []
    for side, n, entry, s, tR in itertools.product((1, -1), SMAS, ENTRIES, STOPS, TARGETS):
        for a, P in Ps.items():
            tr = run(P, side, n, entry, s, tR, COST[a])
            sp = pd.Timestamp(SPLIT[a])
            tr_in = [x["R"] for x in tr if x["t"] < sp]
            tr_out = [x["R"] for x in tr if x["t"] >= sp]
            si, so = summarize(tr_in), summarize(tr_out)
            yrs_in = (sp - P["t"][200]).days / 365.25
            rows.append(dict(side=side, sma=n, entry=str(entry), stop=s, tR=tR, asset=a,
                             **{f"in_{k}": v for k, v in si.items()},
                             **{f"out_{k}": v for k, v in so.items()},
                             per_year=si["n"] / yrs_in))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    cache = sys.argv[1] if len(sys.argv) > 1 else None
    df = evaluate(load_all(cache))
    df.to_csv((cache or ".") + "/lab_results.csv", index=False)
    key = ["side", "sma", "entry", "stop", "tR"]
    g = df.groupby(key).agg(in_exp=("in_exp", "mean"), in_min_n=("in_n", "min"), in_min_exp=("in_exp", "min"),
                            out_exp=("out_exp", "mean"), out_min_exp=("out_exp", "min"),
                            out_n=("out_n", "sum"), per_year=("per_year", "sum")).reset_index()
    g = g[g.in_min_n >= 20]
    pd.set_option("display.width", 200)
    for side in (1, -1):
        gs = g[g.side == side].sort_values("in_exp", ascending=False)
        print(f"\n===== {'買い' if side == 1 else '売り'}: 学習期間の期待値（3銘柄平均, R/回）上位15 =====")
        print(gs.head(15).round(3).to_string(index=False))
        print(f"  全{len(gs)}通りのうち 学習期間プラス: {(gs.in_exp > 0).mean()*100:.0f}%  "
              f"検証期間プラス: {(gs.out_exp > 0).mean()*100:.0f}%  "
              f"学習上位20%の検証プラス率: {(gs.head(len(gs)//5).out_exp > 0).mean()*100:.0f}%")
        base = gs[gs.sma == 0]
        filt = gs[gs.sma > 0]
        print(f"  トレンドフィルターなし 検証期間平均 {base.out_exp.mean():.3f}R / あり {filt.out_exp.mean():.3f}R")
