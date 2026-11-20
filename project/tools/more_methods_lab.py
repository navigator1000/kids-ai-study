#!/usr/bin/env python3
"""ほかの手法の検証（BTC・ETH、2018年〜）。結果は project/research/results.csv に記録する。

  A スーパートレンド(ATR10・3倍)   B ケルトナーチャネル上抜け(EMA20＋ATR10×2、EMAで決済)
  C 平均足が陽線の間は保有          D デュアル・スラスト（4日間の値幅で値幅ブレイク、当日決済）
  E グリッド取引（1時間足。一定間隔ごとに同じ量を買い、その間隔だけ上がったら売る）
  F ETH/BTC比率の平均回帰（zスコア±2で入り、±0.5か10日で決済）
  G 大きく動いた翌日（前日の値動きがATRの2倍以上：急落の翌日に買う／急騰の翌日に買う、当日決済）
  （値幅ブレイクの決済の工夫は exit_combo_lab.py に移した。15分足まで使い、足の中の値動きの順番の扱いも見直した）
共通: 1R＝ATR1本分＝資金の0.5%（単利）、Vantage CFD想定（往復0.05%、買いスワップ年25〜35%、売り0%）、含み損込み。
使い方: python3 project/tools/more_methods_lab.py <キャッシュ（BTC.csv, ETH.csv, BTC_1h.csv, ETH_1h.csv）>
"""
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
import portfolio_lab as PF  # noqa: E402
from btceth_lab import ind, load, state  # noqa: E402
from registry import metrics_row, record  # noqa: E402

RISK, START = 0.5, pd.Timestamp("2018-01-01")
END = pd.Timestamp("2026-11-16")
SPEC = PF.SPEC


def clip(x):
    return x[(x.index >= START) & (x.index <= END)]


def both(fn):
    s = None
    for a in ("BTC", "ETH"):
        x = fn(a)
        s = x if s is None else s.add(x, fill_value=0)
    return clip(s.fillna(0) * RISK)


def show(group, name, x, n=None, memo=""):
    yrs = max((x.index[-1] - x.index[0]).days / 365.25, 0.5)
    r = metrics_row(x, n / yrs if n else np.nan)
    print(f"  {name:40s} 年{r['年平均%']:+6.1f}% 落込{r['最大の落ち込み%']:5.1f}% 効率{r['効率']:5.2f} マイナスの年 {r['マイナスの年']:22s}"
          f" プラスの月{r['プラスの月%']:4.0f}% ｜ 前半{r['前半の年平均%']:+.1f}%→後半{r['後半の年平均%']:+.1f}%")
    record(グループ=group, 名前=name, 銘柄="BTC,ETH", 時間足="日足" if group != "グリッド" else "1時間足", メモ=memo, **r)
    return r


# ---------------------------------------------------------------- A〜C 指標で持つ/持たない
def supertrend(I, n=10, m=3.0):
    h, l, c = I["h"], I["l"], I["c"]
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    atr = pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    hl2 = (h + l) / 2
    up, dn = hl2 + m * atr, hl2 - m * atr
    fu, fl, tr_ = up.copy(), dn.copy(), np.ones(len(c))
    for i in range(1, len(c)):
        fu[i] = up[i] if (up[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = dn[i] if (dn[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        tr_[i] = 1 if c[i] > fu[i - 1] else (-1 if c[i] < fl[i - 1] else tr_[i - 1])
    return tr_


def keltner(I, side=1):
    c = pd.Series(I["c"])
    mid = c.ewm(span=20, adjust=False).mean().to_numpy()
    h, l, cc = I["h"], I["l"], I["c"]
    pc = np.r_[cc[0], cc[:-1]]
    atr = pd.Series(np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))).rolling(10).mean().to_numpy()
    if side == 1:
        return state(cc > mid + 2 * atr, cc < mid)
    return state(cc < mid - 2 * atr, cc > mid)


def heikin(I):
    o, h, l, c = I["o"], I["h"], I["l"], I["c"]
    hc = (o + h + l + c) / 4
    ho = np.zeros(len(c))
    ho[0] = (o[0] + c[0]) / 2
    for i in range(1, len(c)):
        ho[i] = (ho[i - 1] + hc[i - 1]) / 2
    return (hc > ho).astype(float)


# ---------------------------------------------------------------- D デュアル・スラスト
def dual_thrust(I, side, a, k=0.5, N=4, direction=False):
    o, h, l, c = I["o"], I["h"], I["l"], I["c"]
    H, L, C = pd.Series(h), pd.Series(l), pd.Series(c)
    rng = np.maximum(H.rolling(N).max() - C.rolling(N).min(), C.rolling(N).max() - L.rolling(N).min()).shift(1).to_numpy()
    atr = np.r_[np.nan, I["atr"][:-1]]
    trig = o + side * k * rng
    hit = (h >= trig) if side == 1 else (l <= trig)
    if direction:
        ma = np.r_[np.nan, C.rolling(5).mean().to_numpy()[:-1]]
        hit &= (o > ma) if side == 1 else (o < ma)
    hit &= (atr > 0) & (rng > 0)
    sw = SPEC[a][1] if side == 1 else SPEC[a][2]
    R = np.where(hit, side * (c - trig) / atr - SPEC[a][0] * trig / atr - sw / 36500 * trig / atr, 0.0)
    return pd.Series(np.nan_to_num(R), index=I["idx"]), int(np.nansum(hit))


# ---------------------------------------------------------------- E グリッド（1時間足）
@njit(cache=True)
def grid_engine(o, h, l, c, roll, step, M, stop, cost, swap_day):
    """同じ量を M 段まで。買った値の step 上で売る（後入れ先出し）。全部持っている状態で最後の買値から stop 下がったら全部損切り。
    資金比の通算損益（含み込み）と取引回数を返す。1段の量＝資金の 1/M（全部持つと資金の1倍）。"""
    N = len(c)
    out = np.zeros(N)
    held = np.zeros(M)
    k, ref, realized, trades = 0, o[0], 0.0, 0
    unit = 1.0 / M
    for t in range(N):
        # 売り（いちばん新しく買った分から）
        while k > 0 and h[t] >= held[k - 1] * (1 + step):
            px = max(o[t], held[k - 1] * (1 + step))
            realized += unit * (px / held[k - 1] - 1) - cost / 2 * unit
            k -= 1
            trades += 1
            if k == 0:
                ref = px
        # 買い
        while k < M:
            nxt = (held[k - 1] if k > 0 else ref) * (1 - step)
            if l[t] <= nxt:
                held[k] = min(o[t], nxt)
                realized -= cost / 2 * unit
                k += 1
            else:
                break
        # 全部持っていて、さらに下がったら損切り
        if k == M and stop > 0 and l[t] <= held[k - 1] * (1 - stop):
            px = min(o[t], held[k - 1] * (1 - stop))
            for i in range(k):
                realized += unit * (px / held[i] - 1) - cost / 2 * unit
            trades += k
            k, ref = 0, px
        # 何も持っていなくて上がったら、基準を引き上げる
        if k == 0 and c[t] > ref * (1 + step):
            ref = c[t]
        if roll[t]:
            realized -= swap_day * unit * k
        flt = 0.0
        for i in range(k):
            flt += unit * (c[t] / held[i] - 1)
        out[t] = realized + flt
    return out, trades


def grid(cache, a, step, M, stop):
    df = pd.read_csv(os.path.join(cache, f"{a}_1h.csv"), index_col=0, parse_dates=True)
    df = df[~df.index.duplicated()].sort_index()
    df = df[df.index >= START]
    roll = np.asarray(df.index.hour == 21, dtype=np.bool_)
    tot, n = grid_engine(df.open.to_numpy(float), df.high.to_numpy(float), df.low.to_numpy(float), df.close.to_numpy(float),
                         roll, step, M, stop, SPEC[a][0], SPEC[a][1] / 36500)
    eq = pd.Series(tot, index=df.index).resample("1D").last().ffill()
    return eq.diff().fillna(eq.iloc[0]) * 100, n


# ---------------------------------------------------------------- F ペアの平均回帰
def pair_mr(I, n=20, zin=2.0, zout=0.5, maxd=10):
    e = pd.Series(I["ETH"]["c"], index=I["ETH"]["idx"])
    b = pd.Series(I["BTC"]["c"], index=I["BTC"]["idx"])
    df = pd.concat({"e": e, "b": b}, axis=1).dropna()
    lr = np.log(df.e / df.b)
    z = ((lr - lr.rolling(n).mean()) / lr.rolling(n).std()).to_numpy()
    spread = (df.e.pct_change() - df.b.pct_change()).fillna(0).to_numpy()
    unit = (pd.Series(spread).abs().rolling(20).mean().shift(1) * 1.25).to_numpy()
    pos, side, held, trades = np.zeros(len(df)), 0, 0, 0
    for i in range(1, len(df)):
        if side == 0:
            if z[i - 1] > zin:
                side, held = -1, 0
                trades += 1
            elif z[i - 1] < -zin:
                side, held = 1, 0
                trades += 1
        else:
            held += 1
            if abs(z[i - 1]) < zout or held >= maxd:
                side = 0
        pos[i] = side
    flip = np.abs(np.diff(np.r_[0, pos]))
    long_swap = np.where(pos > 0, SPEC["ETH"][1], np.where(pos < 0, SPEC["BTC"][1], 0)) / 36500
    R = (pos * spread - flip * 2 * SPEC["ETH"][0] - np.abs(pos) * long_swap) / unit
    return pd.Series(np.nan_to_num(R, posinf=0, neginf=0), index=df.index), trades


# ---------------------------------------------------------------- G 大きく動いた翌日
def big_day(I, a, kind, th=2.0):
    o, c, atr = I["o"], I["c"], I["atr"]
    pc = np.r_[np.nan, c[:-1]]
    move = np.r_[np.nan, (c - pc)[:-1]]
    a1 = np.r_[np.nan, atr[:-1]]
    sig = (move < -th * a1) if kind == "rebound" else (move > th * a1)
    sig &= a1 > 0
    R = np.where(sig, (c - o) / a1 - SPEC[a][0] * o / a1 - SPEC[a][1] / 36500 * o / a1, 0.0)
    return pd.Series(np.nan_to_num(R), index=I["idx"]), int(np.nansum(sig))


def main(cache):
    btc = load(cache, "BTC")
    I = {a: ind(load(cache, a), btc) for a in ("BTC", "ETH")}

    print("==== A〜C 指標で持つ／持たない（翌日の始値で売買）")
    show("人気の指標", "スーパートレンド・買いだけ", both(lambda a: PF.pos_side(I[a], (supertrend(I[a]) == 1).astype(float), 1, SPEC[a][0], SPEC[a][1])))
    show("人気の指標", "スーパートレンド・買い＋売り", both(lambda a: PF.pos_side(I[a], (supertrend(I[a]) == 1).astype(float), 1, SPEC[a][0], SPEC[a][1])
                                                     .add(PF.pos_side(I[a], (supertrend(I[a]) == -1).astype(float), -1, SPEC[a][0], SPEC[a][2]), fill_value=0)))
    show("人気の指標", "スーパートレンド・買い（BTCが200日平均の上）", both(lambda a: PF.pos_side(I[a], (supertrend(I[a]) == 1) * I[a]["btc200"], 1, SPEC[a][0], SPEC[a][1])))
    show("人気の指標", "ケルトナー上抜け・買いだけ", both(lambda a: PF.pos_side(I[a], keltner(I[a], 1), 1, SPEC[a][0], SPEC[a][1])))
    show("人気の指標", "ケルトナー・買い＋売り", both(lambda a: PF.pos_side(I[a], keltner(I[a], 1), 1, SPEC[a][0], SPEC[a][1])
                                                .add(PF.pos_side(I[a], keltner(I[a], -1), -1, SPEC[a][0], SPEC[a][2]), fill_value=0)))
    show("人気の指標", "平均足が陽線の間は保有", both(lambda a: PF.pos_side(I[a], heikin(I[a]), 1, SPEC[a][0], SPEC[a][1])))

    print("\n==== D デュアル・スラスト（当日決済）")
    for k in (0.4, 0.6):
        for direction in (False, True):
            n = [0]

            def f(a, k=k, direction=direction):
                L_, nl = dual_thrust(I[a], 1, a, k, direction=direction)
                S_, ns = dual_thrust(I[a], -1, a, k, direction=direction)
                n[0] += nl + ns
                return L_ + 0.5 * S_
            x = both(f)
            show("値幅ブレイクの仲間", f"デュアル・スラスト k={k}{'・5日平均の向き' if direction else ''}（買い＋売り×0.5）", x, n[0])

    print("\n==== E グリッド取引（1時間足・同じ量・全部持つと資金の1倍）")
    for step, M, stop in ((0.01, 10, 0.0), (0.02, 10, 0.0), (0.03, 10, 0.0), (0.02, 10, 0.05), (0.03, 20, 0.0), (0.05, 10, 0.0)):
        xs, n = [], 0
        for a in ("BTC", "ETH"):
            s, nn = grid(cache, a, step, M, stop)
            xs.append(s / 2)                                   # BTCとETHに資金を半分ずつ
            n += nn
        x = clip(pd.concat(xs, axis=1).fillna(0).sum(axis=1))
        show("グリッド", f"グリッド {step*100:.0f}%間隔・{M}段・{'損切り' + str(int(stop*100)) + '%' if stop else '損切りなし'}", x, n,
             "資金をBTCとETHに半分ずつ。全部持つと資金の1倍")

    print("\n==== F ETH/BTC比率の平均回帰")
    for zin in (1.5, 2.0, 2.5):
        R, n = pair_mr(I, zin=zin)
        show("ペア", f"ETH/BTCの平均回帰 z±{zin}", clip(R * RISK), n)

    print("\n==== G 大きく動いた翌日（前日の値動きがATRの2倍以上、当日決済）")
    for kind, lab in (("rebound", "急落の翌日に買う"), ("follow", "急騰の翌日に買う")):
        n = [0]

        def g(a, kind=kind):
            s, nn = big_day(I[a], a, kind)
            n[0] += nn
            return s
        show("大きく動いた翌日", lab, both(g), n[0])


if __name__ == "__main__":
    main(sys.argv[1])
