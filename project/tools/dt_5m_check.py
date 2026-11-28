#!/usr/bin/env python3
"""デイトレ第1回で合格した TA03（ニューヨークの寄り付きのレンジのブレイク）を、5分足で計算し直す（計画書 TP03 の一部）。

15分足では、約定した足で損切りにも届いていたら「損切りが先」とみなした（厳しめ）。5分足なら、その重なりが3分の1ほどに減る。
5分足でも同じ「厳しめ」と、約定した足では損切りを見ない「甘め」の両方で計算し、本当の値の幅を見る。
レンジは米国株の始まり（UTC 13時半、冬は14時半）からの30分・60分、決済は米国株の終わり（UTC 20時、冬は21時）。
1R＝日足のATR1本分＝資金の0.5%、往復0.05%（0%・0.10%も）。結果は results.csv（グループ「計画・デイトレ第1回」、計画ID TP03）。
使い方: python3 project/tools/dt_5m_check.py <キャッシュ（BTC_5m.csv・ETH_5m.csv・BTC_15m_vol.csv・ETH_15m_vol.csv）>
"""
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dt1_lab as DT  # noqa: E402
import vantage_exit_time as VX  # noqa: E402

P = 288


@njit
def brk5(O, H, L, C, start, end, xbar, up, dn, slu, sld, ok, strict):
    nd = start.shape[0]
    N = O.shape[0]
    rd = np.empty(nd, np.int64)
    rs = np.empty(nd, np.int64)
    re = np.empty(nd)
    rx = np.empty(nd)
    rb = np.empty(nd, np.int64)
    rxb = np.empty(nd, np.int64)
    nt = 0
    for d in range(nd):
        if not ok[d]:
            continue
        b0 = d * 288
        last = b0 + xbar[d]
        if last > N:
            continue
        k = -1
        side = 0
        e = 0.0
        for j in range(b0 + start[d], min(b0 + end[d], last)):
            u = H[j] >= up[d]
            v = L[j] <= dn[d]
            if u and v:
                break
            if u:
                k, side, e = j, 1, max(O[j], up[d])
                break
            if v:
                k, side, e = j, -1, min(O[j], dn[d])
                break
        if k < 0:
            continue
        sl = slu[d] if side == 1 else sld[d]
        x = C[last - 1]
        xb = last - 1
        for j in range(k if strict else k + 1, last):
            if side == 1 and L[j] <= sl:
                x = sl if j == k else min(O[j], sl)
                xb = j
                break
            if side == -1 and H[j] >= sl:
                x = sl if j == k else max(O[j], sl)
                xb = j
                break
        rd[nt], rs[nt], re[nt], rx[nt], rb[nt], rxb[nt] = d, side, e, x, k, xb
        nt += 1
    return rd[:nt], rs[:nt], re[:nt], rx[:nt], rb[:nt], rxb[:nt]


def load5(cache, days15):
    X = {}
    for a in DT.COINS:
        df = pd.read_csv(os.path.join(cache, f"{a}_5m.csv"), index_col=0, parse_dates=True)
        df = df[~df.index.duplicated()].sort_index()
        full = pd.date_range(days15[0], days15[-1] + pd.Timedelta(days=1), freq="5min", inclusive="left")
        df = df.reindex(full)
        for c in ("open", "high", "low", "close"):
            df[c] = df[c].ffill().bfill()
        X[a] = {k: np.ascontiguousarray(df[c].to_numpy()) for k, c in (("O", "open"), ("H", "high"), ("L", "low"), ("C", "close"))}
    return X


def main(cache):
    DT.GROUP = "計画・デイトレ第1回・5分足"
    X15 = DT.load(cache)
    idx = X15["BTC"]["idx"]
    X5 = load5(cache, idx)
    us = VX.summer(idx)
    nd = len(idx)
    ctx = dict(X=X15, idx=idx)
    import round1_lab as R1
    import round7_lab as R7
    D = R1.prep(cache)
    parts = R7.s1_parts(D, DT.COINS, D["BTC"])
    per, _ = R7.build(parts, R7.size_rules(parts, DT.COINS)["S4案"])
    ctx.update(s4=R1.total(per), s4r=R1.stats(R1.total(per)))
    print("==== TP03（一部）TA03 を5分足で：約定した足で損切りを先とみなす（厳しめ）／見ない（甘め）")
    for rl15 in (2, 4):
        rl = rl15 * 3
        s0 = np.where(us, 162, 174)
        xb = np.where(us, 240, 252)
        for strict in (True, False):
            out = {}
            for a in DT.COINS:
                x5, x15 = X5[a], X15[a]
                Hm, Lm = x5["H"].reshape(nd, P), x5["L"].reshape(nd, P)
                cols = s0[:, None] + np.arange(rl)[None, :]
                hi = np.take_along_axis(Hm, cols, 1).max(1)
                lo = np.take_along_axis(Lm, cols, 1).min(1)
                ok = np.isfinite(x15["atr"])
                d, s, e, xp, eb, xbb = brk5(x5["O"], x5["H"], x5["L"], x5["C"], (s0 + rl).astype(np.int64), (xb - 1).astype(np.int64),
                                            xb.astype(np.int64), hi, lo, lo, hi, ok, strict)
                # res_trades は15分足の本数でスワップを見るが、決済はどれもスワップの前なので、本数を15分足に直して渡す
                out[a] = DT.res_trades(x15, a, d, s, e, xp, np.ones(len(d)), eb // 3, xbb // 3)
            DT.show(ctx, "TP03", f"TA03を5分足で：ニューヨーク{rl15 * 15}分・損切りあり・{'厳しめ' if strict else '甘め'}", out,
                    memo="5分足で計算し直した確かめ")


if __name__ == "__main__":
    main(sys.argv[1])
