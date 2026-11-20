#!/usr/bin/env python3
"""ナンピン・マーチンゲールの検証（BTC・ETH 1時間足、2018年〜）。まだ実践はしない前提で、結果を results.csv に記録する。

仕組み（買いの場合。売りは逆向き）
  ・1段目を買い、価格が「間隔」だけ下がるごとに、前の段の「倍率」倍の量を買い足す（最大「段数」まで）。
  ・平均取得価格から「利確幅」だけ上がったら全部決済して、すぐ次のサイクルを始める。
  ・最大段数のさらに1間隔先まで下がったら全部損切り（「損切りなし」はそのまま持ち続ける）。
  ・量：最大段数まで買い足した時の合計が「資金×建玉倍率」になるように1段目を決める。単利。
  ・口座が飛ぶ：含み損込みの残高が資金の20%以下になったら強制決済。資金を入れ直して続け、回数を数える。
  ・同じ1時間に買い足しと利確の両方の価格に届いたら、買い足しが先（厳しめ）。
コスト: 往復0.05%。スワップ（UTC21時をまたぐごと）: 買い BTC年25%・ETH年35%、売り0%。
使い方: python3 project/tools/martingale_lab.py <キャッシュ（BTC_1h.csv, ETH_1h.csv）> [gold]
  gold: 金（PAXG 1時間足）で主なパターン
"""
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
from registry import metrics_row, record  # noqa: E402

START = pd.Timestamp("2018-01-01")
SWAP = {"BTC": 25.35, "ETH": 35.49}
COST = 0.0005


@njit(cache=True)
def engine(o, h, l, c, atr, allow, roll, side, step_atr, step, mult, levels, tp_kind, tp, stop_at_max,
           lev, cost, swap_day, max_hold, ruin, anti):
    """戻り値: 毎時の通算損益(資金比、含み込み), サイクル(開始, 終了, 段数, 損益, 最大含み損), 飛んだ回数, 最大建玉"""
    N = len(c)
    wsum = 0.0
    for i in range(levels):
        wsum += mult ** i
    base = lev / wsum
    total = np.zeros(N)
    cs, ce, ck = np.zeros(N, np.int64), np.zeros(N, np.int64), np.zeros(N, np.int64)
    cp, cw = np.zeros(N), np.zeros(N)
    ncyc, ruins = 0, 0
    epx, en = np.zeros(levels), np.zeros(levels)
    realized, acct_start = 0.0, 0.0
    incyc, k, start, stepabs, nxt, tpx, worst, peak = False, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0
    for t in range(N):
        if not incyc and allow[t] and atr[t] > 0:
            incyc, k, start, worst, peak = True, 1, t, 0.0, o[t]
            epx[0], en[0] = o[t], base
            realized -= cost / 2 * base
            stepabs = step * atr[t] if step_atr else step * o[t]
            nxt = o[t] + side * stepabs if anti else o[t] - side * stepabs
        if incyc:
            avg = 0.0
            tot = 0.0
            for i in range(k):
                avg += epx[i] * en[i]
                tot += en[i]
            avg /= tot
            if tp_kind == 0:
                tpx = avg * (1 + side * tp)
            else:
                tpx = avg + side * tp * stepabs
            added, closed, exitpx = False, False, 0.0
            if not anti:
                # 逆行 → 買い足し
                while k < levels and ((side == 1 and l[t] <= nxt) or (side == -1 and h[t] >= nxt)):
                    f = nxt
                    if (side == 1 and o[t] < nxt) or (side == -1 and o[t] > nxt):
                        f = o[t]
                    epx[k], en[k] = f, base * mult ** k
                    realized -= cost / 2 * en[k]
                    k += 1
                    added = True
                    nxt = f * (1 - side * step) if not step_atr else f - side * stepabs
                if k == levels and stop_at_max and ((side == 1 and l[t] <= nxt) or (side == -1 and h[t] >= nxt)):
                    exitpx = nxt
                    if (side == 1 and o[t] < nxt) or (side == -1 and o[t] > nxt):
                        exitpx = o[t]
                    closed = True
                if not closed and not added and ((side == 1 and h[t] >= tpx) or (side == -1 and l[t] <= tpx)):
                    exitpx = tpx
                    if (side == 1 and o[t] > tpx) or (side == -1 and o[t] < tpx):
                        exitpx = o[t]
                    closed = True
            else:
                # 逆マーチン：順行で買い増し、「最後の段−1間隔」か「最も有利な価格−2間隔」の近い方まで戻ったら全部決済
                stop_px = epx[k - 1] - side * stepabs
                trail = peak - side * 2 * stepabs
                if (side == 1 and trail > stop_px) or (side == -1 and trail < stop_px):
                    stop_px = trail
                if (side == 1 and l[t] <= stop_px) or (side == -1 and h[t] >= stop_px):
                    exitpx = stop_px
                    if (side == 1 and o[t] < stop_px) or (side == -1 and o[t] > stop_px):
                        exitpx = o[t]
                    closed = True
                else:
                    while k < levels and ((side == 1 and h[t] >= nxt) or (side == -1 and l[t] <= nxt)):
                        epx[k], en[k] = nxt, base * mult ** k
                        realized -= cost / 2 * en[k]
                        k += 1
                        nxt = nxt + side * stepabs
            if side == 1 and h[t] > peak:
                peak = h[t]
            elif side == -1 and l[t] < peak:
                peak = l[t]
            if not closed and max_hold > 0 and t - start >= max_hold:
                exitpx, closed = c[t], True
            if closed:
                pnl = 0.0
                for i in range(k):
                    pnl += en[i] * side * (exitpx / epx[i] - 1) - cost / 2 * en[i]
                realized += pnl
                cs[ncyc], ce[ncyc], ck[ncyc], cp[ncyc], cw[ncyc] = start, t, k, pnl, worst
                ncyc += 1
                incyc = False
            elif roll[t]:
                tot = 0.0
                for i in range(k):
                    tot += en[i]
                realized -= swap_day * tot
        flt = 0.0
        if incyc:
            for i in range(k):
                flt += en[i] * side * (c[t] / epx[i] - 1)
            if flt < worst:
                worst = flt
        total[t] = realized + flt
        if 1.0 + (total[t] - acct_start) <= ruin:
            # 口座が飛ぶ：強制決済、資金を入れ直して続ける
            if incyc:
                realized += flt
                cs[ncyc], ce[ncyc], ck[ncyc], cp[ncyc], cw[ncyc] = start, t, k, flt, worst
                ncyc += 1
                incyc = False
            total[t] = realized
            acct_start = realized
            ruins += 1
    return total, cs[:ncyc], ce[:ncyc], ck[:ncyc], cp[:ncyc], cw[:ncyc], ruins, lev


def load_hourly(cache, a):
    df = pd.read_csv(os.path.join(cache, f"{a}_1h.csv"), index_col=0, parse_dates=True)
    df = df[~df.index.duplicated()].sort_index()
    d = df.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    pc = d.close.shift()
    tr = pd.concat([d.high - d.low, (d.high - pc).abs(), (d.low - pc).abs()], axis=1).max(axis=1)
    atr_d = tr.rolling(14).mean().shift(1)
    sma200 = d.close.rolling(200).mean().shift(1)
    above = (d.close.shift(1) > sma200)
    hours = df.index.normalize()
    df = df.assign(atr=atr_d.reindex(hours).to_numpy(), above=above.reindex(hours).fillna(False).to_numpy(),
                   below=(~above).reindex(hours).fillna(False).to_numpy() & sma200.reindex(hours).notna().to_numpy())
    return df[df.index >= START]


PATTERNS = {
    # 名前: dict(side, step_atr, step, mult, levels, tp_kind, tp, stop, lev, filt, max_hold, anti)
    "定番：2%ごと2倍・6段・+1%利確・損切りあり": dict(step=0.02, mult=2.0, levels=6, tp=0.01, stop=1),
    "定番・損切りなし": dict(step=0.02, mult=2.0, levels=6, tp=0.01, stop=0),
    "定番・建玉3倍": dict(step=0.02, mult=2.0, levels=6, tp=0.01, stop=1, lev=3.0),
    "深め：2%ごと2倍・8段": dict(step=0.02, mult=2.0, levels=8, tp=0.01, stop=1),
    "間隔広め：5%ごと2倍・5段": dict(step=0.05, mult=2.0, levels=5, tp=0.015, stop=1),
    "ゆるやか：3%ごと1.5倍・8段": dict(step=0.03, mult=1.5, levels=8, tp=0.01, stop=1),
    "同じ量：3%ごと等倍・10段（グリッド）": dict(step=0.03, mult=1.0, levels=10, tp=0.01, stop=1),
    "同じ量・損切りなし": dict(step=0.03, mult=1.0, levels=10, tp=0.01, stop=0),
    "値動き連動：ATR1本ごと1.5倍・8段・利確ATR0.5": dict(step_atr=True, step=1.0, mult=1.5, levels=8, tp_kind=1, tp=0.5, stop=1),
    "値動き連動：ATR1本ごと2倍・6段": dict(step_atr=True, step=1.0, mult=2.0, levels=6, tp_kind=1, tp=0.5, stop=1),
    "利確大きめ：2%ごと1.5倍・8段・+3%利確": dict(step=0.02, mult=1.5, levels=8, tp=0.03, stop=1),
    "トレンドの向き（買い）：200日平均の上だけ": dict(step=0.02, mult=1.5, levels=8, tp=0.01, stop=1, filt="above"),
    "トレンドの向き（売り）：200日平均の下だけ": dict(side=-1, step=0.02, mult=1.5, levels=8, tp=0.01, stop=1, filt="below"),
    "期限つき：7日で強制決済": dict(step=0.02, mult=1.5, levels=8, tp=0.01, stop=1, max_hold=168),
    "逆マーチン：3%上がるごと1.5倍・5段（高値から2間隔戻ったら決済）": dict(step=0.03, mult=1.5, levels=5, stop=1, anti=1),
    "ナンピン1回だけ：3%下で同量を1回・+1.5%利確・その3%下で損切り": dict(step=0.03, mult=1.0, levels=2, tp=0.015, stop=1),
    "24時間で強制決済：2%ごと1.5倍・8段": dict(step=0.02, mult=1.5, levels=8, tp=0.01, stop=1, max_hold=24),
    "深くゆるやか：2%ごと1.2倍・12段": dict(step=0.02, mult=1.2, levels=12, tp=0.01, stop=1),
}


def run_pattern(df, a, p):
    side = p.get("side", 1)
    filt = p.get("filt")
    allow = np.ones(len(df), np.bool_) if filt is None else df[filt].to_numpy(np.bool_)
    roll = np.asarray(df.index.hour == 21, dtype=np.bool_)
    out = engine(df.open.to_numpy(float), df.high.to_numpy(float), df.low.to_numpy(float), df.close.to_numpy(float),
                 np.nan_to_num(df.atr.to_numpy(float)), allow, roll, side, p.get("step_atr", False), p["step"], p["mult"],
                 p["levels"], p.get("tp_kind", 0), p.get("tp", 0.01), p.get("stop", 1), p.get("lev", 1.0), COST,
                 (SWAP[a] if side == 1 else 0.0) / 36500, p.get("max_hold", 0), 0.2, p.get("anti", 0))
    total, cs, ce, ck, cp, cw, ruins, lev = out
    eq = pd.Series(total, index=df.index)
    daily = eq.resample("1D").last().ffill().diff().fillna(eq.iloc[0]) * 100       # 資金に対する％（含み込み）
    cyc = pd.DataFrame({"start": df.index[cs], "end": df.index[ce], "k": ck, "pnl": cp * 100, "worst": cw * 100})
    return daily, cyc, ruins


def describe(name, a, daily, cyc, ruins, levels):
    r = metrics_row(daily, len(cyc) / max((daily.index[-1] - daily.index[0]).days / 365.25, 0.5))
    win = (cyc.pnl > 0).mean() * 100 if len(cyc) else np.nan
    deep = (cyc.k >= levels).mean() * 100 if len(cyc) else np.nan
    hours = ((cyc.end - cyc.start).dt.total_seconds() / 3600).median() if len(cyc) else np.nan
    print(f"  {a}: 年{r['年平均%']:+7.1f}% 落込{r['最大の落ち込み%']:7.1f}% 口座が飛んだ{ruins}回 マイナスの年 {r['マイナスの年']:15s}"
          f" ｜ 年{r['年の取引回数']}サイクル 勝率{win:5.1f}% 最大段数まで{deep:4.1f}% 最悪のサイクル{cyc.pnl.min():+.1f}%"
          f" 最大の含み損{cyc.worst.min():+.1f}% 保有の中央値{hours:.0f}時間 プラスの月{r['プラスの月%']:.0f}%")
    return r, win, ruins


def bet_martingale(cache):
    """賭け金のマーチンゲール：値幅ブレイク（BTC+ETH、日足）で負けたら次は2倍、勝ったら1倍に戻す（最大2^n倍）。"""
    from btceth_lab import ind, load
    import portfolio_lab as PF
    btc = load(cache, "BTC")
    seqs = []
    for a in ("BTC", "ETH"):
        I = ind(load(cache, a), btc)
        r = PF.vbo_side(I, 1, PF.SPEC[a][0], PF.SPEC[a][1])
        seqs.append(r[r != 0])
    trades = pd.concat(seqs).sort_index()
    trades = trades[trades.index >= START]
    print("\n==== 賭け金のマーチンゲール（値幅ブレイク買い・BTC+ETH、1R＝資金の0.5%）")
    for cap in (0, 2, 4, 6):
        size, pnl = 1.0, []
        for R in trades.to_numpy():
            pnl.append(R * size * 0.5)
            size = 1.0 if R > 0 else min(size * 2, 2.0 ** cap)
        x = pd.Series(pnl, index=trades.index).groupby(level=0).sum()
        x = x.reindex(pd.date_range(x.index[0], x.index[-1], freq="D")).fillna(0)
        r = metrics_row(x)
        lab = "倍にしない（普通）" if cap == 0 else f"負けたら倍・最大{2**cap}倍"
        print(f"  {lab:18s} 年{r['年平均%']:+6.1f}% 落込{r['最大の落ち込み%']:5.1f}% 効率{r['効率']:4.2f} マイナスの年 {r['マイナスの年']}")
        record(グループ="マーチン", 名前=f"賭け金マーチン（値幅ブレイク）：{lab}", 銘柄="BTC,ETH", 時間足="日足",
               メモ="負けたら次の量を2倍、勝ったら1倍に戻す", **r)


def main(cache):
    data = {a: load_hourly(cache, a) for a in ("BTC", "ETH")}
    for name, p in PATTERNS.items():
        print(f"\n== {name}（最大で資金の{p.get('lev', 1.0):g}倍の建玉）")
        res = {}
        for a, df in data.items():
            daily, cyc, ruins = run_pattern(df, a, p)
            r, win, ruins = describe(name, a, daily, cyc, ruins, p["levels"])
            res[a] = daily
            record(グループ="マーチン", 名前=f"{name}（{a}）", 銘柄=a, 時間足="1時間足",
                   メモ=f"サイクル勝率{win:.1f}%・口座が飛んだ{ruins}回・最大建玉は資金の{p.get('lev', 1.0):g}倍", **r)
    bet_martingale(cache)


if __name__ == "__main__" and len(sys.argv) == 2:
    main(sys.argv[1])


def gold(cache):
    """金（PAXG 1時間足で代用、2020年9月〜）で主なパターン。週末も値がつく点に注意。"""
    SWAP["PAXG"] = 6.53
    df = load_hourly(cache, "PAXG")
    global COST
    keep, COST = COST, 0.0003
    try:
        for name in ("定番：2%ごと2倍・6段・+1%利確・損切りあり", "定番・損切りなし", "ゆるやか：3%ごと1.5倍・8段",
                     "同じ量・損切りなし", "深くゆるやか：2%ごと1.2倍・12段"):
            p = PATTERNS[name]
            daily, cyc, ruins = run_pattern(df, "PAXG", p)
            print(f"\n== 金：{name}")
            r, win, ruins = describe(name, "金", daily, cyc, ruins, p["levels"])
            record(グループ="マーチン", 名前=f"{name}（金・PAXG）", 銘柄="金(PAXG)", 時間足="1時間足",
                   コストの前提="往復0.03%・買いスワップ年6.5%", メモ=f"2020年9月〜。サイクル勝率{win:.1f}%・口座が飛んだ{ruins}回", **r)
    finally:
        COST = keep


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "gold":
    gold(sys.argv[1])
