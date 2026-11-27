#!/usr/bin/env python3
"""ETHだけで回す場合の、入金額ごとのロットと成績（Vantage の実際の条件、毎日8時20分に決済と注文）。

・形：S4案・R02（どちらも ETH の勝ち負けだけで量を決める）と M08。R03 は BTC の勝ち負けも使うので外す。
  向きは S1 と同じく BTC の5日平均（ETH 自身の5日平均にした場合も見る）。
・8時20分は、8時15分と8時30分の区切りではさむ（vantage_cycle_time.py と同じ計算）。
・ロット＝倍率 × 入金額 × 1回の量（0.5%か1.0%）÷（ATR × ドル円）を0.01に丸める（最小0.01）。ドル円は日ごと。入金額は固定（単利）。
・スワップは日本時間7時（夏は6時）・金曜3日分・土日なし。往復0.05%。
使い方: python3 project/tools/vantage_eth_lots.py <キャッシュ（BTC_15m.csv・ETH_15m.csv・USDJPY.csv）>
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import more_methods_lab as MM  # noqa: E402
import round1_lab as R1  # noqa: E402
import round4_lab as R4  # noqa: E402
import vantage_cycle_time as VC  # noqa: E402
import vantage_exit_time as VX  # noqa: E402

DEPOSITS = (10_000, 30_000, 50_000, 100_000, 200_000, 300_000, 500_000, 1_000_000)
RISKS = (0.005, 0.01)


def eth_trades(D, off, own=False):
    """ETH の取引（S1 の向き＝BTCの5日平均、own=True なら ETH 自身の5日平均）と M08。8時15分なら off=93。"""
    X = {a: (VC.shifted(D[a], off) if off else dict(D[a], jst=D[a]["idx"], wd=D[a]["idx"].dayofweek.to_numpy())) for a in VX.COINS}
    n = min(len(X[a]["dO"]) for a in VX.COINS)
    for a in VX.COINS:
        for k, v in list(X[a].items()):
            if k == "M":
                X[a]["M"] = {kk: vv[:n] for kk, vv in v.items()}
            elif isinstance(v, (np.ndarray, pd.Index)) and len(v) >= n:
                X[a][k] = v[:n]
    x, bx = X["ETH"], X["BTC"]
    days, jst = x["idx"], pd.DatetimeIndex(x["jst"])
    hour = np.where(VX.summer(jst), 21, 22)
    roll = np.clip(np.asarray(((jst + pd.to_timedelta(hour, unit="h")) - days) / pd.Timedelta(minutes=15), float), 0, 96).astype(int)
    offa, xbar, sched = np.zeros(n, int), np.full(n, 96), VX.SCHED["A"]
    ref = x if own else bx
    bdir = np.sign(ref["dO"] - R1.sma_prev(ref, 5))
    q = R4.mult(x, "静かな日だけ") > 0
    hi, lo = x["dO"] + 0.8 * x["rng"], x["dO"] - 0.8 * x["rng"]
    ok = ~np.isnan(x["atr"]) & ~np.isnan(x["rng"])
    nan = np.full(n, np.nan)
    T = pd.concat([VX.trades(x, "ETH", hi, nan, ok & (np.nan_to_num(bdir) > 0), 92, 1.0, 0.5, 1.0, offa, xbar, roll, sched, both=False),
                   VX.trades(x, "ETH", nan, lo, ok & (np.nan_to_num(bdir) < 0), 92, 1.0, 0.5, 1.0, offa, xbar, roll, sched, both=False)],
                  ignore_index=True)
    T["q"] = q[T.d.to_numpy()]
    m8 = [VX.trades(x, "ETH", hi, lo, q & ok, 80, 1.0, 0.5, 1.0, offa, xbar, roll, sched)]
    with np.errstate(invalid="ignore"):
        m = (x["dH"] < np.r_[np.nan, x["dH"][:-1]]) & (x["dL"] > np.r_[np.nan, x["dL"][:-1]])
    m8.append(VX.trades(x, "ETH", np.r_[np.nan, x["dH"][:-1]], np.r_[np.nan, x["dL"][:-1]], np.r_[False, m[:-1]], 92, 1.0, 0.5, 0.5,
                        offa, xbar, roll, sched))
    hh, ll = pd.Series(x["dH"]).rolling(5).max(), pd.Series(x["dL"]).rolling(5).min()
    with np.errstate(invalid="ignore"):
        m = ((hh < hh.shift(5)) & (ll > ll.shift(5))).to_numpy()
    m8.append(VX.trades(x, "ETH", np.r_[np.nan, hh.to_numpy()[:-1]], np.r_[np.nan, ll.to_numpy()[:-1]], np.r_[False, m[:-1]], 92, 1.0, 0.5,
                        0.5, offa, xbar, roll, sched))
    TM = pd.concat(m8, ignore_index=True)
    TM["q"] = True
    # 量の倍率（ETH の勝ち負けだけ）
    t = T[T.q & (days[T.d.to_numpy()] >= MM.START)]
    cnt = np.bincount(t.d, minlength=n).astype(float)
    win = np.bincount(t.d, weights=(t.R > 0).astype(float), minlength=n)
    wk = np.zeros(n, bool)
    lost = (VX.known_window(cnt, wk, 2) >= 1) & (VX.known_window(win, wk, 2) == 0)
    n1, w1 = VX.known_window(cnt, wk, 1), VX.known_window(win, wk, 1)
    won = (n1 >= 1) & (w1 == n1)
    mult = {"S1": np.ones(n), "S4案": np.where(lost, 2.0, 1.0), "R02": np.where(lost, 2.0, np.where(won, 0.5, 1.0))}
    return days, x, T, TM, mult


def yen(days, x, T, m, dep, risk, fx):
    """入金額 dep・1回の量 risk で、0.01ロットに丸めた時の毎日の損益（円）と、ロット・ほしい量。"""
    d = T.d.to_numpy()
    k = T.q.to_numpy() & ((np.ones(len(T)) if m is None else m[d]) > 0)
    d, R, w, cw = d[k], T.R.to_numpy()[k], T.w.to_numpy()[k], T.cw.to_numpy()[k]
    mm = np.ones(len(d)) if m is None else m[d]
    A, f = x["atr"][d], fx[d]
    tgt = mm * cw * w * dep * risk / (A * f)
    lots = np.maximum(np.round(tgt / 0.01) * 0.01, 0.01)
    p = lots * R * A * f
    pnl = pd.Series(np.bincount(d, weights=p, minlength=len(days)), index=days)
    margin = pd.Series(np.bincount(d, weights=lots * x["dO"][d] * f * 0.002, minlength=len(days)), index=days)
    return MM.clip(pnl), MM.clip(margin), lots, tgt, d


def stats(p, dep):
    yrs = (MM.END - MM.START).days / 365.25
    eq = p.cumsum()
    dd = float((eq.cummax().clip(lower=0) - eq).max())
    yearly = p.groupby(p.index.year).sum()
    mon = p.resample("ME").sum()
    return dict(a=p.sum() / yrs, ap=p.sum() / yrs / dep * 100, dd=dd, ddp=dd / dep * 100, st=mon.mean() / mon.std(ddof=1) * np.sqrt(12),
                neg=",".join(str(y) for y, v in yearly.items() if v < 0) or "なし", worst_day=float(p.min()), worst_year=float(yearly.min()),
                yearly=yearly)


def main(cache):
    D = R1.prep(cache)
    fx = pd.read_csv(os.path.join(cache, "USDJPY.csv"), index_col=0, parse_dates=True)["close"]
    fx = fx[~fx.index.duplicated()].sort_index()
    res = {}
    for off, lab in ((93, "8時15分"), (94, "8時30分"), (0, "9時")):
        for own in (False, True):
            days, x, T, TM, mult = eth_trades(D, off, own)
            fxd = fx.reindex(pd.DatetimeIndex(days).normalize().union(fx.index)).ffill().reindex(pd.DatetimeIndex(days).normalize()).bfill().to_numpy()
            res[(off, own)] = (days, x, T, TM, mult, fxd)
            print(f"\n==== {lab}・向きは{'ETH自身' if own else 'BTC'}の5日平均（丸めなし、入金100万円・1回1.0%）")
            for key, TT, m in (("S4案", T, mult["S4案"]), ("R02", T, mult["R02"]), ("M08", TM, None), ("S1", T, mult["S1"])):
                p, *_ = yen(days, x, TT, m, 1_000_000, 0.01, fxd)
                # 丸めなしの値：ロットの丸めの代わりに、ほしい量のまま
                d = TT.d.to_numpy()
                k = TT.q.to_numpy() & ((np.ones(len(TT)) if m is None else m[d]) > 0)
                mm = (np.ones(len(TT)) if m is None else m[d])[k]
                raw = pd.Series(np.bincount(d[k], weights=0.01 * mm * TT.cw.to_numpy()[k] * TT.w.to_numpy()[k] * TT.R.to_numpy()[k] * 100,
                                            minlength=len(days)), index=days)
                r = R1.stats(MM.clip(raw))
                R1.line(f"{key}", r)
    print("\n==== 入金額ごと（ETHだけ・向きはBTCの5日平均・実際のロット）：8時15分 ／ 8時30分")
    for key in ("R02", "S4案", "M08"):
        for risk in RISKS:
            print(f"-- {key}・1回{risk * 100:.1f}%")
            for dep in DEPOSITS:
                cells = []
                for off in (93, 94):
                    days, x, T, TM, mult, fxd = res[(off, False)]
                    TT, m = (TM, None) if key == "M08" else (T, mult[key])
                    p, mg, lots, tgt, d = yen(days, x, TT, m, dep, risk, fxd)
                    s = stats(p, dep)
                    over = np.mean(lots / tgt >= 1.5) * 100
                    cells.append(f"年{s['a']:+,.0f}円（{s['ap']:+.1f}%）落込{s['dd']:,.0f}円（{s['ddp']:.1f}%）安定度{s['st']:.2f} 負け年{s['neg']} "
                                 f"最悪の日{s['worst_day']:+,.0f}円 丸めで1.5倍以上{over:.0f}% 証拠金最大{mg.max():,.0f}円")
                print(f"  {dep:>9,}円: " + " ／ ".join(cells))
    # 直近1年の ETH の ATR（8時15分区切り・取引した日）
    days, x, T, TM, mult, fxd = res[(93, False)]
    t = T[T.q & (days[T.d.to_numpy()] >= MM.END - pd.Timedelta(days=365)) & (days[T.d.to_numpy()] <= MM.END)]
    A = x["atr"][t.d.to_numpy()]
    print(f"\n==== 直近1年で取引した日の ETH の ATR：中央値{np.median(A):.1f}ドル・下から1割{np.quantile(A, 0.1):.1f}・上から1割{np.quantile(A, 0.9):.1f}"
          f"（値段の中央値{np.median(x['dO'][t.d.to_numpy()]):,.0f}ドル）")


if __name__ == "__main__":
    main(sys.argv[1])
