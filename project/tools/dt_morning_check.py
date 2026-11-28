#!/usr/bin/env python3
"""デイトレ第2回の TA05（前日の高値・安値のブレイク）を、朝8時20分の操作だけで回す形で確かめる（計画書 TP06 の一部）。

毎朝8時20分に、前の日の建玉を決済して、直近24時間（前日8時20分〜今日8時20分）の高値に買いの逆指値、安値に売りの逆指値を置く
（先に約定した方。損切りは24時間の中値、利確は ATR×2 か、なし）。決済は翌朝8時20分。スワップの時刻（7時、夏は6時）をまたぐ。
8時20分は、8時15分と8時30分の区切りではさむ。15分足と5分足の両方で計算する（5分足が本当の値に近い）。
1R＝その区切りで計算した日足のATR1本分＝資金の0.5%、往復0.05%。結果は results.csv（グループ「計画・デイトレ第2回・朝の形」、計画ID TP06）。
使い方: python3 project/tools/dt_morning_check.py <キャッシュ（BTC_15m_vol.csv・ETH_15m_vol.csv・BTC_5m.csv・ETH_5m.csv・BTC_15m.csv・ETH_15m.csv）>
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dt1_lab as DT  # noqa: E402
import dt2_lab as D2  # noqa: E402
import round1_lab as R1  # noqa: E402
import round7_lab as R7  # noqa: E402
import vantage_cycle_time as VC  # noqa: E402
import vantage_exit_time as VX  # noqa: E402

DT.GROUP = "計画・デイトレ第2回・朝の形"


def shifted15(x, off):
    """15分足を off 本目から24時間ずつに区切り直し、dt1_lab の形（スワップの時刻・曜日つき）にする。"""
    y = VC.shifted(x, off)
    nd = len(y["dO"])
    y["nd"] = nd
    jst = pd.DatetimeIndex(y["jst"])
    hour = np.where(VX.summer(jst), 21, 22)
    roll = ((jst + pd.to_timedelta(hour, unit="h")) - y["idx"]) / pd.Timedelta(minutes=15)
    y["roll"] = np.clip(np.asarray(roll, float), 0, 96).astype(int)
    y["bd"] = np.repeat(np.arange(nd), 96)
    y["bod"] = np.tile(np.arange(96), nd)
    for k, c in (("O", "open"), ("H", "high"), ("L", "low"), ("C", "close")):
        y[k] = np.ascontiguousarray(y["M"][c].ravel())
    return y


def main(cache):
    X = DT.load(cache)
    idx = X["BTC"]["idx"]
    X5 = D2.C5.load5(cache, idx)
    D = R1.prep(cache)
    parts = R7.s1_parts(D, DT.COINS, D["BTC"])
    per, _ = R7.build(parts, R7.size_rules(parts, DT.COINS)["S4案"])
    s4 = R1.total(per)
    print("==== TP06（一部）TA05 を朝8時20分の操作だけで：直近24時間の高値・安値に逆指値、損切りは中値、翌朝8時20分に決済")
    for off, lab in ((93, "8時15分"), (94, "8時30分")):
        Y = {a: shifted15(X[a], off) for a in DT.COINS}
        nd = min(len(Y[a]["dO"]) for a in DT.COINS)
        ctx = dict(X=Y, idx=Y["BTC"]["idx"][:nd], s4=s4, s4r=R1.stats(s4))
        for use5 in (False, True):
            for tp in (None, 2.0):
                out = {}
                for a in DT.COINS:
                    y = Y[a]
                    ph, pl = np.r_[np.nan, y["dH"][:-1]], np.r_[np.nan, y["dL"][:-1]]
                    mid = (ph + pl) / 2
                    tpu = ph + tp * y["atr"] if tp else np.full(len(ph), np.nan)
                    tpd = pl - tp * y["atr"] if tp else np.full(len(ph), np.nan)
                    ok = np.isfinite(y["atr"]) & np.isfinite(ph)
                    if use5:
                        o5 = off * 3
                        n5 = (len(X5[a]["O"]) - o5) // 288
                        m = min(n5, len(ph))
                        a5 = {k: np.ascontiguousarray(X5[a][k][o5:o5 + m * 288]) for k in ("O", "H", "L", "C")}
                        d, s, e, xp, eb, xb = D2.brk_p(a5["O"], a5["H"], a5["L"], a5["C"], 288, np.zeros(m, np.int64), np.full(m, 287, np.int64),
                                                       np.full(m, 288, np.int64), ph[:m], pl[:m], mid[:m], mid[:m], tpu[:m], tpd[:m],
                                                       np.full(m, np.nan), np.zeros(m, np.int64), ok[:m])
                        eb, xb = eb // 3, xb // 3
                    else:
                        d, s, e, xp, eb, xb = D2.brk_p(y["O"], y["H"], y["L"], y["C"], 96, np.zeros(len(ph), np.int64),
                                                       np.full(len(ph), 95, np.int64), np.full(len(ph), 96, np.int64), ph, pl, mid, mid, tpu, tpd,
                                                       np.full(len(ph), np.nan), np.zeros(len(ph), np.int64), ok)
                    out[a] = DT.res_trades(y, a, d, s, e, xp, np.ones(len(d)), eb, xb)
                r = DT.show(ctx, "TP06", f"TA05を{lab}の操作だけで・{'5分足' if use5 else '15分足'}・利確{'ATR×2' if tp else 'なし'}", out,
                            memo="朝の注文だけで回す形。スワップの時刻をまたぐ")
                x = r["_x"]
                sm, xm = s4.resample("ME").sum(), x.resample("ME").sum()
                corr = sm.corr(xm)
                sc = 0.5 * sm.std() / xm.std()
                mix = R1.stats(s4.add(sc * x, fill_value=0))
                yt = x.groupby(x.index.year).sum().round(1)
                print(f"      S4案と月の相関{corr:+.2f}・S4案に足すと{mix['安定度']:.2f}（前半{mix['_h1s']:.2f}・後半{mix['_h2s']:.2f}）｜年ごと "
                      + " ".join(f"{y_}:{v:+.1f}" for y_, v in yt.items()))


if __name__ == "__main__":
    main(sys.argv[1])
