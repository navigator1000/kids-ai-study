#!/usr/bin/env python3
"""1日の区切り（注文と決済の時刻）を、日本時間9時から動かした場合。ランキング上位3つ（R03・M08・R02）とS4案・S1。

毎朝、区切りの時刻に「前の日の建玉を決済 → その日の注文を出す」。値幅・ATR・5日平均・静かな日の判定も、
すべてその時刻で区切った24時間の足で計算する（日本時間8時20分なら、前日8時20分〜今日8時20分の高値・安値と、今の値段）。
15分足しかないので、8時20分は8時15分と8時30分ではさむ。7時〜9時を15分ずつ動かして、近くでも成績が保たれるかも見る。
約定は区切りの1時間前まで（M08 の値幅ブレイクは4時間前まで）。量の決め方・コスト（往復0.05%）・1R＝資金の0.5%は今まで通り。
スワップ（Vantage）：日本時間7時（夏時間は6時）をまたいだ買いに1日分、金曜に入った分は3日分、土日に入った分はなし（A）。
結果は results.csv（グループ「確認・注文と決済の時刻」）。
使い方: python3 project/tools/vantage_cycle_time.py <キャッシュ（BTC_15m.csv・ETH_15m.csv）>
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
import vantage_exit_time as VX  # noqa: E402
from registry import drop, record  # noqa: E402

COINS = VX.COINS
GROUP = "確認・注文と決済の時刻"


def shifted(x, off):
    """15分足を off 本目（UTC 0時から数える）から24時間ずつに区切り直す。off=93 なら日本時間8時15分区切り。"""
    n = (len(x["O"]) - off) // 96
    sl = slice(off, off + n * 96)
    M = {k: np.ascontiguousarray(x[c][sl]).reshape(n, 96) for k, c in (("open", "O"), ("high", "H"), ("low", "L"), ("close", "C"))}
    dO, dH, dL, dC = M["open"][:, 0], M["high"].max(1), M["low"].min(1), M["close"][:, -1]
    pc = np.r_[np.nan, dC[:-1]]
    tr = np.nanmax(np.vstack([dH - dL, np.abs(dH - pc), np.abs(dL - pc)]), axis=0)
    idx = pd.DatetimeIndex(x["idx"][0] + pd.Timedelta(minutes=15 * off) + pd.to_timedelta(np.arange(n), unit="D"))
    jst = (idx + pd.Timedelta(hours=9)).normalize()          # 区切りの日本時間の日付
    return dict(M=M, dO=dO, dH=dH, dL=dL, dC=dC, prevC=pc, atr=pd.Series(tr).rolling(14).mean().shift(1).to_numpy(),
                rng=np.r_[np.nan, (dH - dL)[:-1]], idx=idx, jst=jst, wd=jst.dayofweek.to_numpy())


def build(D, off, swap="A"):
    X = {a: shifted(D[a], off) for a in COINS}
    n = min(len(X[a]["dO"]) for a in COINS)
    for a in COINS:
        for k, v in list(X[a].items()):
            if k == "M":
                X[a]["M"] = {kk: vv[:n] for kk, vv in v.items()}
            elif hasattr(v, "__len__") and len(v) >= n:
                X[a][k] = v[:n]
    days = X["BTC"]["idx"]
    jst = X["BTC"]["jst"]
    # スワップの時刻：その日の日本時間の日付 J の UTC 22時（夏は21時）＝ 翌日の日本時間7時（夏は6時）。区切りからの本数
    hour = np.where(VX.summer(jst), 21, 22)
    roll = ((jst + pd.to_timedelta(hour, unit="h")) - days) / pd.Timedelta(minutes=15)
    roll = np.clip(np.asarray(roll, float), 0, 96).astype(int)
    off_a, xbar = np.zeros(n, int), np.full(n, 96)
    sched = VX.SCHED[swap]
    bx = X["BTC"]
    bdir = np.sign(bx["dO"] - R1.sma_prev(bx, 5))
    s1, m8 = [], []
    nan = np.full(n, np.nan)
    for a in COINS:
        x = X[a]
        q = R4.mult(x, "静かな日だけ") > 0
        hi, lo = x["dO"] + 0.8 * x["rng"], x["dO"] - 0.8 * x["rng"]
        ok = ~np.isnan(x["atr"]) & ~np.isnan(x["rng"])
        tl = VX.trades(x, a, hi, nan, ok & (np.nan_to_num(bdir) > 0), 92, 1.0, 0.5, 1.0, off_a, xbar, roll, sched, both=False)
        ts = VX.trades(x, a, nan, lo, ok & (np.nan_to_num(bdir) < 0), 92, 1.0, 0.5, 1.0, off_a, xbar, roll, sched, both=False)
        T = pd.concat([tl, ts], ignore_index=True)
        T["q"] = q[T.d.to_numpy()]
        s1.append(T)
        m8.append(VX.trades(x, a, hi, lo, q & ok, 80, 1.0, 0.5, 1.0, off_a, xbar, roll, sched))
        with np.errstate(invalid="ignore"):
            m = (x["dH"] < np.r_[np.nan, x["dH"][:-1]]) & (x["dL"] > np.r_[np.nan, x["dL"][:-1]])
        m8.append(VX.trades(x, a, np.r_[np.nan, x["dH"][:-1]], np.r_[np.nan, x["dL"][:-1]], np.r_[False, m[:-1]], 92, 1.0, 0.5, 0.5,
                            off_a, xbar, roll, sched))
        hh, ll = pd.Series(x["dH"]).rolling(5).max(), pd.Series(x["dL"]).rolling(5).min()
        with np.errstate(invalid="ignore"):
            m = ((hh < hh.shift(5)) & (ll > ll.shift(5))).to_numpy()
        m8.append(VX.trades(x, a, np.r_[np.nan, hh.to_numpy()[:-1]], np.r_[np.nan, ll.to_numpy()[:-1]], np.r_[False, m[:-1]], 92, 1.0, 0.5,
                            0.5, off_a, xbar, roll, sched))
    return days, pd.concat(s1, ignore_index=True), pd.concat(m8, ignore_index=True)


def evaluate(D, off, swap="A"):
    days, TS, TM = build(D, off, swap)
    mult = VX.sizing(days, TS)
    yrs = (MM.END - MM.START).days / 365.25
    out = {}
    for key, T, m in (("R03", TS, mult["R03"]), ("M08", TM, None), ("R02", TS, mult["R02"]), ("S4案", TS, mult["S4案"]), ("S1", TS, mult["S1"])):
        x, sw, n = VX.daily(days, T, m)
        r = R1.stats(x)
        r["年の取引回数"] = round(n / yrs, 1)
        out[key] = (r, sw.sum() / yrs)
    return out


def label(off):
    t = (pd.Timestamp("2000-01-01 09:00") + pd.Timedelta(minutes=15 * off))
    return f"{t.hour}時{t.minute:02d}分" if t.minute else f"{t.hour}時"


def main(cache):
    print(f"（古い「{GROUP}」の行を {drop(GROUP)} 行消して、計算し直す）")
    D = R1.prep(cache)
    # 確かめ：区切り0本（9時）は vantage_exit_time の「毎日9時」と同じ数字になるはず
    base = evaluate(D, 0)
    for key, (r, sw) in base.items():
        R1.line(f"9時（確かめ）{key}", r, f" ｜スワップ 年{sw:.2f}%")
    print("\n==== 区切りの時刻を7時〜9時で動かす（安定度：R03・M08・R02・S4案・S1）")
    grid = {}
    for off in range(88, 96):
        res = evaluate(D, off)
        grid[off] = res
        print(f"  日本時間{label(off):7s} " + "  ".join(f"{k} {v[0]['安定度']:.2f}（年{v[0]['年平均%']:+.1f}%・落込{v[0]['最大の落ち込み%']:.1f}%）" for k, v in res.items()))
    grid[96] = base
    print(f"  日本時間9時     " + "  ".join(f"{k} {v[0]['安定度']:.2f}（年{v[0]['年平均%']:+.1f}%・落込{v[0]['最大の落ち込み%']:.1f}%）" for k, v in base.items()))
    print("\n==== 8時15分・8時30分（8時20分をはさむ）と9時の詳しい数字")
    for off in (93, 94, 96):
        res = grid[off]
        for key, (r, sw) in res.items():
            R1.line(f"{label(off % 96 if off != 96 else 0) if off != 96 else '9時'} {key}", r, f" ｜スワップ 年{sw:.2f}%・取引 年{r['年の取引回数']:.0f}回")
            record(グループ=GROUP, 計画ID="", 名前=f"{key}：毎日{label(off) if off != 96 else '9時'}に決済して注文（その時刻で区切った日足）",
                   銘柄="BTC,ETH", 時間足="15分足", メモ=f"スワップは日本時間7時（夏6時）・金曜3日分・土日なし。スワップ 年{sw:.2f}%",
                   **{k: v for k, v in r.items() if not k.startswith("_")})
    print("\n==== 土日にもスワップがつく場合（B）：8時15分・8時30分")
    for off in (93, 94):
        res = evaluate(D, off, "B")
        print(f"  {label(off)} " + "  ".join(f"{k} {v[0]['安定度']:.2f}（年{v[0]['年平均%']:+.1f}%）" for k, v in res.items()))
    print("\n==== 年ごとの損益（%）：9時と8時15分・8時30分（R03）")
    yt = {}
    for off in (96, 93, 94):
        days, TS, TM = build(D, off % 96)
        mult = VX.sizing(days, TS)
        x, _, _ = VX.daily(days, TS, mult["R03"])
        yt["9時" if off == 96 else label(off)] = x.groupby(x.index.year).sum().round(1)
    print(pd.DataFrame(yt).T.to_string())


if __name__ == "__main__":
    main(sys.argv[1])
