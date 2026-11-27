#!/usr/bin/env python3
"""ランキング上位3つ（R03・M08・R02）とS4案で、決済の時刻を変えた場合。Vantage のスワップの時刻をつけて計算する。

Vantage のスワップ（2026-11-27 に確認）: サーバーの時刻は GMT+2（米国の夏時間の間は GMT+3）。サーバーの0時、
  つまり UTC 22時（夏は21時）＝日本時間7時（夏は6時）をまたいで持っている買いにつく。金曜の分は3日分。
  土日の分は、つかない（A）とする。土日にも1日分ずつつく（B）場合も見る。売りは0。
決済の時刻（注文は今まで通り日本時間9時。約定は決済の時刻より前だけ）:
  ① 毎日9時（今の手順）
  ② 毎日6時半
  ③ 月〜金の朝は6時半、土日は12時半（金曜・土曜に入った取引は、翌日の12時半まで持つ）
  ④ 土日だけ12時半（月〜金の朝は9時）
量の決め方（S4案・R03・R02）は、決済が終わって勝ち負けがわかっている取引だけで判定する
  （③④では、土曜・日曜の朝9時には、前日の取引がまだ決済前なので数えない）。
1R＝ATR1本分＝資金の0.5%（単利）、往復0.05%。結果は results.csv（グループ「確認・決済の時刻」）。
使い方: python3 project/tools/vantage_exit_time.py <キャッシュ（BTC_15m.csv・ETH_15m.csv）>
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import more_methods_lab as MM  # noqa: E402
import portfolio_lab as PF  # noqa: E402
import round1_lab as R1  # noqa: E402
import round4_lab as R4  # noqa: E402
from registry import drop, metrics_row, record  # noqa: E402

COINS = ("BTC", "ETH")
GROUP = "確認・決済の時刻"
COST = 0.0005
SWD = {a: PF.SPEC[a][1] / 36500 for a in COINS}
SCHED = {"A": np.array([1, 1, 1, 1, 3, 0, 0]), "B": np.array([1, 1, 1, 1, 3, 1, 1]), "検証": np.ones(7, int)}
# 決済：(何日後, 何本目の始まり)。値段はその1本前の足の終値。96本目＝翌朝9時、86本目＝6時半、翌日の14本目＝翌日の12時半
X9, X630, X1230 = (0, 96), (0, 86), (1, 14)
PLANS = {
    "毎日9時": lambda dow: X9,
    "毎日6時半": lambda dow: X630,
    "月〜金の朝は6時半・土日は12時半": lambda dow: X1230 if dow in (4, 5) else X630,
    "土日だけ12時半（月〜金の朝は9時）": lambda dow: X1230 if dow in (4, 5) else X9,
}


def summer(days):
    """米国の夏時間（3月の第2日曜〜11月の第1日曜）の日。この間、Vantage の0時は UTC 21時になる。"""
    out = np.zeros(len(days), bool)
    for y in range(days[0].year, days[-1].year + 1):
        mar, nov = pd.Timestamp(y, 3, 1), pd.Timestamp(y, 11, 1)
        start = mar + pd.Timedelta(days=(6 - mar.dayofweek) % 7 + 7)
        end = nov + pd.Timedelta(days=(6 - nov.dayofweek) % 7)
        out |= np.asarray((days >= start) & (days < end))
    return out


def exits(days, plan):
    dow = days.dayofweek.to_numpy()
    xs = np.array([PLANS[plan](w) for w in dow])
    return xs[:, 0], xs[:, 1]


def trades(x, a, Lu, Ld, ok, last, wl, ws, cw, off, xbar, roll, sched, both=True):
    """逆指値（both=True なら上下で先に約定した方だけ、False なら Lu・Ld の片方だけ置いた日）。決済は (off, xbar)。
    スワップは、その日の0時（roll 本目）より前に入り、決済がそれより後なら、曜日の日数分。"""
    H, L, O, C = x["M"]["high"], x["M"]["low"], x["M"]["open"], x["M"]["close"]
    nd = len(x["dO"])
    lim = np.minimum(last, np.where(off == 0, xbar, 96))
    allow = np.arange(96)[None, :] < lim[:, None]
    with np.errstate(invalid="ignore"):
        up = (H >= Lu[:, None]) & allow
        dn = (L <= Ld[:, None]) & allow
    big = 10 ** 6
    ju = np.where(up.any(1), up.argmax(1), big)
    jd = np.where(dn.any(1), dn.argmax(1), big)
    ok = ok & (x["atr"] > 0) & (np.arange(nd) + off < nd)
    if both:
        ok &= ~np.isnan(Lu) & ~np.isnan(Ld) & (ju != jd)
    d = np.flatnonzero(ok & (np.minimum(ju, jd) < big))
    sd = np.where(ju[d] < jd[d], 1, -1)
    j = np.minimum(ju[d], jd[d])
    o = O[d, j]
    e = np.where(sd == 1, np.maximum(o, Lu[d]), np.minimum(o, Ld[d]))
    xp = C[d + off[d], xbar[d] - 1]
    A = x["atr"][d]
    held = (j < roll[d]) & ((off[d] > 0) | (xbar[d] > roll[d]))
    sw = np.where((sd == 1) & held, sched[x["idx"][d].dayofweek.to_numpy()] * SWD[a] * e / A, 0.0)
    R = sd * (xp - e) / A - COST * e / A - sw
    return pd.DataFrame(dict(coin=a, d=d, side=sd, w=np.where(sd == 1, wl, ws), cw=cw, R=R, sw=sw, wk=off[d] > 0))


def build(D, plan, swap):
    days = D["BTC"]["idx"]
    off, xbar = exits(days, plan)
    if swap == "検証":
        roll = np.full(len(days), 84)
    else:
        roll = np.where(summer(days), 84, 88)
    sched = SCHED[swap]
    bdir = np.sign(D["BTC"]["dO"] - R1.sma_prev(D["BTC"], 5))
    s1, m8 = [], []
    for a in COINS:
        x = D[a]
        q = R4.mult(x, "静かな日だけ") > 0
        hi, lo = x["dO"] + 0.8 * x["rng"], x["dO"] - 0.8 * x["rng"]
        base_ok = ~np.isnan(x["atr"]) & ~np.isnan(x["rng"])
        # S1 の向き（BTCの5日平均）。買い×1・売り×0.5、約定は8時（92本目）まで。静かかどうかは後で量として掛ける
        nan = np.full(len(days), np.nan)
        tl = trades(x, a, hi, nan, base_ok & (np.nan_to_num(bdir) > 0), 92, 1.0, 0.5, 1.0, off, xbar, roll, sched, both=False)
        ts = trades(x, a, nan, lo, base_ok & (np.nan_to_num(bdir) < 0), 92, 1.0, 0.5, 1.0, off, xbar, roll, sched, both=False)
        T = pd.concat([tl, ts], ignore_index=True)
        T["q"] = q[T.d.to_numpy()]
        s1.append(T)
        # M08：向きなし（静かな日だけ・5時まで）＋F02×0.5＋F08×0.5
        m8.append(trades(x, a, hi, lo, q & base_ok, 80, 1.0, 0.5, 1.0, off, xbar, roll, sched))
        with np.errstate(invalid="ignore"):
            m = (x["dH"] < np.r_[np.nan, x["dH"][:-1]]) & (x["dL"] > np.r_[np.nan, x["dL"][:-1]])
        m8.append(trades(x, a, np.r_[np.nan, x["dH"][:-1]], np.r_[np.nan, x["dL"][:-1]], np.r_[False, m[:-1]], 92, 1.0, 0.5, 0.5,
                         off, xbar, roll, sched))
        hh, ll = pd.Series(x["dH"]).rolling(5).max(), pd.Series(x["dL"]).rolling(5).min()
        with np.errstate(invalid="ignore"):
            m = ((hh < hh.shift(5)) & (ll > ll.shift(5))).to_numpy()
        m8.append(trades(x, a, np.r_[np.nan, hh.to_numpy()[:-1]], np.r_[np.nan, ll.to_numpy()[:-1]], np.r_[False, m[:-1]], 92, 1.0, 0.5,
                         0.5, off, xbar, roll, sched))
    return days, pd.concat(s1, ignore_index=True), pd.concat(m8, ignore_index=True)


def known_window(n, wk, look):
    """前の look 日の合計。ただし前日の取引が翌日の昼まで持つ形（wk）なら、朝9時にはまだ決済前なので数えない。"""
    c = np.r_[0.0, np.cumsum(n)]
    d = np.arange(len(n))
    tot = c[d] - c[np.maximum(0, d - look)]
    prev = np.r_[0.0, n[:-1]] * np.r_[False, wk[:-1]]
    return tot - prev


def sizing(days, T):
    """S1・S4案・R03・R02（勝った翌日0.5倍）の量の倍率（銘柄ごと・日ごと）。判定は実際に取った（静かな日の）取引で。"""
    nd = len(days)
    wkday = np.isin(days.dayofweek.to_numpy(), [])  # あとで上書き
    st = {}
    for a in COINS:
        t = T[(T.coin == a) & T.q & (days[T.d.to_numpy()] >= MM.START)]
        n = np.bincount(t.d, minlength=nd).astype(float)
        w = np.bincount(t.d, weights=(t.R > 0).astype(float), minlength=nd)
        st[a] = (n, w)
    wk = np.zeros(nd, bool)
    wk[T.d[T.wk].unique()] = True
    out = {k: {} for k in ("S1", "S4案", "R03", "R02")}
    pn = sum(st[a][0] for a in COINS)
    pw = sum(st[a][1] for a in COINS)
    pooled_lost = (known_window(pn, wk, 2) >= 1) & (known_window(pw, wk, 2) == 0)
    for a in COINS:
        n, w = st[a]
        lost = (known_window(n, wk, 2) >= 1) & (known_window(w, wk, 2) == 0)
        n1, w1 = known_window(n, wk, 1), known_window(w, wk, 1)
        won = (n1 >= 1) & (w1 == n1)
        out["S1"][a] = np.ones(nd)
        out["S4案"][a] = np.where(lost, 2.0, 1.0)
        out["R03"][a] = np.where(pooled_lost, 2.0, 1.0)
        out["R02"][a] = np.where(lost, 2.0, np.where(won, 0.5, 1.0))
    del wkday
    return out


def daily(days, T, mult=None):
    m = np.ones(len(T)) if mult is None else np.array([mult[c][d] for c, d in zip(T.coin, T.d)])
    k = T.q.to_numpy() if "q" in T else np.ones(len(T), bool)
    p = 0.5 * T.R.to_numpy() * T.w.to_numpy() * T.cw.to_numpy() * m * k
    s = 0.5 * T.sw.to_numpy() * T.w.to_numpy() * T.cw.to_numpy() * m * k
    nd = len(days)
    pnl = pd.Series(np.bincount(T.d, weights=p, minlength=nd), index=days)
    swp = pd.Series(np.bincount(T.d, weights=s, minlength=nd), index=days)
    n = ((m * k) > 0) & (days[T.d.to_numpy()] >= MM.START) & (days[T.d.to_numpy()] <= MM.END)
    return MM.clip(pnl), MM.clip(swp), int(n.sum())


def main(cache):
    print(f"（古い「{GROUP}」の行を {drop(GROUP)} 行消して、計算し直す）")
    D = R1.prep(cache)
    yrs = (MM.END - MM.START).days / 365.25
    res = {}
    for swap in ("検証", "A", "B"):
        plans = ("毎日9時",) if swap == "検証" else tuple(PLANS)
        for plan in plans:
            days, TS, TM = build(D, plan, swap)
            mult = sizing(days, TS)
            print(f"\n==== 決済：{plan}／スワップ：{ {'検証': '検証と同じ（毎晩6時に1日分）', 'A': '7時（夏6時）・金曜3日分・土日なし', 'B': '7時（夏6時）・金曜3日分・土日も1日分'}[swap] }")
            for key, T, m in (("R03", TS, mult["R03"]), ("M08", TM, None), ("R02", TS, mult["R02"]), ("S4案", TS, mult["S4案"]),
                              ("S1", TS, mult["S1"])):
                x, sw, n = daily(days, T, m)
                r = R1.stats(x)
                r["年の取引回数"] = round(n / yrs, 1)
                R1.line(f"{key}", r, f" ｜スワップ 年{sw.sum() / yrs:.2f}%・取引 年{n / yrs:.0f}回")
                res[(swap, plan, key)] = (r, sw.sum() / yrs)
                if swap != "検証":
                    record(グループ=GROUP, 計画ID="", 名前=f"{key}：決済は{plan}（スワップ{swap}）", 銘柄="BTC,ETH", 時間足="15分足",
                           メモ=f"スワップは日本時間7時（夏6時）・金曜3日分・土日{'なし' if swap == 'A' else '1日分'}。スワップ 年{sw.sum() / yrs:.2f}%",
                           **{k: v for k, v in r.items() if not k.startswith("_")})
    return res


if __name__ == "__main__":
    main(sys.argv[1])
