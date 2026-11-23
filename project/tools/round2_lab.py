#!/usr/bin/env python3
"""計画書（project/research/plan.md）第2回・前半：値幅ブレイクの派生（A）と値動き（E）の優先「中」24件（BTC・ETH、2018年〜）。

  A04 値幅を実体で測る／A05 値幅を数日の平均で測る／A09 長い流れで向きを決める／A10 区切りの時刻を24通り／A12 12時半区切り／
  A13 2日・3日持つ／A14 直近に勝った時だけ／A17 はらみ足の翌日だけ／A18 約定の時間帯で選ぶ／A19 同時約定の量を抑える／
  A23 2段目の買い増し／A24 だましの逆張り（始値に戻ったら撤退・反対に入る）／A25 ウップス／A26 週の値幅ブレイク／
  A28 4時間の値幅ブレイク／A30 前日の高値・安値で逆指値
  E02 TTMスクイーズ／E08 静かな日が続いた長さ／E09 ボラの目標／E10 ATRが上がった日だけ／E11 値幅の拡大の翌日に乗る／
  E12 静かな日の値幅ブレイクを持ち越す／E14 短い期間と長い期間のボラの比／E15 終値の位置
共通: Vantage CFD（往復0.05%、買いのスワップはUTC21時をまたぐごと）、1R＝ATR1本分＝資金の0.5%（単利）、含み損込みの毎日の損益。
      基準は A01（買い×1＋売り×0.5、5日平均の向き、k=0.8、翌朝9時に決済）。「前半で選び、後半で確かめる」。
結果は results.csv（グループ「計画・第2回」、計画IDつき）と plan.csv。
使い方: python3 project/tools/round2_lab.py <キャッシュ（BTC_15m.csv, ETH_15m.csv）> [followup]
        followup: 区切り24通りの均等分散と、「静か」の判定方法を前半で選んで後半で確かめる
"""
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
import more_methods_lab as MM  # noqa: E402
import plan_tool as PT  # noqa: E402
import portfolio_lab as PF  # noqa: E402
import round1_lab as R1  # noqa: E402
from registry import drop, metrics_row, record  # noqa: E402

COINS = ("BTC", "ETH")
SPLIT = PF.SPLIT
GROUP = "計画・第2回"
DAY = 96
OUT = {}                                 # 計画ID → 判定に使う数字（結果の文章は最後にまとめて書く）


# ---------------------------------------------------------------- エンジン
@njit
def eng(O, H, L, C, P, last, trig, ok, A, start, side, cost, swd, roll_flag, hold, trig2):
    """期間ごと（長さP本）に trig に逆指値。hold[d] 期間持って決済。trig2 に届いたら同じ量を足す（NaNなら足さない）。
    roll_flag[j]：その足の始まりにスワップがかかるか。返り値：毎期間の損益（資金の%）、取引ごとの期間・約定した足・R・足した分のR。"""
    nd = trig.shape[0]
    N = nd * P
    pnl = np.zeros(nd)
    te = np.empty(nd, np.int64)
    tp = np.empty(nd, np.int64)
    tr = np.empty(nd)
    t2 = np.zeros(nd)
    nt = 0
    d = 0
    while d < nd:
        if not ok[d] or not (A[d] > 0) or np.isnan(trig[d]):
            d += 1
            continue
        b0 = d * P
        s0 = b0 + start[d]
        k = -1
        e = 0.0
        for j in range(s0, b0 + last):
            if j == s0 and side * (O[j] - trig[d]) > 0:
                break
            if (side == 1 and H[j] >= trig[d]) or (side == -1 and L[j] <= trig[d]):
                k = j
                e = max(O[j], trig[d]) if side == 1 else min(O[j], trig[d])
                break
        if k < 0:
            d += 1
            continue
        Ad = A[d]
        cR = cost * e / Ad
        sR = swd * e / Ad
        pnl[d] -= cR * 0.5
        has2 = not np.isnan(trig2[d])
        in2 = False
        e2 = 0.0
        c2 = 0.0
        s2 = 0.0
        mark = 0.0
        mark2 = 0.0
        nroll = 0
        nroll2 = 0
        held = 0
        x = 0.0
        out = -1
        j = k
        while j < N:
            dd = j // P
            pos = j - dd * P
            if j > k and roll_flag[j]:
                nroll += 1
                pnl[dd] -= sR * 0.5
                if in2:
                    nroll2 += 1
                    pnl[dd] -= s2 * 0.5
            if has2 and not in2 and dd == d and pos < last:
                if (side == 1 and H[j] >= trig2[d]) or (side == -1 and L[j] <= trig2[d]):
                    in2 = True
                    e2 = max(O[j], trig2[d]) if side == 1 else min(O[j], trig2[d])
                    c2 = cost * e2 / Ad
                    s2 = swd * e2 / Ad
                    pnl[dd] -= c2 * 0.5
            if pos == P - 1:
                held += 1
                m = side * (C[j] - e) / Ad
                m2 = side * (C[j] - e2) / Ad if in2 else 0.0
                if held < hold[d]:
                    pnl[dd] += (m - mark) * 0.5 + (m2 - mark2) * 0.5
                    mark = m
                    mark2 = m2
                else:
                    x = C[j]
                    out = dd
                    break
            j += 1
        if out < 0:
            break
        rx = side * (x - e) / Ad
        pnl[out] += (rx - mark) * 0.5
        r2 = 0.0
        if in2:
            rx2 = side * (x - e2) / Ad
            pnl[out] += (rx2 - mark2) * 0.5
            r2 = rx2 - c2 - nroll2 * s2
        te[nt] = d
        tp[nt] = k - b0
        tr[nt] = rx - cR - nroll * sR
        t2[nt] = r2
        nt += 1
        d = out + 1
    return pnl, te[:nt], tp[:nt], tr[:nt], t2[:nt]


@njit
def eng_rev(O, H, L, C, P, last, trig, ok, A, dO, side, cost, swd, swd_r, roll_flag, reverse):
    """値幅ブレイクで入り、その日のうちに始値まで戻ったら撤退。reverse=1 なら、そこで反対向きに入り直して日の終わりに決済。
    約定した15分足の中は、終値が始値を越えて戻っていた時だけ撤退とみなす（安値が約定の前かもしれないため）。返り値は1期間＝1日の損益（R）。"""
    nd = trig.shape[0]
    R = np.zeros(nd)
    for d in range(nd):
        if not ok[d] or not (A[d] > 0) or np.isnan(trig[d]):
            continue
        b0 = d * P
        k = -1
        e = 0.0
        for j in range(b0, b0 + last):
            if (side == 1 and H[j] >= trig[d]) or (side == -1 and L[j] <= trig[d]):
                k = j
                e = max(O[j], trig[d]) if side == 1 else min(O[j], trig[d])
                break
        if k < 0:
            continue
        Ad = A[d]
        stop = dO[d]
        x = C[b0 + P - 1]
        hit = -1
        xs = 0.0
        for j in range(k, b0 + P):
            if j == k:
                if side * (C[j] - stop) <= 0:
                    hit, xs = j, stop
                    break
            elif (side == 1 and L[j] <= stop) or (side == -1 and H[j] >= stop):
                hit = j
                xs = min(O[j], stop) if side == 1 else max(O[j], stop)
                break
        roll = (k - b0) < 84
        if hit < 0:
            R[d] = side * (x - e) / Ad - cost * e / Ad - (swd * e / Ad if roll else 0.0)
        else:
            r = side * (xs - e) / Ad - cost * e / Ad - (swd * e / Ad if (roll and (hit - b0) >= 84) else 0.0)
            if reverse == 1:
                r += -side * (x - xs) / Ad - cost * xs / Ad - (swd_r * xs / Ad if (hit - b0) < 84 else 0.0)
            R[d] = r
    return R


@njit
def eng_oops(O, H, L, C, P, last, lvl, ok, A, side, cost, swd):
    """買い：始まった後に前日安値（lvl）を一度割り、前日安値まで戻ったら買い（売りは前日高値で逆）。日の終わりに決済。1日の損益（R）。"""
    nd = lvl.shape[0]
    R = np.zeros(nd)
    for d in range(nd):
        if not ok[d] or not (A[d] > 0) or np.isnan(lvl[d]):
            continue
        b0 = d * P
        below = False
        k = -1
        e = 0.0
        for j in range(b0, b0 + last):
            if not below:
                if (side == 1 and L[j] < lvl[d]) or (side == -1 and H[j] > lvl[d]):
                    below = True
                    if side * (C[j] - lvl[d]) > 0:        # 同じ足の中で戻って引けた
                        k, e = j, lvl[d]
                        break
            elif (side == 1 and H[j] >= lvl[d]) or (side == -1 and L[j] <= lvl[d]):
                k = j
                e = max(O[j], lvl[d]) if side == 1 else min(O[j], lvl[d])
                break
        if k < 0:
            continue
        Ad = A[d]
        x = C[b0 + P - 1]
        R[d] = side * (x - e) / Ad - cost * e / Ad - (swd * e / Ad if (k - b0) < 84 else 0.0)
    return R


# ---------------------------------------------------------------- 期間の作り方
def periods(x, P, off):
    """15分足を、off 本目から P 本ずつの期間に分ける（P=96で1日、672で1週間、16で4時間）。"""
    O, H, L, C = x["O"], x["H"], x["L"], x["C"]
    n = (len(O) - off) // P
    sl = slice(off, off + n * P)
    fl = {k: np.ascontiguousarray(v[sl]) for k, v in (("O", O), ("H", H), ("L", L), ("C", C))}
    M = {k: v.reshape(n, P) for k, v in fl.items()}
    pO, pH, pL, pC = M["O"][:, 0], M["H"].max(1), M["L"].min(1), M["C"][:, -1]
    pc = np.r_[np.nan, pC[:-1]]
    tr = np.nanmax(np.vstack([pH - pL, np.abs(pH - pc), np.abs(pL - pc)]), axis=0)
    t0 = x["idx"][0] + pd.Timedelta(minutes=15 * off)
    return dict(fl=fl, M=M, P=P, n=n, pO=pO, pH=pH, pL=pL, pC=pC,
                atr=pd.Series(tr).rolling(14).mean().shift(1).to_numpy(), rng=np.r_[np.nan, (pH - pL)[:-1]],
                sma5=pd.Series(pC).rolling(5).mean().shift(1).to_numpy(),
                roll=((np.arange(n * P) + off) % DAY == 84),
                idx=pd.DatetimeIndex(t0 + pd.to_timedelta(np.arange(n) * P * 15, unit="min")))


def run(Pd, a, k=0.8, ma=True, trig_fn=None, extra_ok=None, hold=1, trig2_m=None, short_w=0.5, last=None):
    """期間 Pd で値幅ブレイク（買い×1＋売り×short_w）。期間ごとの損益（資金の%）と取引一覧。"""
    P = Pd["P"]
    last = P - 4 if last is None else last
    tot = np.zeros(Pd["n"])
    ts = []
    for side, w in ((1, 1.0), (-1, short_w)):
        if w == 0:
            continue
        trig = trig_fn(Pd, side) if trig_fn else Pd["pO"] + side * k * Pd["rng"]
        with np.errstate(invalid="ignore"):
            ok = ~np.isnan(Pd["atr"]) & ~np.isnan(trig)
            if ma:
                ok &= side * (Pd["pO"] - Pd["sma5"]) > 0
        if extra_ok is not None:
            ok &= extra_ok(Pd, side)
        hd = np.full(Pd["n"], hold, np.int64) if np.isscalar(hold) else hold.astype(np.int64)
        t2 = np.full(Pd["n"], np.nan) if trig2_m is None else trig + side * trig2_m * Pd["rng"]
        swd = (PF.SPEC[a][1] if side == 1 else PF.SPEC[a][2]) / 36500
        fl = Pd["fl"]
        pnl, te, tp, tr, r2 = eng(fl["O"], fl["H"], fl["L"], fl["C"], P, last, trig.astype(np.float64), ok.astype(np.bool_),
                                  Pd["atr"], np.zeros(Pd["n"], np.int64), side, PF.SPEC[a][0], swd, Pd["roll"], hd, t2)
        tot += w * pnl
        ts.append(pd.DataFrame({"coin": a, "day": Pd["idx"][te], "pos": tp, "side": side, "w": w, "R": tr, "R2": r2}))
    T = pd.concat(ts, ignore_index=True)
    return pd.Series(tot, index=Pd["idx"]), T[(T.day >= MM.START) & (T.day <= MM.END)]


def daily(s):
    """期間の損益を日ごとにまとめる（4時間・週の期間用）。"""
    return s.groupby(s.index.normalize()).sum()


def combo(D, P=DAY, off=0, **kw):
    per, Ts = {}, []
    for a in COINS:
        Pd = periods(D[a], P, off)
        s, T = run(Pd, a, **kw)
        per[a] = daily(s) if P != DAY else s
        Ts.append(T)
    return per, pd.concat(Ts, ignore_index=True)


def show(pid, name, per, T=None, memo="", tf="15分足"):
    x = R1.total(per)
    r = R1.stats(x, T)
    cb, ce = (metrics_row(MM.clip(per[a])) for a in COINS)
    r["_coins"] = (cb, ce)
    R1.line(name, r, f" ｜BTC{cb['年平均%']:+.1f}% ETH{ce['年平均%']:+.1f}%")
    record(グループ=GROUP, 計画ID=pid, 名前=f"{pid} {name}", 銘柄="BTC,ETH", 時間足=tf,
           メモ=(memo + "。" if memo else "") + f"BTCだけ年{cb['年平均%']:+.1f}%・ETHだけ年{ce['年平均%']:+.1f}%",
           **{k: v for k, v in r.items() if not k.startswith("_")})
    return r


def from_trades(T, keep, scale=None):
    """取引一覧から毎日の損益を作り直す（当日決済の取引だけ使える）。keep：残す取引、scale：量の倍率。"""
    U = T[keep].copy()
    U["p"] = U.w * 0.5 * U.R * (1.0 if scale is None else scale[keep])
    per = {}
    for a in COINS:
        s = U[U.coin == a].groupby("day").p.sum()
        per[a] = s
    return per, U


def verdict(r, base, pid, grid=None):
    s1, s2, pf = R1.judge(r, list(r.get("_coins", ())), grid)
    return f"{R1.mark(s1, s2)}・{R1.compare(r, base)}"


# ---------------------------------------------------------------- 本体
def main(cache):
    print(f"（古い「{GROUP}」の行を {drop(GROUP)} 行消して、計算し直す）")
    D = R1.prep(cache)
    per0, T0 = combo(D)
    base = show("A01", "A01（基準・確認）", per0, T0)
    Pdays = {a: periods(D[a], DAY, 0) for a in COINS}

    def best_by_h1(res):
        return max(res, key=lambda k: res[k]["_h1s"])

    # ---- A04・A05・A30 値幅の測り方
    print("\n==== A04 値幅を前日の実体で測る")
    res = {}
    for k in (0.5, 0.75, 1.0, 1.5, 2.0):
        per, T = combo(D, trig_fn=lambda Pd, s, k=k: Pd["pO"] + s * k * np.abs(np.r_[np.nan, (Pd["pC"] - Pd["pO"])[:-1]]))
        res[k] = show("A04", f"実体×{k}", per, T)
    OUT["A04"] = (res, best_by_h1(res))
    print("\n==== A05 値幅を直近N日の平均で測る")
    res = {}
    for n in (3, 5, 10):
        for k in (0.6, 0.8, 1.0):
            def tf(Pd, s, n=n, k=k):
                avg = pd.Series(Pd["pH"] - Pd["pL"]).rolling(n).mean().shift(1).to_numpy()
                return Pd["pO"] + s * k * avg
            per, T = combo(D, trig_fn=tf)
            res[(n, k)] = show("A05", f"{n}日の平均×{k}", per, T)
    OUT["A05"] = (res, best_by_h1(res))
    print("\n==== A30 前日の高値・安値で逆指値（5日平均の向き）")
    per, T = combo(D, trig_fn=lambda Pd, s: np.r_[np.nan, (Pd["pH"] if s == 1 else Pd["pL"])[:-1]])
    OUT["A30"] = show("A30", "前日の高値（買い）・安値（売り）", per, T)

    # ---- A09 長い流れで向き
    print("\n==== A09 長い流れ（N日平均）で向きを決める")
    res = {}
    for n in (50, 100, 200):
        def lo(Pd, s, n=n):
            m = pd.Series(Pd["pC"]).rolling(n).mean().shift(1).to_numpy()
            with np.errstate(invalid="ignore"):
                return s * (Pd["pO"] - m) > 0
        per, T = combo(D, ma=False, extra_ok=lo)
        res[(n, "だけ")] = show("A09", f"{n}日平均の向きだけ", per, T)
        per, T = combo(D, ma=True, extra_ok=lo)
        res[(n, "かつ")] = show("A09", f"5日平均と{n}日平均の向きがそろう時だけ", per, T)
    OUT["A09"] = (res, best_by_h1(res))

    # ---- A10・A12 区切りの時刻
    print("\n==== A10 1日の区切りの時刻（日本時間）")
    res = {}
    for jst in range(24):
        off = ((jst - 9) % 24) * 4
        per, T = combo(D, off=off)
        res[jst] = show("A10", f"{jst}時区切り", per, T, tf="15分足")
    OUT["A10"] = res
    print("\n==== A12 12時半区切り")
    per, T = combo(D, off=(3 * 4 + 2))
    OUT["A12"] = show("A12", "12時半区切り", per, T)

    # ---- A13・E12 持つ日数
    print("\n==== A13 約定したらN日持つ")
    res = {}
    for h in (2, 3):
        per, T = combo(D, hold=h)
        res[h] = show("A13", f"{h}日持つ", per, T)
    OUT["A13"] = res
    print("\n==== E12 静かな日に約定した分だけ2日持つ")
    def quiet_arr(Pd):
        atrp = pd.Series(Pd["atr"] / Pd["pO"])
        med = atrp.shift(1).rolling(365, min_periods=200).median()
        return (atrp < med).to_numpy() & med.notna().to_numpy()
    res = {}
    for lab, qh, vh, vskip in (("静かな日は2日・荒い日は1日", 2, 1, False), ("静かな日は2日・荒い日は見送り", 2, 1, True)):
        per, Ts = {}, []
        for a in COINS:
            Pd = Pdays[a]
            q = quiet_arr(Pd)
            hold = np.where(q, qh, vh)
            s, T = run(Pd, a, hold=hold, extra_ok=(lambda P_, side, q=q: q) if vskip else None)
            per[a] = s
            Ts.append(T)
        res[lab] = show("E12", lab, per, pd.concat(Ts))
    per, Ts = {}, []
    for a in COINS:
        q = quiet_arr(Pdays[a])
        s, T = run(Pdays[a], a, extra_ok=lambda P_, side, q=q: q)
        per[a] = s
        Ts.append(T)
    res["参考：静かな日だけ・1日（E03）"] = show("E12", "参考：静かな日だけ・1日（E03と同じ形）", per, pd.concat(Ts))
    OUT["E12"] = res

    # ---- A14・A17・A18・A19・E08・E10・E14・E15 取引一覧から絞り込む（当日決済なので、同じ取引を残すか外すかだけ）
    T = T0.copy()
    T["d"] = T.day.dt.normalize()
    T = T.sort_values(["coin", "day"]).reset_index(drop=True)

    def filt(pid, name, keep, scale=None, memo=""):
        per, U = from_trades(T, keep, scale)
        return show(pid, name, per, U, memo)

    print("\n==== A14 直近の取引が勝った時だけ（銘柄ごと）")
    prev_R = T.groupby("coin").R.shift(1)
    prev_day = T.groupby("coin").day.shift(1)
    prev2_R = T.groupby("coin").R.shift(2)
    res = {}
    res["前日に取引があって勝った"] = filt("A14", "前日に取引があって勝った時だけ", ((T.day - prev_day).dt.days == 1) & (prev_R > 0))
    res["直近の取引が勝った"] = filt("A14", "直近の取引が勝った時だけ", prev_R > 0)
    res["直近の取引が負けた"] = filt("A14", "直近の取引が負けた時だけ（逆）", prev_R <= 0)
    res["直近2回とも勝った"] = filt("A14", "直近2回とも勝った時だけ", (prev_R > 0) & (prev2_R > 0))
    OUT["A14"] = res

    def daymap(fn):
        """日ごとの条件（銘柄ごと）を取引一覧に写す。fn(Pd, side) → 日ごとの真偽。"""
        out = np.zeros(len(T), bool)
        for a in COINS:
            Pd = Pdays[a]
            for side in (1, -1):
                m = pd.Series(fn(Pd, side), index=Pd["idx"])
                sel = (T.coin == a) & (T.side == side)
                out[sel.to_numpy()] = m.reindex(T.loc[sel, "d"]).fillna(False).to_numpy().astype(bool)
        return out

    print("\n==== A17 はらみ足（前日の高値・安値が前々日の内側）の翌日だけ")
    def inside(Pd, side):
        h1, l1 = np.r_[np.nan, Pd["pH"][:-1]], np.r_[np.nan, Pd["pL"][:-1]]
        h2, l2 = np.r_[np.nan, np.nan, Pd["pH"][:-2]], np.r_[np.nan, np.nan, Pd["pL"][:-2]]
        with np.errstate(invalid="ignore"):
            return (h1 < h2) & (l1 > l2)
    m = daymap(inside)
    OUT["A17"] = {"だけ": filt("A17", "はらみ足の翌日だけ", m), "以外": filt("A17", "はらみ足の翌日以外", ~m)}

    print("\n==== A18 約定の時間帯（日本時間、4時間ごと）で選ぶ")
    jst = ((T.pos * 15) // 60 + 9) % 24
    T["band"] = (((jst - 9) % 24) // 4)
    labels = ["9〜13時", "13〜17時", "17〜21時", "21〜1時", "1〜5時", "5〜8時"]
    h1m = T[T.day < SPLIT].groupby("band").R.mean()
    h2m = T[T.day >= SPLIT].groupby("band").R.mean()
    cnt = T.groupby("band").R.size() / R1.YEARS
    for b in range(6):
        print(f"    {labels[b]:8s} 1回あたり 前半{h1m.get(b, np.nan):+.3f}R → 後半{h2m.get(b, np.nan):+.3f}R（年{cnt.get(b, 0):.0f}回）")
    good = [b for b in range(6) if h1m.get(b, -1) > 0]
    OUT["A18"] = (filt("A18", "前半で1回あたりプラスだった時間帯だけ（" + "・".join(labels[b] for b in good) + "）", T.band.isin(good)),
                  {labels[b]: (h1m.get(b, np.nan), h2m.get(b, np.nan)) for b in range(6)}, [labels[b] for b in good])

    print("\n==== A19 BTCとETHが同じ日に同じ向きで約定したら量を減らす")
    both = T.groupby(["d", "side"]).coin.transform("nunique") == 2
    print(f"    同じ日・同じ向きの取引の割合 {both.mean() * 100:.0f}%")
    res = {}
    for f in (0.5, 0.75):
        res[f] = filt("A19", f"同時の時は{f}倍", pd.Series(True, index=T.index), scale=np.where(both, f, 1.0))
    OUT["A19"] = (res, both.mean())

    print("\n==== E08 静かな日が続いた長さ")
    def run_len(Pd):
        q = quiet_arr(Pd).astype(int)
        r = np.zeros(len(q), int)
        for i in range(len(q)):
            r[i] = (r[i - 1] + 1) if (q[i] and i > 0) else q[i]
        return r
    res = {}
    for L in (1, 3, 5, 10):
        m = daymap(lambda Pd, side, L=L: run_len(Pd) >= L)
        res[L] = filt("E08", f"静かな日が{L}日以上続いた時だけ", m)
    m5 = daymap(lambda Pd, side: run_len(Pd) >= 5)
    mq = daymap(lambda Pd, side: run_len(Pd) >= 1)
    res["段階"] = filt("E08", "荒い×0.5・静か1〜4日目×1.0・5日目から×1.5", pd.Series(True, index=T.index),
                     scale=np.where(m5, 1.5, np.where(mq, 1.0, 0.5)))
    OUT["E08"] = res

    print("\n==== E10 ATRが前日より増えた日だけ")
    def atr_up(Pd, side):
        a_ = Pd["atr"]
        with np.errstate(invalid="ignore"):
            return a_ > np.r_[np.nan, a_[:-1]]
    m = daymap(atr_up)
    OUT["E10"] = {"増えた日": filt("E10", "ATRが前日より増えた日だけ", m), "減った日": filt("E10", "ATRが前日より減った日だけ", ~m)}

    print("\n==== E14 5日のボラ÷60日のボラ（前日まで）が小さい日だけ")
    def ratio(Pd):
        lr = np.log(pd.Series(Pd["pC"])).diff()
        return (lr.rolling(5).std() / lr.rolling(60).std()).shift(1).to_numpy()
    res = {}
    for th in (0.8, 1.0):
        m = daymap(lambda Pd, side, th=th: ratio(Pd) < th)
        res[th] = filt("E14", f"比が{th}未満の日だけ", m)
    m = daymap(lambda Pd, side: ratio(Pd) >= 1.0)
    res["1以上"] = filt("E14", "比が1以上の日だけ（逆）", m)
    OUT["E14"] = res

    print("\n==== E15 終値の位置（前日の値幅のどこで引けたか）")
    def clv(Pd):
        with np.errstate(invalid="ignore", divide="ignore"):
            v = (Pd["pC"] - Pd["pL"]) / (Pd["pH"] - Pd["pL"])
        return np.r_[np.nan, v[:-1]]
    res = {}
    m = daymap(lambda Pd, side: (side * (clv(Pd) - 0.5)) > 0)
    res["値幅ブレイクの向きと合う時だけ"] = filt("E15", "値幅ブレイク：前日の引けが向きと同じ側（買いは上半分）の時だけ", m)
    for th in (0.8, 0.9):
        per, Ts = {}, []
        for a in COINS:
            Pd = Pdays[a]
            c = clv(Pd)
            side_arr = np.where(c >= th, 1, np.where(c <= 1 - th, -1, 0))
            Rr = side_arr * (Pd["pC"] - Pd["pO"]) / Pd["atr"] - (side_arr != 0) * PF.SPEC[a][0] * Pd["pO"] / Pd["atr"] \
                - (side_arr == 1) * PF.SPEC[a][1] / 36500 * Pd["pO"] / Pd["atr"]
            Rr = np.nan_to_num(Rr)
            w = np.where(side_arr == 1, 1.0, 0.5)
            per[a] = pd.Series(Rr * 0.5 * w, index=Pd["idx"])
            Ts.append(pd.DataFrame({"day": Pd["idx"][side_arr != 0], "R": Rr[side_arr != 0]}))
        U = pd.concat(Ts)
        res[th] = show("E15", f"前日の引けが上{int((1 - th) * 100)}%なら翌日買い（下なら売り）、9時〜翌9時", per,
                       U[(U.day >= MM.START) & (U.day <= MM.END)], tf="日足")
    OUT["E15"] = res

    # ---- A23 2段目の買い増し
    print("\n==== A23 約定後、さらに値幅×m進んだら同じ量を足す")
    res = {}
    for m_ in (0.3, 0.5, 1.0):
        per, T2 = combo(D, trig2_m=m_)
        res[m_] = show("A23", f"さらに値幅×{m_}進んだら足す", per, T2)
    OUT["A23"] = res

    # ---- A24 だましの逆張り
    print("\n==== A24 約定後、その日のうちに始値まで戻ったら撤退（反対に入る）")
    res = {}
    for rev, lab in ((0, "始値に戻ったら撤退"), (1, "始値に戻ったら撤退して反対に入る")):
        per = {}
        for a in COINS:
            Pd = Pdays[a]
            tot = np.zeros(Pd["n"])
            for side, w in ((1, 1.0), (-1, 0.5)):
                trig = Pd["pO"] + side * 0.8 * Pd["rng"]
                with np.errstate(invalid="ignore"):
                    ok = ~np.isnan(Pd["atr"]) & ~np.isnan(trig) & (side * (Pd["pO"] - Pd["sma5"]) > 0)
                fl = Pd["fl"]
                sw = (PF.SPEC[a][1] if side == 1 else PF.SPEC[a][2]) / 36500
                swr = (PF.SPEC[a][2] if side == 1 else PF.SPEC[a][1]) / 36500
                Rr = eng_rev(fl["O"], fl["H"], fl["L"], fl["C"], DAY, DAY - 4, trig, ok, Pd["atr"], Pd["pO"], side,
                             PF.SPEC[a][0], sw, swr, Pd["roll"], rev)
                tot += Rr * 0.5 * w
            per[a] = pd.Series(tot, index=Pd["idx"])
        res[lab] = show("A24", lab, per)
    OUT["A24"] = res

    # ---- A25 ウップス
    print("\n==== A25 ウップス（前日安値を割ってから戻ったら買い、前日高値を超えてから戻ったら売り）")
    res = {}
    for use_ma in (False, True):
        per = {}
        for a in COINS:
            Pd = Pdays[a]
            tot = np.zeros(Pd["n"])
            for side, w in ((1, 1.0), (-1, 0.5 if use_ma else 1.0)):
                lvl = np.r_[np.nan, (Pd["pL"] if side == 1 else Pd["pH"])[:-1]]
                with np.errstate(invalid="ignore"):
                    ok = ~np.isnan(Pd["atr"]) & ~np.isnan(lvl)
                    if use_ma:
                        ok &= side * (Pd["pO"] - Pd["sma5"]) > 0
                fl = Pd["fl"]
                sw = (PF.SPEC[a][1] if side == 1 else PF.SPEC[a][2]) / 36500
                Rr = eng_oops(fl["O"], fl["H"], fl["L"], fl["C"], DAY, DAY - 4, lvl, ok, Pd["atr"], side, PF.SPEC[a][0], sw)
                tot += Rr * 0.5 * w
            per[a] = pd.Series(tot, index=Pd["idx"])
        lab = "5日平均の向き（売り×0.5）" if use_ma else "向きなし（買い・売りとも×1）"
        res[lab] = show("A25", f"ウップス・{lab}", per)
    OUT["A25"] = res

    # ---- A26 週・A28 4時間
    print("\n==== A26 週の値幅ブレイク（月曜9時〜翌月曜9時、R＝週足のATR）")
    first_mon = int(np.flatnonzero(D["BTC"]["idx"].weekday == 0)[0]) * DAY
    res = {}
    for k in (0.3, 0.5, 0.8):
        for ma in (True, False):
            per, T = combo(D, P=DAY * 7, off=first_mon, k=k, ma=ma)
            res[(k, ma)] = show("A26", f"k={k}・{'5週平均の向き' if ma else '向きなし'}", per, T, tf="15分足（週）")
    OUT["A26"] = res
    print("\n==== A28 4時間の値幅ブレイク（R＝4時間足のATR）")
    res = {}
    for k in (0.5, 0.8, 1.2):
        per, T = combo(D, P=16, off=0, k=k)
        res[k] = show("A28", f"k={k}・直近5本の平均の向き", per, T, tf="15分足（4時間）")
    OUT["A28"] = res

    # ---- E02 TTMスクイーズ・E09 ボラの目標・E11 値幅の拡大の翌日
    print("\n==== E02 TTMスクイーズ（ボリンジャーがケルトナーの内側→外に出た日、20日平均に対する終値の向きに乗る）")
    res = {}
    for km in (1.5, 2.0):
        per, Ts = {}, []
        for a in COINS:
            x = D[a]
            s = pd.Series(x["dC"])
            m20, sd = s.rolling(20).mean(), s.rolling(20).std()
            ema = s.ewm(span=20, adjust=False).mean()
            tr = pd.Series(np.nanmax(np.vstack([x["dH"] - x["dL"], np.abs(x["dH"] - x["prevC"]), np.abs(x["dL"] - x["prevC"])]), axis=0))
            at20 = tr.rolling(20).mean()
            sq = ((m20 + 2 * sd) < (ema + km * at20)) & ((m20 - 2 * sd) > (ema - km * at20))
            fire = (sq.shift(1, fill_value=False) & ~sq).to_numpy()
            tot = np.zeros(len(s))
            for side in (1, -1):
                with np.errstate(invalid="ignore"):
                    sig = fire & (side * (s - m20) > 0).to_numpy()
                    xs = (side * (s - m20) < 0).to_numpy()
                sw = (PF.SPEC[a][1] if side == 1 else PF.SPEC[a][2]) / 36500
                pnl, te, tr_ = R1.swing_engine(x["O"], x["H"], x["L"], x["C"], x["atr"], sig, xs, side, 2.0, 20, PF.SPEC[a][0], sw)
                tot += pnl
                Ts.append(pd.DataFrame({"day": x["idx"][te], "R": tr_}))
            per[a] = pd.Series(tot, index=x["idx"])
        U = pd.concat(Ts)
        res[km] = show("E02", f"ケルトナー×{km}", per, U[(U.day >= MM.START) & (U.day <= MM.END)], tf="日足（約定は15分足）")
    OUT["E02"] = res

    print("\n==== E09 ボラの目標（値幅ブレイクの損益の振れが年率○%になるように量を調整、0.25〜2倍）")
    v0 = R1.total(per0)
    res = {}
    for look in (30, 60):
        vol = v0.rolling(look).std().shift(1) * np.sqrt(365)
        for tgt in (4, 6, 8):
            w = (tgt / vol).clip(0.25, 2.0).fillna(1.0)
            x = v0 * w
            r = R1.stats(x)
            R1.line(f"過去{look}日・年率{tgt}%", r, f" ｜平均の量{w.mean():.2f}倍")
            record(グループ=GROUP, 計画ID="E09", 名前=f"E09 過去{look}日の振れで年率{tgt}%に合わせる", 銘柄="BTC,ETH", 時間足="15分足",
                   メモ=f"平均の量{w.mean():.2f}倍", **{k: v for k, v in r.items() if not k.startswith("_")})
            res[(look, tgt)] = r
    OUT["E09"] = res

    print("\n==== E11 前日の値幅が前々日のm倍以上なら、前日の終値の向き（陽線なら買い）に9時〜翌9時持つ")
    res = {}
    for mlt in (1.5, 2.0, 3.0):
        per, Ts = {}, []
        for a in COINS:
            Pd = Pdays[a]
            rng1 = np.r_[np.nan, (Pd["pH"] - Pd["pL"])[:-1]]
            rng2 = np.r_[np.nan, np.nan, (Pd["pH"] - Pd["pL"])[:-2]]
            dirn = np.sign(np.r_[np.nan, (Pd["pC"] - Pd["pO"])[:-1]])
            with np.errstate(invalid="ignore"):
                side_arr = np.where(rng1 >= mlt * rng2, np.nan_to_num(dirn), 0)
            Rr = side_arr * (Pd["pC"] - Pd["pO"]) / Pd["atr"] - (side_arr != 0) * PF.SPEC[a][0] * Pd["pO"] / Pd["atr"] \
                - (side_arr == 1) * PF.SPEC[a][1] / 36500 * Pd["pO"] / Pd["atr"]
            Rr = np.nan_to_num(Rr)
            per[a] = pd.Series(Rr * 0.5 * np.where(side_arr == 1, 1.0, 0.5), index=Pd["idx"])
            Ts.append(pd.DataFrame({"day": Pd["idx"][side_arr != 0], "R": Rr[side_arr != 0]}))
        U = pd.concat(Ts)
        res[mlt] = show("E11", f"前日の値幅が前々日の{mlt}倍以上", per, U[(U.day >= MM.START) & (U.day <= MM.END)], tf="日足")
    OUT["E11"] = res
    return base


if __name__ == "__main__" and len(sys.argv) == 2:
    main(sys.argv[1])


def followup(cache):
    """第2回・前半の続き：(1) 区切り24通りを1/24ずつ（選ばない分散） (2) 「静か」の判定方法を前半で1つ選び、後半で確かめる。"""
    D = R1.prep(cache)
    per0, T0 = combo(D)
    print("==== 区切りの時刻24通りを1/24ずつ（どれも選ばない）")
    acc = None
    for jst in range(24):
        per, T = combo(D, off=((jst - 9) % 24) * 4)
        x = R1.total(per)
        acc = x / 24 if acc is None else acc.add(x / 24, fill_value=0)
    r = R1.stats(MM.clip(acc))
    R1.line("24通りを1/24ずつ", r)
    record(グループ=GROUP, 計画ID="A10", 名前="A10 区切り24通りを1/24ずつ", 銘柄="BTC,ETH", 時間足="15分足",
           メモ="どの時刻も選ばずに均等に分ける", **{k: v for k, v in r.items() if not k.startswith("_")})

    print("\n==== 「静か」の判定方法を、前半の安定度で1つ選ぶ（量は静かな日だけ1倍）")
    T = T0.copy()
    T["d"] = T.day.dt.normalize()
    defs = {}
    for a in COINS:
        Pd = periods(D[a], DAY, 0)
        atrp = pd.Series(Pd["atr"] / Pd["pO"])
        lr = np.log(pd.Series(Pd["pC"])).diff()
        rv10 = lr.rolling(10).std().shift(1)
        ratio = (lr.rolling(5).std() / lr.rolling(60).std()).shift(1)
        a_ = Pd["atr"]
        cand = {
            "ATRが過去1年の中央値より小さい（E03）": (atrp < atrp.shift(1).rolling(365, min_periods=200).median()),
            "10日の実現ボラが過去1年の中央値より小さい（E05）": (rv10 < rv10.rolling(365, min_periods=200).median()),
            "ATRが前日より減った（E10の逆）": pd.Series(a_ < np.r_[np.nan, a_[:-1]]),
            "5日÷60日のボラが0.8未満（E14）": (ratio < 0.8),
            "5日÷60日のボラが1.0未満（E14）": (ratio < 1.0),
        }
        for name, m in cand.items():
            defs.setdefault(name, {})[a] = pd.Series(m.to_numpy(), index=Pd["idx"])
    res = {}
    for name, mm in defs.items():
        keep = np.zeros(len(T), bool)
        for a in COINS:
            sel = (T.coin == a).to_numpy()
            keep[sel] = mm[a].reindex(T.loc[sel, "d"]).fillna(False).to_numpy().astype(bool)
        per, U = from_trades(T, keep)
        x = R1.total(per)
        r = R1.stats(x, U)
        R1.line(name, r)
        res[name] = r
    best = max(res, key=lambda k: res[k]["_h1s"])
    print(f"  → 前半で一番良いのは「{best}」（前半の安定度{res[best]['_h1s']:.2f}）。後半の安定度{res[best]['_h2s']:.2f}（A01は1.11）")
    return res, best


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "followup":
    followup(sys.argv[1])
