#!/usr/bin/env python3
"""計画書（project/research/plan.md）第2回・後半：時間帯（B）と決済の工夫（K）の優先「中」17件（BTC・ETH、2018年〜）。

  B03 時間帯の総当たり（日本時間○時に入って○時に出る、24×23通り。前半と後半の成績の並びが似ているか）
  B04 アジア時間の流れに乗る／B05 アジア時間の逆張り（9〜17時の上げ下げを見て、17時〜翌9時を持つ）
  B06 ロンドンの始まりのブレイク／B07 ニューヨークの始まりのブレイク（最初の30・60分の高値・安値を抜けたら、翌朝9時まで）
  B08 米国株の時間だけ持つ／B14 週末の値動きの続き／B15 週末の値動きの戻り／B20 半減期のサイクル／
  B22 曜日×時間帯の表（前半と後半で同じ時間が強いか）／B24 前日の向きに1日乗る
  K07 シャンデリア（最高値からATR×k）／K16 トレンドの決済の線（EMA20・EMA50・前日安値）／K17 タイム・ストップ … 9日EMA（C12）に付ける
  K09 半分を利確／K11 夕方に含み損なら決済／K14 静か・荒いで利確の距離を変える … 値幅ブレイク（A01）に付ける
共通: Vantage CFD（往復0.05%、買いのスワップはUTC21時をまたぐごと）、1R＝ATR1本分＝資金の0.5%（9日EMAは損切りまで＝0.5%）、単利。
結果は results.csv（グループ「計画・第2回後半」、計画IDつき）と plan.csv。
使い方: python3 project/tools/round2b_lab.py <キャッシュ（BTC_15m.csv, ETH_15m.csv）>
"""
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
import ema9_lab as E  # noqa: E402
import more_methods_lab as MM  # noqa: E402
import portfolio_lab as PF  # noqa: E402
import round1_lab as R1  # noqa: E402
import round2_lab as R2  # noqa: E402
from registry import drop, metrics_row, record  # noqa: E402

COINS = ("BTC", "ETH")
SPLIT = PF.SPLIT
GROUP = "計画・第2回後半"
DAY = 96
OUT = {}


# ---------------------------------------------------------------- エンジン
@njit
def trend_eng(O, H, L, C, P, dL, dH, dC, xline, sig, atr, side, cost, swd, roll, chand_k, trail_prev, tstop_n, tstop_min):
    """9日EMA（C12）の入り方（前日の信号で翌朝の始値、損切りは信号の足の安値）に、決済の工夫をつける。
    xline：終値がこの線を逆に抜けたら翌朝決済（NaNなら使わない）。chand_k：最高値からATR×k（0なら使わない）。
    trail_prev：損切りを前日の安値へ上げる。tstop_n：N日目の朝に含み益がATR×tstop_min未満なら決済（0なら使わない）。"""
    nd = dC.shape[0]
    N = nd * P
    pnl = np.zeros(nd)
    te = np.empty(nd, np.int64)
    tr = np.empty(nd)
    nt = 0
    d = 1
    while d < nd:
        if not sig[d - 1]:
            d += 1
            continue
        stop = dL[d - 1] if side == 1 else dH[d - 1]
        k = d * P
        e = O[k]
        risk = side * (e - stop)
        if risk <= 0.0 or not (atr[d] > 0):
            d += 1
            continue
        w = min(1.0, risk / e / 0.005)
        scale = 0.5 * w
        cR = cost * e / risk
        sR = swd * e / risk
        a0 = atr[d]
        pnl[d] -= cR * scale
        mark = 0.0
        nroll = 0
        held = 0
        x = 0.0
        out = -1
        best = e
        j = k
        while j < N:
            dd = j // P
            pos = j - dd * P
            if j > k and pos == 0:
                held += 1
                pv = dd - 1
                hit = False
                if not np.isnan(xline[pv]) and side * (dC[pv] - xline[pv]) < 0:
                    hit = True
                if tstop_n > 0 and held >= tstop_n and side * (dC[pv] - e) < tstop_min * a0:
                    hit = True
                if hit:
                    x = O[j]
                    out = dd
                    break
                if trail_prev:
                    nl = dL[pv] if side == 1 else dH[pv]
                    if side * (nl - stop) > 0:
                        stop = nl
                if chand_k > 0 and atr[dd] > 0:
                    cl = best - side * chand_k * atr[dd]
                    if side * (cl - stop) > 0:
                        stop = cl
            if j > k and pos == roll:
                nroll += 1
                pnl[dd] -= sR * scale
            adv = L[j] if side == 1 else H[j]
            if side * (adv - stop) <= 0:
                if j == k:
                    x = stop
                elif side == 1:
                    x = min(O[j], stop)
                else:
                    x = max(O[j], stop)
                out = dd
                break
            fav = H[j] if side == 1 else L[j]
            if side * (fav - best) > 0:
                best = fav
            if pos == P - 1:
                m = side * (C[j] - e) / risk
                pnl[dd] += (m - mark) * scale
                mark = m
            j += 1
        if out < 0:
            break
        rx = side * (x - e) / risk
        pnl[out] += (rx - mark) * scale
        te[nt] = d
        tr[nt] = rx - cR - nroll * sR
        nt += 1
        d = out + 1
    return pnl, te[:nt], tr[:nt]


@njit
def vbo_x(O, H, L, C, P, last, trig, ok, A, side, cost, swd, tp_k, tp_frac, chk):
    """値幅ブレイク（A01）の決済の工夫。tp_k[d]：利確の距離（ATR倍、NaNなら置かない）、tp_frac：利確で閉じる割合、
    chk：その日のこの15分足の始まりに含み損なら全部決済（-1なら使わない）。残りは日の終わりに決済。1日の損益（R）。"""
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
        tp = e + side * tp_k[d] * Ad if not np.isnan(tp_k[d]) else np.nan
        left = 1.0
        got = 0.0
        hit = False
        at_roll = 0.0
        for j in range(k, b0 + P):
            pos = j - b0
            if j > k and pos == 84:
                at_roll = left
            if chk >= 0 and j > k and pos == chk and side * (O[j] - e) < 0:
                got += left * side * (O[j] - e) / Ad
                left = 0.0
                break
            if not hit and not np.isnan(tp):
                if (side == 1 and H[j] >= tp) or (side == -1 and L[j] <= tp):
                    px = tp if j == k else (max(O[j], tp) if side == 1 else min(O[j], tp))
                    got += tp_frac * side * (px - e) / Ad
                    left -= tp_frac
                    hit = True
                    if left <= 1e-9:
                        break
        if left > 1e-9:
            got += left * side * (C[b0 + P - 1] - e) / Ad
        R[d] = got - cost * e / Ad - at_roll * swd * e / Ad
    return R


# ---------------------------------------------------------------- 決まった時刻に入って出る
def window(x, a, i1, i2, side, w):
    """日ごとに、全体の15分足の番号 i1 で入り i2 で出る（始値どうし）。side・w は日ごと。1R＝ATR＝資金の0.5%。"""
    O, n = x["O"], len(x["O"])
    A = x["atr"]
    valid = (side != 0) & (i2 < n) & (i1 >= 0) & ~np.isnan(A) & (A > 0)
    p1, p2 = O[np.clip(i1, 0, n - 1)], O[np.clip(i2, 0, n - 1)]
    r0 = (i1 // DAY) * DAY + 84
    r0 = np.where(r0 <= i1, r0 + DAY, r0)
    rolls = np.where(r0 < i2, (i2 - 1 - r0) // DAY + 1, 0)
    swd = np.where(side > 0, PF.SPEC[a][1], PF.SPEC[a][2]) / 36500
    with np.errstate(invalid="ignore", divide="ignore"):
        R = side * (p2 - p1) / A - PF.SPEC[a][0] * p1 / A - rolls * swd * p1 / A
    R = np.where(valid, R, 0.0)
    T = pd.DataFrame({"day": x["idx"][valid], "R": R[valid]})
    return pd.Series(R * 0.5 * w, index=x["idx"]), T


def both(D, fn):
    """銘柄ごとの fn(x, a) → (日ごとの損益, 取引一覧) を合わせる。"""
    per, Ts = {}, []
    for a in COINS:
        s, T = fn(D[a], a)
        per[a] = s
        Ts.append(T)
    T = pd.concat(Ts, ignore_index=True)
    return per, T[(T.day >= MM.START) & (T.day <= MM.END)]


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


def ma_side(x, short=True):
    with np.errstate(invalid="ignore"):
        s = np.sign(x["dO"] - R1.sma_prev(x, 5))
    s = np.nan_to_num(s)
    return s if short else np.clip(s, 0, None)


# ---------------------------------------------------------------- 本体
def main(cache):
    print(f"（古い「{GROUP}」の行を {drop(GROUP)} 行消して、計算し直す）")
    D = R1.prep(cache)
    nd = {a: len(D[a]["dO"]) for a in COINS}
    day0 = {a: np.arange(nd[a]) * DAY for a in COINS}

    # ---- B24・B04・B05・B08・B14・B15 決まった時刻に入って出る
    print("==== B24 前日の向き（陽線なら買い）に、9時〜翌9時を持つ")
    def b24(x, a):
        side = np.nan_to_num(np.sign(np.r_[np.nan, (x["dC"] - x["dO"])[:-1]]))
        return window(x, a, day0[a], day0[a] + DAY, side, np.ones(nd[a]))
    per, T = both(D, b24)
    OUT["B24"] = show("B24", "前日の向きに1日乗る（買い・売りとも×1）", per, T, tf="日足")

    print("\n==== B04・B05 日本9〜17時の上げ下げを見て、17時〜翌9時を持つ")
    res = {}
    for th in (0.0, 0.25):
        for sgn, pid, lab in ((1, "B04", "同じ向き"), (-1, "B05", "反対の向き")):
            def fn(x, a, th=th, sgn=sgn):
                mv = x["O"][np.clip(day0[a] + 32, 0, len(x["O"]) - 1)] - x["dO"]
                with np.errstate(invalid="ignore"):
                    side = np.where(np.abs(mv) > th * x["atr"], sgn * np.sign(mv), 0)
                return window(x, a, day0[a] + 32, day0[a] + DAY, np.nan_to_num(side), np.ones(nd[a]))
            per, T = both(D, fn)
            res[(pid, th)] = show(pid, f"アジア時間の{lab}（動きがATR×{th}より大きい日）", per, T)
    OUT["B04"], OUT["B05"] = {k: v for k, v in res.items() if k[0] == "B04"}, {k: v for k, v in res.items() if k[0] == "B05"}

    print("\n==== B08 米国株の時間だけ、5日平均の向きに持つ")
    res = {}
    for s_bar, e_bar, lab in ((54, 80, "日本22時半〜翌5時（夏時間）"), (58, 84, "日本23時半〜翌6時（冬時間）")):
        def fn(x, a, s_bar=s_bar, e_bar=e_bar):
            sd = ma_side(x)
            return window(x, a, day0[a] + s_bar, day0[a] + e_bar, sd, np.where(sd > 0, 1.0, 0.5))
        per, T = both(D, fn)
        res[lab] = show("B08", lab, per, T)
    OUT["B08"] = res

    print("\n==== B14・B15 週末（金曜21時〜月曜9時、日本）の上げ下げを見て、月曜9時〜火曜9時を持つ")
    res = {}
    for sgn, pid, lab in ((1, "B14", "同じ向き"), (-1, "B15", "反対の向き")):
        def fn(x, a, sgn=sgn):
            mon = (x["idx"].weekday == 0)
            i1 = day0[a].copy()
            ifri = i1 - 3 * DAY + 48
            mv = np.where(ifri >= 0, x["dO"] - x["O"][np.clip(ifri, 0, None)], np.nan)
            side = np.where(mon, sgn * np.nan_to_num(np.sign(mv)), 0)
            return window(x, a, i1, i1 + DAY, side, np.ones(nd[a]))
        per, T = both(D, fn)
        res[pid] = show(pid, f"週末の値動きと{lab}に、月曜を持つ", per, T)
    OUT["B14"], OUT["B15"] = res["B14"], res["B15"]

    # ---- B06・B07 セッションの始まりのブレイク
    print("\n==== B06・B07 セッションの最初の値幅を抜けたら順張り、翌朝9時に決済")
    res = {}
    for pid, s_bar, lab in (("B06", 28, "ロンドン（日本16時）"), ("B06", 32, "ロンドン（日本17時）"),
                            ("B07", 54, "ニューヨーク（日本22時半）"), ("B07", 58, "ニューヨーク（日本23時半）")):
        for mins in (30, 60):
            nb = mins // 15
            for use_ma in (True, False):
                def fn(x, a, s_bar=s_bar, nb=nb, use_ma=use_ma):
                    Pd = R2.periods(x, DAY, 0)
                    M = Pd["M"]
                    tot = np.zeros(Pd["n"])
                    Ts = []
                    for side, w in ((1, 1.0), (-1, 0.5 if use_ma else 1.0)):
                        trig = M["H"][:, s_bar:s_bar + nb].max(1) if side == 1 else M["L"][:, s_bar:s_bar + nb].min(1)
                        with np.errstate(invalid="ignore"):
                            ok = ~np.isnan(Pd["atr"])
                            if use_ma:
                                ok &= side * (Pd["pO"] - Pd["sma5"]) > 0
                        fl = Pd["fl"]
                        sw = (PF.SPEC[a][1] if side == 1 else PF.SPEC[a][2]) / 36500
                        pnl, te, tp, tr, _ = R2.eng(fl["O"], fl["H"], fl["L"], fl["C"], DAY, DAY - 4, trig.astype(np.float64),
                                                    ok.astype(np.bool_), Pd["atr"], np.full(Pd["n"], s_bar + nb, np.int64), side,
                                                    PF.SPEC[a][0], sw, Pd["roll"], np.ones(Pd["n"], np.int64), np.full(Pd["n"], np.nan))
                        tot += w * pnl
                        Ts.append(pd.DataFrame({"day": Pd["idx"][te], "R": tr}))
                    return pd.Series(tot, index=Pd["idx"]), pd.concat(Ts)
                per, T = both(D, fn)
                name = f"{lab}・最初の{mins}分・{'5日平均の向き' if use_ma else '向きなし'}"
                res[(pid, s_bar, mins, use_ma)] = show(pid, name, per, T)
    OUT["B06"] = {k: v for k, v in res.items() if k[0] == "B06"}
    OUT["B07"] = {k: v for k, v in res.items() if k[0] == "B07"}

    # ---- B03 時間帯の総当たり
    print("\n==== B03 日本時間○時に入って○時間後に出る（24×23通り）")
    b03 = {}
    for mode in ("買いだけ", "5日平均の向き"):
        rows = []
        for u1 in range(24):
            for Lh in range(1, 24):
                xs = None
                for a in COINS:
                    x = D[a]
                    sd = np.ones(nd[a]) if mode == "買いだけ" else ma_side(x)
                    s, _ = window(x, a, day0[a] + u1 * 4, day0[a] + u1 * 4 + Lh * 4, sd, np.where(sd > 0, 1.0, 0.5))
                    xs = s if xs is None else xs.add(s, fill_value=0)
                xs = MM.clip(xs)
                h1, h2 = xs[xs.index < SPLIT], xs[xs.index >= SPLIT]
                m1, m2 = h1.resample("ME").sum(), h2.resample("ME").sum()
                rows.append(dict(jst=(u1 + 9) % 24, L=Lh, a1=h1.mean() * 365, a2=h2.mean() * 365,
                                 s1=m1.mean() / m1.std() * np.sqrt(12), s2=m2.mean() / m2.std() * np.sqrt(12)))
        df = pd.DataFrame(rows)
        rho = df[["a1", "a2"]].corr(method="spearman").iloc[0, 1]
        bothpos = ((df.a1 > 0) & (df.a2 > 0)).mean() * 100
        top = df.sort_values("s1", ascending=False).head(5)
        print(f"  -- {mode}：前半と後半の年平均の並びの相関（順位）{rho:+.2f}、前半・後半ともプラスは{bothpos:.0f}%")
        for r in top.itertuples():
            print(f"     前半の上位：{r.jst:2d}時から{r.L:2d}時間  前半の安定度{r.s1:5.2f}（年{r.a1:+5.1f}%） → 後半{r.s2:5.2f}（年{r.a2:+5.1f}%）")
        b = top.iloc[0]
        xs = None
        for a in COINS:
            x = D[a]
            u1 = int((b.jst - 9) % 24)
            sd = np.ones(nd[a]) if mode == "買いだけ" else ma_side(x)
            s, _ = window(x, a, day0[a] + u1 * 4, day0[a] + u1 * 4 + int(b.L) * 4, sd, np.where(sd > 0, 1.0, 0.5))
            xs = s if xs is None else xs.add(s, fill_value=0)
        r = R1.stats(MM.clip(xs))
        record(グループ=GROUP, 計画ID="B03", 名前=f"B03 {mode}：前半で一番良い{int(b.jst)}時から{int(b.L)}時間", 銘柄="BTC,ETH",
               時間足="15分足", メモ=f"24×23通り。前半と後半の並びの相関{rho:+.2f}、前半・後半ともプラス{bothpos:.0f}%",
               **{k: v for k, v in r.items() if not k.startswith("_")})
        b03[mode] = (rho, bothpos, b, r)
    OUT["B03"] = b03

    # ---- B22 曜日×時間帯
    print("\n==== B22 曜日×時間帯（日本時間）の1時間の値動き（R）")
    cells = {}
    for a in COINS:
        x = D[a]
        Oh = x["O"][::4]
        t = x["idx"][0] + pd.to_timedelta(np.arange(len(Oh)) * 60, unit="min")
        dayi = np.arange(len(Oh)) // 24
        A = x["atr"][np.clip(dayi, 0, len(x["atr"]) - 1)]
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.r_[(Oh[1:] - Oh[:-1]), np.nan] / A * 0.5
        jt = t + pd.Timedelta(hours=9)
        df = pd.DataFrame({"r": r, "wd": jt.weekday, "hr": jt.hour, "t": t}).dropna()
        df = df[(df.t >= MM.START) & (df.t <= MM.END)]
        cells[a] = df
    df = pd.concat(cells.values())
    g1 = df[df.t < SPLIT].groupby(["wd", "hr"]).r.mean()
    g2 = df[df.t >= SPLIT].groupby(["wd", "hr"]).r.mean()
    corr = np.corrcoef(g1.to_numpy(), g2.to_numpy())[0, 1]
    up, dn = g1.sort_values(ascending=False).index[:24], g1.sort_values().index[:24]
    wk = SPLIT.year
    def strat(part, cost_on):
        p = part.copy()
        sgn = np.where(p.set_index(["wd", "hr"]).index.isin(up), 1, np.where(p.set_index(["wd", "hr"]).index.isin(dn), -1, 0))
        c = np.where(sgn != 0, 0.0005 * 0.5 / 0.035 if cost_on else 0.0, 0.0)
        return (sgn * p.r.to_numpy() - c).sum() / ((p.t.max() - p.t.min()).days / 365.25)
    h2 = df[df.t >= SPLIT]
    print(f"  168のマスの平均の、前半と後半の相関 {corr:+.2f}")
    print(f"  前半で上位24マスを買い・下位24マスを売り → 後半の年平均（資金の%）：手数料なし{strat(h2, False):+.1f}%、手数料ありの目安{strat(h2, True):+.1f}%")
    wdn = "月火水木金土日"
    print("  前半の上位5マス：" + "・".join(f"{wdn[w]}{h}時（{g1[(w, h)]:+.3f}→{g2.get((w, h), np.nan):+.3f}）" for w, h in up[:5]))
    print("  前半の下位5マス：" + "・".join(f"{wdn[w]}{h}時（{g1[(w, h)]:+.3f}→{g2.get((w, h), np.nan):+.3f}）" for w, h in dn[:5]))
    OUT["B22"] = (corr, strat(h2, False), strat(h2, True))

    # ---- B20 半減期のサイクル
    print("\n==== B20 半減期（2016-07-09・2020-05-11・2024-04-20）からの年数で、買い持ちの成績を分ける")
    halv = pd.to_datetime(["2016-07-09", "2020-05-11", "2024-04-20"])
    ph_res = {}
    hold = {}
    for a in COINS:
        x = D[a]
        held = np.ones(nd[a])
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.nan_to_num(0.5 / x["atr"])
        pnl = np.nan_to_num(x["dC"] - x["prevC"]) * scale - PF.SPEC[a][1] / 36500 * x["dO"] * scale
        hold[a] = pd.Series(pnl, index=x["idx"])
    idx = hold["BTC"].index
    last_h = np.array([halv[halv <= t].max() for t in idx])
    yrs = np.array([(t - h).days // 365 + 1 if pd.notna(h) else 0 for t, h in zip(idx, last_h)])
    yrs = np.clip(yrs, 0, 4)
    phase = pd.Series(yrs, index=idx)
    x = MM.clip(hold["BTC"].add(hold["ETH"], fill_value=0))
    ph = phase.reindex(x.index)
    for p in (1, 2, 3, 4):
        a1 = x[(ph == p) & (x.index < SPLIT)]
        a2 = x[(ph == p) & (x.index >= SPLIT)]
        ph_res[p] = (a1.mean() * 365 if len(a1) else np.nan, a2.mean() * 365 if len(a2) else np.nan, len(a1), len(a2))
        print(f"  半減期から{p}年目{'以降' if p == 4 else ''}：買い持ちの年率 前半{ph_res[p][0]:+6.1f}%（{ph_res[p][2]}日） 後半{ph_res[p][1]:+6.1f}%（{ph_res[p][3]}日）")
    res = {}
    for keep, lab in (({1}, "1年目だけ持つ"), ({1, 2}, "1・2年目だけ持つ"), ({4, 1}, "4年目以降と1年目だけ持つ")):
        per = {a: hold[a] * phase.reindex(hold[a].index).isin(keep).astype(float) for a in COINS}
        res[lab] = show("B20", lab, per, tf="日足")
    per = {a: hold[a] for a in COINS}
    res["ずっと持つ"] = show("B20", "参考：ずっと持つ（CFD）", per, tf="日足")
    OUT["B20"] = (ph_res, res)

    # ---- K07・K16・K17 9日EMA（C12）の決済
    print("\n==== K07・K16・K17 9日EMA（C12：実体で抜けたら翌朝入り、50日平均で確認）の決済を変える")
    def trend(xl_kind="ema9", chand=0.0, trail=False, tstop=0, tmin=0.5):
        per, Ts = {}, []
        for a in COINS:
            x = D[a]
            B = E.daily(x)
            O, H, L, C = B["flat"]
            s = pd.Series(B["dC"])
            xl = {"ema9": B["ema"], "ema20": s.ewm(span=20, adjust=False).mean().to_numpy(),
                  "ema50": s.ewm(span=50, adjust=False).mean().to_numpy(), "none": np.full(len(s), np.nan)}[xl_kind]
            tot = np.zeros(len(s))
            for side in (1, -1):
                sig = E.signal(B, side, "50日平均の上")
                sw = (PF.SPEC[a][1] if side == 1 else PF.SPEC[a][2]) / 36500
                pnl, te, tr = trend_eng(O, H, L, C, DAY, B["dL"], B["dH"], B["dC"], xl, sig, x["atr"], side, PF.SPEC[a][0], sw,
                                        84, chand, trail, tstop, tmin)
                tot += pnl
                Ts.append(pd.DataFrame({"day": x["idx"][te], "R": tr}))
            per[a] = pd.Series(tot, index=x["idx"])
        T = pd.concat(Ts)
        return per, T[(T.day >= MM.START) & (T.day <= MM.END)]
    per, T = trend()
    base_t = show("K16", "基準：C12（終値が9日EMAを戻ったら翌朝）", per, T, tf="日足（約定は15分足）")
    res16 = {"基準": base_t}
    for kind, lab in (("ema20", "終値が20日EMAを戻ったら"), ("ema50", "終値が50日EMAを戻ったら")):
        per, T = trend(kind)
        res16[lab] = show("K16", lab, per, T, tf="日足（約定は15分足）")
    per, T = trend("none", trail=True)
    res16["損切りを前日の安値へ上げるだけ"] = show("K16", "損切りを前日の安値へ上げるだけ（EMAは見ない）", per, T, tf="日足（約定は15分足）")
    OUT["K16"] = res16
    res07 = {}
    for kc in (2.0, 3.0, 4.0):
        per, T = trend("none", chand=kc)
        res07[kc] = show("K07", f"シャンデリア：最高値からATR×{kc}（EMAは見ない）", per, T, tf="日足（約定は15分足）")
    per, T = trend("ema9", chand=3.0)
    res07["9日EMA＋ATR×3"] = show("K07", "9日EMAの決済＋シャンデリアATR×3の両方", per, T, tf="日足（約定は15分足）")
    OUT["K07"] = (base_t, res07)
    res17 = {}
    for n in (2, 3, 5):
        per, T = trend("ema9", tstop=n)
        res17[n] = show("K17", f"9日EMA＋{n}日目の朝に含み益がATR×0.5未満なら決済", per, T, tf="日足（約定は15分足）")
    OUT["K17"] = (base_t, res17)

    # ---- K09・K11・K14 値幅ブレイク（A01）の決済
    print("\n==== K09・K11・K14 値幅ブレイク（A01）の決済を変える")
    def vbo_variant(tpk_fn=None, frac=1.0, chk=-1):
        per = {}
        for a in COINS:
            Pd = R2.periods(D[a], DAY, 0)
            fl = Pd["fl"]
            tot = np.zeros(Pd["n"])
            for side, w in ((1, 1.0), (-1, 0.5)):
                trig = Pd["pO"] + side * 0.8 * Pd["rng"]
                with np.errstate(invalid="ignore"):
                    ok = ~np.isnan(Pd["atr"]) & ~np.isnan(trig) & (side * (Pd["pO"] - Pd["sma5"]) > 0)
                tpk = np.full(Pd["n"], np.nan) if tpk_fn is None else tpk_fn(Pd)
                sw = (PF.SPEC[a][1] if side == 1 else PF.SPEC[a][2]) / 36500
                Rr = vbo_x(fl["O"], fl["H"], fl["L"], fl["C"], DAY, DAY - 4, trig, ok, Pd["atr"], side, PF.SPEC[a][0], sw,
                           tpk.astype(np.float64), frac, chk)
                tot += Rr * 0.5 * w
            per[a] = pd.Series(tot, index=Pd["idx"])
        return per
    base_v = show("K09", "基準：A01（翌朝9時に全部決済）", vbo_variant())
    res09 = {}
    for kk in (1.0, 1.5, 2.0):
        res09[kk] = show("K09", f"ATR×{kk}で半分を利確、残りは翌朝", vbo_variant(lambda Pd, kk=kk: np.full(Pd["n"], kk), 0.5))
    OUT["K09"] = (base_v, res09)
    res11 = {}
    for jst, bar in ((15, 24), (18, 36), (21, 48)):
        res11[jst] = show("K11", f"日本時間{jst}時に含み損なら決済", vbo_variant(chk=bar))
    OUT["K11"] = (base_v, res11)
    def quiet(Pd):
        atrp = pd.Series(Pd["atr"] / Pd["pO"])
        med = atrp.shift(1).rolling(365, min_periods=200).median()
        return (atrp < med).to_numpy()
    res14 = {}
    for kq, kv, lab in ((np.nan, 1.0, "静かな日は利確なし・荒い日はATR×1"), (3.0, 1.0, "静かな日はATR×3・荒い日はATR×1"),
                        (2.0, 1.5, "静かな日はATR×2・荒い日はATR×1.5"), (np.nan, 1.5, "静かな日は利確なし・荒い日はATR×1.5")):
        res14[lab] = show("K14", lab, vbo_variant(lambda Pd, kq=kq, kv=kv: np.where(quiet(Pd), kq, kv)))
    OUT["K14"] = (base_v, res14)
    return OUT


if __name__ == "__main__":
    main(sys.argv[1])
