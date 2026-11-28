#!/usr/bin/env python3
"""デイトレードの計画書（project/research/daytrade_plan.md）第2回：日中のブレイクとトレンドの残り、決済・量の工夫（17件）。

  TA05 前日の高値・安値のブレイク（朝の注文だけ）／TA06 値幅ブレイクを1日2回（12時間ごと）／
  TA09 狭いレンジの後のブレイク／TA11 ボリンジャーバンドの収縮の後のブレイク
  TB04 土日だけ・平日だけ／TB05 1時間の動きが次の1時間に続くか／TB08 夜だけの値幅ブレイク（S1の形）
  TC04 スーパートレンド（15分足）／TC06 日中の押し目買い・戻り売り／TC07 最初の数時間の高値・安値のブレイク
  TK01 利確と損切り／TK02 追いかける損切り／TK03 時間で決済／TK06 朝5時に決済するか、翌朝まで持つか
  TL01 損切りの幅で量を決める／TL02 1日の損失の上限／TL04 負けの後に量を増やす（日中）
  TK・TL は、第1回で合格した TA03（ニューヨークの寄り付きのレンジのブレイク。5分足で計算）と TD01（急な動きの後に同じ向き）に当てはめる。
共通の決まりは第1回（dt1_lab.py）と同じ。結果は results.csv（グループ「計画・デイトレ第2回」）と daytrade_plan.csv。
使い方: python3 project/tools/dt2_lab.py <キャッシュ（BTC_15m_vol.csv・ETH_15m_vol.csv・BTC_5m.csv・ETH_5m.csv・BTC_15m.csv・ETH_15m.csv）> [項目]
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dt1_lab as DT  # noqa: E402
import dt_5m_check as C5  # noqa: E402
import more_methods_lab as MM  # noqa: E402
import round1_lab as R1  # noqa: E402
import round7_lab as R7  # noqa: E402
import vantage_exit_time as VX  # noqa: E402
from registry import drop  # noqa: E402

warnings.filterwarnings("ignore")
GROUP = "計画・デイトレ第2回"
DT.GROUP = GROUP
COINS, DAY, END, COST = DT.COINS, DT.DAY, DT.END, DT.COST
NAN = None


def merge(*rs):
    return dict(g=sum(r["g"] for r in rs), c=sum(r["c"] for r in rs), s=sum(r["s"] for r in rs),
                ret=np.concatenate([r["ret"] for r in rs]), hold=np.concatenate([r["hold"] for r in rs]),
                eday=np.concatenate([np.asarray(r["eday"], int) for r in rs]))


def run(ctx, pid, title, variants, coins=COINS, family=None, key=None):
    o = DT.run_variants(ctx, pid, title, variants, coins, family)
    if key:
        DT.SUM[key] = DT.SUM.pop(pid)
        DT.BEST[key] = DT.BEST.pop(pid)
        DT.BUILD[key] = DT.BUILD.pop(pid)
    return o


# ---------------------------------------------------------------- エンジン
@njit
def brk_p(O, H, L, C, P, start, end, xbar, up, dn, slu, sld, tpu, tpd, trail, hold, ok):
    """1日 P 本の足で、日ごとに上下の逆指値の先に届いた方に入る。損切り・利確・追いかける損切り・持つ本数の上限つき。
    約定した足で損切りにも届いていたら、損切りを先とみなす。追いかける損切りは、約定の次の足から。"""
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
        b0 = d * P
        last = b0 + xbar[d]
        if last > N:
            continue
        k = -1
        side = 0
        e = 0.0
        for j in range(b0 + start[d], min(b0 + end[d], last)):
            u = (not np.isnan(up[d])) and H[j] >= up[d]
            v = (not np.isnan(dn[d])) and L[j] <= dn[d]
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
        lastx = last
        if hold[d] > 0 and k + hold[d] < lastx:
            lastx = k + hold[d]
        sl = slu[d] if side == 1 else sld[d]
        tp = tpu[d] if side == 1 else tpd[d]
        tr = trail[d]
        best = e
        x = C[lastx - 1]
        xb = lastx - 1
        for j in range(k, lastx):
            if side == 1:
                if (not np.isnan(sl)) and L[j] <= sl:
                    x = sl if j == k else min(O[j], sl)
                    xb = j
                    break
                if j > k:
                    if (not np.isnan(tp)) and H[j] >= tp:
                        x = max(O[j], tp)
                        xb = j
                        break
                    if (not np.isnan(tr)) and L[j] <= best - tr:
                        x = min(O[j], best - tr)
                        xb = j
                        break
                best = max(best, H[j])
            else:
                if (not np.isnan(sl)) and H[j] >= sl:
                    x = sl if j == k else max(O[j], sl)
                    xb = j
                    break
                if j > k:
                    if (not np.isnan(tp)) and L[j] <= tp:
                        x = min(O[j], tp)
                        xb = j
                        break
                    if (not np.isnan(tr)) and H[j] >= best + tr:
                        x = max(O[j], best + tr)
                        xb = j
                        break
                best = min(best, L[j])
        rd[nt], rs[nt], re[nt], rx[nt], rb[nt], rxb[nt] = d, side, e, x, k, xb
        nt += 1
    return rd[:nt], rs[:nt], re[:nt], rx[:nt], rb[:nt], rxb[:nt]


@njit
def sig_engine(O, H, L, C, sig, A, bd, hold, slk, tpk, trk, dlim, win):
    """足 t の終値で出た合図（+1・−1）で、次の足の始値に入り hold 本持つ（同じ向きの合図で数え直し、逆向きなら入れ替え）。
    損切り・利確・追いかける損切りは ATR の倍数（0なら使わない）。同じ足で損切りと利確に届いたら損切りを先とみなす。
    dlim：その日（UTC）の損益がこれ以下（資金の%）になったら、その日は新しく入らない。取引時間の外は持たない。"""
    n = O.shape[0]
    rd = np.empty(n, np.int64)
    rs = np.empty(n, np.int64)
    re = np.empty(n)
    rx = np.empty(n)
    rb = np.empty(n, np.int64)
    rxb = np.empty(n, np.int64)
    nt = 0
    side = 0
    e = 0.0
    sl = np.nan
    tp = np.nan
    best = 0.0
    left = 0
    eb = 0
    pend = 0.0
    cur_day = -1
    dpnl = 0.0
    for t in range(n):
        if bd[t] != cur_day:
            cur_day = bd[t]
            dpnl = 0.0
        if pend != 0.0 and win[t] and A[t] > 0 and dpnl > -dlim:
            if side == 0 or side != pend:
                if side != 0:
                    rd[nt], rs[nt], re[nt], rx[nt], rb[nt], rxb[nt] = bd[eb], side, e, O[t], eb, t - 1
                    dpnl += 0.5 * side * (O[t] - e) / A[eb]
                    nt += 1
                side = int(pend)
                e = O[t]
                eb = t
                best = e
                sl = e - side * slk * A[t] if slk > 0 else np.nan
                tp = e + side * tpk * A[t] if tpk > 0 else np.nan
            left = hold
        pend = 0.0
        if side != 0:
            x = np.nan
            if side == 1:
                if (not np.isnan(sl)) and L[t] <= sl:
                    x = min(O[t], sl)
                elif (not np.isnan(tp)) and H[t] >= tp:
                    x = max(O[t], tp)
                elif trk > 0 and t > eb and L[t] <= best - trk * A[eb]:
                    x = min(O[t], best - trk * A[eb])
                best = max(best, H[t])
            else:
                if (not np.isnan(sl)) and H[t] >= sl:
                    x = max(O[t], sl)
                elif (not np.isnan(tp)) and L[t] <= tp:
                    x = min(O[t], tp)
                elif trk > 0 and t > eb and H[t] >= best + trk * A[eb]:
                    x = max(O[t], best + trk * A[eb])
                best = min(best, L[t])
            if np.isnan(x):
                left -= 1
                if left <= 0 or t + 1 >= n or not win[t + 1]:
                    x = C[t]
            if not np.isnan(x):
                rd[nt], rs[nt], re[nt], rx[nt], rb[nt], rxb[nt] = bd[eb], side, e, x, eb, t
                dpnl += 0.5 * side * (x - e) / A[eb]
                nt += 1
                side = 0
        if win[t] and sig[t] != 0.0:
            pend = sig[t]
    return rd[:nt], rs[:nt], re[:nt], rx[:nt], rb[:nt], rxb[:nt]


@njit
def setup_engine(O, H, L, C, setup, up, dn, sll, sls, valid, hold, win):
    """足 t の終値で条件がそろったら、up（買い）・dn（売り）の逆指値を valid 本だけ置く。先に届いた方に入り、
    損切り（買いは sll、売りは sls）か hold 本で決済。同じ足で上下の両方に届いたら見送り。約定した足でも損切りを先に見る。"""
    n = O.shape[0]
    rs = np.empty(n, np.int64)
    re = np.empty(n)
    rx = np.empty(n)
    rb = np.empty(n, np.int64)
    rxb = np.empty(n, np.int64)
    nt = 0
    side = 0
    e = 0.0
    sl = np.nan
    eb = 0
    pend = 0
    pu = np.nan
    pdn = np.nan
    psl = np.nan
    pss = np.nan
    for t in range(n):
        if side != 0:
            x = np.nan
            if side == 1 and L[t] <= sl:
                x = min(O[t], sl)
            elif side == -1 and H[t] >= sl:
                x = max(O[t], sl)
            elif t >= eb + hold - 1 or t + 1 >= n or not win[t + 1]:
                x = C[t]
            if not np.isnan(x):
                rs[nt], re[nt], rx[nt], rb[nt], rxb[nt] = side, e, x, eb, t
                nt += 1
                side = 0
            continue
        if pend > 0:
            if not win[t]:
                pend = 0
            else:
                u = (not np.isnan(pu)) and H[t] >= pu
                v = (not np.isnan(pdn)) and L[t] <= pdn
                if u and v:
                    pend = 0
                elif u or v:
                    side = 1 if u else -1
                    e = max(O[t], pu) if u else min(O[t], pdn)
                    sl = psl if u else pss
                    eb = t
                    pend = 0
                    if (side == 1 and L[t] <= sl) or (side == -1 and H[t] >= sl):
                        rs[nt], re[nt], rx[nt], rb[nt], rxb[nt] = side, e, sl, t, t
                        nt += 1
                        side = 0
                    continue
                else:
                    pend -= 1
        if side == 0 and pend == 0 and setup[t] and win[t]:
            pend = valid
            pu, pdn, psl, pss = up[t], dn[t], sll[t], sls[t]
    return rs[:nt], re[:nt], rx[:nt], rb[:nt], rxb[:nt]


@njit
def supertrend(H, L, C, atr, mult):
    n = C.shape[0]
    tr = np.zeros(n)
    fub = np.zeros(n)
    flb = np.zeros(n)
    tr[0] = 1.0
    hl2 = (H[0] + L[0]) / 2
    fub[0] = hl2 + mult * atr[0]
    flb[0] = hl2 - mult * atr[0]
    for i in range(1, n):
        hl2 = (H[i] + L[i]) / 2
        bu, bl = hl2 + mult * atr[i], hl2 - mult * atr[i]
        if np.isnan(bu):
            fub[i], flb[i], tr[i] = fub[i - 1], flb[i - 1], tr[i - 1]
            continue
        fub[i] = bu if (bu < fub[i - 1] or C[i - 1] > fub[i - 1] or np.isnan(fub[i - 1])) else fub[i - 1]
        flb[i] = bl if (bl > flb[i - 1] or C[i - 1] < flb[i - 1] or np.isnan(flb[i - 1])) else flb[i - 1]
        if tr[i - 1] > 0 and C[i] < flb[i]:
            tr[i] = -1.0
        elif tr[i - 1] < 0 and C[i] > fub[i]:
            tr[i] = 1.0
        else:
            tr[i] = tr[i - 1]
    return tr


@njit
def pullback(O, H, L, C, side_day, A, k, use_tp, start, endb):
    """日ごとに、その日の向き（side_day）の側だけ。買いは9時からの高値（その足の前まで）から ATR×k 下げた値段に指値、
    売りは安値から ATR×k 上げた値段に指値。決済は endb 本目の始まり（その前の足の終値）か、指値した時の高値・安値（use_tp）。"""
    nd = side_day.shape[0]
    rd = np.empty(nd, np.int64)
    rs = np.empty(nd, np.int64)
    re = np.empty(nd)
    rx = np.empty(nd)
    rb = np.empty(nd, np.int64)
    rxb = np.empty(nd, np.int64)
    nt = 0
    for d in range(nd):
        s = side_day[d]
        if s == 0 or not (A[d] > 0):
            continue
        b0 = d * 96
        ext = H[b0] if s > 0 else L[b0]
        k0 = -1
        e = 0.0
        tp = np.nan
        for j in range(b0 + 1, b0 + endb - 1):
            lvl = ext - s * k * A[d]
            if j >= b0 + start and ((s > 0 and L[j] <= lvl) or (s < 0 and H[j] >= lvl)):
                k0 = j
                e = min(O[j], lvl) if s > 0 else max(O[j], lvl)
                tp = ext if use_tp else np.nan
                break
            ext = max(ext, H[j]) if s > 0 else min(ext, L[j])
        if k0 < 0:
            continue
        x = C[b0 + endb - 1]
        xb = b0 + endb - 1
        for j in range(k0 + 1, b0 + endb):
            if (not np.isnan(tp)) and ((s > 0 and H[j] >= tp) or (s < 0 and L[j] <= tp)):
                x = max(O[j], tp) if s > 0 else min(O[j], tp)
                xb = j
                break
        rd[nt], rs[nt], re[nt], rx[nt], rb[nt], rxb[nt] = d, s, e, x, k0, xb
        nt += 1
    return rd[:nt], rs[:nt], re[:nt], rx[:nt], rb[:nt], rxb[:nt]


# ---------------------------------------------------------------- 第1回の合格形（TA03 は5分足、TD01 は15分足）
def ta03(ctx, mask=None, rl15=2, tp=None, sl="反対側", trail=None, hold_h=None, wfn=None):
    """TA03：米国株の始まりから rl15×15分のレンジのブレイク（5分足）。tp：('range', k) か ('atr', k)、sl：'反対側'・'中値'、
    trail：ATR×k、hold_h：持つ時間の上限、wfn(a, d, e, sl) → 量の倍率（取引ごと）。"""
    X, X5 = ctx["X"], ctx["X5"]
    us = ctx["us"]
    nd = len(ctx["idx"])
    rl = rl15 * 3
    s0 = np.where(us, 162, 174)
    xb = np.where(us, 240, 252)
    out = {}
    for a in COINS:
        x5, x = X5[a], X[a]
        Hm, Lm = x5["H"].reshape(nd, 288), x5["L"].reshape(nd, 288)
        cols = s0[:, None] + np.arange(rl)[None, :]
        hi = np.take_along_axis(Hm, cols, 1).max(1)
        lo = np.take_along_axis(Lm, cols, 1).min(1)
        A = x["atr"]
        slu, sld = (lo, hi) if sl == "反対側" else ((hi + lo) / 2, (hi + lo) / 2)
        if tp is None:
            tpu = tpd = np.full(nd, np.nan)
        elif tp[0] == "range":
            tpu, tpd = hi + tp[1] * (hi - lo), lo - tp[1] * (hi - lo)
        else:
            tpu, tpd = hi + tp[1] * A, lo - tp[1] * A
        tr = np.full(nd, np.nan) if trail is None else trail * A
        hold = np.zeros(nd, np.int64) if hold_h is None else np.full(nd, int(hold_h * 12), np.int64)
        ok = np.isfinite(A) & (np.ones(nd, bool) if mask is None else np.asarray(mask[a]))
        d, s, e, xp, eb, xbb = brk_p(x5["O"], x5["H"], x5["L"], x5["C"], 288, (s0 + rl).astype(np.int64), (xb - 1).astype(np.int64),
                                     xb.astype(np.int64), hi, lo, slu, sld, tpu, tpd, tr, hold, ok)
        w = np.ones(len(d)) if wfn is None else wfn(a, d, e, np.where(s == 1, slu[d], sld[d]))
        out[a] = DT.res_trades(x, a, d, s, e, xp, w, eb // 3, xbb // 3)
    return out


def td01(ctx, mask=None, k=3.0, H=16, slk=0.0, tpk=0.0, trk=0.0, dlim=1e9, end=END, wday=None):
    """TD01：15分の値動きがふだんの k 倍を超えたら、同じ向きに H 本（取引の一覧にして、損切りなどを足せる形）。"""
    X = ctx["X"]
    out = {}
    for a in COINS:
        x = X[a]
        lr, z = DT.zscore(x)
        sig = np.where(np.abs(np.nan_to_num(z)) > k, np.sign(lr), 0.0)
        w = DT.window(x, end) & DT.day_mask(x, None if mask is None else mask[a])
        A = x["atr"][x["bd"]]
        d, s, e, xp, eb, xb = sig_engine(x["O"], x["H"], x["L"], x["C"], sig, np.nan_to_num(A), x["bd"], H, slk, tpk, trk, dlim, w)
        wt = np.ones(len(d)) if wday is None else np.asarray(wday[a])[d]
        out[a] = DT.res_trades(x, a, d, s, e, xp, wt, eb, xb)
    return out


def streak_mult(res_base, look, mlt, nd):
    """取引の結果（コスト込み）から、直近 look 日の取引（1回以上）が全部負けなら翌日 mlt 倍。銘柄ごと。"""
    out = {}
    for a, r in res_base.items():
        d = np.asarray(r["eday"], int)
        win = (np.asarray(r["ret"]) - COST) > 0
        n = np.bincount(d, minlength=nd).astype(float)
        w = np.bincount(d, weights=win.astype(float), minlength=nd)
        out[a] = 1 + (mlt - 1) * R7.lost_last(n, w, look)
    return out


# ---------------------------------------------------------------- 各項目
def sec_ta(ctx):
    X = ctx["X"]
    v = {}
    for xb, xl in ((80, "翌5時"), (93, "翌朝8時15分")):
        for tp in (None, 1.0, 2.0):
            for sl in (True, False):
                def fn(mask, xb=xb, tp=tp, sl=sl):
                    out = {}
                    for a in COINS:
                        x = X[a]
                        ph, pl = np.r_[np.nan, x["dH"][:-1]], np.r_[np.nan, x["dL"][:-1]]
                        mid = (ph + pl) / 2
                        out[a] = DT.brk(x, a, 0, xb - 1, xb, ph, pl, mid if sl else NAN, mid if sl else NAN,
                                        ph + tp * x["atr"] if tp else NAN, pl - tp * x["atr"] if tp else NAN,
                                        mask=None if mask is None else mask[a])
                    return out
                v[f"決済{xl}・利確{'ATR×' + str(tp) if tp else 'なし'}・損切り{'前日の中値' if sl else 'なし'}"] = fn
    run(ctx, "TA05", "前日の高値・安値のブレイク（9時に上下の逆指値、先に約定した方）", v)
    print("    ---- 確かめ：TA05 を5分足で（損切りは前日の中値）")
    X5, nd = ctx["X5"], len(ctx["idx"])
    for xb, xl in ((240, "翌5時"), (279, "翌朝8時15分")):
        for tp in (None, 2.0):
            out = {}
            for a in COINS:
                x, x5 = X[a], X5[a]
                ph, pl = np.r_[np.nan, x["dH"][:-1]], np.r_[np.nan, x["dL"][:-1]]
                mid = (ph + pl) / 2
                tpu = ph + tp * x["atr"] if tp else np.full(nd, np.nan)
                tpd = pl - tp * x["atr"] if tp else np.full(nd, np.nan)
                d, s, e, xp, eb, xbb = brk_p(x5["O"], x5["H"], x5["L"], x5["C"], 288, np.zeros(nd, np.int64), np.full(nd, xb - 1, np.int64),
                                             np.full(nd, xb, np.int64), ph, pl, mid, mid, tpu, tpd, np.full(nd, np.nan),
                                             np.zeros(nd, np.int64), np.isfinite(x["atr"]) & np.isfinite(ph))
                out[a] = DT.res_trades(x, a, d, s, e, xp, np.ones(len(d)), eb // 3, xbb // 3)
            DT.show(ctx, "TA05", f"5分足で：決済{xl}・利確{'ATR×' + str(tp) if tp else 'なし'}・損切り前日の中値", out, memo="5分足で計算し直した確かめ")
    v = {}
    for k in (0.5, 0.8, 1.0):
        for own in (True, False):
            def fn(mask, k=k, own=own):
                out = {}
                for a in COINS:
                    x = X[a]
                    M = x["M"]
                    nd = x["nd"]
                    h1, l1, h2, l2 = M["H"][:, :48].max(1), M["L"][:, :48].min(1), M["H"][:, 48:].max(1), M["L"][:, 48:].min(1)
                    o1, o2 = M["O"][:, 0], M["O"][:, 48]
                    cl = np.empty(2 * nd)
                    cl[0::2], cl[1::2] = M["C"][:, 47], M["C"][:, 95]
                    ma = pd.Series(cl).rolling(5).mean().shift(1).to_numpy()
                    d1, d2 = (np.sign(o1 - ma[0::2]), np.sign(o2 - ma[1::2])) if own else (x["sdir"], x["sdir"])
                    r1, r2 = np.r_[np.nan, (h2 - l2)[:-1]], h1 - l1
                    m = None if mask is None else mask[a]
                    a1 = DT.brk(x, a, 0, 47, 48, np.where(d1 > 0, o1 + k * r1, np.nan), np.where(d1 < 0, o1 - k * r1, np.nan), ws=0.5, mask=m)
                    a2 = DT.brk(x, a, 48, 95, 96, np.where(d2 > 0, o2 + k * r2, np.nan), np.where(d2 < 0, o2 - k * r2, np.nan), ws=0.5, mask=m)
                    out[a] = merge(a1, a2)
                return out
            v[f"k={k}・向きは{'12時間の5本平均' if own else 'S1の向き'}"] = fn
    run(ctx, "TA06", "値幅ブレイクを12時間ごとに（9時と21時に区切り、前の12時間の値幅×k。売りは半分の量）", v)

    def setup_run(mask, kind, p1, p2, filt):
        out = {}
        for a in COINS:
            x = X[a]
            H_, L_, C_ = pd.Series(x["H"]), pd.Series(x["L"]), pd.Series(x["C"])
            if kind == "NR":
                hi, lo = H_.rolling(p1).max(), L_.rolling(p1).min()
                rg = hi - lo
                setup = (rg <= rg.rolling(p2).min().shift(1)).to_numpy()
                up, dn, sll, sls = hi.to_numpy(), lo.to_numpy(), lo.to_numpy(), hi.to_numpy()
                valid, hold = p1, 16
            else:
                mid, sd = C_.rolling(p1).mean(), C_.rolling(p1).std()
                wd = 4 * sd / mid
                setup = (wd <= wd.rolling(480, min_periods=200).quantile(p2).shift(1)).to_numpy()
                up, dn = (mid + 2 * sd).to_numpy(), (mid - 2 * sd).to_numpy()
                sll = sls = mid.to_numpy()
                valid, hold = 8, 16
            if filt:
                s = x["sdir"][x["bd"]]
                up = np.where(s > 0, up, np.nan)
                dn = np.where(s < 0, dn, np.nan)
            w = DT.window(x) & DT.day_mask(x, None if mask is None else mask[a])
            s_, e, xp, eb, xb = setup_engine(x["O"], x["H"], x["L"], x["C"], np.nan_to_num(setup).astype(np.bool_), up, dn, sll, sls,
                                             valid, hold, w)
            out[a] = DT.res_trades(x, a, x["bd"][eb], s_, e, xp, np.ones(len(e)), eb, xb)
        return out
    v = {}
    for N in (8, 16):
        for M_ in (96, 480):
            for filt in (False, True):
                v[f"{N}本の値幅が直近{M_}本で一番狭い・{'S1の向き' if filt else '向きの絞りなし'}"] = \
                    (lambda mask, N=N, M_=M_, filt=filt: setup_run(mask, "NR", N, M_, filt))
    run(ctx, "TA09", "狭いレンジの後のブレイク（その高値・安値に逆指値、損切りは反対側、4時間か翌5時で決済）", v)
    v = {}
    for n in (20, 80):
        for q in (0.1, 0.2):
            for filt in (False, True):
                v[f"{n}本のバンド・幅が下位{int(q * 100)}%・{'S1の向き' if filt else '向きの絞りなし'}"] = \
                    (lambda mask, n=n, q=q, filt=filt: setup_run(mask, "BB", n, q, filt))
    run(ctx, "TA11", "ボリンジャーバンドの収縮の後のブレイク（±2σに逆指値、損切りは中心、4時間か翌5時で決済）", v)


def sec_tb(ctx):
    X = ctx["X"]
    wd = ctx["idx"].dayofweek.to_numpy()
    wk = {a: wd >= 5 for a in COINS}
    wdy = {a: wd < 5 for a in COINS}

    def vbo(mask):
        out = {}
        for a in COINS:
            x = X[a]
            up = np.where(x["sdir"] > 0, x["dO"] + 0.8 * x["rng"], np.nan)
            dn = np.where(x["sdir"] < 0, x["dO"] - 0.8 * x["rng"], np.nan)
            out[a] = DT.brk(x, a, 0, END - 1, END, up, dn, ws=0.5, mask=mask[a])
        return out

    def fade(mask, k=2.0, H=4):
        out = {}
        for a in COINS:
            x = X[a]
            lr, z = DT.zscore(x)
            sig = np.where(np.abs(np.nan_to_num(z)) > k, -np.sign(lr), 0.0)
            w = DT.window(x) & DT.day_mask(x, mask[a])
            out[a] = DT.res_pos(x, a, DT.hold_after(sig, H, w))
        return out
    v = {}
    for nm, m in (("土日だけ", wk), ("平日だけ", wdy)):
        v[f"値幅ブレイク（S1の向き・翌5時）・{nm}"] = (lambda mask, m=m: vbo({a: m[a] & (True if mask is None else mask[a]) for a in COINS}))
        v[f"急な動きの後に同じ向き（TD01）・{nm}"] = (lambda mask, m=m: td01(ctx, {a: m[a] & (True if mask is None else mask[a]) for a in COINS}))
        v[f"急な動きの戻り（2倍・1時間）・{nm}"] = (lambda mask, m=m: fade({a: m[a] & (True if mask is None else mask[a]) for a in COINS}))
    run(ctx, "TB04", "土日だけ・平日だけで、値幅ブレイク・急な動きの後・戻りを比べる", v, family=lambda k: k.split("・")[0])
    v = {}
    for thr in (0.0, 1.0, 2.0):
        for fol in (True, False):
            def fn(mask, thr=thr, fol=fol):
                out = {}
                for a in COINS:
                    x = X[a]
                    C = x["C"]
                    nh = len(C) // 4
                    ch = C[3::4][:nh]
                    hr = np.r_[np.nan, np.diff(np.log(ch))]
                    sd = pd.Series(hr).rolling(168, min_periods=48).std().shift(1).to_numpy()
                    s = np.where(np.abs(np.nan_to_num(hr)) > thr * np.nan_to_num(sd, nan=np.inf), np.sign(hr), 0.0) * (1 if fol else -1)
                    sig_blk = np.r_[0.0, s[:-1]]
                    pos = np.repeat(sig_blk, 4)[:len(C)]
                    out[a] = DT.res_pos(x, a, pos * DT.window(x) * DT.day_mask(x, None if mask is None else mask[a]))
                return out
            v[f"1時間の動きが{thr:g}σ超・{'同じ向き' if fol else '逆'}に次の1時間"] = fn
    run(ctx, "TB05", "直前の1時間の動きの向き（と逆）に、次の1時間を持つ", v, family=lambda k: k.split("・")[-1])
    v = {}
    for (s, e), wl in (((48, 64), "21時〜翌1時"), ((32, 64), "17時〜翌1時"), ((48, 80), "21時〜翌5時")):
        for xb, xl in ((80, "翌5時"), (96, "翌9時")):
            def fn(mask, s=s, e=e, xb=xb):
                out = {}
                for a in COINS:
                    x = X[a]
                    up = np.where(x["sdir"] > 0, x["dO"] + 0.8 * x["rng"], np.nan)
                    dn = np.where(x["sdir"] < 0, x["dO"] - 0.8 * x["rng"], np.nan)
                    m = x["q"] & (True if mask is None else np.asarray(mask[a]))
                    out[a] = DT.brk(x, a, s, min(e, xb - 1), xb, up, dn, ws=0.5, mask=m)
                return out
            v[f"約定は{wl}・決済{xl}"] = fn
    run(ctx, "TB08", "夜だけの値幅ブレイク（S1の形：BTCの向き・静かな日だけ・売りは半分）", v)
    ref = {}
    for a in COINS:
        x = X[a]
        up = np.where(x["sdir"] > 0, x["dO"] + 0.8 * x["rng"], np.nan)
        dn = np.where(x["sdir"] < 0, x["dO"] - 0.8 * x["rng"], np.nan)
        ref[a] = DT.brk(x, a, 0, 92, 96, up, dn, ws=0.5, mask=x["q"])
    DT.show(ctx, "TB08", "参考：S1（9時〜翌8時に約定・翌9時に決済）", ref)


def sec_tc(ctx):
    X = ctx["X"]
    v = {}
    for n, mlt in ((10, 3.0), (20, 3.0)):
        for filt in (False, True):
            for lo in (False, True):
                def fn(mask, n=n, mlt=mlt, filt=filt, lo=lo):
                    out = {}
                    for a in COINS:
                        x = X[a]
                        H_, L_, C_ = x["H"], x["L"], x["C"]
                        pc = np.r_[C_[0], C_[:-1]]
                        tr = np.maximum(H_ - L_, np.maximum(np.abs(H_ - pc), np.abs(L_ - pc)))
                        at = pd.Series(tr).rolling(n).mean().to_numpy()
                        trend = supertrend(H_, L_, C_, at, mlt)
                        pos = np.r_[0.0, trend[:-1]]
                        if filt:
                            s = x["sdir"][x["bd"]]
                            pos = np.where(pos * s > 0, pos, 0.0)
                        if lo:
                            pos = np.clip(pos, 0, None)
                        out[a] = DT.res_pos(x, a, pos * DT.window(x) * DT.day_mask(x, None if mask is None else mask[a]))
                    return out
                v[f"ATR{n}本×{mlt:g}・{'S1の向き' if filt else '向きの絞りなし'}・{'買いだけ' if lo else '買い＋売り'}"] = fn
    run(ctx, "TC04", "15分足のスーパートレンド（9時〜翌5時だけ持つ）", v)
    v = {}
    for k in (0.3, 0.5, 1.0):
        for tp in (False, True):
            def fn(mask, k=k, tp=tp):
                out = {}
                for a in COINS:
                    x = X[a]
                    sd = x["sdir"] * (np.ones(x["nd"]) if mask is None else np.asarray(mask[a]))
                    d, s, e, xp, eb, xb = pullback(x["O"], x["H"], x["L"], x["C"], sd.astype(np.float64), np.nan_to_num(x["atr"]), k, tp, 4, END)
                    out[a] = DT.res_trades(x, a, d, s.astype(np.int64), e, xp, np.ones(len(d)), eb, xb)
                return out
            v[f"高値（安値）からATR×{k}・決済{'元の高値・安値か翌5時' if tp else '翌5時'}"] = fn
    run(ctx, "TC06", "S1の向きの日に、日中の押し目で買う・戻りで売る（指値）", v)
    v = {}
    for N in (2, 4):
        for sl in (True, False):
            for filt in (False, True):
                def fn(mask, N=N, sl=sl, filt=filt):
                    out = {}
                    nb = 4 * N
                    for a in COINS:
                        x = X[a]
                        hi, lo = x["M"]["H"][:, :nb].max(1), x["M"]["L"][:, :nb].min(1)
                        up, dn = hi, lo
                        if filt:
                            up = np.where(x["sdir"] > 0, hi, np.nan)
                            dn = np.where(x["sdir"] < 0, lo, np.nan)
                        out[a] = DT.brk(x, a, nb, END - 1, END, up, dn, lo if sl else NAN, hi if sl else NAN,
                                        mask=None if mask is None else mask[a])
                    return out
                v[f"最初の{N}時間の高値・安値・損切り{'あり' if sl else 'なし'}・{'S1の向き' if filt else '向きの絞りなし'}"] = fn
    run(ctx, "TC07", "9時から N 時間の高値・安値を、その後に抜けたらその向き（翌5時に決済）", v)


def sec_tk(ctx):
    v = {"基準（損切りは反対側・米国株の終わり）": lambda mask: ta03(ctx, mask)}
    for tp in (("range", 1.0), ("range", 2.0), ("atr", 0.5), ("atr", 1.0)):
        v[f"利確は{'レンジ' if tp[0] == 'range' else 'ATR'}×{tp[1]:g}"] = (lambda mask, tp=tp: ta03(ctx, mask, tp=tp))
    v["損切りをレンジの中値に"] = lambda mask: ta03(ctx, mask, sl="中値")
    v["損切りを中値に・利確はレンジ×1"] = lambda mask: ta03(ctx, mask, sl="中値", tp=("range", 1.0))
    run(ctx, "TK01", "TA03（5分足）の利確と損切り", v, key="TK01・TA03")
    v = {"基準（4時間・損切りなし）": lambda mask: td01(ctx, mask)}
    for slk in (0.5, 1.0):
        v[f"損切りATR×{slk:g}"] = (lambda mask, slk=slk: td01(ctx, mask, slk=slk))
        v[f"損切りATR×{slk:g}・利確ATR×1"] = (lambda mask, slk=slk: td01(ctx, mask, slk=slk, tpk=1.0))
    v["利確ATR×1"] = lambda mask: td01(ctx, mask, tpk=1.0)
    run(ctx, "TK01", "TD01の利確と損切り", v, key="TK01・TD01")
    v = {"基準": lambda mask: ta03(ctx, mask)}
    for k in (0.3, 0.5, 1.0):
        v[f"高値からATR×{k:g}戻ったら決済"] = (lambda mask, k=k: ta03(ctx, mask, trail=k))
    run(ctx, "TK02", "TA03（5分足）の追いかける損切り", v, key="TK02・TA03")
    v = {"基準": lambda mask: td01(ctx, mask)}
    for k in (0.3, 0.5, 1.0):
        v[f"高値からATR×{k:g}戻ったら決済"] = (lambda mask, k=k: td01(ctx, mask, trk=k))
    run(ctx, "TK02", "TD01の追いかける損切り", v, key="TK02・TD01")
    v = {"基準（米国株の終わり）": lambda mask: ta03(ctx, mask)}
    for h in (1, 2, 4):
        v[f"{h}時間で決済"] = (lambda mask, h=h: ta03(ctx, mask, hold_h=h))
    run(ctx, "TK03", "TA03（5分足）を時間で決済", v, key="TK03・TA03")
    v = {}
    for H in (4, 8, 16, 32):
        v[f"{H * 0.25:g}時間"] = (lambda mask, H=H: td01(ctx, mask, H=H))
    run(ctx, "TK03", "TD01を持つ時間を変える", v, key="TK03・TD01")
    # TK06：朝5時に決済するか、翌朝8時15分まで持つか（第1回の形）
    X = ctx["X"]
    v = {}
    for xb, xl in ((80, "翌5時"), (93, "翌朝8時15分")):
        def ta04(mask, xb=xb):
            out = {}
            for a in COINS:
                x = X[a]
                hi, lo = x["M"]["H"][:, :28].max(1), x["M"]["L"][:, :28].min(1)
                out[a] = DT.brk(x, a, 28, END - 1, xb, hi, lo, lo, hi, mask=None if mask is None else mask[a])
            return out
        v[f"TA04（アジア時間のレンジ）・決済{xl}"] = ta04
        v[f"TD01・決済{xl}"] = (lambda mask, xb=xb: td01(ctx, mask, end=xb))

        def tc02(mask, xb=xb):
            out = {}
            for a in COINS:
                x = X[a]
                Cs = pd.Series(x["C"])
                dd = (Cs.ewm(span=24, adjust=False).mean() - Cs.ewm(span=96, adjust=False).mean()).shift(1).to_numpy()
                out[a] = DT.res_pos(x, a, np.sign(np.nan_to_num(dd)) * DT.window(x, xb) * DT.day_mask(x, None if mask is None else mask[a]))
            return out
        v[f"TC02（平均24/96本）・決済{xl}"] = tc02
    run(ctx, "TK06", "朝5時に決済するか、翌朝8時15分まで持つか（スワップの時刻をまたぐ）", v, family=lambda k: k.split("・")[0])


def sec_tl(ctx):
    nd = len(ctx["idx"])
    v = {"基準（1R＝ATR）": lambda mask: ta03(ctx, mask)}
    for cap in (1.0, 2.0, 3.0):
        v[f"損切りまでで資金の0.5%（上限ATRの{cap:g}倍の量）"] = \
            (lambda mask, cap=cap: ta03(ctx, mask, wfn=lambda a, d, e, sl, cap=cap: np.clip(ctx["X"][a]["atr"][d] / np.abs(e - sl), 0, cap)))
    run(ctx, "TL01", "TA03（5分足）の量を、損切りまでの幅で決める", v, key="TL01・TA03")
    v = {"基準": lambda mask: td01(ctx, mask)}
    for lim in (0.25, 0.5, 1.0):
        v[f"その日の損失が{lim:g}%を超えたら止める"] = (lambda mask, lim=lim: td01(ctx, mask, dlim=lim))
    run(ctx, "TL02", "TD01に1日の損失の上限", v, key="TL02・TD01")
    base_a, base_d = ta03(ctx), td01(ctx)
    v = {"基準": lambda mask: ta03(ctx, mask)}
    for look in (1, 2):
        for m in (1.5, 2.0):
            def fn(mask, look=look, m=m):
                mult = streak_mult(base_a, look, m, nd)
                return ta03(ctx, mask, wfn=lambda a, d, e, sl, mult=mult: mult[a][d])
            v[f"直近{look}日が全敗なら{m:g}倍"] = fn
    run(ctx, "TL04", "TA03（5分足）で負けの後に量を増やす", v, key="TL04・TA03")
    v = {"基準": lambda mask: td01(ctx, mask)}
    for look in (1, 2):
        for m in (1.5, 2.0):
            v[f"直近{look}日が全敗なら{m:g}倍"] = (lambda mask, look=look, m=m: td01(ctx, mask, wday=streak_mult(base_d, look, m, nd)))
    run(ctx, "TL04", "TD01で負けの後に量を増やす", v, key="TL04・TD01")


SECTIONS = {"ta": sec_ta, "tb": sec_tb, "tc": sec_tc, "tk": sec_tk, "tl": sec_tl}


def main(cache, only=None):
    names = list(SECTIONS) if not only else only.split(",")
    if not only:
        print(f"（古い「{GROUP}」の行を {drop(GROUP)} 行消して、計算し直す）")
    X = DT.load(cache)
    idx = X["BTC"]["idx"]
    X5 = C5.load5(cache, idx)
    D = R1.prep(cache)
    parts = R7.s1_parts(D, COINS, D["BTC"])
    per, _ = R7.build(parts, R7.size_rules(parts, COINS)["S4案"])
    s4 = R1.total(per)
    s4r = R1.stats(s4)
    R1.line("S4案（比べる相手）", s4r)
    ctx = dict(X=X, X5=X5, idx=idx, us=VX.summer(idx), s4=s4, s4r=s4r)
    print("\n==== 確かめ：TA03（5分足・30分）と TD01（取引の一覧にした形）を作り直す")
    DT.show(ctx, "TA03", "確かめ：TA03（5分足・30分のレンジ）", ta03(ctx))
    DT.show(ctx, "TD01", "確かめ：TD01（取引の一覧にした形）", td01(ctx))
    for n in names:
        SECTIONS[n](ctx)
    print("\n==== まとめ（前半で一番良い形の判定）")
    for pid, o in DT.SUM.items():
        if "verdict" not in o:
            continue
        r = o["r"]
        print(f"  {pid} {o['verdict']:12s}「{o['best']}」：年{r['年平均%']:+.2f}% 落込{r['最大の落ち込み%']:.1f}% 安定度{r['安定度']:.2f}"
              f"（前半{r['_h1s']:.2f}・後半{r['_h2s']:.2f}）・倍のコストで年{r['_r2']['年平均%']:+.2f}%・1回{r['_gross']:+.3f}%・年{r['_n']:.0f}回"
              f"｜{o['n']}通り中{o['npos']}通りがプラス｜S4案と月の相関{o['corr']:+.2f}・足すと{o['mix']['安定度']:.2f}")
    return DT.SUM


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
