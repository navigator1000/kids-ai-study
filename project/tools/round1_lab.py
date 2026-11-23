#!/usr/bin/env python3
"""計画書（project/research/plan.md）第1回：優先「高」の21件を、共通の物差しで検証する（BTC・ETH、2018年〜）。

  値幅ブレイクの改善 : A02 kの細かいずらし／A03 値幅をATRで測る／A07 向きの判定の日数／A08 向きの判定なし／
                       A16 NR4・NR7の翌日だけ／A31 寄り付き後の値幅を抜けたら／K10 勝っている時だけ持ち越す
  新しい部品         : C24 時系列モメンタム／C25 複数期間のモメンタム／E01 スクイーズのブレイク
  量の調整           : E05 静かの判定を実現ボラで／E06 週足のATRで／E07 ATRの順位で段階的に／L08 資金の落ち込みで減らす／L09 資金曲線で休む
  選び直し           : N10 ウォーク・フォワード（毎年、過去N年でkを選び直す）
  確かめ             : N15 ランダムな売買との比較／P01 手数料を上げる／P03 逆指値の滑り／P04 注文が遅れる／P08 モンテカルロ
共通: Vantage CFD（往復0.05%、買いのスワップはUTC21時をまたぐごと）、1R＝ATR1本分＝資金の0.5%（単利）、含み損込みの毎日の損益。
      値幅ブレイクは「買い×1＋売り×0.5、5日平均の向き、k=0.8、翌朝9時に決済」（A01）が基準。
結果は results.csv（グループ「計画・第1回」、計画IDつき）と plan.csv（状態・結果）に書く。
使い方: python3 project/tools/round1_lab.py <キャッシュ（BTC_15m.csv, ETH_15m.csv）> [followup]
        followup: 良さそうな形の銘柄別・年ごと・改善の重ね合わせ
"""
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
import exit_combo_lab as EX  # noqa: E402
import more_methods_lab as MM  # noqa: E402
import plan_tool as PT  # noqa: E402
import portfolio_lab as PF  # noqa: E402
from registry import drop, metrics_row, record  # noqa: E402

COINS = ("BTC", "ETH")
SPLIT = PF.SPLIT
GROUP = "計画・第1回"
P = 96
LAST = P - P // 24                       # 最後の1時間には入らない
ROLL = EX.ROLL_UTC * P // 24
YEARS = (MM.END - MM.START).days / 365.25
RESULTS = {}                             # 計画ID → plan.csv に書く結果


# ---------------------------------------------------------------- エンジン
@njit
def vbo_engine(O, H, L, C, trig, ok, A, start, side, late_mode, slip, cost, swd, carry_max):
    """1日ごとに trig に逆指値（start 本目から有効）。約定したら日の終わりに決済。
    carry_max>0 なら、日の終わりに含み益があれば最大 carry_max 日まで持ち越す。1R＝A（ATR）＝資金の0.5%。"""
    nd = trig.shape[0]
    N = nd * P
    pnl = np.zeros(nd)
    te = np.empty(nd, np.int64)
    tr = np.empty(nd)
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
        for j in range(s0, b0 + LAST):
            if j == s0 and side * (O[j] - trig[d]) > 0:   # 注文を出す時点で、もう逆指値を越えている
                if late_mode == 1:
                    k = j
                    e = O[j]
                break
            if (side == 1 and H[j] >= trig[d]) or (side == -1 and L[j] <= trig[d]):
                k = j
                e = max(O[j], trig[d]) if side == 1 else min(O[j], trig[d])
                break
        if k < 0:
            d += 1
            continue
        e = e * (1 + side * slip)
        Ad = A[d]
        cR = cost * e / Ad
        sR = swd * e / Ad
        pnl[d] -= cR * 0.5
        mark = 0.0
        nroll = 0
        carried = 0
        x = 0.0
        out = -1
        j = k
        while j < N:
            dd = j // P
            pos = j - dd * P
            if j > k and pos == ROLL:
                nroll += 1
                pnl[dd] -= sR * 0.5
            if pos == P - 1:
                m = side * (C[j] - e) / Ad
                if carry_max > 0 and m > 0 and carried < carry_max:
                    pnl[dd] += (m - mark) * 0.5
                    mark = m
                    carried += 1
                else:
                    x = C[j]
                    out = dd
                    break
            j += 1
        if out < 0:
            break
        rx = side * (x - e) / Ad
        pnl[out] += (rx - mark) * 0.5
        te[nt] = d
        tr[nt] = rx - cR - nroll * sR
        nt += 1
        d = out + 1
    return pnl, te[:nt], tr[:nt]


@njit
def oco_engine(O, H, L, C, tl, ts, ok, A, wl, ws, cost, swl, sws):
    """上下に逆指値。先に約定した方だけ持ち、日の終わりに決済。同じ15分足で両方に届いた日は見送る。"""
    nd = tl.shape[0]
    pnl = np.zeros(nd)
    te = np.empty(nd, np.int64)
    tr = np.empty(nd)
    nt = 0
    for d in range(nd):
        if not ok[d] or not (A[d] > 0) or np.isnan(tl[d]):
            continue
        b0 = d * P
        k = -1
        side = 0
        e = 0.0
        for j in range(b0, b0 + LAST):
            up = H[j] >= tl[d]
            dn = L[j] <= ts[d]
            if up and dn:
                break
            if up:
                k, side, e = j, 1, max(O[j], tl[d])
                break
            if dn:
                k, side, e = j, -1, min(O[j], ts[d])
                break
        if k < 0:
            continue
        Ad = A[d]
        w = wl if side == 1 else ws
        swd = swl if side == 1 else sws
        x = C[b0 + P - 1]
        R = side * (x - e) / Ad - cost * e / Ad - (swd * e / Ad if (k - b0) < ROLL else 0.0)
        pnl[d] += R * 0.5 * w
        te[nt] = d
        tr[nt] = R
        nt += 1
    return pnl, te[:nt], tr[:nt]


@njit
def swing_engine(O, H, L, C, atr, sig, xsig, side, stop_k, maxd, cost, swd):
    """前日の終値で sig なら翌朝の始値で入る。損切りATR×stop_k、xsig か maxd 日で翌朝決済。"""
    nd = sig.shape[0]
    N = nd * P
    pnl = np.zeros(nd)
    te = np.empty(nd, np.int64)
    tr = np.empty(nd)
    nt = 0
    d = 1
    while d < nd:
        Ad = atr[d]
        if not sig[d - 1] or not (Ad > 0):
            d += 1
            continue
        k = d * P
        e = O[k]
        stop = e - side * stop_k * Ad
        cR = cost * e / Ad
        sR = swd * e / Ad
        pnl[d] -= cR * 0.5
        mark = 0.0
        held = 0
        nroll = 0
        x = 0.0
        out = -1
        j = k
        while j < N:
            dd = j // P
            pos = j - dd * P
            if j > k and pos == 0:
                held += 1
                if xsig[dd - 1] or held >= maxd:
                    x = O[j]
                    out = dd
                    break
            if j > k and pos == ROLL:
                nroll += 1
                pnl[dd] -= sR * 0.5
            adv = L[j] if side == 1 else H[j]
            if side * (adv - stop) <= 0:
                x = stop if j == k else (min(O[j], stop) if side == 1 else max(O[j], stop))
                out = dd
                break
            if pos == P - 1:
                m = side * (C[j] - e) / Ad
                pnl[dd] += (m - mark) * 0.5
                mark = m
            j += 1
        if out < 0:
            break
        rx = side * (x - e) / Ad
        pnl[out] += (rx - mark) * 0.5
        te[nt] = d
        tr[nt] = rx - cR - nroll * sR
        nt += 1
        d = out + 1
    return pnl, te[:nt], tr[:nt]


# ---------------------------------------------------------------- 下ごしらえ
def prep(cache):
    D = {}
    for a in COINS:
        x = EX.day_matrix_n(os.path.join(cache, f"{a}_15m.csv"), P)
        M = x["M"]
        x.update(O=np.ascontiguousarray(M["open"].ravel()), H=np.ascontiguousarray(M["high"].ravel()),
                 L=np.ascontiguousarray(M["low"].ravel()), C=np.ascontiguousarray(M["close"].ravel()),
                 dH=M["high"].max(1), dL=M["low"].min(1), dC=M["close"][:, -1], idx=pd.DatetimeIndex(x["days"]))
        x["prevC"] = np.r_[np.nan, x["dC"][:-1]]
        D[a] = x
    return D


def sma_prev(x, n):
    """前日までの終値のn日平均（その日の朝にわかる値）。"""
    return pd.Series(x["dC"]).rolling(n).mean().shift(1).to_numpy()


def vbo(D, k=0.8, ma=5, rng="prev", extra_ok=None, start=None, late_mode=0, slip=0.0, cost=None, carry=0,
        trig_fn=None, short_w=0.5, sides=(1, -1)):
    """値幅ブレイク（A01の形）を条件を変えて回す。銘柄ごとの毎日の損益（資金の%）と取引一覧。"""
    out, trades = {}, []
    for a in COINS:
        x = D[a]
        nd = len(x["dO"])
        base = x["rng"] if rng == "prev" else x["atr"]
        mav = sma_prev(x, ma) if ma else None
        tot = np.zeros(nd)
        for side, w in ((1, 1.0), (-1, short_w)):
            if side not in sides or w == 0:
                continue
            trig = trig_fn(x, side) if trig_fn else x["dO"] + side * k * base
            with np.errstate(invalid="ignore"):
                ok = ~np.isnan(x["atr"]) & ~np.isnan(base)
                if mav is not None:
                    ok &= side * (x["dO"] - mav) > 0
            if extra_ok is not None:
                ok &= extra_ok(x, side)
            st = np.zeros(nd, np.int64) if start is None else start(x) if callable(start) else np.full(nd, start, np.int64)
            c = PF.SPEC[a][0] if cost is None else cost
            swd = (PF.SPEC[a][1] if side == 1 else PF.SPEC[a][2]) / 36500
            pnl, te, tr = vbo_engine(x["O"], x["H"], x["L"], x["C"], trig.astype(np.float64), ok.astype(np.bool_), x["atr"],
                                     st, side, late_mode, slip, c, swd, carry)
            tot += w * pnl
            trades.append(pd.DataFrame({"coin": a, "day": x["idx"][te], "R": tr, "side": side}))
        out[a] = pd.Series(tot, index=x["idx"])
    T = pd.concat(trades, ignore_index=True)
    return out, T[(T.day >= MM.START) & (T.day <= MM.END)]


def total(per):
    return MM.clip(per["BTC"].add(per["ETH"], fill_value=0))


def stats(x, T=None):
    """記録用の数字と、判定に使う数字。"""
    n = len(T) / YEARS if T is not None else np.nan
    r = metrics_row(x, n)
    h1, h2 = metrics_row(x[x.index < SPLIT]), metrics_row(x[x.index >= SPLIT])
    r["_h1s"], r["_h2s"], r["_n"] = h1["安定度"], h2["安定度"], n
    r["_wr"] = (T.R > 0).mean() * 100 if T is not None and len(T) else np.nan
    return r


def line(label, r, extra=""):
    print(f"    {label:30s} 年{r['年平均%']:+6.2f}% 落込{r['最大の落ち込み%']:5.1f}% 安定度{r['安定度']:5.2f}"
          f"（前半{r['_h1s']:5.2f}・後半{r['_h2s']:5.2f}） 負け年{r['マイナスの年']:14s} 後半{r['後半の年平均%']:+5.1f}%"
          + (f" 年{r['_n']:4.0f}回" if r["_n"] == r["_n"] else "") + extra)


def rec(pid, name, r, memo="", tf="15分足"):
    record(グループ=GROUP, 計画ID=pid, 名前=f"{pid} {name}", 銘柄="BTC,ETH", 時間足=tf, メモ=memo,
           **{k: v for k, v in r.items() if not k.startswith("_")})


def judge(r, coin_rs, grid_rs=None):
    """計画書の1次・2次の判定。"""
    s1 = r["年平均%"] > 0 and r["前半の年平均%"] > 0 and r["後半の年平均%"] > 0 and (r["_n"] != r["_n"] or r["_n"] >= 20)
    neg = 0 if r["マイナスの年"] == "なし" else len(str(r["マイナスの年"]).split(","))
    pos_frac = np.mean([g["年平均%"] > 0 for g in grid_rs]) if grid_rs else 1.0
    s2 = s1 and pos_frac >= 0.7 and all(c["年平均%"] > 0 for c in coin_rs) and r["安定度"] >= 0.8 and neg <= 2
    return s1, s2, pos_frac


def mark(s1, s2):
    return "2次合格" if s2 else ("1次合格・2次不合格" if s1 else "1次不合格")


def compare(r, base):
    """A01（基準）と比べる：安定度が全体・後半とも上なら「改善」。"""
    d_all, d_h2 = r["安定度"] - base["安定度"], r["_h2s"] - base["_h2s"]
    if d_all > 0.03 and d_h2 > 0.03:
        return "A01より良い"
    if d_all < -0.03 and d_h2 < -0.03:
        return "A01より悪い"
    return "A01と同じくらい"


def per_coin(per):
    return [metrics_row(MM.clip(per[a])) for a in COINS]


# ---------------------------------------------------------------- 本体
def main(cache):
    print(f"（古い「{GROUP}」の行を {drop(GROUP)} 行消して、計算し直す）")
    D = prep(cache)

    print("==== 基準 A01：値幅ブレイク（k=0.8・5日平均・買い×1＋売り×0.5）")
    per0, T0 = vbo(D)
    v0 = total(per0)
    base = stats(v0, T0)
    line("A01", base)
    print(f"    （exit_combo_lab の結果 年+8.51% と同じになるかの確認）")

    # ---- A02 kの細かいずらし
    print("\n==== A02 kの細かいずらし")
    kgrid = [round(0.3 + 0.1 * i, 1) for i in range(10)]
    kser, krs = {}, {}
    for k in kgrid:
        per, T = vbo(D, k=k)
        kser[k] = total(per)
        krs[k] = stats(kser[k], T)
        line(f"k={k}", krs[k])
        rec("A02", f"k={k}", krs[k])
    best = max(kgrid, key=lambda k: krs[k]["_h1s"])
    s1, s2, pf = judge(krs[0.8], per_coin(per0), list(krs.values()))
    nolose = sum(1 for k in kgrid if krs[k]["マイナスの年"] == "なし")
    RESULTS["A02"] = (f"k=0.3〜1.2の10通り中{sum(krs[k]['年平均%'] > 0 for k in kgrid)}通りがプラス、{nolose}通りが負け年なし。"
                      f"前半で一番良いk={best}は後半の安定度{krs[best]['_h2s']:.2f}（k=0.8は{base['_h2s']:.2f}）。{mark(s1, s2)}")

    # ---- A03 値幅をATRで測る
    print("\n==== A03 値幅をATRで測る（始値±ATR×k）")
    ars = {}
    for k in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0):
        per, T = vbo(D, k=k, rng="atr")
        ars[k] = (stats(total(per), T), per)
        line(f"ATR×{k}", ars[k][0])
        rec("A03", f"ATR×{k}", ars[k][0])
    bk = max(ars, key=lambda k: ars[k][0]["_h1s"])
    r, per = ars[bk]
    s1, s2, pf = judge(r, per_coin(per), [v[0] for v in ars.values()])
    RESULTS["A03"] = (f"7通り中{sum(v[0]['年平均%'] > 0 for v in ars.values())}通りがプラス。前半で一番良いATR×{bk}：年{r['年平均%']:+.1f}%・"
                      f"落ち込み{r['最大の落ち込み%']:.1f}%・後半の安定度{r['_h2s']:.2f}。{compare(r, base)}。{mark(s1, s2)}")

    # ---- A07 向きの判定の日数
    print("\n==== A07 向きの判定に使う平均の日数")
    mrs = {}
    for n in (3, 5, 10, 20, 50):
        per, T = vbo(D, ma=n)
        mrs[n] = (stats(total(per), T), per)
        line(f"{n}日平均", mrs[n][0])
        rec("A07", f"{n}日平均の向き", mrs[n][0])
    bn = max(mrs, key=lambda n: mrs[n][0]["_h1s"])
    r, per = mrs[bn]
    s1, s2, pf = judge(r, per_coin(per), [v[0] for v in mrs.values()])
    RESULTS["A07"] = (f"3・5・10・20・50日を比べ、前半で一番良いのは{bn}日（後半の安定度{r['_h2s']:.2f}、5日は{base['_h2s']:.2f}）。"
                      f"10日以上は{'負け年が出る' if any(mrs[n][0]['マイナスの年'] != 'なし' for n in (10, 20, 50)) else '負け年なし'}。"
                      f"{compare(r, base) if bn != 5 else '5日のままでよい'}。{mark(s1, s2)}")

    # ---- A08 向きの判定なし
    print("\n==== A08 向きの判定なし（上下に逆指値）")
    a08 = {}
    for k in (0.6, 0.8, 1.0):
        per = {}
        ts = []
        for a in COINS:
            x = D[a]
            ok = (~np.isnan(x["atr"]) & ~np.isnan(x["rng"]))
            pnl, te, tr = oco_engine(x["O"], x["H"], x["L"], x["C"], x["dO"] + k * x["rng"], x["dO"] - k * x["rng"], ok,
                                     x["atr"], 1.0, 0.5, PF.SPEC[a][0], PF.SPEC[a][1] / 36500, PF.SPEC[a][2] / 36500)
            per[a] = pd.Series(pnl, index=x["idx"])
            ts.append(pd.DataFrame({"day": x["idx"][te], "R": tr}))
        T = pd.concat(ts)
        T = T[(T.day >= MM.START) & (T.day <= MM.END)]
        r = stats(total(per), T)
        line(f"先に約定した方だけ k={k}", r)
        rec("A08", f"先に約定した方だけ（OCO）k={k}", r)
        per2, T2 = vbo(D, k=k, ma=None)
        r2 = stats(total(per2), T2)
        line(f"上下とも置く k={k}", r2)
        rec("A08", f"上下とも置く k={k}", r2)
        a08[k] = (r, r2)
    r, r2 = a08[0.8]
    RESULTS["A08"] = (f"k=0.8で、先に約定した方だけ：年{r['年平均%']:+.1f}%・落ち込み{r['最大の落ち込み%']:.1f}%・安定度{r['安定度']:.2f}、"
                      f"上下とも：年{r2['年平均%']:+.1f}%・安定度{r2['安定度']:.2f}（A01は{base['安定度']:.2f}）。"
                      f"{'向きの判定はあった方が良い' if max(r['安定度'], r2['安定度']) < base['安定度'] else '向きの判定なしでも同じくらい'}")

    # ---- A16 NR4・NR7
    print("\n==== A16 前日の値幅が直近N日で一番小さい日（NR4・NR7）")
    a16 = {}
    for n in (4, 7):
        def nr(x, side, n=n):
            rr = pd.Series(x["rng"])
            return (rr <= rr.rolling(n).min()).to_numpy()
        def nn(x, side, n=n):
            rr = pd.Series(x["rng"])
            return ~(rr <= rr.rolling(n).min()).to_numpy()
        per, T = vbo(D, extra_ok=nr)
        r = stats(total(per), T)
        line(f"NR{n}の翌日だけ", r)
        rec("A16", f"NR{n}の翌日だけ", r)
        per2, T2 = vbo(D, extra_ok=nn)
        r2 = stats(total(per2), T2)
        line(f"NR{n}以外の日だけ", r2)
        rec("A16", f"NR{n}以外の日だけ", r2)
        a16[n] = (r, r2, per)
    r, r2, per = a16[7]
    s1, s2, pf = judge(r, per_coin(per))
    RESULTS["A16"] = (f"NR7の翌日だけ：年{r['年平均%']:+.1f}%・年{r['_n']:.0f}回・安定度{r['安定度']:.2f}、NR4：年{a16[4][0]['年平均%']:+.1f}%・"
                      f"安定度{a16[4][0]['安定度']:.2f}。1回あたりの成績は{'良い' if r['年平均%'] / max(r['_n'], 1) > base['年平均%'] / base['_n'] else '変わらない'}が、"
                      f"回数が減る。{mark(s1, s2)}")

    # ---- A31 寄り付き後の値幅
    print("\n==== A31 9時からN分の高値・安値を抜けたら（オープニング・レンジ・ブレイク）")
    a31 = {}
    for mins in (15, 30, 60):
        nb = mins // 15
        def orb(x, side, nb=nb):
            M = x["M"]
            return M["high"][:, :nb].max(1) if side == 1 else M["low"][:, :nb].min(1)
        for filt in (5, None):
            per, T = vbo(D, ma=filt, trig_fn=orb, start=nb)
            r = stats(total(per), T)
            lab = f"{mins}分・{'5日平均の向き' if filt else '向きなし'}"
            line(lab, r)
            rec("A31", lab, r)
            a31[(mins, filt)] = (r, per)
    bkey = max(a31, key=lambda k: a31[k][0]["_h1s"])
    r, per = a31[bkey]
    s1, s2, pf = judge(r, per_coin(per), [v[0] for v in a31.values()])
    RESULTS["A31"] = (f"6通り中{sum(v[0]['年平均%'] > 0 for v in a31.values())}通りがプラス。前半で一番良い{bkey[0]}分・"
                      f"{'向きあり' if bkey[1] else '向きなし'}：年{r['年平均%']:+.1f}%・落ち込み{r['最大の落ち込み%']:.1f}%・安定度{r['安定度']:.2f}・"
                      f"年{r['_n']:.0f}回。{compare(r, base)}。{mark(s1, s2)}")

    # ---- K10 勝っている時だけ持ち越す
    print("\n==== K10 翌朝に含み益なら持ち越す（最大N日）")
    k10 = {}
    for c in (1, 2, 3):
        per, T = vbo(D, carry=c)
        k10[c] = (stats(total(per), T), per)
        line(f"最大{c}日持ち越す", k10[c][0])
        rec("K10", f"含み益なら最大{c}日持ち越す", k10[c][0])
    r = k10[1][0]
    RESULTS["K10"] = (f"1日：年{r['年平均%']:+.1f}%・落ち込み{r['最大の落ち込み%']:.1f}%・安定度{r['安定度']:.2f}、"
                      f"3日：年{k10[3][0]['年平均%']:+.1f}%・安定度{k10[3][0]['安定度']:.2f}（A01は年{base['年平均%']:+.1f}%・{base['安定度']:.2f}）。"
                      f"{compare(r, base)}")

    # ---- C24・C25 モメンタム、E01 スクイーズ（新しい部品）
    def pos_daily(x, a, pos):
        held = np.r_[0.0, pos[:-1]]
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.nan_to_num(0.5 / x["atr"])
        dpx = np.nan_to_num(x["dC"] - x["prevC"])
        change = np.abs(np.diff(np.r_[0.0, held]))
        swd = np.where(held > 0, PF.SPEC[a][1], PF.SPEC[a][2]) / 36500
        pnl = held * dpx * scale - change * PF.SPEC[a][0] / 2 * x["dO"] * scale - np.abs(held) * swd * x["dO"] * scale
        seg = np.cumsum(change > 0)
        on = held != 0
        T = pd.DataFrame({"seg": seg[on], "p": pnl[on], "day": x["idx"][on]}).groupby("seg").agg(R=("p", "sum"), day=("day", "first"))
        return pd.Series(pnl, index=x["idx"]), T

    def component(pid, title, make, settings, label):
        print(f"\n==== {pid} {title}")
        res = {}
        for sset in settings:
            per, Ts = {}, []
            for a in COINS:
                p, T = make(D[a], a, sset)
                per[a] = p
                Ts.append(T)
            T = pd.concat(Ts)
            T = T[(T.day >= MM.START) & (T.day <= MM.END)]
            x = total(per)
            r = stats(x, T)
            line(label(sset), r)
            rec(pid, label(sset), r, tf="日足（約定は15分足）")
            res[sset] = (r, per, x)
        bset = max(res, key=lambda k: res[k][0]["_h1s"])
        r, per, x = res[bset]
        s1, s2, pf = judge(r, per_coin(per), [v[0] for v in res.values()])
        corr = v0.resample("ME").sum().corr(x.resample("ME").sum())
        mix = stats(v0.add(0.5 * x, fill_value=0))
        s3 = s2 and corr < 0.5 and mix["安定度"] > base["安定度"]
        line(f"値幅ブレイク＋{label(bset)}×0.5", mix, f" ｜月の相関{corr:+.2f}")
        rec(pid, f"値幅ブレイク＋{label(bset)}×0.5", mix, memo=f"月の相関{corr:+.2f}", tf="日足（約定は15分足）")
        RESULTS[pid] = (f"{len(res)}通り中{sum(v[0]['年平均%'] > 0 for v in res.values())}通りがプラス。前半で一番良い{label(bset)}：年{r['年平均%']:+.1f}%・"
                        f"落ち込み{r['最大の落ち込み%']:.1f}%・負け年{r['マイナスの年']}・後半{r['後半の年平均%']:+.1f}%。"
                        f"値幅ブレイクとの月の相関{corr:+.2f}、×0.5で足すと安定度{mix['安定度']:.2f}（値幅ブレイクだけ{base['安定度']:.2f}）。"
                        f"{'3次合格' if s3 else mark(s1, s2) + ('・3次不合格' if s2 else '')}")

    def tsmom(x, a, sset):
        n, longonly = sset
        mom = x["dC"] - np.r_[np.full(n, np.nan), x["dC"][:-n]]
        pos = np.nan_to_num(np.sign(mom))
        if longonly:
            pos = np.clip(pos, 0, None)
        return pos_daily(x, a, pos)
    component("C24", "時系列モメンタム（過去N日の騰落の向きに持つ）", tsmom,
              [(n, lo) for lo in (False, True) for n in (7, 14, 30, 60, 90)],
              lambda s: f"{s[0]}日{'・買いだけ' if s[1] else '・買い＋売り'}")

    def multimom(x, a, sset):
        longonly = sset
        votes = sum(np.nan_to_num(np.sign(x["dC"] - np.r_[np.full(n, np.nan), x["dC"][:-n]])) for n in (7, 30, 90))
        pos = np.sign(votes)
        if longonly:
            pos = np.clip(pos, 0, None)
        return pos_daily(x, a, pos)
    component("C25", "複数期間のモメンタム（7・30・90日の多数決）", multimom, [False, True],
              lambda s: "7・30・90日の多数決" + ("・買いだけ" if s else "・買い＋売り"))

    def squeeze(x, a, sset):
        n, win = sset
        s = pd.Series(x["dC"])
        m20, sd = s.rolling(20).mean(), s.rolling(20).std()
        width = (4 * sd / m20)
        sq = (width <= width.rolling(n).min()).to_numpy()
        armed = pd.Series(sq).rolling(win, min_periods=1).max().fillna(0).to_numpy().astype(bool)
        tot, Ts = np.zeros(len(s)), []
        for side in (1, -1):
            with np.errstate(invalid="ignore"):
                sig = armed & (side * (s - (m20 + side * 2 * sd)) > 0).to_numpy()
                xs = (side * (s - m20) < 0).to_numpy()
            swd = (PF.SPEC[a][1] if side == 1 else PF.SPEC[a][2]) / 36500
            pnl, te, tr = swing_engine(x["O"], x["H"], x["L"], x["C"], x["atr"], sig, xs, side, 2.0, 20, PF.SPEC[a][0], swd)
            tot += pnl
            Ts.append(pd.DataFrame({"day": x["idx"][te], "R": tr}))
        return pd.Series(tot, index=x["idx"]), pd.concat(Ts)
    component("E01", "スクイーズのブレイク（バンドの幅が過去N日で最小→N日以内に2σの外で引けたら乗る）", squeeze,
              [(n, w) for n in (60, 120, 250) for w in (5, 10)], lambda s: f"{s[0]}日で最小・{s[1]}日以内")

    # ---- 量の調整（E05・E06・E07・L08・L09）
    def weighted(wfn):
        out = {}
        for a in COINS:
            w = wfn(D[a], a)
            out[a] = per0[a] * w
        return total(out)

    def sizing(pid, title, rules):
        print(f"\n==== {pid} {title}")
        res = {}
        for lab, fn in rules:
            x = weighted(fn)
            r = stats(x)
            line(lab, r)
            rec(pid, lab, r)
            res[lab] = r
        return res

    def quiet_by(series_fn):
        def make(qs, vs):
            def fn(x, a):
                q = series_fn(x)
                return np.where(np.isnan(q), 1.0, np.where(q > 0, qs, vs))
            return fn
        return make

    def rv_quiet(n):
        def f(x):
            lr = np.log(pd.Series(x["dC"])).diff()
            rv = lr.rolling(n).std().shift(1)
            med = rv.rolling(365, min_periods=200).median()
            return np.where(med.isna(), np.nan, (rv < med).astype(float) * 2 - 1)
        return f
    e05 = sizing("E05", "静かの判定を実現ボラ（N日）で", [
        (f"{n}日の値動き：{lab}", quiet_by(rv_quiet(n))(qs, vs)) for n in (10, 20, 60) for lab, qs, vs in
        (("静かな日だけ", 1.0, 0.0), ("静か×1.5・荒い×0.5", 1.5, 0.5))])

    def atr_quiet(x):
        atrp = pd.Series(x["atr"] / x["dO"])
        med = atrp.shift(1).rolling(365, min_periods=200).median()
        return np.where(med.isna(), np.nan, (atrp < med).astype(float) * 2 - 1)

    def week_quiet(x):
        df = pd.DataFrame({"h": x["dH"], "l": x["dL"], "c": x["dC"]}, index=x["idx"])
        wk = df.resample("W-SUN").agg({"h": "max", "l": "min", "c": "last"})
        pc = wk.c.shift(1)
        tr = pd.concat([wk.h - wk.l, (wk.h - pc).abs(), (wk.l - pc).abs()], axis=1).max(axis=1)
        atrp = tr.rolling(14).mean() / wk.c
        q = (atrp < atrp.rolling(52, min_periods=30).median()).astype(float) * 2 - 1
        q[atrp.rolling(52, min_periods=30).median().isna()] = np.nan
        q = q.shift(1)                                       # 終わった週の値を次の週に使う
        return q.reindex(x["idx"], method="ffill").to_numpy()
    e06 = sizing("E06", "静かの判定を週足のATRで", [
        (f"週足：{lab}", quiet_by(week_quiet)(qs, vs)) for lab, qs, vs in (("静かな日だけ", 1.0, 0.0), ("静か×1.5・荒い×0.5", 1.5, 0.5))]
        + [(f"参考（日足ATR・E04）：静か×1.5・荒い×0.5", quiet_by(atr_quiet)(1.5, 0.5))])

    def pct_w(levels):
        def fn(x, a):
            atrp = pd.Series(x["atr"] / x["dO"])
            pct = atrp.rolling(365, min_periods=200).apply(lambda v: (v[:-1] < v[-1]).mean(), raw=True).to_numpy()
            if levels == "連続":
                w = 1.5 - pct
            else:
                cuts = np.linspace(0, 1, levels + 1)[1:-1]
                ws = np.linspace(1.5, 0.5, levels) if levels == 3 else np.array([1.6, 1.3, 1.0, 0.7, 0.4])
                w = ws[np.searchsorted(cuts, np.nan_to_num(pct, nan=0.5))]
            return np.where(np.isnan(pct), 1.0, w)
        return fn
    e07 = sizing("E07", "ATRの順位（過去1年）で量を段階的に変える", [
        ("3段階（1.5・1.0・0.5）", pct_w(3)), ("5段階（1.6〜0.4）", pct_w(5)), ("連続（1.5−順位）", pct_w("連続"))])

    shadow = v0.cumsum()
    peak = shadow.cummax()
    def eq_rule(wser):
        return stats(v0 * wser.shift(1).fillna(1.0))
    print("\n==== L08 資金（値幅ブレイクの損益の累計）が高値から下がったら量を半分")
    l08 = {}
    for th in (1, 2, 3, 4):
        r = eq_rule(pd.Series(np.where(peak - shadow >= th, 0.5, 1.0), index=v0.index))
        line(f"高値から{th}%下がったら半分", r)
        rec("L08", f"高値から{th}%下がったら量を半分", r)
        l08[th] = r
    print("\n==== L09 資金曲線がN日平均の下なら休む（または半分）")
    l09 = {}
    for n in (10, 20, 40, 60):
        ma = shadow.rolling(n).mean()
        for lab, low in (("休む", 0.0), ("半分", 0.5)):
            r = eq_rule(pd.Series(np.where(shadow < ma, low, 1.0), index=v0.index))
            line(f"{n}日平均の下で{lab}", r)
            rec("L09", f"資金曲線が{n}日平均の下で{lab}", r)
            l09[(n, lab)] = r

    def size_result(pid, res, ref_note):
        bl = max(res, key=lambda k: res[k]["_h1s"])
        r = res[bl]
        better = sum(1 for v in res.values() if v["_h2s"] > base["_h2s"])
        RESULTS[pid] = (f"{len(res)}通り中{better}通りで後半の安定度がA01（{base['_h2s']:.2f}）より上。前半で一番良い「{bl}」：年{r['年平均%']:+.1f}%・"
                        f"落ち込み{r['最大の落ち込み%']:.1f}%・安定度{r['安定度']:.2f}・後半の安定度{r['_h2s']:.2f}。{ref_note}")
    size_result("E05", e05, "日足ATRで判定するE04（後半の安定度1.32）と比べる")
    size_result("E06", e06, "日足ATRで判定するE04と比べる")
    size_result("E07", e07, "2段階のE04（静か×1.5・荒い×0.5）と比べる")
    size_result("L08", l08, "")
    size_result("L09", l09, "")

    # ---- N10 ウォーク・フォワード
    print("\n==== N10 毎年、過去N年の安定度が一番良いkを選び直す（k=0.5〜1.1）")
    ks = [k for k in kgrid if 0.5 <= k <= 1.1]
    n10 = {}
    for n in (2, 3, 4):
        parts, picks = [], []
        for y in range(2018 + n, 2027):
            lo, hi = pd.Timestamp(f"{y - n}-01-01"), pd.Timestamp(f"{y}-01-01")
            score = {k: metrics_row(kser[k][(kser[k].index >= lo) & (kser[k].index < hi)])["安定度"] for k in ks}
            kb = max(score, key=score.get)
            picks.append(f"{y}:{kb}")
            parts.append(kser[kb][(kser[kb].index >= hi) & (kser[kb].index < pd.Timestamp(f"{y + 1}-01-01"))])
        x = pd.concat(parts)
        fixed = kser[0.8][kser[0.8].index >= x.index[0]]
        r, rf = stats(x), stats(fixed)
        line(f"過去{n}年で選ぶ（{x.index[0]:%Y}年〜）", r, " ｜選んだk " + " ".join(picks))
        line(f"同じ期間のk=0.8固定", rf)
        rec("N10", f"過去{n}年でkを選び直す", r, memo="選んだk " + " ".join(picks))
        n10[n] = (r, rf)
    r, rf = n10[3]
    RESULTS["N10"] = (f"過去3年で選び直す形（2021年〜）：年{r['年平均%']:+.1f}%・安定度{r['安定度']:.2f}。同じ期間のk=0.8固定は年{rf['年平均%']:+.1f}%・"
                      f"安定度{rf['安定度']:.2f}。{'選び直す方が良い' if r['安定度'] > rf['安定度'] + 0.03 else '選び直しても良くならない'}")

    # ---- N15 ランダムな売買との比較
    print("\n==== N15 ランダムな売買との比較（1,000回）")
    rng_ = np.random.default_rng(1)
    sims = {"時刻だけランダム（向きは5日平均）": np.zeros(1000), "時刻も向きもランダム": np.zeros(1000)}
    for a in COINS:
        x = D[a]
        mav = sma_prev(x, 5)
        m = (x["idx"] >= MM.START) & (x["idx"] <= MM.END)
        Mx = x["M"]
        for side, w in ((1, 1.0), (-1, 0.5)):
            with np.errstate(invalid="ignore"):
                ok = (side * (x["dO"] - mav) > 0) & ~np.isnan(x["atr"]) & m
            n_tr = int(((T0.coin == a) & (T0.side == side)).sum())
            p = n_tr / ok.sum()
            days = np.flatnonzero(ok)
            A = x["atr"][days]
            for key in sims:
                take = rng_.random((1000, len(days))) < p
                bar = rng_.integers(0, LAST, (1000, len(days)))
                sd = side if key.startswith("時刻だけ") else np.where(rng_.random((1000, len(days))) < 0.5, 1, -1)
                ww = w if key.startswith("時刻だけ") else np.where(sd == 1, 1.0, 0.5)
                e = Mx["open"][days][np.arange(len(days)), bar] if False else Mx["open"][days[None, :], bar]
                xx = Mx["close"][days, -1][None, :]
                swa = np.where(sd == 1, PF.SPEC[a][1], PF.SPEC[a][2]) / 36500
                R = sd * (xx - e) / A - PF.SPEC[a][0] * e / A - np.where(bar < ROLL, swa * e / A, 0.0)
                sims[key] += (np.where(take, R, 0.0) * 0.5 * ww).sum(axis=1) / YEARS
    act = base["年平均%"]
    n15 = []
    for key, s in sims.items():
        q = (s >= act).mean() * 100
        print(f"    {key:24s} 年平均の分布：中央値{np.median(s):+5.2f}%・上位5%{np.percentile(s, 95):+5.2f}%・一番良い{s.max():+5.2f}%"
              f" ｜A01（{act:+.2f}%）以上になった割合 {q:.1f}%")
        n15.append(f"{key}：中央値{np.median(s):+.1f}%、A01以上は{q:.1f}%")
    RESULTS["N15"] = "ランダムな売買1,000回。" + "／".join(n15) + "。値幅ブレイクの成績は偶然では説明できない" if max(
        (s >= act).mean() for s in sims.values()) < 0.05 else "ランダムでも同じくらいの成績が出る"

    # ---- P01・P03・P04 実務の確認
    print("\n==== P01 手数料を上げる（往復）")
    p01 = {}
    for c in (0.0005, 0.001, 0.002):
        per, T = vbo(D, cost=c)
        p01[c] = stats(total(per), T)
        line(f"往復{c * 100:.2f}%", p01[c])
        rec("P01", f"往復{c * 100:.2f}%", p01[c])
    RESULTS["P01"] = (f"往復0.1%で年{p01[0.001]['年平均%']:+.1f}%（負け年{p01[0.001]['マイナスの年']}）、0.2%で年{p01[0.002]['年平均%']:+.1f}%"
                      f"（負け年{p01[0.002]['マイナスの年']}）。手数料が2倍になっても{'プラス' if p01[0.001]['年平均%'] > 0 else 'マイナス'}")
    print("\n==== P03 逆指値の滑り（約定が不利になる分）")
    p03 = {}
    for s in (0.0, 0.0005, 0.001):
        per, T = vbo(D, slip=s)
        p03[s] = stats(total(per), T)
        line(f"滑り{s * 100:.2f}%", p03[s])
        rec("P03", f"逆指値の滑り{s * 100:.2f}%", p03[s])
    RESULTS["P03"] = (f"滑り0.05%で年{p03[0.0005]['年平均%']:+.1f}%・安定度{p03[0.0005]['安定度']:.2f}、0.1%で年{p03[0.001]['年平均%']:+.1f}%・"
                      f"安定度{p03[0.001]['安定度']:.2f}（負け年{p03[0.001]['マイナスの年']}）")
    print("\n==== P04 注文が遅れる（9時ちょうどの代わりに）")
    p04 = {}
    for nb, lab in ((1, "9時15分"), (4, "10時")):
        for lm, how in ((0, "越えていたら見送る"), (1, "越えていたら成行で入る")):
            per, T = vbo(D, start=nb, late_mode=lm)
            r = stats(total(per), T)
            line(f"{lab}に注文・{how}", r)
            rec("P04", f"{lab}に注文・{how}", r)
            p04[(nb, lm)] = r
    RESULTS["P04"] = (f"9時15分に注文：年{p04[(1, 0)]['年平均%']:+.1f}%（見送る）・{p04[(1, 1)]['年平均%']:+.1f}%（成行）、10時：年{p04[(4, 0)]['年平均%']:+.1f}%・"
                      f"{p04[(4, 1)]['年平均%']:+.1f}%（A01は{base['年平均%']:+.1f}%）")

    # ---- P08 モンテカルロ
    print("\n==== P08 モンテカルロ（A01の毎日の損益）")
    vals = v0.to_numpy()
    rs = np.random.default_rng(2)
    dd = []
    for _ in range(10000):
        eq = np.cumsum(rs.permutation(vals))
        dd.append((np.maximum.accumulate(eq) - eq).max())
    dd = np.array(dd)
    blocks = len(vals) // 20
    yr, ydd = [], []
    for _ in range(10000):
        st = rs.integers(0, len(vals) - 20, 18)
        path = np.concatenate([vals[s:s + 20] for s in st])[:365]
        eq = np.cumsum(path)
        yr.append(eq[-1])
        ydd.append((np.maximum.accumulate(eq) - eq).max())
    yr, ydd = np.array(yr), np.array(ydd)
    print(f"    並べ替え1万回の最大の落ち込み：中央値{np.median(dd):.1f}%・95%{np.percentile(dd, 95):.1f}%・99%{np.percentile(dd, 99):.1f}%（実際は{base['最大の落ち込み%']:.1f}%）")
    print(f"    20日ずつ組み合わせた1年1万回：年がマイナスになる確率{(yr < 0).mean() * 100:.1f}%・下位5%の年{np.percentile(yr, 5):+.1f}%・"
          f"1年の落ち込みの95%{np.percentile(ydd, 95):.1f}%")
    RESULTS["P08"] = (f"並べ替え1万回の最大の落ち込みは中央値{np.median(dd):.1f}%・99%で{np.percentile(dd, 99):.1f}%（実際{base['最大の落ち込み%']:.1f}%）。"
                      f"1年を1万回作ると、マイナスの年になる確率{(yr < 0).mean() * 100:.1f}%・下位5%で年{np.percentile(yr, 5):+.1f}%")

    # ---- 計画書に書き込む
    rows = PT.load()
    for r in rows:
        if r["ID"] in RESULTS:
            r["状態"] = "済"
            r["結果"] = RESULTS[r["ID"]]
    PT.save(rows)
    PT.render(rows)
    print(f"\n計画書に{len(RESULTS)}件を書き込んだ")
    for k, v in RESULTS.items():
        print(f"  {k}: {v}")


if __name__ == "__main__" and len(sys.argv) == 2:
    main(sys.argv[1])


def followup(cache):
    """第1回の続き：良さそうな形を、銘柄別・年ごと・組み合わせで確かめる（記録は「計画・第1回」に追加）。"""
    D = prep(cache)
    per0, T0 = vbo(D)
    v0 = total(per0)
    base = stats(v0, T0)

    def show(pid, name, per, T=None, record_it=True):
        x = total(per)
        r = stats(x, T)
        cb, ce = (metrics_row(MM.clip(per[a])) for a in COINS)
        line(name, r, f" ｜BTCだけ{cb['年平均%']:+.1f}%（負け年{cb['マイナスの年']}） ETHだけ{ce['年平均%']:+.1f}%（負け年{ce['マイナスの年']}）")
        if record_it:
            rec(pid, name, r, memo=f"BTCだけ年{cb['年平均%']:+.1f}%・ETHだけ年{ce['年平均%']:+.1f}%")
        return x, r

    def oco(k=0.8, wl=1.0, ws=0.5, wfn=None):
        per, ts = {}, []
        for a in COINS:
            x = D[a]
            ok = (~np.isnan(x["atr"]) & ~np.isnan(x["rng"]))
            pnl, te, tr = oco_engine(x["O"], x["H"], x["L"], x["C"], x["dO"] + k * x["rng"], x["dO"] - k * x["rng"], ok,
                                     x["atr"], wl, ws, PF.SPEC[a][0], PF.SPEC[a][1] / 36500, PF.SPEC[a][2] / 36500)
            w = 1.0 if wfn is None else wfn(x)
            per[a] = pd.Series(pnl * w, index=x["idx"])
            ts.append(pd.DataFrame({"day": x["idx"][te], "R": tr}))
        T = pd.concat(ts)
        return per, T[(T.day >= MM.START) & (T.day <= MM.END)]

    def rvw(n, qs, vs):
        def fn(x):
            lr = np.log(pd.Series(x["dC"])).diff()
            rv = lr.rolling(n).std().shift(1)
            med = rv.rolling(365, min_periods=200).median()
            return np.where(med.isna(), 1.0, np.where(rv < med, qs, vs))
        return fn

    def pctw(x):
        atrp = pd.Series(x["atr"] / x["dO"])
        pct = atrp.rolling(365, min_periods=200).apply(lambda v: (v[:-1] < v[-1]).mean(), raw=True).to_numpy()
        w = np.array([1.5, 1.0, 0.5])[np.searchsorted([1 / 3, 2 / 3], np.nan_to_num(pct, nan=0.5))]
        return np.where(np.isnan(pct), 1.0, w)

    print("==== 確かめ：銘柄別")
    show("A01", "A01（基準）", per0, T0, record_it=False)
    per, T = oco()
    xo, ro = show("A08", "向きの判定なし（先に約定した方だけ）k=0.8", per, T)
    xq = {}
    for n in (10, 20):
        p = {a: per0[a] * rvw(n, 1.0, 0.0)(D[a]) for a in COINS}
        xq[n], _ = show("E05", f"{n}日の値動きで静かな日だけ", p)
    p = {a: per0[a] * pctw(D[a]) for a in COINS}
    xp, _ = show("E07", "ATRの順位で3段階（1.5・1.0・0.5）", p)

    print("\n==== 確かめ：改善を重ねる")
    per, T = oco(wfn=rvw(10, 1.0, 0.0))
    show("A08", "向きの判定なし＋10日の値動きで静かな日だけ", per, T)
    per, T = oco(wfn=pctw)
    show("A08", "向きの判定なし＋ATRの順位で3段階", per, T)
    p = {a: per0[a] * rvw(10, 1.0, 0.0)(D[a]) * 1.5 for a in COINS}
    show("E05", "10日の値動きで静かな日だけ×1.5（量を増やして年平均をそろえる）", p)

    print("\n==== 確かめ：年ごとの損益（%）")
    rows = {"A01": v0, "A08 向きなし": xo, "E05 静かな日だけ（10日）": xq[10], "E07 3段階": xp}
    print(pd.DataFrame({k: s.groupby(s.index.year).sum() for k, s in rows.items()}).round(1).T.to_string())

    print("\n==== 確かめ：C24 30日の時系列モメンタムを値幅ブレイクに足す")
    for n in (14, 30):
        per = {}
        for a in COINS:
            x = D[a]
            mom = x["dC"] - np.r_[np.full(n, np.nan), x["dC"][:-n]]
            held = np.r_[0.0, np.nan_to_num(np.sign(mom))[:-1]]
            with np.errstate(invalid="ignore", divide="ignore"):
                scale = np.nan_to_num(0.5 / x["atr"])
            change = np.abs(np.diff(np.r_[0.0, held]))
            swd = np.where(held > 0, PF.SPEC[a][1], PF.SPEC[a][2]) / 36500
            pnl = held * np.nan_to_num(x["dC"] - x["prevC"]) * scale - change * PF.SPEC[a][0] / 2 * x["dO"] * scale - np.abs(held) * swd * x["dO"] * scale
            per[a] = pd.Series(pnl, index=x["idx"])
        xm = total(per)
        corr = v0.resample("ME").sum().corr(xm.resample("ME").sum())
        for wgt in (0.25, 0.5):
            mix = stats(v0.add(wgt * xm, fill_value=0))
            line(f"値幅ブレイク＋{n}日モメンタム×{wgt}", mix, f" ｜月の相関{corr:+.2f}")
            rec("C24", f"値幅ブレイク＋{n}日・買い＋売り×{wgt}", mix, memo=f"月の相関{corr:+.2f}", tf="日足")


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "followup":
    followup(sys.argv[1])
