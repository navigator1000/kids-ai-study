#!/usr/bin/env python3
"""計画書（project/research/plan.md）第4回・前半：組み合わせ（M）と資金管理（L）（BTC・ETH、2018年〜）。

組み合わせの土台（ここまでの回で合格・改善した部品）:
  値幅ブレイクの向き : 自分の5日平均（A01）／ETHはBTCの5日平均（H05）／向きなし・先に約定した方（A08）
  量の調整         : なし／静かな日だけ（E05：10日の実現ボラが過去1年の中央値より小さい日）／ATRの順位で3段階（E07）
  約定の時間       : いつでも／日本時間5〜8時の約定は見送る（A18）
  足す部品         : C29 値動きで割ったモメンタム（z>1.0、買い＋売り）×0・0.25・0.5／F02 はらみ足のブレイク×0・0.5／
                     F08 三角持ち合いのブレイク×0・0.5
  → 3×3×2×3×2×2＝216通り。前半（2018〜2022年）だけで選び、後半（2023年〜）で確かめる。
M06 リスクパリティ／M07 落ち込みが最小の配分／M08 部品を増やす／M09 負ける年をなくす配分（いずれも前半で決める）／
M11 口座を分ける（値幅ブレイク＋現物の持ちっぱなし）／M12 静かな日は値幅ブレイク・荒い日はトレンド／
M13 手法の多数決で向きを決める／M14 値幅ブレイクと逆に動く部品を探す／M15 静か×1.5の値幅ブレイク＋9日EMA×0.5
L02 複利／L03 ケリー基準の一部／L04 固定比率／L05 連敗で量を半分／L10 1日の損失上限／L11 月の損失上限／
L12 年の目標で量を減らす／L13 前月の成績で量を変える／L14 最小ロットの丸め／L15 建玉の上限
共通: Vantage CFD（往復0.05%、買いのスワップ年25〜35%、売り0%）、1R＝ATR（日足）1本分＝資金の0.5%（単利）、含み損込み。
結果は results.csv（グループ「計画・第4回」、計画IDつき）と plan.csv。
使い方: python3 project/tools/round4_lab.py <キャッシュ（BTC_15m.csv, ETH_15m.csv, USDJPY.csv）> [followup]
        followup: 候補の形の年ごとの損益と、今の価格での0.01ロットへの丸めのずれ
"""
import itertools
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
import round2b_lab as R2B  # noqa: E402
import round3_lab as R3  # noqa: E402
from registry import drop, metrics_row, record  # noqa: E402

COINS = ("BTC", "ETH")
SPLIT = PF.SPLIT
GROUP = "計画・第4回"
DAY = 96
R3.GROUP = GROUP


@njit
def oco_last(O, H, L, C, tl, ts, ok, A, wl, ws, cost, swl, sws, last):
    """上下に逆指値、先に約定した方だけ持ち、日の終わりに決済（last 本目より後には入らない）。"""
    nd = tl.shape[0]
    pnl = np.zeros(nd)
    te = np.empty(nd, np.int64)
    tr = np.empty(nd)
    tsd = np.empty(nd, np.int64)
    nt = 0
    for d in range(nd):
        if not ok[d] or not (A[d] > 0) or np.isnan(tl[d]):
            continue
        b0 = d * DAY
        k = -1
        side = 0
        e = 0.0
        for j in range(b0, b0 + last):
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
        x = C[b0 + DAY - 1]
        R = side * (x - e) / Ad - cost * e / Ad - (swd * e / Ad if (k - b0) < 84 else 0.0)
        pnl[d] += R * 0.5 * w
        te[nt], tr[nt], tsd[nt] = d, R, side
        nt += 1
    return pnl, te[:nt], tr[:nt], tsd[:nt]


# ---------------------------------------------------------------- 部品を作る
def vbo_var(D, dirmode, skip):
    """値幅ブレイク（k=0.8・買い×1＋売り×0.5）の向きの決め方と、約定の時間の制限を変える。銘柄ごとの毎日の損益と取引一覧。"""
    last = 80 if skip else DAY - 4
    bs = np.sign(D["BTC"]["dO"] - R1.sma_prev(D["BTC"], 5))
    per, Ts = {}, []
    for a in COINS:
        x = D[a]
        Pd = R2.periods(x, DAY, 0)
        if dirmode == "oco":
            fl = Pd["fl"]
            ok = ~np.isnan(Pd["atr"]) & ~np.isnan(Pd["rng"])
            pnl, te, tr, sd = oco_last(fl["O"], fl["H"], fl["L"], fl["C"], Pd["pO"] + 0.8 * Pd["rng"], Pd["pO"] - 0.8 * Pd["rng"],
                                       ok, Pd["atr"], 1.0, 0.5, PF.SPEC[a][0], PF.SPEC[a][1] / 36500, PF.SPEC[a][2] / 36500, last)
            s = pd.Series(pnl, index=Pd["idx"])
            T = pd.DataFrame({"coin": a, "day": Pd["idx"][te], "side": sd, "w": np.where(sd == 1, 1.0, 0.5), "R": tr})
        elif dirmode == "btc5" and a == "ETH":
            s, T = R2.run(Pd, a, ma=False, extra_ok=lambda P_, side: (side * np.nan_to_num(bs)) > 0, last=last)
        else:
            s, T = R2.run(Pd, a, last=last)
        per[a] = s
        Ts.append(T)
    T = pd.concat(Ts, ignore_index=True)
    return per, T[(T.day >= MM.START) & (T.day <= MM.END)]


def mult(x, kind):
    """量の調整（その日の朝にわかる値）。"""
    n = len(x["dO"])
    if kind == "なし":
        return np.ones(n)
    if kind == "静かな日だけ":
        lr = np.log(pd.Series(x["dC"])).diff()
        rv = lr.rolling(10).std().shift(1)
        med = rv.rolling(365, min_periods=200).median()
        return np.where(med.isna(), 1.0, (rv < med).astype(float))
    atrp = pd.Series(x["atr"] / x["dO"])
    pct = atrp.rolling(365, min_periods=200).apply(lambda v: (v[:-1] < v[-1]).mean(), raw=True).to_numpy()
    w = np.array([1.5, 1.0, 0.5])[np.searchsorted([1 / 3, 2 / 3], np.nan_to_num(pct, nan=0.5))]
    return np.where(np.isnan(pct), 1.0, w)


def comp_c29(D):
    per = {}
    for a in COINS:
        x = D[a]
        lr = np.log(pd.Series(x["dC"])).diff()
        z = (lr.rolling(30).sum() / (lr.rolling(30).std() * np.sqrt(30))).to_numpy()
        per[a] = R3.pos_daily(x, a, R3.ls(z > 1.0, z < -1.0, False))[0]
    return per


def comp_oco(D, kind):
    per = {}
    for a in COINS:
        x = D[a]
        if kind == "F02":
            with np.errstate(invalid="ignore"):
                m = (x["dH"] < np.r_[np.nan, x["dH"][:-1]]) & (x["dL"] > np.r_[np.nan, x["dL"][:-1]])
            tl, ts = np.r_[np.nan, x["dH"][:-1]], np.r_[np.nan, x["dL"][:-1]]
        else:
            hh, ll = pd.Series(x["dH"]).rolling(5).max(), pd.Series(x["dL"]).rolling(5).min()
            with np.errstate(invalid="ignore"):
                m = ((hh < hh.shift(5)) & (ll > ll.shift(5))).to_numpy()
            tl, ts = np.r_[np.nan, hh.to_numpy()[:-1]], np.r_[np.nan, ll.to_numpy()[:-1]]
        ok = np.r_[False, m[:-1]] & ~np.isnan(x["atr"])
        pnl, te, tr = R1.oco_engine(x["O"], x["H"], x["L"], x["C"], tl.astype(np.float64), ts.astype(np.float64), ok, x["atr"],
                                    1.0, 0.5, PF.SPEC[a][0], PF.SPEC[a][1] / 36500, PF.SPEC[a][2] / 36500)
        per[a] = pd.Series(pnl, index=x["idx"])
    return per


def comp_ema9(D):
    per = {}
    for a in COINS:
        x = D[a]
        B = E.daily(x)
        O, H, L, C = B["flat"]
        tot = np.zeros(len(B["dC"]))
        for side in (1, -1):
            sw = (PF.SPEC[a][1] if side == 1 else PF.SPEC[a][2]) / 36500
            pnl, te, tr = R2B.trend_eng(O, H, L, C, DAY, B["dL"], B["dH"], B["dC"], B["ema"], E.signal(B, side, "50日平均の上"),
                                        x["atr"], side, PF.SPEC[a][0], sw, 84, 0.0, False, 0, 0.5)
            tot += pnl
        per[a] = pd.Series(tot, index=x["idx"])
    return per


def tot(per, m=None):
    """銘柄ごとの損益（と量の倍率）を合計して、期間を切る。"""
    s = None
    for a in COINS:
        v = per[a] * (1.0 if m is None else m[a])
        s = v if s is None else s.add(v, fill_value=0)
    return MM.clip(s)


def rec(pid, name, r, memo=""):
    record(グループ=GROUP, 計画ID=pid, 名前=f"{pid} {name}", 銘柄="BTC,ETH", 時間足="15分足・日足", メモ=memo,
           **{k: v for k, v in r.items() if not k.startswith("_")})


def eff(r):
    return r["年平均%"] / r["最大の落ち込み%"] if r["最大の落ち込み%"] > 0 else np.nan


def line(name, r, extra=""):
    R1.line(name, r, f" ｜効率{eff(r):.2f}" + extra)


# ---------------------------------------------------------------- 本体
def main(cache):
    print(f"（古い「{GROUP}」の行を {drop(GROUP)} 行消して、計算し直す）")
    D = R1.prep(cache)
    per0, T0 = R2.combo(D)
    v0 = tot(per0)
    base = R1.stats(v0, T0)
    line("A01（基準）", base)
    h1 = lambda s: s[s.index < SPLIT]
    h2 = lambda s: s[s.index >= SPLIT]
    b1, b2 = metrics_row(h1(v0)), metrics_row(h2(v0))

    # ---- 部品
    vv = {}
    for dm in ("own5", "btc5", "oco"):
        for sk in (False, True):
            vv[(dm, sk)] = vbo_var(D, dm, sk)
    M = {kind: {a: mult(D[a], kind) for a in COINS} for kind in ("なし", "静かな日だけ", "3段階")}
    c29, f02, f08, ema9 = tot(comp_c29(D)), tot(comp_oco(D, "F02")), tot(comp_oco(D, "F08")), tot(comp_ema9(D))

    # ---- 216通り（M07・M08・M09）
    print("\n==== 216通りの組み合わせ（前半だけで選び、後半で確かめる）")
    names = {"own5": "向き：自分の5日平均", "btc5": "向き：ETHはBTCの5日平均", "oco": "向きなし"}
    rows = []
    for (dm, sk), km, w29, w02, w08 in itertools.product(vv.keys(), M.keys(), (0, 0.25, 0.5), (0, 0.5), (0, 0.5)):
        x = tot(vv[(dm, sk)][0], M[km]).add(w29 * c29, fill_value=0).add(w02 * f02, fill_value=0).add(w08 * f08, fill_value=0)
        r1, r2, ra = metrics_row(h1(x)), metrics_row(h2(x)), metrics_row(x)
        rows.append(dict(dm=dm, sk=sk, km=km, w29=w29, w02=w02, w08=w08, x=x,
                         s1=r1["安定度"], a1=r1["年平均%"], dd1=r1["最大の落ち込み%"], neg1=r1["マイナスの年"],
                         s2=r2["安定度"], a2=r2["年平均%"], dd2=r2["最大の落ち込み%"],
                         s=ra["安定度"], a=ra["年平均%"], dd=ra["最大の落ち込み%"], neg=ra["マイナスの年"]))
    G = pd.DataFrame(rows)
    def label(rw):
        parts = [names[rw.dm], f"量：{rw.km}", "5〜8時は見送る" if rw.sk else "時間の制限なし"]
        for w, n in ((rw.w29, "C29"), (rw.w02, "F02"), (rw.w08, "F08")):
            if w:
                parts.append(f"{n}×{w}")
        return "・".join(parts)
    print(f"  A01：前半の安定度{b1['安定度']:.2f}・後半{b2['安定度']:.2f}")
    print(f"  216通りの後半の安定度：中央値{G.s2.median():.2f}、A01（{b2['安定度']:.2f}）より上は{(G.s2 > b2['安定度']).mean() * 100:.0f}%、"
          f"1.3より上は{(G.s2 > 1.3).mean() * 100:.0f}%。前半と後半の安定度の順位の相関{G[['s1', 's2']].corr(method='spearman').iloc[0, 1]:+.2f}")
    picks = {}
    picks["M08"] = G.sort_values("s1", ascending=False).iloc[0]
    cand = G[G.a1 >= b1["年平均%"]]
    picks["M07"] = cand.sort_values("dd1").iloc[0]
    cand = G[G.neg1 == "なし"].copy()
    cand["e1"] = cand.a1 / cand.dd1
    picks["M09"] = cand.sort_values("e1", ascending=False).iloc[0]
    rules = {"M08": "前半の安定度が一番高い形", "M07": "前半の年平均がA01以上の中で、前半の落ち込みが一番小さい形",
             "M09": "前半に負けた年がない中で、前半の効率が一番高い形"}
    res = {}
    for pid, rw in picks.items():
        r = R1.stats(rw.x)
        line(f"{pid}（{rules[pid]}）", r, f"\n        中身：{label(rw)}")
        rec(pid, f"前半で選ぶ：{rules[pid]}（{label(rw)}）", r, memo="216通りから前半だけで選んだ形")
        res[pid] = (r, label(rw))
    print("  前半の安定度の上位5つ → 後半")
    for rw in G.sort_values("s1", ascending=False).head(5).itertuples():
        print(f"    前半{rw.s1:.2f} → 後半{rw.s2:.2f}（年{rw.a2:+.1f}%・落込{rw.dd2:.1f}%） {label(rw)}")

    print("\n==== 選ばずに決めた形（ここまでの合格をそのまま重ねる）")
    fixed = {
        "S1 ETHはBTCの向き＋静かな日だけ": (("btc5", False), "静かな日だけ", 0, 0, 0),
        "S2 ETHはBTCの向き＋3段階＋C29×0.25": (("btc5", False), "3段階", 0.25, 0, 0),
        "S3 全部（ETHはBTCの向き＋3段階＋5〜8時は見送る＋C29×0.25＋F02×0.5＋F08×0.5）": (("btc5", True), "3段階", 0.25, 0.5, 0.5),
    }
    fixed_x = {}
    for name, (key, km, w29, w02, w08) in fixed.items():
        x = tot(vv[key][0], M[km]).add(w29 * c29, fill_value=0).add(w02 * f02, fill_value=0).add(w08 * f08, fill_value=0)
        r = R1.stats(x)
        line(name, r)
        rec("M08", name, r, memo="選ばずに、合格した改善をそのまま重ねた形")
        fixed_x[name] = (x, r)
    res["固定"] = {k: v[1] for k, v in fixed_x.items()}

    # ---- M06 リスクパリティ
    print("\n==== M06 リスクパリティ（前半の毎日の損益の振れが同じになるように、部品の量を決める）")
    vb = tot(vv[("btc5", False)][0], M["3段階"])
    comps = {"値幅ブレイク（ETHはBTCの向き＋3段階）": vb, "C29": c29, "F02": f02, "F08": f08}
    sig = {k: h1(v).std() for k, v in comps.items()}
    w = {k: sig["値幅ブレイク（ETHはBTCの向き＋3段階）"] / s_ for k, s_ in sig.items()}
    x = sum(w[k] * v for k, v in comps.items())
    r = R1.stats(x)
    line("リスクパリティ", r, " ｜量 " + "・".join(f"{k.split('（')[0]}×{v:.2f}" for k, v in w.items()))
    rec("M06", "リスクパリティ（値幅ブレイク・C29・F02・F08）", r, memo="量 " + "・".join(f"{k.split('（')[0]}×{v:.2f}" for k, v in w.items()))
    res["M06"] = (r, w)

    # ---- M11 口座を分ける
    print("\n==== M11 資金の一部で値幅ブレイク（CFD、S2の形）、残りでBTC・ETHを現物で持ちっぱなし（半分ずつ）")
    x2 = fixed_x["S2 ETHはBTCの向き＋3段階＋C29×0.25"][0]
    px = {a: pd.Series(D[a]["dC"], index=D[a]["idx"]) for a in COINS}
    hold = 0.5 * sum(px[a] / px[a][px[a].index >= MM.START].iloc[0] for a in COINS)
    hold = hold[(hold.index >= MM.START) & (hold.index <= MM.END)]
    res11 = {}
    for share in (1.0, 0.8, 0.5, 0.0):
        acct = share * (1 + x2.cumsum() / 100) + (1 - share) * hold.reindex(x2.index).ffill()
        yrs = (acct.index[-1] - acct.index[0]).days / 365.25
        ye = acct.groupby(acct.index.year).last()
        yr = ye / np.r_[1.0, ye.to_numpy()[:-1]] - 1
        neg = [str(y) for y, v in yr.items() if v < 0]
        dd = ((acct.cummax() - acct) / acct.cummax()).max() * 100
        cagr = (acct.iloc[-1] ** (1 / yrs) - 1) * 100
        print(f"    値幅ブレイク{share * 100:.0f}%・現物{(1 - share) * 100:.0f}%：年率{cagr:+.1f}%・口座の最大の落ち込み{dd:.1f}%・"
              f"負け年{','.join(neg) or 'なし'}・最悪の年{yr.min() * 100:+.1f}%")
        record(グループ=GROUP, 計画ID="M11", 名前=f"M11 値幅ブレイク{share * 100:.0f}%・現物{(1 - share) * 100:.0f}%", 銘柄="BTC,ETH",
               時間足="日足", メモ=f"口座全体：年率{cagr:+.1f}%・最大の落ち込み{dd:.1f}%・負け年{','.join(neg) or 'なし'}・最悪の年{yr.min() * 100:+.1f}%")
        res11[share] = (cagr, dd, ",".join(neg) or "なし", yr.min() * 100)
    res["M11"] = res11

    # ---- M12 静かな日は値幅ブレイク、荒い日はトレンド
    print("\n==== M12 静かな日（10日の実現ボラ）は値幅ブレイク、荒い日はトレンド（9日EMA・C29）")
    q = M["静かな日だけ"]
    per_e9, per_c29 = comp_ema9(D), comp_c29(D)
    r_ = {}
    for nm, pc in (("9日EMA", per_e9), ("C29", per_c29)):
        x = tot(per0, q).add(tot(pc, {a: 1 - q[a] for a in COINS}), fill_value=0)
        r_[nm] = R1.stats(x)
        line(f"静かな日は値幅ブレイク・荒い日は{nm}", r_[nm])
        rec("M12", f"静かな日は値幅ブレイク・荒い日は{nm}", r_[nm])
    r_["静かな日だけ"] = R1.stats(tot(per0, q))
    line("参考：静かな日だけ値幅ブレイク（E05）", r_["静かな日だけ"])
    res["M12"] = r_

    # ---- M13 手法の多数決で向きを決める
    print("\n==== M13 値幅ブレイクの向きを、3つ（5日平均・前日の終値と9日EMA・30日の騰落）の多数決で決める")
    per, Ts = {}, []
    for a in COINS:
        x = D[a]
        s5 = np.sign(x["dO"] - R1.sma_prev(x, 5))
        ema = pd.Series(x["dC"]).ewm(span=9, adjust=False).mean().to_numpy()
        se = np.sign(np.r_[np.nan, (x["dC"] - ema)[:-1]])
        sm = np.sign(np.r_[np.nan, x["dC"][:-1]] - np.r_[np.full(31, np.nan), x["dC"][:-31]])
        vote = np.sign(np.nan_to_num(s5) + np.nan_to_num(se) + np.nan_to_num(sm))
        s, T = R2.run(R2.periods(x, DAY, 0), a, ma=False, extra_ok=lambda P_, side, vote=vote: (side * vote) > 0)
        per[a] = s
        Ts.append(T)
    T = pd.concat(Ts)
    r = R1.stats(tot(per), T[(T.day >= MM.START) & (T.day <= MM.END)])
    line("3つの多数決で向きを決める", r)
    rec("M13", "値幅ブレイクの向きを3つの多数決で決める", r)
    res["M13"] = r

    # ---- M14 値幅ブレイクと逆に動く部品
    print("\n==== M14 値幅ブレイク（A01）と月の損益が逆に動く部品を探す")
    def d10(x, a):
        c = x["dC"]
        st = np.zeros(len(c))
        for i in range(1, len(c)):
            st[i] = st[i - 1] + 1 if c[i] < c[i - 1] else 0
        roll = (np.arange(len(x["O"])) % DAY == 84)
        pnl, te, tr, th = R3.swing_p(x["O"], x["H"], x["L"], x["C"], DAY, x["atr"], st >= 4, np.zeros(len(c), bool), 1, 99.0, 2,
                                     PF.SPEC[a][0], PF.SPEC[a][1] / 36500, roll)
        return pd.Series(pnl, index=x["idx"])
    def tsm(x, a, n=30):
        mom = x["dC"] - np.r_[np.full(n, np.nan), x["dC"][:-n]]
        return R3.pos_daily(x, a, np.nan_to_num(np.sign(mom)))[0]
    def hold_spot(x, a):
        with np.errstate(invalid="ignore", divide="ignore"):
            return pd.Series(np.nan_to_num(0.5 * (x["dC"] - x["prevC"]) / x["atr"]), index=x["idx"])
    table = {"C29 値動きで割ったモメンタム": c29, "F02 はらみ足のブレイク": f02, "F08 三角持ち合いのブレイク": f08,
             "C12 9日EMA": ema9, "C24 30日の時系列モメンタム": tot({a: tsm(D[a], a) for a in COINS}),
             "D10 4日続けて下げた翌日に買う": tot({a: d10(D[a], a) for a in COINS}),
             "参考：現物の持ちっぱなし（同じ量の物差し）": tot({a: hold_spot(D[a], a) for a in COINS})}
    mv0 = v0.resample("ME").sum()
    out14 = {}
    for nm, s in table.items():
        rr = metrics_row(s)
        c = mv0.corr(s.resample("ME").sum())
        mix = R1.stats(v0.add(0.25 * s, fill_value=0))
        print(f"    {nm:28s} 単独：年{rr['年平均%']:+6.2f}%・安定度{rr['安定度']:5.2f} ｜月の相関{c:+.2f} ｜値幅ブレイク＋×0.25の安定度{mix['安定度']:.2f}")
        out14[nm] = (rr["年平均%"], rr["安定度"], c, mix["安定度"])
    res["M14"] = out14

    # ---- M15 静か×1.5の値幅ブレイク＋9日EMA×0.5
    print("\n==== M15 値動きで量を変えた値幅ブレイク＋9日EMA×0.5")
    r15 = {}
    for km, lab in (("3段階", "ATRの順位で3段階"),):
        x = tot(per0, M[km]).add(0.5 * ema9, fill_value=0)
        r15[lab] = R1.stats(x)
        line(f"{lab}＋9日EMA×0.5", r15[lab])
        rec("M15", f"値幅ブレイク（{lab}）＋9日EMA×0.5", r15[lab])
    atrq = {}
    for a in COINS:
        atrp = pd.Series(D[a]["atr"] / D[a]["dO"])
        med = atrp.shift(1).rolling(365, min_periods=200).median()
        atrq[a] = np.where(med.isna(), 1.0, np.where(atrp < med, 1.5, 0.5))
    x = tot(per0, atrq).add(0.5 * ema9, fill_value=0)
    r15["静か×1.5・荒い×0.5"] = R1.stats(x)
    line("静か×1.5・荒い×0.5（E04）＋9日EMA×0.5", r15["静か×1.5・荒い×0.5"])
    rec("M15", "値幅ブレイク（静か×1.5・荒い×0.5）＋9日EMA×0.5", r15["静か×1.5・荒い×0.5"])
    res["M15"] = r15

    # ================================================================ 資金管理（L）
    base_sys = fixed_x["S2 ETHはBTCの向き＋3段階＋C29×0.25"][0]
    sys_r = R1.stats(base_sys)
    print("\n==== 資金管理（L）：S2（ETHはBTCの向き＋3段階＋C29×0.25）と A01 に当てはめる")
    line("S2（そのまま）", sys_r)
    L = {}

    def compound(x):
        eq = (1 + x / 100).cumprod()
        yrs = (x.index[-1] - x.index[0]).days / 365.25
        return (eq.iloc[-1] ** (1 / yrs) - 1) * 100, ((eq.cummax() - eq) / eq.cummax()).max() * 100, eq.iloc[-1]
    print("\n  -- L02 複利（毎回、その時の資金の0.5%で量を決める）")
    L["L02"] = {}
    for nm, x in (("A01", v0), ("S2", base_sys)):
        cg, dd, mlt = compound(x)
        print(f"    {nm}：単利の年平均{metrics_row(x)['年平均%']:+.2f}% → 複利の年率{cg:+.2f}%・最大の落ち込み{dd:.1f}%・"
              f"2018年1月の資金が{mlt:.2f}倍")
        record(グループ=GROUP, 計画ID="L02", 名前=f"L02 複利（{nm}）", 銘柄="BTC,ETH", 時間足="15分足・日足",
               メモ=f"複利の年率{cg:+.2f}%・最大の落ち込み{dd:.1f}%・資金{mlt:.2f}倍（単利の年平均{metrics_row(x)['年平均%']:+.2f}%）")
        L["L02"][nm] = (cg, dd, mlt)

    print("\n  -- L03 ケリー基準（前半の毎日の損益の平均÷分散から、量の倍率を出す）")
    L["L03"] = {}
    for nm, x in (("A01", v0), ("S2", base_sys)):
        mu, var = h1(x).mean() / 100, h1(x).var() / 1e4
        kelly = mu / var
        out = []
        for frac in (0.25, 0.5):
            xs = x * kelly * frac
            cg, dd, mlt = compound(xs)
            out.append((frac, kelly * frac, cg, dd))
            print(f"    {nm}：ケリーの倍率{kelly:.1f}倍 → その{frac}は{kelly * frac:.1f}倍（1回{0.5 * kelly * frac:.1f}%）："
                  f"複利の年率{cg:+.1f}%・最大の落ち込み{dd:.1f}%")
        record(グループ=GROUP, 計画ID="L03", 名前=f"L03 ケリー基準（{nm}）", 銘柄="BTC,ETH", 時間足="15分足・日足",
               メモ="・".join(f"{f}ケリー＝{k:.1f}倍：複利の年率{c:+.1f}%・落ち込み{d:.1f}%" for f, k, c, d in out))
        L["L03"][nm] = (kelly, out)

    def rule(nm, pid, x, wfn, lab):
        w = wfn(x)
        xs = x * w
        r = R1.stats(xs)
        line(f"{nm}：{lab}", r, f" ｜平均の量{w.mean():.2f}倍")
        rec(pid, f"{lab}（{nm}）", r, memo=f"平均の量{w.mean():.2f}倍")
        return r

    def fixed_ratio(x):
        cum = x.cumsum().shift(1).fillna(0)
        return 1 + 0.25 * np.floor(np.maximum(cum, 0) / 10)
    def loss_streak(n):
        def f(x):
            w = np.ones(len(x))
            run = 0
            for i, v in enumerate(x.to_numpy()):
                w[i] = 0.5 if run >= n else 1.0
                run = run + 1 if v < 0 else (0 if v > 0 else run)
            return pd.Series(w, index=x.index)
        return f
    def month_cap(cap):
        def f(x):
            w = np.ones(len(x))
            mtd, cur = 0.0, None
            for i, (t, v) in enumerate(zip(x.index, x.to_numpy())):
                if cur != (t.year, t.month):
                    cur, mtd = (t.year, t.month), 0.0
                w[i] = 0.0 if mtd <= -cap else 1.0
                mtd += v * w[i]
            return pd.Series(w, index=x.index)
        return f
    def year_target(tgt):
        def f(x):
            w = np.ones(len(x))
            ytd, cur = 0.0, None
            for i, (t, v) in enumerate(zip(x.index, x.to_numpy())):
                if cur != t.year:
                    cur, ytd = t.year, 0.0
                w[i] = 0.5 if ytd >= tgt else 1.0
                ytd += v * w[i]
            return pd.Series(w, index=x.index)
        return f
    def prev_month(x):
        m = x.resample("ME").sum()
        sgn = np.where(m.shift(1) > 0, 1.2, np.where(m.shift(1) < 0, 0.8, 1.0))
        mm = pd.Series(sgn, index=m.index.to_period("M"))
        return pd.Series(mm.reindex(x.index.to_period("M")).to_numpy(), index=x.index)

    for pid, title, items in (
        ("L04", "固定比率（資金が10%増えるごとに量を0.25倍ずつ増やす）", [("10%ごとに+0.25倍", fixed_ratio)]),
        ("L05", "連敗で量を半分（勝った日に戻す）", [(f"{n}日続けて負けたら半分", loss_streak(n)) for n in (2, 3, 5)]),
        ("L11", "月の損失上限（その月の損失が届いたら月末まで休む）", [(f"月−{c}%で休む", month_cap(c)) for c in (1, 2, 3)]),
        ("L12", "年の目標に届いたら量を半分", [(f"年+{t}%で半分", year_target(t)) for t in (5, 10)]),
        ("L13", "前月の成績で量を変える（プラスなら1.2倍、マイナスなら0.8倍）", [("前月で1.2倍・0.8倍", prev_month)]),
    ):
        print(f"\n  -- {pid} {title}")
        L[pid] = {}
        for lab, wfn in items:
            for nm, x in (("A01", v0), ("S2", base_sys)):
                L[pid][(lab, nm)] = rule(nm, pid, x, wfn, lab)

    print("\n  -- L10 1日の損失上限（BTCとETHの含み損の合計が届いたら、その日は両方とも決済）")
    L["L10"] = {}
    ents = {}
    for a in COINS:
        Pd = R2.periods(D[a], DAY, 0)
        for side, w in ((1, 1.0), (-1, 0.5)):
            trig = Pd["pO"] + side * 0.8 * Pd["rng"]
            with np.errstate(invalid="ignore"):
                ok = ~np.isnan(Pd["atr"]) & ~np.isnan(trig) & (side * (Pd["pO"] - Pd["sma5"]) > 0)
            fl = Pd["fl"]
            pnl, te, tp, tr, _ = R2.eng(fl["O"], fl["H"], fl["L"], fl["C"], DAY, DAY - 4, trig.astype(np.float64), ok.astype(np.bool_),
                                        Pd["atr"], np.zeros(Pd["n"], np.int64), side, PF.SPEC[a][0],
                                        (PF.SPEC[a][1] if side == 1 else PF.SPEC[a][2]) / 36500, Pd["roll"],
                                        np.ones(Pd["n"], np.int64), np.full(Pd["n"], np.nan))
            for d_, p_, r_k in zip(te, tp, tr):
                ents.setdefault(d_, []).append((a, side, w, p_, r_k, Pd))
    for cap in (0.5, 1.0, 1.5):
        daily = np.zeros(len(D["BTC"]["dO"]))
        for d_, lst in ents.items():
            b0 = d_ * DAY
            total_end = sum(w * 0.5 * r_k for a, side, w, p_, r_k, Pd in lst)
            stopped = None
            for j in range(min(p_ for _, _, _, p_, _, _ in lst), DAY):
                mtm = 0.0
                for a, side, w, p_, r_k, Pd in lst:
                    if j < p_:
                        continue
                    fl = Pd["fl"]
                    trig = Pd["pO"][d_] + side * 0.8 * Pd["rng"][d_]
                    e = max(fl["O"][b0 + p_], trig) if side == 1 else min(fl["O"][b0 + p_], trig)
                    mtm += w * 0.5 * (side * (fl["C"][b0 + j] - e) / Pd["atr"][d_] - PF.SPEC[a][0] * e / Pd["atr"][d_])
                if mtm <= -cap:
                    stopped = mtm
                    break
            daily[d_] = stopped if stopped is not None else total_end
        x = MM.clip(pd.Series(daily, index=D["BTC"]["idx"]))
        r = R1.stats(x)
        line(f"A01：1日の損失が−{cap}%に届いたら決済", r)
        rec("L10", f"1日の損失上限−{cap}%（A01）", r, memo="15分足の終値で判定")
        L["L10"][cap] = r

    print("\n  -- L14 最小ロット（BTC・ETHとも0.01）に丸めた時（A01、資金は円で固定、ドル円は日ごと）")
    L["L14"] = {}
    try:
        fx = pd.read_csv(os.path.join(cache, "USDJPY.csv"), index_col=0, parse_dates=True)
        fx = fx[fx.columns[-1] if "close" not in fx.columns else "close"]
        fx.index = pd.DatetimeIndex(fx.index).tz_localize(None).normalize()
    except Exception:
        fx = pd.Series(dtype=float)
    for cap_jpy in (300_000, 1_000_000, 3_000_000):
        for pol in ("四捨五入（0なら見送り）", "最低0.01に切り上げ"):
            per = {}
            for a in COINS:
                Pd = R2.periods(D[a], DAY, 0)
                s, T = R2.run(Pd, a)
                rate = fx.reindex(Pd["idx"].normalize(), method="ffill").fillna(150).to_numpy() if len(fx) else np.full(Pd["n"], 150.0)
                ratio_by_day = np.ones(Pd["n"])
                usd = cap_jpy / rate
                want = 0.005 * usd / Pd["atr"]                         # 1R（ATR）＝資金の0.5% になる枚数
                for side, w in ((1, 1.0), (-1, 0.5)):
                    lots = w * want
                    rl = np.round(lots / 0.01) * 0.01
                    if pol.startswith("最低"):
                        rl = np.maximum(rl, 0.01)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        rt = np.where(lots > 0, rl / lots, 1.0)
                    ratio_by_day = np.where(side * (Pd["pO"] - Pd["sma5"]) > 0, rt, ratio_by_day)
                per[a] = s * np.nan_to_num(ratio_by_day, nan=1.0)
            x = tot(per)
            r = R1.stats(x)
            line(f"資金{cap_jpy // 10000}万円・{pol}", r)
            rec("L14", f"資金{cap_jpy // 10000}万円・{pol}（A01）", r)
            L["L14"][(cap_jpy, pol)] = r

    print("\n  -- L15 建玉の大きさ（S2で使う建玉の合計が、資金の何倍になるか）")
    notional = None
    for a in COINS:
        x = D[a]
        with np.errstate(invalid="ignore", divide="ignore"):
            per_unit = 0.005 / (x["atr"] / x["dO"])                    # 1倍の量の建玉（資金に対する倍率）
        n_ = pd.Series(np.nan_to_num(per_unit * M["3段階"][a] * 1.0 + 0.25 * per_unit), index=x["idx"])
        notional = n_ if notional is None else notional.add(n_, fill_value=0)
    notional = MM.clip(notional)
    print(f"    建玉の合計（値幅ブレイクの買いとC29が両方とも最大の日）：中央値で資金の{notional.median():.2f}倍、"
          f"99%で{notional.quantile(0.99):.2f}倍、最大{notional.max():.2f}倍。1倍・2倍・3倍の上限には届かない")
    record(グループ=GROUP, 計画ID="L15", 名前="L15 S2の建玉の大きさ", 銘柄="BTC,ETH", 時間足="日足",
           メモ=f"中央値で資金の{notional.median():.2f}倍・99%で{notional.quantile(0.99):.2f}倍・最大{notional.max():.2f}倍")
    L["L15"] = (notional.median(), notional.quantile(0.99), notional.max())
    return res, L, base, sys_r


if __name__ == "__main__" and len(sys.argv) == 2:
    main(sys.argv[1])


def followup(cache):
    """第4回・前半の続き：(1) 候補の形の年ごとの損益 (2) 今の価格での、0.01ロットへの丸めのずれ。"""
    D = R1.prep(cache)
    per0, T0 = R2.combo(D)
    v0 = tot(per0)
    Mq = {a: mult(D[a], "静かな日だけ") for a in COINS}
    M3 = {a: mult(D[a], "3段階") for a in COINS}
    c29, f02 = tot(comp_c29(D)), tot(comp_oco(D, "F02"))
    vb5 = vbo_var(D, "btc5", False)[0]
    rows = {"A01（今の第一候補）": v0, "S1 ETHはBTCの向き＋静かな日だけ": tot(vb5, Mq),
            "S2 ＋3段階＋C29×0.25": tot(vb5, M3).add(0.25 * c29, fill_value=0),
            "M09 ETHはBTCの向き＋静かな日だけ＋F02×0.5": tot(vb5, Mq).add(0.5 * f02, fill_value=0)}
    print("==== 年ごとの損益（%）")
    print(pd.DataFrame({k: s.groupby(s.index.year).sum() for k, s in rows.items()}).round(1).T.to_string())
    for k, s in rows.items():
        r = metrics_row(s)
        print(f"  {k:28s} プラスの月{r['プラスの月%']:.0f}%・最悪の年{r['最悪の年%']:+.1f}%・最長連続マイナス月{r['最長連続マイナス月']}・"
              f"高値回復の最長{r['高値回復の最長日数']}日")

    print("\n==== 今の価格（直近30日）で、1回0.5%（ATR1本分）の量を0.01ロットに丸めた時のずれ")
    fx = pd.read_csv(os.path.join(cache, "USDJPY.csv"), index_col=0, parse_dates=True)["close"]
    rate = float(fx.iloc[-1])
    for a in COINS:
        x = D[a]
        last = slice(-30, None)
        px, atr = x["dC"][last], x["atr"][last]
        for cap in (300_000, 1_000_000, 3_000_000):
            usd = cap / rate
            for side, w in ((1, 1.0), (-1, 0.5)):
                want = w * 0.005 * usd / atr
                rl = np.maximum(np.round(want / 0.01) * 0.01, 0.0)
                print(f"  {a} 資金{cap // 10000:3d}万円・{'買い' if side == 1 else '売り（半分）'}：ほしい量 {np.median(want):.4f}枚"
                      f"（約{np.median(want * px) * rate / 10000:.0f}万円分） → 0.01刻みで{np.median(rl):.2f}枚（ずれ{np.median(rl / want):.2f}倍）")
    print(f"  （ドル円 {rate:.1f}円、BTC {D['BTC']['dC'][-1]:,.0f}ドル・ETH {D['ETH']['dC'][-1]:,.0f}ドル）")


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "followup":
    followup(sys.argv[1])
