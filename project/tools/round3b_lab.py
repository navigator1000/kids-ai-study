#!/usr/bin/env python3
"""計画書（project/research/plan.md）第3回・後半：逆張り（D）とローソク足の形（F）の優先「中」20件（BTC・ETH、2018年〜）。

  D10 連続陰線の翌日に買う／D11 連続陽線の翌日に売る／D14 20日平均からの乖離で買う／D15 日中の逆張り（始値からATR×k）／
  D16 前日の高値・安値で逆張り／D20 暴落の後に買う／D21 長い下ヒゲの翌日に買う／D24 静かな日だけRSI(2)の逆張り／
  D28 20日安値を割った直後に買う
  F01 包み足／F02 はらみ足のブレイク／F03 ピンバー／F04 十字線のブレイク／F08 三角持ち合いのブレイク／F10 前週の高値・安値／
  F12 スマッシュ・デイ／F13 上げた日の比率／F16 中値より上の始まり／F17 強い引け／F20 週末の窓を埋めにいく
共通: Vantage CFD（往復0.05%、買いのスワップ年25〜35%、売り0%）、1R＝ATR（日足）1本分＝資金の0.5%（単利）、含み損込み。
      買い×1、売り×0.5（値幅ブレイクと同じ）。「前半の安定度で一番良い形」を選び、1〜3次で判定する（round3_lab.component）。
結果は results.csv（グループ「計画・第3回後半」、計画IDつき）と plan.csv。
使い方: python3 project/tools/round3b_lab.py <キャッシュ（BTC_15m.csv, ETH_15m.csv）>
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
import more_methods_lab as MM  # noqa: E402
import portfolio_lab as PF  # noqa: E402
import round1_lab as R1  # noqa: E402
import round2_lab as R2  # noqa: E402
import round2b_lab as R2B  # noqa: E402
import round3_lab as R3  # noqa: E402
from registry import drop, metrics_row, record  # noqa: E402

COINS = ("BTC", "ETH")
GROUP = "計画・第3回後半"
DAY = 96
R3.GROUP = GROUP                         # show・component の記録先をこの回にする


def prev(a, k=1):
    """k日前の値（k=0ならそのまま）。"""
    return a if k == 0 else np.r_[np.full(k, np.nan), a[:-k]]


def roll_of(x):
    return (np.arange(len(x["O"])) % DAY == 84)


def swing(x, a, sig, side, maxd, xsig=None, stop_k=99.0):
    """前日の終値で sig なら翌朝の始値で入り、maxd 日（か xsig）で翌朝決済。"""
    xs = np.zeros(len(sig), bool) if xsig is None else np.nan_to_num(xsig).astype(bool)
    sw = (PF.SPEC[a][1] if side == 1 else PF.SPEC[a][2]) / 36500
    pnl, te, tr, th = R3.swing_p(x["O"], x["H"], x["L"], x["C"], DAY, x["atr"], np.nan_to_num(sig).astype(bool), xs, side,
                                 stop_k, maxd, PF.SPEC[a][0], sw, roll_of(x))
    return pnl, pd.DataFrame({"day": x["idx"][te], "R": tr, "hold": th})


def limit(x, a, ok, lvl, side, maxd, stop_k=99.0, xsig=None):
    """ok の日に lvl の指値（買いは下で待つ）。maxd 日で翌朝決済（maxd=1 なら当日中＝翌朝9時）。"""
    xs = np.zeros(len(ok), bool) if xsig is None else np.nan_to_num(xsig).astype(bool)
    sw = (PF.SPEC[a][1] if side == 1 else PF.SPEC[a][2]) / 36500
    pnl, te, tr, th = R3.limit_swing(x["O"], x["H"], x["L"], x["C"], DAY, x["atr"], np.nan_to_num(ok).astype(bool),
                                     np.asarray(lvl, dtype=np.float64), xs, side, stop_k, maxd, PF.SPEC[a][0], sw, roll_of(x))
    return pnl, pd.DataFrame({"day": x["idx"][te], "R": tr, "hold": th})


def two_sides(fn_long, fn_short, w_short=0.5):
    """買いと売り（×w_short）の損益を足す関数を作る。fn_*(x, a) → (pnl, T)。"""
    def fn(x, a):
        tot = np.zeros(len(x["dO"]))
        Ts = []
        for f, w in ((fn_long, 1.0), (fn_short, w_short)):
            if f is None or w == 0:
                continue
            p, T = f(x, a)
            tot += w * p
            Ts.append(T)
        return pd.Series(tot, index=x["idx"]), pd.concat(Ts)
    return fn


def run(pid, name, fn, tf="日足（約定は15分足）"):
    per, T = R3.both(D_, fn)
    return R3.show(pid, name, per, T, tf=tf)


def oco(x, a, tl, ts, ok, wl=1.0, ws=0.5):
    pnl, te, tr = R1.oco_engine(x["O"], x["H"], x["L"], x["C"], np.asarray(tl, np.float64), np.asarray(ts, np.float64),
                                np.nan_to_num(ok).astype(bool), x["atr"], wl, ws, PF.SPEC[a][0], PF.SPEC[a][1] / 36500,
                                PF.SPEC[a][2] / 36500)
    return pd.Series(pnl, index=x["idx"]), pd.DataFrame({"day": x["idx"][te], "R": tr, "hold": 0})


def main(cache):
    global D_
    print(f"（古い「{GROUP}」の行を {drop(GROUP)} 行消して、計算し直す）")
    D_ = D = R1.prep(cache)
    per0, T0 = R2.combo(D)
    v0 = R1.total(per0)
    base = R1.stats(v0, T0)
    R1.line("A01（基準）", base)
    comp = lambda pid, res: R3.component(pid, res, v0, base)

    # ---------------- D 逆張り
    print("\n==== D10 N日続けて下げた翌日に買い、2日持つ ／ D11 N日続けて上げた翌日に売り、2日持つ")
    def streak(c, up):
        s = np.zeros(len(c))
        for i in range(1, len(c)):
            ok = (c[i] > c[i - 1]) if up else (c[i] < c[i - 1])
            s[i] = s[i - 1] + 1 if ok else 0
        return s
    for pid, up, side in (("D10", False, 1), ("D11", True, -1)):
        res = {}
        for n in (3, 4, 5):
            def fn(x, a, n=n, up=up, side=side):
                p, T = swing(x, a, streak(x["dC"], up) >= n, side, 2)
                return pd.Series(p * (1.0 if side == 1 else 1.0), index=x["idx"]), T
            res[f"{n}日"] = run(pid, f"{n}日続けて{'上げた' if up else '下げた'}翌日に{'売り' if side == -1 else '買い'}・2日", fn)
        comp(pid, res)

    print("\n==== D14 終値が20日平均から一定以上下に離れたら翌朝買い（平均に戻るか20日で決済、損切りATR×2）")
    res = {}
    for dev in (0.05, 0.10, 0.15, 0.20):
        def fn(x, a, dev=dev):
            m = R3.sma(x["dC"], 20)
            with np.errstate(invalid="ignore"):
                sig = x["dC"] < m * (1 - dev)
                xs = x["dC"] > m
            p, T = swing(x, a, sig, 1, 20, xs, 2.0)
            return pd.Series(p, index=x["idx"]), T
        res[f"−{int(dev * 100)}%"] = run("D14", f"20日平均から−{int(dev * 100)}%以上", fn)
    comp("D14", res)

    print("\n==== D15 日中の逆張り：始値からATR×k下で買い・上で売りの指値、当日決済")
    res = {}
    for k in (0.5, 1.0, 1.5, 2.0):
        for use_ma in (False, True):
            def fn(x, a, k=k, use_ma=use_ma):
                sd = np.sign(x["dO"] - R1.sma_prev(x, 5)) if use_ma else np.ones(len(x["dO"]))
                okl = ~np.isnan(x["atr"]) & ((sd > 0) if use_ma else True)
                oks = ~np.isnan(x["atr"]) & ((sd < 0) if use_ma else True)
                return two_sides(lambda x_, a_: limit(x_, a_, okl, x["dO"] - k * x["atr"], 1, 1),
                                 lambda x_, a_: limit(x_, a_, oks, x["dO"] + k * x["atr"], -1, 1),
                                 0.5 if use_ma else 1.0)(x, a)
            res[f"ATR×{k}{'・5日平均の向き' if use_ma else ''}"] = run("D15", f"ATR×{k}・{'5日平均の向きだけ' if use_ma else '向きなし'}", fn, "15分足")
    comp("D15", res)

    print("\n==== D16 前日安値で買い・前日高値で売りの指値、当日決済")
    res = {}
    for use_ma in (False, True):
        def fn(x, a, use_ma=use_ma):
            sd = np.sign(x["dO"] - R1.sma_prev(x, 5)) if use_ma else np.ones(len(x["dO"]))
            okl = ~np.isnan(x["atr"]) & ((sd > 0) if use_ma else True)
            oks = ~np.isnan(x["atr"]) & ((sd < 0) if use_ma else True)
            return two_sides(lambda x_, a_: limit(x_, a_, okl, prev(x["dL"]), 1, 1),
                             lambda x_, a_: limit(x_, a_, oks, prev(x["dH"]), -1, 1), 0.5 if use_ma else 1.0)(x, a)
        res["5日平均の向き" if use_ma else "向きなし"] = run("D16", "5日平均の向きだけ" if use_ma else "向きなし", fn, "15分足")
    comp("D16", res)

    print("\n==== D20 1日で大きく下げた後、w日目の朝に買って10日持つ")
    res = {}
    for drop_ in (0.08, 0.10, 0.15):
        for w in (1, 2, 3):
            def fn(x, a, drop_=drop_, w=w):
                with np.errstate(invalid="ignore", divide="ignore"):
                    crash = (x["dC"] / x["prevC"] - 1) <= -drop_
                sig = prev(crash.astype(float), w - 1) if w > 1 else crash.astype(float)
                p, T = swing(x, a, sig > 0, 1, 10)
                return pd.Series(p, index=x["idx"]), T
            res[f"−{int(drop_ * 100)}%・{w}日目"] = run("D20", f"−{int(drop_ * 100)}%以上の下げ・{w}日目に買い", fn)
    comp("D20", res)

    print("\n==== D21 下ヒゲが実体のm倍以上（ATR×0.5以上）の翌日に買い、N日持つ")
    res = {}
    for m in (2, 3, 5):
        for hold in (1, 3):
            def fn(x, a, m=m, hold=hold):
                o, h, l, c = x["dO"], x["dH"], x["dL"], x["dC"]
                wick = np.minimum(o, c) - l
                body = np.maximum(np.abs(c - o), 0.1 * (h - l))
                with np.errstate(invalid="ignore"):
                    sig = (wick >= m * body) & (wick >= 0.5 * x["atr"])
                p, T = swing(x, a, sig, 1, hold)
                return pd.Series(p, index=x["idx"]), T
            res[f"{m}倍・{hold}日"] = run("D21", f"下ヒゲが実体の{m}倍以上・{hold}日持つ", fn)
    comp("D21", res)

    print("\n==== D24 静かな日（ATRが過去1年の中央値より小さい）だけ、RSI(2)の逆張り")
    res = {}
    for only_quiet in (True, False):
        def fn(x, a, only_quiet=only_quiet):
            c = x["dC"]
            d = np.diff(c, prepend=c[0])
            up = pd.Series(np.clip(d, 0, None)).ewm(alpha=0.5, adjust=False).mean()
            dn = pd.Series(np.clip(-d, 0, None)).ewm(alpha=0.5, adjust=False).mean()
            r2 = (100 - 100 / (1 + up / dn)).to_numpy()
            atrp = pd.Series(x["atr"] / x["dO"])
            q = (atrp < atrp.shift(1).rolling(365, min_periods=200).median()).to_numpy()
            # 信号はその日の終値で出て、翌日に入る。翌日が静かかどうかはその朝にわかる
            qn = np.r_[q[1:], False] if only_quiet else np.ones(len(c), bool)
            m5 = R3.sma(c, 5)
            with np.errstate(invalid="ignore"):
                return two_sides(lambda x_, a_: swing(x_, a_, (r2 < 10) & qn, 1, 5, c > m5, 2.0),
                                 lambda x_, a_: swing(x_, a_, (r2 > 90) & qn, -1, 5, c < m5, 2.0), 0.5)(x, a)
        res["静かな日だけ" if only_quiet else "参考：いつも"] = run("D24", "静かな日だけ" if only_quiet else "参考：いつも（D02の形）", fn)
    comp("D24", res)

    print("\n==== D28 直近20日の安値を割ったところで買い（指値）、当日か3日で決済")
    res = {}
    for hold in (1, 3):
        for stop_k in (99.0, 1.0):
            def fn(x, a, hold=hold, stop_k=stop_k):
                lvl = pd.Series(x["dL"]).rolling(20).min().shift(1).to_numpy()
                p, T = limit(x, a, ~np.isnan(lvl), lvl, 1, hold, stop_k)
                return pd.Series(p, index=x["idx"]), T
            lab = f"{'当日' if hold == 1 else '3日'}・{'損切りなし' if stop_k > 50 else '損切りATR×1'}"
            res[lab] = run("D28", lab, fn, "15分足")
    comp("D28", res)

    # ---------------- F ローソク足の形
    print("\n==== F01 包み足（陽の包み足で買い・陰の包み足で売り）の翌朝に入り、N日持つ")
    res = {}
    for hold in (1, 3, 5):
        def fn(x, a, hold=hold):
            o, c = x["dO"], x["dC"]
            po, pc = prev(o), prev(c)
            with np.errstate(invalid="ignore"):
                bull = (pc < po) & (c > o) & (c > po) & (o <= pc * 1.001)
                bear = (pc > po) & (c < o) & (c < po) & (o >= pc * 0.999)
            return two_sides(lambda x_, a_: swing(x_, a_, bull, 1, hold), lambda x_, a_: swing(x_, a_, bear, -1, hold))(x, a)
        res[f"{hold}日"] = run("F01", f"包み足・{hold}日持つ", fn)
    comp("F01", res)

    def nextday_oco(pid, title, mask_fn):
        print(f"\n==== {pid} {title}")
        res = {}
        def fn(x, a):
            m = mask_fn(x)
            ok = np.r_[False, m[:-1]] & ~np.isnan(x["atr"])
            return oco(x, a, prev(x["dH"]), prev(x["dL"]), ok)
        res["上下に逆指値（先に約定した方）"] = run(pid, "前日の高値・安値の先に抜けた方に乗る、当日決済", fn, "15分足")
        def fn2(x, a):
            m = mask_fn(x)
            okd = np.r_[False, m[:-1]]
            per_, T = R2.run(R2.periods(x, DAY, 0), a, trig_fn=lambda Pd, s: prev(Pd["pH"] if s == 1 else Pd["pL"]),
                             extra_ok=lambda Pd, s: okd)
            return per_, T.assign(hold=0)
        res["5日平均の向きだけ"] = run(pid, "5日平均の向きだけ、当日決済", fn2, "15分足")
        comp(pid, res)

    with np.errstate(invalid="ignore"):
        nextday_oco("F02", "はらみ足（高値・安値が前日の内側）の翌日、その高値・安値を抜けたら乗る",
                    lambda x: (x["dH"] < prev(x["dH"])) & (x["dL"] > prev(x["dL"])))
        nextday_oco("F04", "十字線（実体が値幅の10%以下）の翌日、その高値・安値を抜けたら乗る",
                    lambda x: np.abs(x["dC"] - x["dO"]) <= 0.1 * (x["dH"] - x["dL"]))

    print("\n==== F08 三角持ち合い：直近N日の高値・安値が、その前のN日より両方内側（高値の切り下げ・安値の切り上げ）の翌日、"
          "直近N日の高値・安値を抜けたら乗る、当日決済")
    print("    （4日続けて内側に入る形は9年間で0回だったので、N日ずつの高値・安値で判定する）")
    res = {}
    for n in (5, 10):
        def fn(x, a, n=n):
            hh = pd.Series(x["dH"]).rolling(n).max()
            ll = pd.Series(x["dL"]).rolling(n).min()
            with np.errstate(invalid="ignore"):
                m = ((hh < hh.shift(n)) & (ll > ll.shift(n))).to_numpy()
            ok = np.r_[False, m[:-1]] & ~np.isnan(x["atr"])
            return oco(x, a, prev(hh.to_numpy()), prev(ll.to_numpy()), ok)
        res[f"{n}日"] = run("F08", f"{n}日ずつで内側に縮んだ翌日、直近{n}日の高値・安値を抜けたら乗る", fn, "15分足")
    comp("F08", res)

    print("\n==== F03 ピンバー（下ヒゲが値幅の2/3以上で上の1/3で引け）の翌日、その高値を超えたら買い（売りは逆）、当日決済")
    res = {}
    for use_ma in (False, True):
        def fn(x, a, use_ma=use_ma):
            o, h, l, c = x["dO"], x["dH"], x["dL"], x["dC"]
            rg = h - l
            with np.errstate(invalid="ignore", divide="ignore"):
                bull = ((np.minimum(o, c) - l) >= 2 / 3 * rg) & ((c - l) >= 2 / 3 * rg) & (rg >= 0.5 * x["atr"])
                bear = ((h - np.maximum(o, c)) >= 2 / 3 * rg) & ((h - c) >= 2 / 3 * rg) & (rg >= 0.5 * x["atr"])
            okb, oks = np.r_[False, bull[:-1]], np.r_[False, bear[:-1]]
            per_, T = R2.run(R2.periods(x, DAY, 0), a, ma=use_ma, trig_fn=lambda Pd, s: prev(Pd["pH"] if s == 1 else Pd["pL"]),
                             extra_ok=lambda Pd, s: okb if s == 1 else oks)
            return per_, T.assign(hold=0)
        res["5日平均の向き" if use_ma else "向きなし"] = run("F03", "5日平均の向きだけ" if use_ma else "向きなし", fn, "15分足")
    comp("F03", res)

    print("\n==== F10 前週の高値を超えたら買い（前週の安値を割ったら売り）、週末に決済")
    first_mon = int(np.flatnonzero(D["BTC"]["idx"].weekday == 0)[0]) * DAY
    res = {}
    for use_ma in (True, False):
        def fn(x, a, use_ma=use_ma):
            Pd = R2.periods(x, DAY * 7, first_mon)
            s, T = R2.run(Pd, a, ma=use_ma, trig_fn=lambda P_, sd: prev(P_["pH"] if sd == 1 else P_["pL"]))
            return R2.daily(s), T.assign(hold=7)
        res["5週平均の向き" if use_ma else "向きなし"] = run("F10", "5週平均の向きだけ" if use_ma else "向きなし", fn, "15分足（週）")
    comp("F10", res)

    print("\n==== F12 スマッシュ・デイ（前日安値を割って引けた日）の翌日、その日の高値を超えたら買い（売りは逆）、当日決済")
    res = {}
    for use_ma in (False, True):
        def fn(x, a, use_ma=use_ma):
            with np.errstate(invalid="ignore"):
                sb = x["dC"] < prev(x["dL"])
                ss = x["dC"] > prev(x["dH"])
            okb, oks = np.r_[False, sb[:-1]], np.r_[False, ss[:-1]]
            per_, T = R2.run(R2.periods(x, DAY, 0), a, ma=use_ma, trig_fn=lambda Pd, s: prev(Pd["pH"] if s == 1 else Pd["pL"]),
                             extra_ok=lambda Pd, s: okb if s == 1 else oks)
            return per_, T.assign(hold=0)
        res["5日平均の向き" if use_ma else "向きなし"] = run("F12", "5日平均の向きだけ" if use_ma else "向きなし", fn, "15分足")
    comp("F12", res)

    print("\n==== F13 直近10日のうちN日以上上げていれば買い（10−N日以下なら売り）")
    res = {}
    for n in (6, 7, 8):
        def fn(x, a, n=n):
            up = pd.Series((x["dC"] > x["prevC"]).astype(float)).rolling(10).sum().to_numpy()
            return R3.pos_daily(x, a, R3.ls(up >= n, up <= 10 - n, False))
        res[f"{n}日"] = run("F13", f"10日中{n}日以上", fn, "日足")
    comp("F13", res)

    print("\n==== F16 始値が前日の中値より上なら買い・下なら売り（9時〜翌9時）／ F17 強い引け（値幅の上位10%）の翌日に買い")
    res = {}
    def fn16(x, a):
        mid = prev((x["dH"] + x["dL"]) / 2)
        with np.errstate(invalid="ignore"):
            side = np.where(x["dO"] > mid, 1, np.where(x["dO"] < mid, -1, 0))
        d0 = np.arange(len(x["dO"])) * DAY
        s, T = R2B.window(x, a, d0, d0 + DAY, side, np.where(side > 0, 1.0, 0.5))
        return s, T.assign(hold=1)
    res["中値"] = run("F16", "始値が前日の中値より上なら買い・下なら売り", fn16, "日足")
    comp("F16", res)
    res = {}
    for hold in (1, 3):
        def fn(x, a, hold=hold):
            with np.errstate(invalid="ignore", divide="ignore"):
                clv = (x["dC"] - x["dL"]) / (x["dH"] - x["dL"])
            p, T = swing(x, a, clv >= 0.9, 1, hold)
            return pd.Series(p, index=x["idx"]), T
        res[f"{hold}日"] = run("F17", f"終値が値幅の上位10%の翌日に買い・{hold}日持つ", fn, "日足")
    comp("F17", res)

    print("\n==== F20 週末（金曜21時〜月曜9時、日本）の動きと反対に月曜9時に入り、金曜21時の値段に戻ったら利確（戻らなければ火曜9時に決済）")
    res = {}
    def fn20(x, a):
        Pd = R2.periods(x, DAY, 0)
        fl = Pd["fl"]
        n = Pd["n"]
        d0 = np.arange(n) * DAY
        fri = np.where(d0 - 3 * DAY + 48 >= 0, fl["O"][np.clip(d0 - 3 * DAY + 48, 0, None)], np.nan)
        mon = (Pd["idx"].weekday == 0)
        mv = Pd["pO"] - fri
        tot = np.zeros(n)
        Ts = []
        for side in (1, -1):
            with np.errstate(invalid="ignore", divide="ignore"):
                ok = mon & ~np.isnan(Pd["atr"]) & (side * mv < 0)          # 週末に下げたら買い、上げたら売り
                tpk = np.abs(mv) / Pd["atr"]
            sw = (PF.SPEC[a][1] if side == 1 else PF.SPEC[a][2]) / 36500
            Rr = R2B.vbo_x(fl["O"], fl["H"], fl["L"], fl["C"], DAY, DAY - 4, Pd["pO"].astype(np.float64), ok, Pd["atr"], side,
                           PF.SPEC[a][0], sw, np.where(ok, tpk, np.nan), 1.0, -1)
            tot += Rr * 0.5
            Ts.append(pd.DataFrame({"day": Pd["idx"][ok], "R": Rr[ok], "hold": 1}))
        return pd.Series(tot, index=Pd["idx"]), pd.concat(Ts)
    res["窓埋め"] = run("F20", "週末の窓を埋めにいく（買い・売りとも×1）", fn20, "15分足")
    comp("F20", res)
    return base


if __name__ == "__main__":
    main(sys.argv[1])
