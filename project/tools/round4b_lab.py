#!/usr/bin/env python3
"""計画書（project/research/plan.md）第4回・後半：ほかの市場・経済指標（I）と VIX で休む（E17）（BTC・ETH、2018年〜）。

  I01 米株（S&P500）が200日平均の上の時だけ／I02 前日の米株の向き／I03 ナスダック／I04 ドル円／I05 ドル指数／I06 米国の10年金利／
  I07 金／I08 原油の急変／I09 日経平均／I10 VIXの急上昇の翌日／I11 リスクオン・オフ（株↑金↓）／I12 株との相関が高い時だけ／
  I13 銀／I15 米国株の時間のBTCの動き／E17 VIXが高い日は値幅ブレイクを休む
使い方は2通り：(a) その指標だけでBTC・ETHを売買する（9時〜翌9時、または条件が続く間）
              (b) 値幅ブレイク（A01）の絞り込みに使う
ほかの市場の値は、その日の朝9時（UTC0時）より前に終わった取引日の終値だけを使う（米国の終値は日本時間の朝5〜6時）。
共通: Vantage CFD（往復0.05%、買いのスワップ年25〜35%、売り0%）、1R＝ATR（日足）1本分＝資金の0.5%（単利）、買い×1・売り×0.5。
データ: キャッシュの S&P500.csv・Nikkei.csv・USDJPY.csv・Gold.csv・Crude.csv・Silver.csv と、Yahooから取った Nasdaq.csv・DXY.csv・US10Y.csv・VIX.csv。
結果は results.csv（グループ「計画・第4回後半」、計画IDつき）と plan.csv。
使い方: python3 project/tools/round4b_lab.py <キャッシュ>
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
import round4_lab as R4  # noqa: E402
from registry import drop, metrics_row  # noqa: E402

COINS = ("BTC", "ETH")
GROUP = "計画・第4回後半"
DAY = 96
R3.GROUP = GROUP


def macro(cache, name):
    df = pd.read_csv(os.path.join(cache, f"{name}.csv"), index_col=0, parse_dates=True)
    s = df["close"].astype(float)
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    return s[~s.index.duplicated(keep="last")].sort_index()


def avail(s, days):
    """取引日の値を、翌日（UTC0時＝日本時間9時）から使えるようにして、暗号資産の日に合わせる。"""
    t = s.copy()
    t.index = t.index + pd.Timedelta(days=1)
    return t.reindex(days, method="ffill").to_numpy()


def daytrade(side_fn):
    """日ごとの向き（その朝にわかる値）で、9時〜翌9時を持つ。買い×1・売り×0.5。"""
    def fn(x, a):
        side = np.nan_to_num(side_fn(x, a))
        d0 = np.arange(len(x["dO"])) * DAY
        s, T = R2B.window(x, a, d0, d0 + DAY, side, np.where(side > 0, 1.0, 0.5))
        return s, T.assign(hold=1)
    return fn


def holdwhile(cond_fn):
    """条件（その朝にわかる値）が成り立つ日だけ買いで持つ。"""
    def fn(x, a):
        c = np.nan_to_num(cond_fn(x, a)).astype(float)
        pos = np.r_[c[1:], 0.0]                 # pos[d-1] を「d日目に持つ」に合わせる
        return R3.pos_daily(x, a, pos)
    return fn


def main(cache):
    print(f"（古い「{GROUP}」の行を {drop(GROUP)} 行消して、計算し直す）")
    D = R1.prep(cache)
    days = D["BTC"]["idx"]
    per0, T0 = R2.combo(D)
    v0 = R1.total(per0)
    base = R1.stats(v0, T0)
    R1.line("A01（基準）", base)
    comp = lambda pid, res: R3.component(pid, res, v0, base)
    run = lambda pid, name, fn, tf="日足": R3.show(pid, name, *R3.both(D, fn), tf=tf)

    mk = {n: macro(cache, n) for n in ("S&P500", "Nasdaq", "USDJPY", "DXY", "US10Y", "Gold", "Crude", "Nikkei", "Silver", "VIX")}
    ret = {n: s.pct_change() for n, s in mk.items()}
    A = {n: avail(s, days) for n, s in mk.items()}                      # その朝にわかる終値
    AR = {n: avail(r, days) for n, r in ret.items()}                    # その朝にわかる前日の騰落率
    ma = lambda n, k: avail(mk[n].rolling(k).mean(), days)

    def vbo_filter(pid, name, okfn):
        """値幅ブレイク（A01）を、okfn(side) が真の日だけ取引する。"""
        per, Ts = {}, []
        for a in COINS:
            s, T = R2.run(R2.periods(D[a], DAY, 0), a, extra_ok=lambda P_, side: np.nan_to_num(okfn(side)).astype(bool))
            per[a] = s
            Ts.append(T)
        return R3.show(pid, name, per, pd.concat(Ts), tf="15分足")

    out = {}
    # ---- I01・I03 株価指数が平均の上の時だけ
    for pid, n in (("I01", "S&P500"), ("I03", "Nasdaq")):
        print(f"\n==== {pid} {n}が平均の上の時だけ買いで持つ（単独）／値幅ブレイクの絞り込み")
        res = {}
        for k in (50, 200):
            with np.errstate(invalid="ignore"):
                cond = A[n] > ma(n, k)
            res[f"{k}日平均の上で買い持ち"] = run(pid, f"{n}が{k}日平均の上の時だけ買い持ち", holdwhile(lambda x, a, cond=cond: cond))
        with np.errstate(invalid="ignore"):
            btc_cond = D["BTC"]["dO"] > R1.sma_prev(D["BTC"], 200)
        res["参考：BTC自身の200日平均"] = run(pid, "参考：BTCが200日平均の上の時だけ買い持ち", holdwhile(lambda x, a: btc_cond))
        comp(pid, res)
        with np.errstate(invalid="ignore"):
            up = A[n] > ma(n, 200)
        out[pid] = {"単独": res,
                    "絞り込み": vbo_filter(pid, f"値幅ブレイク：{n}が200日平均の上の日だけ", lambda side: up),
                    "向き": vbo_filter(pid, f"値幅ブレイク：買いは{n}が200日平均の上、売りは下の日だけ",
                                     lambda side: (up if side == 1 else ~up))}

    # ---- I02・I04・I07・I09・I13 前日の向き（9時〜翌9時）
    for pid, n, lab in (("I02", "S&P500", "米株（S&P500）"), ("I04", "USDJPY", "ドル円（上げ＝円安）"), ("I07", "Gold", "金"),
                        ("I09", "Nikkei", "日経平均"), ("I13", "Silver", "銀")):
        print(f"\n==== {pid} {lab}が前日に上げたら買い・下げたら売り（9時〜翌9時）")
        res = {}
        for th in (0.0, 0.005):
            res[f"{th * 100:.1f}%より大きい動き"] = run(pid, f"{lab}の前日の動きが±{th * 100:.1f}%を超えた向きに",
                                                   daytrade(lambda x, a, n=n, th=th: np.where(np.abs(np.nan_to_num(AR[n])) > th,
                                                                                            np.sign(np.nan_to_num(AR[n])), 0)))
        comp(pid, res)
        out[pid] = res
    print("\n==== I02（続き）値幅ブレイクの向きが、前日の米株の向きと同じ時だけ")
    out["I02_f"] = vbo_filter("I02", "値幅ブレイク：前日の米株の向きと同じ時だけ", lambda side: side * np.nan_to_num(AR["S&P500"]) > 0)

    # ---- I05・I06 ドル指数・金利が下がっている時だけ買い持ち
    for pid, n, lab in (("I05", "DXY", "ドル指数"), ("I06", "US10Y", "米国の10年金利")):
        print(f"\n==== {pid} {lab}が20日平均より下（下げている）時だけ買い持ち／値幅ブレイクの絞り込み")
        res = {}
        for k in (20, 50):
            with np.errstate(invalid="ignore"):
                cond = A[n] < ma(n, k)
            res[f"{k}日平均の下"] = run(pid, f"{lab}が{k}日平均の下の時だけ買い持ち", holdwhile(lambda x, a, cond=cond: cond))
        comp(pid, res)
        with np.errstate(invalid="ignore"):
            dn = A[n] < ma(n, 20)
        out[pid] = {"単独": res, "向き": vbo_filter(pid, f"値幅ブレイク：買いは{lab}が20日平均の下、売りは上の日だけ",
                                                     lambda side: (dn if side == 1 else ~dn))}

    # ---- I08 原油の急変
    print("\n==== I08 原油が前日に大きく動いたら（20日の振れの2倍以上）、同じ向き／反対の向きにBTC・ETHを9時〜翌9時")
    res = {}
    sd = avail(ret["Crude"].rolling(20).std(), days)
    for sgn, lab in ((1, "同じ向き"), (-1, "反対の向き")):
        res[lab] = run("I08", f"原油の急変と{lab}", daytrade(lambda x, a, sgn=sgn: np.where(np.abs(np.nan_to_num(AR["Crude"])) > 2 * np.nan_to_num(sd, nan=9),
                                                                                       sgn * np.sign(np.nan_to_num(AR["Crude"])), 0)))
    comp("I08", res)
    out["I08"] = res

    # ---- I10 VIXの急上昇の翌日に買う
    print("\n==== I10 VIXが前日比+20%以上に上がった翌朝に買い、N日持つ")
    res = {}
    jump = np.nan_to_num(AR["VIX"]) >= 0.20
    for hold in (1, 5):
        def fn(x, a, hold=hold):
            sig = np.r_[jump[1:], False]                  # d日目の朝にわかる → 前の日の終わりの信号として渡す
            roll = (np.arange(len(x["O"])) % DAY == 84)
            pnl, te, tr, th = R3.swing_p(x["O"], x["H"], x["L"], x["C"], DAY, x["atr"], sig, np.zeros(len(sig), bool), 1, 99.0,
                                         hold, PF.SPEC[a][0], PF.SPEC[a][1] / 36500, roll)
            return pd.Series(pnl, index=x["idx"]), pd.DataFrame({"day": x["idx"][te], "R": tr, "hold": th})
        res[f"{hold}日"] = run("I10", f"VIXの急上昇の翌朝に買い・{hold}日持つ", fn)
    comp("I10", res)
    out["I10"] = res

    # ---- I11 リスクオン・オフ
    print("\n==== I11 前日に株が上げて金が下げた（リスクオン）なら買い、株が下げて金が上げた（リスクオフ）なら売り")
    on = (np.nan_to_num(AR["S&P500"]) > 0) & (np.nan_to_num(AR["Gold"]) < 0)
    off = (np.nan_to_num(AR["S&P500"]) < 0) & (np.nan_to_num(AR["Gold"]) > 0)
    res = {"リスクオン・オフ": run("I11", "リスクオンで買い・リスクオフで売り", daytrade(lambda x, a: np.where(on, 1, np.where(off, -1, 0))))}
    comp("I11", res)
    out["I11"] = res

    # ---- I12 株との相関が高い時だけ前日の米株の向きに
    print("\n==== I12 BTCと米株の30日の相関が高い時だけ、前日の米株の向きに9時〜翌9時")
    bt = pd.Series(D["BTC"]["dC"], index=days).pct_change()
    spx = ret["S&P500"].reindex(days)
    corr = bt.where(spx.notna()).rolling(60, min_periods=20).corr(spx).shift(1).to_numpy()
    res = {}
    for th in (0.3, 0.5):
        res[f"相関{th}以上"] = run("I12", f"相関が{th}以上の時だけ前日の米株の向き",
                               daytrade(lambda x, a, th=th: np.where(np.nan_to_num(corr) >= th, np.sign(np.nan_to_num(AR["S&P500"])), 0)))
    comp("I12", res)
    out["I12"] = res

    # ---- I15 米国株の時間のBTCの動き
    print("\n==== I15 前日の米国株の時間（日本22時半〜翌6時）のBTCの動きの向き／反対に、9時〜翌9時を持つ")
    xb = D["BTC"]
    us = np.r_[np.nan, (xb["O"][np.clip(np.arange(len(xb["dO"])) * DAY + 84, 0, len(xb["O"]) - 1)] /
                       xb["O"][np.arange(len(xb["dO"])) * DAY + 54] - 1)[:-1]]
    res = {}
    for sgn, lab in ((1, "同じ向き"), (-1, "反対の向き")):
        res[lab] = run("I15", f"前日の米国株の時間のBTCの動きと{lab}", daytrade(lambda x, a, sgn=sgn: sgn * np.sign(np.nan_to_num(us))))
    comp("I15", res)
    out["I15"] = res

    # ---- E17 VIXが高い日は値幅ブレイクを休む
    print("\n==== E17 VIX（前日の終値）が一定以上の日は、値幅ブレイク（A01）を休む／半分")
    res = {}
    vix = np.nan_to_num(A["VIX"], nan=0)
    for th in (20, 25, 30, 35):
        res[f"{th}以上で休む"] = vbo_filter("E17", f"VIXが{th}以上の日は休む", lambda side, th=th: vix < th)
    share = {th: (vix[(days >= MM.START) & (days <= MM.END)] >= th).mean() * 100 for th in (20, 25, 30, 35)}
    print("    VIXが20・25・30・35以上の日の割合：" + "・".join(f"{th}以上 {v:.0f}%" for th, v in share.items()))
    per = {a: per0[a] * np.where(vix >= 25, 0.5, 1.0) for a in COINS}
    res["25以上で半分"] = R3.show("E17", "VIXが25以上の日は半分", per, tf="15分足")
    out["E17"] = (res, share)
    return out, base


if __name__ == "__main__":
    main(sys.argv[1])
