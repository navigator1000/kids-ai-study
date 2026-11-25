#!/usr/bin/env python3
"""計画書（project/research/plan.md）第6回・後半：残りの B・F・K・Q の31件（BTC・ETH、2018年〜）。これで計画書の「未」は終わり。

  B 時間・カレンダー：B09 月初・月末／B10 月の中旬／B11 四半期末／B12 年末年始／B13 米国の祝日の前後／B16 毎月1日の買い／
                      B17 雇用統計の夜／B21 5月に売れ／B23 オプションの満期／B25 2日の連続性
  F ローソク足の形：F05 赤三兵／F06 明けの明星／F07 ダブルボトム／F09 フラッグ／F11 アウトサイドバー／F14 キリ番の突破／
                    F15 フィボナッチの押し／F18 高値更新の連続日数／F21 投げ売りの後の小さな陽線／F22 ヘッド＆ショルダー
  K 決済の工夫：K08 パラボリックで決済／K12 陰線で撤退／K13 決済を翌日の早朝に／K15 2段階の利確／K18 損切り後の入り直し／
                K19 週末前に決済／K20 損切りの幅を相場で変える
  Q 長期の持ち方（参考）：Q04 積立＋200日平均／Q05 毎月のリバランス／Q06 積立の曜日／Q07 下げた時だけ多く積み立てる
共通: Vantage CFD（往復0.05%、買いのスワップ年25〜35%、売り0%）、1R＝ATR（日足）1本分＝資金の0.5%（単利）。
積立（Q）は国内現物を想定（往復0.15%のうち買う時の0.075%・スワップなし、BTC・ETHを半分ずつ、2018年1月から）。
「休む・だけ」の条件は、同じ割合の日をランダムに選んだ1,000回と比べる。
結果は results.csv（グループ「計画・第6回後半」、計画IDつき）と plan.csv。
使い方: python3 project/tools/round6b_lab.py <キャッシュ> [b,f,k,q のどれか（カンマ区切り）]
"""
import sys
import warnings

import numpy as np
import pandas as pd
from numba import njit

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
import more_methods_lab as MM  # noqa: E402
import portfolio_lab as PF  # noqa: E402
import round1_lab as R1  # noqa: E402
import round2_lab as R2  # noqa: E402
import round2b_lab as R2B  # noqa: E402
import round3_lab as R3  # noqa: E402
import round3b_lab as R3B  # noqa: E402
import round4_lab as R4  # noqa: E402
import round4b_lab as R4B  # noqa: E402
import round5b_lab as R5B  # noqa: E402
import round6_lab as R6  # noqa: E402
from registry import drop, metrics_row, record  # noqa: E402

COINS = ("BTC", "ETH")
GROUP = "計画・第6回後半"
DAY = 96
R3.GROUP = R5B.GROUP = R6.GROUP = GROUP
warnings.filterwarnings("ignore")
SUM = {}
prev = lambda a, k=1: np.asarray(a, float) if k == 0 else R6.prev(a, k)
swing, limit, two = R3B.swing, R3B.limit, R3B.two_sides


def keep(pid, **kw):
    SUM.setdefault(pid, {}).update(kw)


def comp(ctx, pid, res):
    return R6.comp(ctx, pid, res)


def near(days, dates, before, after):
    m = np.zeros(len(days), bool)
    for c in pd.DatetimeIndex(dates):
        m |= (days >= c - pd.Timedelta(days=before)) & (days <= c + pd.Timedelta(days=after))
    return m


def sw_fn(sig_fn, side=1, maxd=3, xsig_fn=None, stop=99.0):
    """終値で合図（sig_fn(x)）→翌朝入り、maxd日（か xsig）で翌朝決済。"""
    def fn(x, a):
        with np.errstate(invalid="ignore"):
            sig = np.nan_to_num(sig_fn(x)).astype(bool)
            xs = None if xsig_fn is None else xsig_fn(x)
            p, T = swing(x, a, sig, side, maxd, xs, stop)
        return pd.Series(p, index=x["idx"]), T
    return fn


# ---------------------------------------------------------------- B 時間・カレンダー
def sec_b(ctx):
    D, days = ctx["D"], ctx["days"]
    dom, mon, wd = days.day.to_numpy(), days.month.to_numpy(), days.dayofweek.to_numpy()
    dim = days.days_in_month.to_numpy()
    vf, hf = R5B.vf, R5B.hf

    print("\n==== B09 月の最初・最後のN日だけ買い持ち")
    out = {}
    for n in (3, 5, 7):
        out[f"最初の{n}日"] = hf(ctx, "B09", f"月の最初の{n}日", dom <= n)
        out[f"最後の{n}日"] = hf(ctx, "B09", f"月の最後の{n}日", dom > dim - n)
    keep("B09", rows={k: R5B.brief(v) for k, v in out.items()})

    print("\n==== B10 月の中旬（10〜20日）")
    mid_m = (dom >= 10) & (dom <= 20)
    keep("B10", rows={"休む": R5B.brief(vf(ctx, "B10", "月の10〜20日は休む", ~mid_m)), "だけ持つ": R5B.brief(hf(ctx, "B10", "月の10〜20日", mid_m))})

    print("\n==== B11 四半期末（3・6・9・12月の最後の5日）")
    qe = np.isin(mon, [3, 6, 9, 12]) & (dom > dim - 5)
    keep("B11", rows={"だけ持つ": R5B.brief(hf(ctx, "B11", "四半期末の5日", qe)), "休む": R5B.brief(vf(ctx, "B11", "四半期末の5日は休む", ~qe))})

    print("\n==== B12 年末年始（12月20日〜1月10日）")
    ny = ((mon == 12) & (dom >= 20)) | ((mon == 1) & (dom <= 10))
    keep("B12", rows={"だけ持つ": R5B.brief(hf(ctx, "B12", "12月20日〜1月10日", ny)), "休む": R5B.brief(vf(ctx, "B12", "12月20日〜1月10日は休む", ~ny))})

    print("\n==== B13 米国の祝日の前日・翌日")
    import holidays
    us = pd.DatetimeIndex(sorted(holidays.US(years=range(2017, 2027)).keys()))
    bef, aft = days.isin(us - pd.Timedelta(days=1)), days.isin(us + pd.Timedelta(days=1))
    keep("B13", rows={"前日": R5B.brief(hf(ctx, "B13", "米国の祝日の前日", bef)), "翌日": R5B.brief(hf(ctx, "B13", "米国の祝日の翌日", aft)),
                      "前後は休む": R5B.brief(vf(ctx, "B13", "米国の祝日の前日・当日・翌日は休む", ~(bef | aft | days.isin(us))))})

    print("\n==== B16 毎月1日の9時に買い、N日後の9時に決済")
    res = {}
    last_day = (dom == dim)
    for n in (1, 3, 5):
        res[f"{n}日"] = R6.run_fn(ctx, "B16", f"毎月1日に買い・{n}日持つ", sw_fn(lambda x, ld=last_day: ld, 1, n))
    comp(ctx, "B16", res)

    print("\n==== B17 雇用統計の夜（毎月第1金曜、日本21時半か22時半）：その日の値幅ブレイクを休む／発表後1時間の向きに翌朝9時まで")
    rel = []
    for y in range(2017, 2027):
        for m_ in range(1, 13):
            d1 = pd.Timestamp(year=y, month=m_, day=1)
            fr = d1 + pd.Timedelta(days=(4 - d1.dayofweek) % 7)
            t = pd.Timestamp(f"{fr:%Y-%m-%d} 08:30", tz="America/New_York").tz_convert("UTC").tz_localize(None)
            rel.append(t)
    rel = pd.DatetimeIndex(rel)
    rd = days.get_indexer(rel.normalize())
    nfp = np.zeros(len(days), bool)
    nfp[rd[rd >= 0]] = True
    r_skip = vf(ctx, "B17", "雇用統計の日は休む", ~nfp)
    res = {}
    for sgn, lab in ((1, "同じ向き"), (-1, "反対の向き")):
        def fn(x, a, sgn=sgn):
            n = len(x["dO"])
            i1 = np.full(n, -1)
            for t, d in zip(rel, rd):
                if d >= 0:
                    i1[d] = d * DAY + (t.hour * 60 + t.minute) // 15
            O = x["O"]
            ok = (i1 >= 0) & (i1 + 4 < len(O))
            mv = np.where(ok, O[np.clip(i1 + 4, 0, len(O) - 1)] - O[np.clip(i1, 0, len(O) - 1)], 0.0)
            side = sgn * np.sign(mv) * ok
            s, T = R2B.window(x, a, i1 + 4, (np.arange(n) + 1) * DAY, side, np.where(side > 0, 1.0, 0.5))
            return s, T.assign(hold=0)
        res[lab] = R6.run_fn(ctx, "B17", f"雇用統計の発表後1時間の{lab}に、翌朝9時まで", fn, tf="15分足")
    comp(ctx, "B17", res)
    keep("B17", skip=R5B.brief(r_skip))

    print("\n==== B21 5月に売れ：11〜4月だけ値幅ブレイク（5〜10月は休む）")
    keep("B21", rows={"11〜4月だけ": R5B.brief(vf(ctx, "B21", "11〜4月だけ（5〜10月は休む）", np.isin(mon, [11, 12, 1, 2, 3, 4])))})

    print("\n==== B23 オプションの満期（毎月最終金曜の日本17時＝UTC8時）の前後")
    lf = np.zeros(len(days), bool)
    for y in range(2017, 2027):
        for m_ in range(1, 13):
            e = pd.Timestamp(year=y, month=m_, day=1) + pd.offsets.MonthEnd(0)
            e = e - pd.Timedelta(days=(e.dayofweek - 4) % 7)
            i = days.get_indexer([e])[0]
            if i >= 0:
                lf[i] = True
    out = {"休む": vf(ctx, "B23", "満期の日（最終金曜）は休む", ~lf)}
    for lab, off in (("満期の後24時間", 0), ("満期の前24時間", -DAY)):
        def fn(x, a, off=off):
            n = len(x["dO"])
            i1 = np.where(lf, np.arange(n) * DAY + 32 + off, -1)
            s, T = R2B.window(x, a, i1, i1 + DAY, lf.astype(float), np.ones(n))
            return s, T.assign(hold=1)
        out[lab] = R6.run_fn(ctx, "B23", f"{lab}だけ買い", fn, tf="15分足")
    mv = {}
    for a in COINS:
        x = D[a]
        O = x["O"]
        idx = np.nonzero(lf)[0] * DAY + 32
        idx = idx[(idx - DAY >= 0) & (idx + DAY < len(O))]
        allr = np.abs(np.log(O[DAY::DAY] / O[:-DAY:DAY]))
        mv[a] = (np.mean(np.abs(np.log(O[idx] / O[idx - DAY]))) * 100, np.mean(np.abs(np.log(O[idx + DAY] / O[idx]))) * 100, np.nanmean(allr) * 100)
        print(f"    {a}：値動きの大きさ（絶対値）の平均 満期の前24時間{mv[a][0]:.2f}%・後24時間{mv[a][1]:.2f}%（ふだんの24時間{mv[a][2]:.2f}%）")
    keep("B23", rows={k: R5B.brief(v) for k, v in out.items()}, mv=mv)

    print("\n==== B25 N日続けて上げたら翌日も買い（下げたら売り）、9時〜翌9時")
    res = {}
    for n in (2, 3):
        sides = {}
        for a in COINS:
            c = D[a]["dC"]
            up = np.ones(len(c), bool)
            dn = np.ones(len(c), bool)
            for k in range(n):
                with np.errstate(invalid="ignore"):
                    up &= prev(c, k + 1) > prev(c, k + 2)
                    dn &= prev(c, k + 1) < prev(c, k + 2)
            sides[a] = np.where(up, 1.0, np.where(dn, -1.0, 0.0))
        res[f"{n}日"] = R6.dt(ctx, "B25", f"{n}日続けて上げたら買い・下げたら売り（9時〜翌9時）", sides)
    comp(ctx, "B25", res)


# ---------------------------------------------------------------- F ローソク足の形
def sec_f(ctx):
    D = ctx["D"]

    def ohlc(x):
        return x["dO"], x["dH"], x["dL"], x["dC"], x["atr"]

    print("\n==== F05 赤三兵：3日続けて陽線で、終値も3日続けて上げたら翌朝買い（N日持つ）")
    res = {}

    def sanpei(x):
        o, h, l, c, A = ohlc(x)
        s = np.ones(len(c), bool)
        for k in range(3):
            with np.errstate(invalid="ignore"):
                s &= (prev(c, k) > prev(o, k)) & (prev(c, k) > prev(c, k + 1))
        return s
    for n in (1, 3, 5):
        res[f"{n}日"] = R6.run_fn(ctx, "F05", f"赤三兵・{n}日持つ", sw_fn(sanpei, 1, n))
    comp(ctx, "F05", res)

    print("\n==== F06 明けの明星：大陰線（実体ATR×0.5以上）・小さな足（実体がその3割以下）・大陽線（1本目の実体の半分より上で引ける）→翌朝買い")
    res = {}

    def star(x):
        o, h, l, c, A = ohlc(x)
        b1o, b1c, b2o, b2c = prev(o, 2), prev(c, 2), prev(o, 1), prev(c, 1)
        with np.errstate(invalid="ignore"):
            big1 = (b1o - b1c) >= 0.5 * A
            small2 = np.abs(b2c - b2o) <= 0.3 * (b1o - b1c)
            big3 = ((c - o) >= 0.5 * A) & (c > (b1o + b1c) / 2)
        return big1 & small2 & big3
    for n in (3, 5):
        res[f"{n}日"] = R6.run_fn(ctx, "F06", f"明けの明星・{n}日持つ", sw_fn(star, 1, n))
    comp(ctx, "F06", res)

    print("\n==== F07 ダブルボトム（簡易）：20日の中の2つの安値が2%以内、その間の高値を終値で抜けたら翌朝買い（損切りATR×2）")
    res = {}

    def dbl(x):
        o, h, l, c, A = ohlc(x)
        s = np.zeros(len(c), bool)
        for d in range(21, len(c)):
            w = l[d - 20:d]
            i1 = int(np.argmin(w))
            for i2 in range(len(w)):
                if abs(i2 - i1) >= 5 and abs(w[i2] / w[i1] - 1) <= 0.02:
                    a_, b_ = sorted((i1, i2))
                    neck = h[d - 20 + a_:d - 20 + b_ + 1].max()
                    if c[d] > neck >= c[d - 1]:
                        s[d] = True
                        break
        return s
    for n in (5, 10):
        res[f"{n}日"] = R6.run_fn(ctx, "F07", f"ダブルボトム・{n}日持つ", sw_fn(dbl, 1, n, stop=2.0))
    comp(ctx, "F07", res)

    print("\n==== F09 フラッグ：3日で+10%以上上げた後の小さな押し（上げ幅の半分まで）から、押しの間の高値を終値で抜けたら翌朝買い（損切りATR×2）")
    res = {}

    def flag(x):
        o, h, l, c, A = ohlc(x)
        s = np.zeros(len(c), bool)
        r3 = c / prev(c, 3) - 1
        for d in range(12, len(c)):
            for p in range(d - 8, d - 1):
                if r3[p] >= 0.10:
                    base, top = c[p - 3], h[p:d].max()
                    pull = l[p + 1:d].min() if d > p + 1 else np.inf
                    if pull > top - 0.5 * (top - base) and c[d] > top >= c[d - 1] and pull < top:
                        s[d] = True
                        break
        return s
    for n in (3, 5):
        res[f"{n}日"] = R6.run_fn(ctx, "F09", f"フラッグ・{n}日持つ", sw_fn(flag, 1, n, stop=2.0))
    comp(ctx, "F09", res)

    print("\n==== F11 アウトサイドバー：前日の高値・安値を両方超えた日の、終値の向きに翌日持つ（売り×0.5）")
    res = {}
    sides = {}
    for a in COINS:
        o, h, l, c, A = ohlc(D[a])
        with np.errstate(invalid="ignore"):
            ob = (h > prev(h)) & (l < prev(l))
        sides[a] = prev(np.where(ob, np.sign(c - o), 0.0))
        sides[a] = np.nan_to_num(sides[a])
    res["1日"] = R6.dt(ctx, "F11", "アウトサイドバーの終値の向きに9時〜翌9時", sides)

    def ob3(x, a):
        o, h, l, c, A = ohlc(x)
        with np.errstate(invalid="ignore"):
            ob = (h > prev(h)) & (l < prev(l))
            return two(lambda x_, a_: swing(x_, a_, ob & (c > o), 1, 3), lambda x_, a_: swing(x_, a_, ob & (c < o), -1, 3), 0.5)(x, a)
    res["3日"] = R6.run_fn(ctx, "F11", "アウトサイドバーの終値の向きに3日", ob3)
    comp(ctx, "F11", res)

    print("\n==== F14 キリ番の突破：始値の上の一番近いキリ番（BTC1万ドル・ETH千ドル刻み）を超えたら買い、翌朝9時に決済（1ATR以内のキリ番だけ）")
    res = {}
    for ma in (False, True):
        per, Ts = {}, []
        for a in COINS:
            step = 10000.0 if a == "BTC" else 1000.0

            def trig(Pd, side, step=step):
                o = Pd["pO"]
                t = (np.floor(o / step) + 1) * step
                with np.errstate(invalid="ignore"):
                    return np.where((t - o) <= Pd["atr"], t, np.nan)
            s, T = R2.run(R2.periods(D[a], DAY, 0), a, ma=ma, trig_fn=trig, short_w=0)
            per[a] = s
            Ts.append(T)
        lab = "5日平均の向きだけ" if ma else "向きの条件なし"
        res[lab] = R6.show("F14", f"キリ番の突破で買い（{lab}）", per, pd.concat(Ts, ignore_index=True), tf="15分足")
    comp(ctx, "F14", res)

    print("\n==== F15 フィボナッチの押し：直近20日の上昇（安値→高値、ATR×2以上）の38.2・50・61.8%に買いの指値、5日で決済（損切りATR×2）")
    res = {}
    for r in (0.382, 0.5, 0.618):
        def fn(x, a, r=r):
            o, h, l, c, A = ohlc(x)
            hh = pd.Series(h).rolling(20).max().shift(1).to_numpy()
            ll = pd.Series(l).rolling(20).min().shift(1).to_numpy()
            ih = pd.Series(h).rolling(20).apply(np.argmax, raw=True).shift(1).to_numpy()
            il = pd.Series(l).rolling(20).apply(np.argmin, raw=True).shift(1).to_numpy()
            lvl = hh - r * (hh - ll)
            with np.errstate(invalid="ignore"):
                ok = (il < ih) & ((hh - ll) >= 2 * A) & (o > lvl)
            p_, T_ = limit(x, a, np.nan_to_num(ok).astype(bool), np.nan_to_num(lvl), 1, 5, 2.0)
            return pd.Series(p_, index=x["idx"]), T_
        res[f"{r * 100:.1f}%"] = R6.run_fn(ctx, "F15", f"フィボナッチの押し{r * 100:.1f}%で買い", fn, tf="15分足")
    comp(ctx, "F15", res)

    print("\n==== F18 N日続けて高値を更新したら翌朝買い、3日持つ")
    res = {}
    for n in (2, 3, 4, 5):
        def sig(x, n=n):
            h = x["dH"]
            s = np.ones(len(h), bool)
            for k in range(n):
                with np.errstate(invalid="ignore"):
                    s &= prev(h, k) > prev(h, k + 1)
            return s
        res[f"{n}日"] = R6.run_fn(ctx, "F18", f"{n}日続けて高値更新・3日持つ", sw_fn(sig, 1, 3))
    comp(ctx, "F18", res)

    print("\n==== F21 投げ売りの後の小さな陽線：前日に実体ATR×1.5以上の陰線、当日は実体ATR×0.5以下の陽線→翌朝買い")
    res = {}

    def capit(x):
        o, h, l, c, A = ohlc(x)
        with np.errstate(invalid="ignore"):
            return ((prev(o) - prev(c)) >= 1.5 * prev(A)) & (c > o) & ((c - o) <= 0.5 * A)
    for n in (3, 5):
        res[f"{n}日"] = R6.run_fn(ctx, "F21", f"投げ売りの後の小さな陽線・{n}日持つ", sw_fn(capit, 1, n))
    comp(ctx, "F21", res)

    print("\n==== F22 ヘッド＆ショルダー（簡易）：30日の中の3つの山（真ん中が一番高く、両肩が3%以内）、谷の安値（ネックライン）を終値で割ったら翌朝売り（損切りATR×2）")
    res = {}

    def hs(x):
        o, h, l, c, A = ohlc(x)
        s = np.zeros(len(c), bool)
        pk = np.zeros(len(c), bool)
        for i in range(3, len(c) - 3):
            pk[i] = h[i] == h[i - 3:i + 4].max()
        for d in range(33, len(c)):
            ps = [i for i in range(d - 30, d - 3) if pk[i]]
            if len(ps) < 3:
                continue
            p1, p2, p3 = ps[-3], ps[-2], ps[-1]
            if h[p2] > h[p1] and h[p2] > h[p3] and abs(h[p1] / h[p3] - 1) <= 0.03:
                neck = l[p1:p3 + 1].min()
                if c[d] < neck <= c[d - 1]:
                    s[d] = True
        return s
    for n in (5, 10):
        res[f"{n}日"] = R6.run_fn(ctx, "F22", f"ヘッド＆ショルダーで売り・{n}日持つ", sw_fn(hs, -1, n, stop=2.0))
    comp(ctx, "F22", res)


# ---------------------------------------------------------------- K 決済の工夫
@njit
def vbo_ex(O, H, L, C, P, trig, ok, A, side, cost, swd, last, xbar, stop_k, reent, tp1, tp2, hmode):
    """値幅ブレイク（A01の形）に決済の工夫をつける。xbar[d]：その日の xbar 本目の前で決済（96なら翌朝9時）。
    stop_k[d]：損切り（ATR倍、NaNなら置かない）。reent：損切りの後、同じ日にもう一度逆指値に届いたら1回だけ入り直す。
    tp1・tp2：利確（ATR倍、0なら置かない。両方ある時は半分ずつ）。hmode：1＝約定後の最初の1時間足、2＝毎時、逆向きの足で終わったら決済。
    同じ足で損切りと利確の両方に届いたら、損切りを先とみなす。返り値：毎日のR（その日の取引の合計）と取引回数。"""
    nd = trig.shape[0]
    Rd = np.zeros(nd)
    nt = np.zeros(nd, np.int64)
    for d in range(nd):
        if not ok[d] or not (A[d] > 0) or np.isnan(trig[d]):
            continue
        b0 = d * P
        Ad = A[d]
        xb = xbar[d]
        lastb = min(last, xb - 4)
        tries = 2 if reent else 1
        j = b0
        first = True
        while tries > 0 and j < b0 + lastb:
            k = -1
            e = 0.0
            while j < b0 + lastb:
                if first and j == b0 and side * (O[j] - trig[d]) > 0:
                    j = b0 + lastb
                    break
                if (side == 1 and H[j] >= trig[d]) or (side == -1 and L[j] <= trig[d]):
                    k = j
                    e = max(O[j], trig[d]) if side == 1 else min(O[j], trig[d])
                    break
                j += 1
            if k < 0:
                break
            tries -= 1
            first = False
            stop = e - side * stop_k[d] * Ad if not np.isnan(stop_k[d]) else np.nan
            t1 = e + side * tp1 * Ad if tp1 > 0 else np.nan
            t2 = e + side * tp2 * Ad if tp2 > 0 else np.nan
            left = 1.0
            got = 0.0
            rolled = 0.0
            h1 = False
            h2 = False
            stopped = False
            hb = ((k - b0) // 4 + 1) * 4 + b0
            checked = False
            out = b0 + xb - 1
            jj = k
            while jj < b0 + xb:
                pos = jj - b0
                if jj > k and pos == 84:
                    rolled = left
                if not np.isnan(stop):
                    adv = L[jj] if side == 1 else H[jj]
                    if side * (adv - stop) <= 0:
                        px = stop if jj == k else (min(O[jj], stop) if side == 1 else max(O[jj], stop))
                        got += left * side * (px - e) / Ad
                        left = 0.0
                        stopped = True
                        out = jj
                        break
                if not h1 and not np.isnan(t1):
                    if (side == 1 and H[jj] >= t1) or (side == -1 and L[jj] <= t1):
                        px = t1 if jj == k else (max(O[jj], t1) if side == 1 else min(O[jj], t1))
                        fr = 0.5 if not np.isnan(t2) else 1.0
                        got += fr * side * (px - e) / Ad
                        left -= fr
                        h1 = True
                if h1 and not h2 and not np.isnan(t2) and left > 1e-9:
                    if (side == 1 and H[jj] >= t2) or (side == -1 and L[jj] <= t2):
                        px = t2 if jj == k else (max(O[jj], t2) if side == 1 else min(O[jj], t2))
                        got += left * side * (px - e) / Ad
                        left = 0.0
                        h2 = True
                if left <= 1e-9:
                    out = jj
                    break
                if hmode > 0 and jj >= hb + 3 and (jj - b0) % 4 == 3 and (hmode == 2 or not checked):
                    checked = True
                    if side * (C[jj] - O[jj - 3]) < 0:
                        got += left * side * (C[jj] - e) / Ad
                        left = 0.0
                        out = jj
                        break
                jj += 1
            if left > 1e-9:
                got += left * side * (C[b0 + xb - 1] - e) / Ad
            Rd[d] += got - cost * e / Ad - rolled * swd * e / Ad
            nt[d] += 1
            if not (stopped and reent):
                break
            j = out + 1
    return Rd, nt


def vbo_k(ctx, xbar=None, stop=None, reent=False, tp1=0.0, tp2=0.0, hmode=0, skip=None):
    """A01（5日平均の向き、買い×1・売り×0.5、k=0.8）に vbo_ex の決済の工夫をつけた、銘柄ごとの毎日の損益と取引一覧。"""
    per, Ts = {}, []
    for a in COINS:
        Pd = R2.periods(ctx["D"][a], DAY, 0)
        fl = Pd["fl"]
        n = Pd["n"]
        tot = np.zeros(n)
        xb = np.full(n, DAY, np.int64) if xbar is None else (np.full(n, xbar, np.int64) if np.isscalar(xbar) else xbar.astype(np.int64))
        sk = np.full(n, np.nan) if stop is None else (np.full(n, float(stop)) if np.isscalar(stop) else np.asarray(stop[a], float))
        for side, w in ((1, 1.0), (-1, 0.5)):
            trig = (Pd["pO"] + side * 0.8 * Pd["rng"]).astype(np.float64)
            with np.errstate(invalid="ignore"):
                ok = ~np.isnan(Pd["atr"]) & ~np.isnan(trig) & (side * (Pd["pO"] - Pd["sma5"]) > 0)
            if skip is not None:
                ok &= ~skip
            swd = (PF.SPEC[a][1] if side == 1 else PF.SPEC[a][2]) / 36500
            Rd, nt = vbo_ex(fl["O"], fl["H"], fl["L"], fl["C"], DAY, trig, ok, Pd["atr"], side, PF.SPEC[a][0], swd, DAY - 4, xb, sk,
                            reent, tp1, tp2, hmode)
            tot += Rd * 0.5 * w
            on = nt > 0
            Ts.append(pd.DataFrame({"coin": a, "day": Pd["idx"][on], "R": Rd[on], "side": side}))
        per[a] = pd.Series(tot, index=Pd["idx"])
    T = pd.concat(Ts, ignore_index=True)
    return per, T[(T.day >= MM.START) & (T.day <= MM.END)]


def sec_k(ctx):
    D, days = ctx["D"], ctx["days"]
    sh = lambda pid, name, pt: R6.show(pid, name, *pt, tf="15分足")
    b = sh("K", "確かめ：vbo_exで A01 と同じ形", vbo_k(ctx))
    print(f"    （A01 は年{ctx['base']['年平均%']:+.2f}%・安定度{ctx['base']['安定度']:.2f}）")

    print("\n==== K08 パラボリックで決済：20日高値を終値で抜けたら買い（安値なら売り）。決済はSARの反転／参考：10日安値（高値）")
    res = {}
    for ex in ("SAR", "10日安値"):
        for lo in (False, True):
            def fn(x, a, ex=ex, lo=lo):
                h, l, c = x["dH"], x["dL"], x["dC"]
                hh, ll = pd.Series(h).rolling(20).max().shift(1).to_numpy(), pd.Series(l).rolling(20).min().shift(1).to_numpy()
                with np.errstate(invalid="ignore"):
                    if ex == "SAR":
                        sd = R6.psar_dir(h, l, 0.02, 0.2)
                        xl, xs = sd < 0, sd > 0
                    else:
                        xl = c < pd.Series(l).rolling(10).min().shift(1).to_numpy()
                        xs = c > pd.Series(h).rolling(10).max().shift(1).to_numpy()
                    pl = R3.state(c > hh, xl, 1)
                    ps = 0 if lo else R3.state(c < ll, xs, -1)
                return pl + ps
            res[("SARで決済" if ex == "SAR" else "参考：10日安値で決済") + ("・買いだけ" if lo else "")] = R6.run_pos(
                ctx, "K08", "20日高値の抜けで入り、" + ("SARの反転で決済" if ex == "SAR" else "10日安値で決済（参考）"), fn, lo)
    comp(ctx, "K08", res)

    print("\n==== K12 陰線で撤退：約定した後の1時間足が、逆向き（買いなら陰線）で終わったら決済")
    out = {"最初の1時間だけ確かめる": sh("K12", "約定後の最初の1時間足が逆向きなら決済", vbo_k(ctx, hmode=1)),
           "毎時確かめる": sh("K12", "約定後、1時間足が逆向きで終わったらその時点で決済", vbo_k(ctx, hmode=2))}
    keep("K12", rows={k: R5B.brief(v) for k, v in out.items()})

    print("\n==== K13 決済を翌日の早朝に：翌朝9時ではなく、日本時間5・6・7・8時に決済（6時をまたぐと買いにスワップ）")
    out = {}
    for hr, xb in ((5, 80), (6, 84), (7, 88), (8, 92)):
        out[f"{hr}時"] = sh("K13", f"日本時間{hr}時に決済", vbo_k(ctx, xbar=xb))
    keep("K13", rows={k: R5B.brief(v) for k, v in out.items()})

    print("\n==== K15 2段階の利確：ATR×1で半分、ATR×2で残り（届かなければ翌朝9時）")
    out = {"1と2で半分ずつ": sh("K15", "ATR×1で半分・ATR×2で残りを利確", vbo_k(ctx, tp1=1.0, tp2=2.0)),
           "参考：ATR×1で全部": sh("K15", "参考：ATR×1で全部利確", vbo_k(ctx, tp1=1.0)),
           "参考：ATR×2で全部": sh("K15", "参考：ATR×2で全部利確", vbo_k(ctx, tp1=2.0))}
    keep("K15", rows={k: R5B.brief(v) for k, v in out.items()})

    print("\n==== K18 損切り後の入り直し：損切り（ATR×0.5・1）にかかった後、同じ日にもう一度逆指値に届いたら1回だけ入り直す")
    out = {}
    for sk in (0.5, 1.0):
        out[f"損切り{sk}・入り直さない"] = sh("K18", f"損切りATR×{sk}（入り直さない）", vbo_k(ctx, stop=sk))
        out[f"損切り{sk}・入り直す"] = sh("K18", f"損切りATR×{sk}＋同じ日に1回入り直す", vbo_k(ctx, stop=sk, reent=True))
    keep("K18", rows={k: R5B.brief(v) for k, v in out.items()})

    print("\n==== K19 週末前に決済：金曜（日本時間）の23時に決済し、土日は取引しない")
    wd = days.dayofweek.to_numpy()
    xb = np.where(wd == 4, 56, DAY)
    wk = (wd == 5) | (wd == 6)
    out = {"金曜23時に決済・土日は休む": sh("K19", "金曜23時に決済・土日は休む", vbo_k(ctx, xbar=xb, skip=wk)),
           "金曜23時に決済だけ": sh("K19", "金曜23時に決済（土日も取引）", vbo_k(ctx, xbar=xb)),
           "参考：土日を休むだけ": sh("K19", "参考：土日を休むだけ（B02）", vbo_k(ctx, skip=wk))}
    keep("K19", rows={k: R5B.brief(v) for k, v in out.items()})

    print("\n==== K20 損切りの幅を相場で変える：静かな日（10日の実現ボラが1年の中央値より小さい）はATR×1、荒い日はATR×2")
    out = {"静か1・荒い2": sh("K20", "損切り：静かな日ATR×1・荒い日ATR×2", vbo_k(ctx, stop={a: np.where(ctx["Mq"][a] > 0, 1.0, 2.0) for a in COINS})),
           "静か2・荒い1": sh("K20", "損切り：静かな日ATR×2・荒い日ATR×1（逆）", vbo_k(ctx, stop={a: np.where(ctx["Mq"][a] > 0, 2.0, 1.0) for a in COINS}))}
    for sk in (1.0, 1.5, 2.0):
        out[f"一律{sk}"] = sh("K20", f"参考：損切りATR×{sk}（一律）", vbo_k(ctx, stop=sk))
    keep("K20", rows={k: R5B.brief(v) for k, v in out.items()}, base=R5B.brief(b))


# ---------------------------------------------------------------- Q 長期の持ち方（参考）
def dca(D, buy, contrib, spend_all=True, cost=0.00075):
    """contrib[d] 円をその日に積み立て、buy[d] の日の朝9時に BTC・ETH を半分ずつ買う（国内現物、買う時のコスト0.075%）。
    spend_all=True なら手元の現金を全部使う（待った分もまとめて買う）。False ならその日の積立分だけ買い、待った分は現金のまま。
    返り値：毎日の評価額（現金込み）と、積み立てた総額。"""
    days = D["BTC"]["idx"]
    w = (days >= MM.START) & (days <= MM.END)
    units = {a: 0.0 for a in COINS}
    cash = inv = 0.0
    val, invs = np.zeros(len(days)), np.zeros(len(days))
    for d in np.nonzero(w)[0]:
        inv += contrib[d]
        cash += contrib[d]
        if buy[d] and cash > 0:
            spend = cash if spend_all else min(cash, contrib[d])
            for a in COINS:
                units[a] += spend / 2 * (1 - cost) / D[a]["dO"][d]
            cash -= spend
        val[d] = sum(units[a] * D[a]["dC"][d] for a in COINS) + cash
        invs[d] = inv
    return pd.Series(val[w], index=days[w]), pd.Series(invs[w], index=days[w])


def dca_stats(v, inv):
    ok = inv > 0
    r = (v[ok] / inv[ok] - 1)
    return dict(mult=float(v.iloc[-1] / inv.iloc[-1]), worst=float(r.min() * 100), final=float(r.iloc[-1] * 100), inv=float(inv.iloc[-1]))


def sec_q(ctx):
    D, days = ctx["D"], ctx["days"]
    print("\n==== Q 長期の持ち方（参考）：国内現物で、BTC・ETHを半分ずつ。毎週7円ずつ積み立てて買う形を基準にする（2018年1月〜）")
    wd = days.dayofweek.to_numpy()
    out = {}

    def run(lab, buy, contrib=None, spend_all=True):
        contrib = np.where(buy | (wd == 0), 7.0, 0.0) if contrib is None else contrib
        v, inv = dca(D, buy, contrib, spend_all)
        st = dca_stats(v, inv)
        out[lab] = st
        print(f"    {lab:34s} 最後に積み立てた総額の{st['mult']:.2f}倍（{st['final']:+.0f}%）・一番悪い時{st['worst']:+.0f}%・積み立てた総額{st['inv']:,.0f}")
        record(グループ=GROUP, 計画ID=lab.split("：")[0] if "：" in lab else "Q", 名前=lab, 銘柄="BTC,ETH", 時間足="日足",
               コストの前提="国内現物（買う時0.075%・スワップなし）", **{"1回の量": "毎週7円ずつ積み立てる（下げた週は2倍など）"},
               メモ=f"2018年1月〜：積み立てた総額の{st['mult']:.2f}倍・含み損益が一番悪い時{st['worst']:+.0f}%")
        return st
    run("Q：参考：毎週月曜に積立", wd == 0)
    print("\n==== Q06 積立の曜日：毎週の買う曜日（日本時間の朝9時）を変える")
    q6 = {}
    for k, nm in enumerate("月火水木金土日"):
        q6[nm] = run(f"Q06：毎週{nm}曜に積立", wd == k, np.where(wd == k, 7.0, 0.0))
    print("\n==== Q04 積立＋200日平均：BTCが200日平均の上の週だけ買う（下の週は現金で待ち、上に戻ったらまとめて買う／待った分は買わない）")
    up = np.nan_to_num(D["BTC"]["dO"] > R1.sma_prev(D["BTC"], 200)).astype(bool)
    mon7 = np.where(wd == 0, 7.0, 0.0)
    run("Q04：200日平均の上の月曜だけ（待った分もまとめて買う）", (wd == 0) & up, mon7, True)
    run("Q04：200日平均の上の月曜だけ（待った分は現金のまま）", (wd == 0) & up, mon7, False)
    print("\n==== Q07 下げた時だけ多く積み立てる：BTCが前の週に−10%（−5%）以上下げた月曜は2倍の金額を積み立てて買う（積み立てた総額1円あたりで比べる）")
    lb = np.log(D["BTC"]["dO"])
    wr = lb - prev(lb, 7)
    for th in (-0.10, -0.05):
        with np.errstate(invalid="ignore"):
            amt = np.where(wd == 0, np.where(np.nan_to_num(wr) <= np.log1p(th), 14.0, 7.0), 0.0)
        run(f"Q07：前週{th * 100:.0f}%以下なら2倍", wd == 0, amt)
    print("\n==== Q05 毎月のリバランス：2018年1月にまとめて買い、BTC・ETH・現金を毎月同じ割合に戻す／戻さない（売り買いに0.15%）")
    q5 = {}
    idx = days[(days >= MM.START) & (days <= MM.END)]
    px = {a: pd.Series(D[a]["dC"], index=days).reindex(idx).to_numpy() for a in COINS}
    me = idx.to_series().groupby(idx.to_period("M")).transform("max").to_numpy() == idx.to_numpy()
    for wb, we_, wc in ((0.3, 0.3, 0.4), (0.5, 0.5, 0.0), (0.2, 0.2, 0.6)):
        for reb in (True, False):
            u = {"BTC": wb / px["BTC"][0], "ETH": we_ / px["ETH"][0]}
            cash = wc
            v = np.zeros(len(idx))
            for t in range(len(idx)):
                v[t] = u["BTC"] * px["BTC"][t] + u["ETH"] * px["ETH"][t] + cash
                if reb and me[t]:
                    tgt = {"BTC": wb * v[t], "ETH": we_ * v[t]}
                    tr = sum(abs(tgt[a] - u[a] * px[a][t]) for a in COINS)
                    v[t] -= tr * 0.0015
                    u = {a: wb_ * v[t] / px[a][t] for a, wb_ in (("BTC", wb), ("ETH", we_))}
                    cash = wc * v[t]
            s = pd.Series(v, index=idx)
            yrs = (idx[-1] - idx[0]).days / 365.25
            cagr = (v[-1] / v[0]) ** (1 / yrs) - 1
            dd = (1 - s / s.cummax()).max()
            mret = s.resample("ME").last().pct_change().dropna()
            st = mret.mean() / mret.std() * np.sqrt(12)
            yr = s.resample("YE").last().pct_change()
            yr.iloc[0] = s.resample("YE").last().iloc[0] / v[0] - 1
            neg = [int(y.year) for y, r in yr.items() if r < 0]
            lab = f"BTC{wb * 10:.0f}・ETH{we_ * 10:.0f}・現金{wc * 10:.0f}・{'毎月戻す' if reb else '戻さない'}"
            q5[lab] = dict(cagr=cagr * 100, dd=dd * 100, s=st, neg=neg)
            print(f"    {lab:26s} 年率{cagr * 100:+6.1f}%・最大の落ち込み{dd * 100:4.0f}%・安定度{st:.2f}・マイナスの年{len(neg)}回")
            record(グループ=GROUP, 計画ID="Q05", 名前=f"Q05 {lab}", 銘柄="BTC,ETH", 時間足="日足", コストの前提="国内現物（売り買い0.15%）",
                   **{"1回の量": "2018年1月にまとめて買う（複利）"}, 安定度=round(st, 2),
                   メモ=f"年率{cagr * 100:+.1f}%・最大の落ち込み{dd * 100:.0f}%（複利）・マイナスの年" + ("・".join(map(str, neg)) or "なし"))
    keep("Q", out=out, q5=q5, q6=q6)


# ---------------------------------------------------------------- 本体
SECTIONS = {"b": sec_b, "f": sec_f, "k": sec_k, "q": sec_q}


def main(cache, only=None):
    names = list(SECTIONS) if not only else only.split(",")
    if not only:
        print(f"（古い「{GROUP}」の行を {drop(GROUP)} 行消して、計算し直す）")
    D = R1.prep(cache)
    days = D["BTC"]["idx"]
    per0, T0 = R2.combo(D)
    v0 = R1.total(per0)
    base = R1.stats(v0, T0)
    R1.line("A01（基準）", base)
    Mq = {a: R4.mult(D[a], "静かな日だけ") for a in COINS}
    v0d = (per0["BTC"] + per0["ETH"]).reindex(days).fillna(0).to_numpy()
    longd = sum(R3.pos_daily(D[a], a, np.ones(len(days)))[0].reindex(days).fillna(0).to_numpy() for a in COINS)
    mid = ((days.year - 2018) * 12 + days.month - 1).to_numpy()
    ctx = dict(cache=cache, D=D, days=days, per0=per0, T0=T0, v0=v0, base=base, Mq=Mq, v0d=v0d, longd=longd, mid=mid)
    for n in names:
        SECTIONS[n](ctx)
    if R3.OUT:
        print("\n==== まとめ（部品としての判定）")
        for pid, o in R3.OUT.items():
            r = o["r"]
            print(f"  {pid} {o['verdict']:12s} 前半で一番良い形「{o['best']}」：年{r['年平均%']:+.2f}% 落込{r['最大の落ち込み%']:.1f}% "
                  f"安定度{r['安定度']:.2f}（前半{r['_h1s']:.2f}・後半{r['_h2s']:.2f}）｜{o['n']}通り中{o['npos']}通りがプラス")
    return SUM


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
