#!/usr/bin/env python3
"""計画書（project/research/plan.md）第6回・前半：残りの優先「低」のうち A・C・D・E・H・M の37件（BTC・ETH、2018年〜）。

  A 値幅ブレイクの派生：A15 前日に負けた日だけ／A21 kを自動で調整／A22 ヒゲの長さで値幅を測る／A27 月の値幅ブレイク／
                        A29 1時間の値幅ブレイク
  C トレンド：C10 タートル（量を足していく）／C15 パラボリックSAR／C17 アルーン／C19 ハル移動平均／C20 KAMA／C21 TRIX／
              C30 トレンドの強さで量／C31 ストキャスティクスの順張り／C32 CCI／C33 ROC／C35 新値足／C36 練行足／
              C37 ポイント＆フィギュア／C39 トレンドの年齢
  D 逆張り：D12 ウィリアムズ%R／D13 ストキャスティクスの逆張り／D17 ピボット／D18 RSIのダイバージェンス／D19 ボリンジャー＋RSI／
            D22 夜の上げを朝に売る／D23 はらみ足の逆張り／D25 週足の逆張り／D26 月足の逆張り／D27 キリ番での逆張り
  E・H・M：E13 パーキンソン・ボラ／E16 ヒゲの比率／H03 ETHが先・BTCが後／H06 相関が下がった時／H07 ETHの出遅れを買う／
           H08 アルトの先行／H11 BTCの急変後のETH／M10 金（PAXG）も加える
共通: Vantage CFD（往復0.05%、買いのスワップ年25〜35%、売り0%）、1R＝ATR（日足）1本分＝資金の0.5%（単利）。
日足の合図は終値で判定して翌朝9時（始値）に入る。トレンドの形は、第3回と同じく「買い＋売り（売りも×1）」と「買いだけ」を見る。
結果は results.csv（グループ「計画・第6回」、計画IDつき）と plan.csv。
使い方: python3 project/tools/round6_lab.py <キャッシュ> [a,c,d,eh のどれか（カンマ区切り）]
"""
import os
import sys
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
from numba import njit

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
import exit_combo_lab as EX  # noqa: E402
import more_methods_lab as MM  # noqa: E402
import portfolio_lab as PF  # noqa: E402
import round1_lab as R1  # noqa: E402
import round2_lab as R2  # noqa: E402
import round2b_lab as R2B  # noqa: E402
import round3_lab as R3  # noqa: E402
import round3b_lab as R3B  # noqa: E402
import round4_lab as R4  # noqa: E402
import round4b_lab as R4B  # noqa: E402
from registry import drop, metrics_row, record  # noqa: E402

COINS = ("BTC", "ETH")
GROUP = "計画・第6回"
DAY = 96
R3.GROUP = GROUP
warnings.filterwarnings("ignore")
SUM = {}


# ---------------------------------------------------------------- 共通
def keep(pid, **kw):
    SUM.setdefault(pid, {}).update(kw)


def show(pid, name, per, T=None, memo="", tf="日足"):
    x = R1.total(per)
    r = R1.stats(x, T)
    cb, ce = (metrics_row(MM.clip(per[a])) for a in COINS)
    r["_coins"] = (cb, ce)
    r["_hold"] = T["hold"].mean() if T is not None and "hold" in T and len(T) else np.nan
    r["_x"] = x
    R1.line(name, r, f" ｜BTC{cb['年平均%']:+.1f}% ETH{ce['年平均%']:+.1f}%" + (f" 保有{r['_hold']:.1f}日" if r["_hold"] == r["_hold"] else ""))
    record(グループ=GROUP, 計画ID=pid, 名前=f"{pid} {name}", 銘柄="BTC,ETH", 時間足=tf,
           メモ=(memo + "。" if memo else "") + f"BTCだけ年{cb['年平均%']:+.1f}%・ETHだけ年{ce['年平均%']:+.1f}%"
           + (f"・平均保有{r['_hold']:.1f}日" if r["_hold"] == r["_hold"] else ""),
           **{k: v for k, v in r.items() if not k.startswith("_")})
    return r


def scaled(ctx, pid, name, per, T, m, memo=""):
    """値幅ブレイクの毎日の損益に、日ごとの倍率 m（0なら休む）をかける。"""
    per2 = {a: per[a] * m[a] for a in COINS}
    T = T.reset_index(drop=True)
    keepm = np.zeros(len(T), bool)
    for a in COINS:
        sel = (T["coin"] == a).to_numpy()
        pos = ctx["D"][a]["idx"].get_indexer(pd.DatetimeIndex(T.loc[sel, "day"]))
        keepm[sel] = np.asarray(m[a])[pos] > 0
    return show(pid, name, per2, T[keepm], memo=memo, tf="15分足")


def comp(ctx, pid, res):
    R3.component(pid, res, ctx["v0"], ctx["base"])
    o = R3.OUT[pid]
    keep(pid, verdict=o["verdict"], best=o["best"], n=o["n"], npos=o["npos"], corr=o["corr"], r=brief(o["r"]), mix=brief(o["mix"]),
         rows={k: brief(v) for k, v in res.items()})
    return o


def brief(r):
    return dict(a=r["年平均%"], dd=r["最大の落ち込み%"], s=r["安定度"], s1=r.get("_h1s"), s2=r.get("_h2s"), a1=r["前半の年平均%"],
                a2=r["後半の年平均%"], neg=r["マイナスの年"], n=r.get("_n"), hold=r.get("_hold"))


def run_pos(ctx, pid, label, posfn, longonly=None):
    """日足の建玉（終値で決め、翌日に持つ）。posfn(x, a) → 建玉の配列。"""
    per, T = R3.both(ctx["D"], lambda x, a: R3.pos_daily(x, a, posfn(x, a)))
    tail = "" if longonly is None else ("・買いだけ" if longonly else "・買い＋売り")
    return show(pid, f"{label}{tail}", per, T)


def run_fn(ctx, pid, name, fn, tf="日足（約定は15分足）"):
    return show(pid, name, *R3.both(ctx["D"], fn), tf=tf)


def dt(ctx, pid, name, sides):
    """日ごとの向き（その朝にわかる値）で9時〜翌9時を持つ。買い×1・売り×0.5。"""
    return show(pid, name, *R3.both(ctx["D"], R4B.daytrade(lambda x, a: sides[a])), tf="日足")


@contextmanager
def spec_gold_roll():
    yield


def prev(a, k=1):
    return np.r_[np.full(k, np.nan), np.asarray(a, float)[:-k]]


def ema(v, n):
    return pd.Series(v).ewm(span=n, adjust=False).mean().to_numpy()


def wma(v, n):
    w = np.arange(1, n + 1, dtype=float)
    out = np.full(len(v), np.nan)
    out[n - 1:] = np.convolve(np.asarray(v, float), w[::-1], mode="valid") / w.sum()
    return out


def rsi(c, n):
    d = np.diff(c, prepend=c[0])
    up = pd.Series(np.clip(d, 0, None)).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    dn = pd.Series(np.clip(-d, 0, None)).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100 - 100 / (1 + up / dn)


def stoch(h, l, c, n=14, d=3):
    hh, ll = pd.Series(h).rolling(n).max().to_numpy(), pd.Series(l).rolling(n).min().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        k = 100 * (c - ll) / (hh - ll)
    return k, R3.sma(k, d)


def dmi(h, l, c, n=14):
    ph, pl, pc = prev(h), prev(l), prev(c)
    up, dn = h - ph, pl - l
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    w = lambda v: pd.Series(np.nan_to_num(v)).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    at = w(tr)
    with np.errstate(invalid="ignore", divide="ignore"):
        pdi, mdi = 100 * w(pdm) / at, 100 * w(mdm) / at
        adx = w(100 * np.abs(pdi - mdi) / (pdi + mdi))
    return pdi, mdi, adx


def ls(cl, cs, longonly):
    return R3.ls(cl, cs, longonly)


# ---------------------------------------------------------------- A 値幅ブレイクの派生
def sec_a(ctx):
    D, days, per0, T0 = ctx["D"], ctx["days"], ctx["per0"], ctx["T0"]
    T = T0.reset_index(drop=True)
    di = days.get_indexer(pd.DatetimeIndex(T.day))
    won = (T.R.to_numpy() > 0).astype(float)

    # ---- A15
    print("\n==== A15 直近N日の値幅ブレイク（休んだ日も取ったはずの結果で数える）が全部負けた翌日だけ取引")

    def after(coin, n, want_loss, TT=None):
        if TT is None:
            sel, dd, ww_ = (T.coin == coin).to_numpy(), di, won
        else:
            TT = TT.reset_index(drop=True)
            sel = (TT.coin == coin).to_numpy()
            dd, ww_ = days.get_indexer(pd.DatetimeIndex(TT.day)), (TT.R.to_numpy() > 0).astype(float)
        cnt = np.bincount(dd[sel], minlength=len(days)).astype(float)
        wv = np.bincount(dd[sel], weights=ww_[sel], minlength=len(days))
        cn, cw = np.r_[0, np.cumsum(cnt)], np.r_[0, np.cumsum(wv)]
        m = np.zeros(len(days))
        for d in range(len(days)):
            lo = max(0, d - n)
            nn, ww = cn[d] - cn[lo], cw[d] - cw[lo]
            m[d] = float(nn >= 1 and ((ww == 0) if want_loss else (ww == nn)))
        return m
    res = {}
    for n in (1, 2, 3):
        res[f"{n}日"] = scaled(ctx, "A15", f"値幅ブレイク：直近{n}日の取引が全敗した翌日だけ", per0, T0, {a: after(a, n, True) for a in COINS})
    res["参考：前日が全勝"] = scaled(ctx, "A15", "参考：値幅ブレイク：前日の取引が全勝した翌日だけ", per0, T0, {a: after(a, 1, False) for a in COINS})
    print("    ---- 負けが続いた翌日だけ量を増やす：直近N日の取引（取ったはずのもの）が全敗なら、翌日はM倍（ほかは1倍）")
    grid = {}
    for n in (1, 2, 3):
        for mlt in (1.25, 1.5, 2.0):
            grid[(n, mlt)] = scaled(ctx, "A15", f"値幅ブレイク：直近{n}日が全敗なら翌日は{mlt}倍", per0, T0,
                                    {a: 1 + (mlt - 1) * after(a, n, True) for a in COINS})
    b0 = ctx["base"]
    both_better = sum(r["_h1s"] > b0["_h1s"] and r["_h2s"] > b0["_h2s"] for r in grid.values())
    bk = max(grid, key=lambda k: grid[k]["_h1s"])
    print(f"      9通り中{both_better}通りが、前半・後半ともA01（{b0['_h1s']:.2f}・{b0['_h2s']:.2f}）より上。前半で一番良いのは直近{bk[0]}日・{bk[1]}倍")
    # S1 で確かめる。判定に使う取引は (a) 休んだ日も取ったはずの結果で数える／(b) 実際に取った取引（静かな日）だけ
    Tb5 = ctx["Tb5"].reset_index(drop=True)
    tk = np.zeros(len(Tb5), bool)
    for a in COINS:
        sel = (Tb5.coin == a).to_numpy()
        tk[sel] = ctx["Mq"][a][days.get_indexer(pd.DatetimeIndex(Tb5.loc[sel, "day"]))] > 0
    Ts1 = Tb5[tk]
    s1g = {"S1": scaled(ctx, "A15", "S1（参考）", ctx["vb5"], ctx["Tb5"], ctx["Mq"])}
    b1 = s1g["S1"]
    n_better = 0
    for n in (1, 2, 3):
        for mlt in (1.5, 2.0):
            for src, TT in (("取ったはずの取引で判定", ctx["Tb5"]), ("実際に取った取引で判定", Ts1)):
                m = {a: (1 + (mlt - 1) * after(a, n, True, TT)) * ctx["Mq"][a] for a in COINS}
                r = scaled(ctx, "A15", f"S1：直近{n}日が全敗なら翌日は{mlt}倍（{src}）", ctx["vb5"], ctx["Tb5"], m)
                s1g[f"{n}日・{mlt}倍・{src}"] = r
                n_better += r["_h1s"] > b1["_h1s"] and r["_h2s"] > b1["_h2s"]
    print(f"      S1で12通り中{n_better}通りが、前半・後半ともS1（{b1['_h1s']:.2f}・{b1['_h2s']:.2f}）より上")
    pick = s1g["2日・2.0倍・実際に取った取引で判定"]
    yt = pd.DataFrame({"S1": b1["_x"].groupby(b1["_x"].index.year).sum(), "2日全敗→2倍": pick["_x"].groupby(pick["_x"].index.year).sum()}).round(1)
    print("      年ごとの損益（%）：\n" + "\n".join("        " + ln for ln in yt.T.to_string().splitlines()))
    dbl = np.mean([after(a, 2, True, Ts1)[(days >= MM.START) & (days <= MM.END)].mean() for a in COINS]) * 100
    rm = metrics_row(pick["_x"])
    print(f"      2倍にする日は{dbl:.0f}%。プラスの月{rm['プラスの月%']:.0f}%・最悪の年{rm['最悪の年%']:+.1f}%・高値回復の最長{rm['高値回復の最長日数']}日")
    keep("A15", years=yt, dbl=dbl, pick_m=dict(posm=rm["プラスの月%"], worst=rm["最悪の年%"], longest=rm["高値回復の最長日数"]))
    keep("A15", grid={f"{k[0]}日・{k[1]}倍": brief(v) for k, v in grid.items()}, both_better=both_better, best=bk,
         s1={k: brief(v) for k, v in s1g.items()}, s1_better=n_better)
    # ランダムに同じ割合の日を選んだ場合と比べる（銘柄ごとに同じ割合）
    w = ((days >= MM.START) & (days <= MM.END))
    mid = ((days.year - 2018) * 12 + days.month - 1).to_numpy()[w]
    pa = {a: per0[a].reindex(days).fillna(0).to_numpy()[w] for a in COINS}
    g = np.random.default_rng(0)
    cal = {}
    for n in (1, 2, 3):
        ms = {a: after(a, n, True)[w] > 0 for a in COINS}

        def st(z):
            m = np.bincount(mid, weights=z)
            return m.mean() / m.std() * np.sqrt(12)
        act = st(sum(pa[a] * ms[a] for a in COINS))
        rnd = np.array([st(sum(pa[a] * (g.random(len(pa[a])) < ms[a].mean()) for a in COINS)) for _ in range(1000)])
        cal[n] = dict(frac=np.mean([ms[a].mean() for a in COINS]) * 100, s=act, rank=(rnd < act).mean() * 100, med=float(np.median(rnd)))
        print(f"      直近{n}日：残した日{cal[n]['frac']:.0f}%。ランダムに同じ割合の日を選んだ1,000回の中で、安定度は下から{cal[n]['rank']:.0f}%（中央値{cal[n]['med']:.2f}）")
    keep("A15", rows={k: brief(v) for k, v in res.items()}, base=brief(ctx["base"]), cal=cal)

    # ---- A21
    print("\n==== A21 kを自動で調整：毎朝、k=0.6〜1.0のうち、直近N回の成績（1回平均のR）が一番良かったkを使う")
    ks = (0.6, 0.7, 0.8, 0.9, 1.0)
    PK, TK = {}, {}
    for k in ks:
        PK[k], TK[k] = R2.combo(D, k=k)
    res = {}
    for N in (10, 20, 50):
        per, Ts, share = {}, [], np.zeros(len(ks))
        for a in COINS:
            score = np.full((len(ks), len(days)), np.nan)
            for i, k in enumerate(ks):
                t = TK[k][TK[k].coin == a]
                td = days.get_indexer(pd.DatetimeIndex(t.day))
                R = t.R.to_numpy()
                o = np.argsort(td, kind="stable")
                td, R = td[o], R[o]
                cs = np.r_[0, np.cumsum(R)]
                pos = np.searchsorted(td, np.arange(len(days)), "left")
                lo = np.maximum(pos - N, 0)
                score[i] = np.where(pos - lo >= N, (cs[pos] - cs[lo]) / np.maximum(pos - lo, 1), np.nan)
            allnan = np.isnan(score).all(0)
            best = np.argmax(np.where(np.isnan(score), -np.inf, score), axis=0)
            best[allnan] = ks.index(0.8)
            pnl = sum((best == i) * PK[k][a].to_numpy() for i, k in enumerate(ks))
            per[a] = pd.Series(pnl, index=PK[0.8][a].index)
            w = (days >= MM.START) & (days <= MM.END)
            share += np.bincount(best[w], minlength=len(ks))
            for i, k in enumerate(ks):
                t = TK[k][TK[k].coin == a]
                Ts.append(t[best[days.get_indexer(pd.DatetimeIndex(t.day))] == i])
        res[f"直近{N}回"] = show("A21", f"値幅ブレイク：kを直近{N}回の成績で0.6〜1.0から選ぶ", per, pd.concat(Ts, ignore_index=True), tf="15分足")
        print("      選ばれたkの割合：" + "・".join(f"k={k} {v / share.sum() * 100:.0f}%" for k, v in zip(ks, share)))
    keep("A21", rows={k: brief(v) for k, v in res.items()})

    # ---- A22
    print("\n==== A22 前日のヒゲの長さで値幅を測る：買いは始値＋上ヒゲ×k、売りは始値−下ヒゲ×k（5日平均の向き、翌朝決済）")

    def wick(k):
        def fn(Pd, side):
            o, h, l, c = Pd["pO"], Pd["pH"], Pd["pL"], Pd["pC"]
            up, dn = prev(h - np.maximum(o, c)), prev(np.minimum(o, c) - l)
            return o + k * up if side == 1 else o - k * dn
        return fn
    res = {}
    for k in (1.0, 2.0, 3.0):
        per, Tk = R2.combo(D, trig_fn=wick(k))
        res[f"k={k:.0f}"] = show("A22", f"値幅ブレイク：前日のヒゲ×{k:.0f}", per, Tk, tf="15分足")
    keep("A22", rows={k: brief(v) for k, v in res.items()})

    # ---- A27
    print("\n==== A27 月の値幅ブレイク：月の始値±前月の値幅×kに逆指値（買い×1・売り×0.5）、月末に決済")

    def monthly(x, a, k, ma):
        o, h, l, c, A = x["dO"], x["dH"], x["dL"], x["dC"], x["atr"]
        mon = (x["idx"].year * 12 + x["idx"].month).to_numpy()
        starts = np.r_[0, np.nonzero(np.diff(mon))[0] + 1]
        ends = np.r_[starts[1:] - 1, len(c) - 1]
        mc = c[ends]
        pnl = np.zeros(len(c))
        rows = []
        for m in range(1, len(starts)):
            s0, s1 = starts[m], ends[m]
            if s1 == len(c) - 1 and m == len(starts) - 1:
                continue
            rng = h[starts[m - 1]:ends[m - 1] + 1].max() - l[starts[m - 1]:ends[m - 1] + 1].min()
            ma5 = mc[max(0, m - 5):m].mean() if m >= 5 else np.nan
            for side, w in ((1, 1.0), (-1, 0.5)):
                if ma and not (side * (mc[m - 1] - ma5) > 0):
                    continue
                trig = o[s0] + side * k * rng
                e, d0 = np.nan, -1
                for d in range(s0, s1 + 1):
                    if (side == 1 and h[d] >= trig) or (side == -1 and l[d] <= trig):
                        e = max(o[d], trig) if side == 1 else min(o[d], trig)
                        d0 = d
                        break
                if d0 < 0 or not (A[d0] > 0):
                    continue
                Ad = A[d0]
                sw = (PF.SPEC[a][1] if side == 1 else PF.SPEC[a][2]) / 36500
                ref = e
                tot = -PF.SPEC[a][0] * e / Ad
                pnl[d0] -= PF.SPEC[a][0] * e / Ad * 0.5 * w
                for d in range(d0, s1 + 1):
                    mv = side * (c[d] - ref) / Ad - sw * e / Ad
                    pnl[d] += mv * 0.5 * w
                    tot += mv
                    ref = c[d]
                rows.append((x["idx"][d0], tot, s1 - d0 + 1))
        T_ = pd.DataFrame(rows, columns=["day", "R", "hold"])
        return pd.Series(pnl, index=x["idx"]), T_
    res = {}
    for k in (0.2, 0.4, 0.6):
        for ma in (False, True):
            lab = f"k={k}" + ("・5か月平均の向きだけ" if ma else "")
            res[lab] = run_fn(ctx, "A27", f"月の値幅ブレイク（{lab}）", lambda x, a, k=k, ma=ma: monthly(x, a, k, ma), tf="日足")
    comp(ctx, "A27", res)

    # ---- A29
    print("\n==== A29 1時間の値幅ブレイク：1時間ごとに、始値±前の1時間の値幅×kに逆指値（5本平均の向き）、その1時間の終わりに決済")
    res = {}
    for k in (0.5, 1.0, 1.5):
        per, Tk = R2.combo(D, P=4, k=k, last=3)
        res[f"k={k}"] = show("A29", f"1時間の値幅ブレイク（k={k}）", per, Tk, tf="15分足")
    keep("A29", rows={k: brief(v) for k, v in res.items()})


# ---------------------------------------------------------------- C トレンド
@njit
def psar_dir(h, l, af0, afmax):
    n = len(h)
    out = np.zeros(n)
    up = True
    ep = h[0]
    s = l[0]
    af = af0
    for i in range(1, n):
        s = s + af * (ep - s)
        if up:
            s = min(s, l[i - 1], l[i - 2] if i >= 2 else l[i - 1])
            if l[i] < s:
                up, s, ep, af = False, ep, l[i], af0
            elif h[i] > ep:
                ep = h[i]
                af = min(af + af0, afmax)
        else:
            s = max(s, h[i - 1], h[i - 2] if i >= 2 else h[i - 1])
            if h[i] > s:
                up, s, ep, af = True, ep, h[i], af0
            elif l[i] < ep:
                ep = l[i]
                af = min(af + af0, afmax)
        out[i] = 1.0 if up else -1.0
    return out


@njit
def tlb_dir(c, n):
    """新値足（n本）の向き。+1 陽線の流れ、-1 陰線の流れ。"""
    m = len(c)
    out = np.zeros(m)
    hi = np.empty(m)
    lo = np.empty(m)
    k = 0
    d = 0
    hi[0], lo[0] = c[0], c[0]
    k = 1
    for i in range(1, m):
        p = c[i]
        if d >= 0:
            if p > hi[k - 1]:
                hi[k], lo[k] = p, hi[k - 1]
                k += 1
                d = 1
            elif d == 1:
                j0 = max(0, k - n)
                if p < lo[j0:k].min():
                    hi[k], lo[k] = lo[k - 1], p
                    k += 1
                    d = -1
            elif p < lo[k - 1]:
                hi[k], lo[k] = lo[k - 1], p
                k += 1
                d = -1
        else:
            if p < lo[k - 1]:
                hi[k], lo[k] = lo[k - 1], p
                k += 1
            else:
                j0 = max(0, k - n)
                if p > hi[j0:k].max():
                    hi[k], lo[k] = p, hi[k - 1]
                    k += 1
                    d = 1
        out[i] = d
    return out


@njit
def pnf_pos(lc, box, longonly):
    """ポイント＆フィギュア（終値・3枠転換、対数の枠）。前のXの列の高値を抜けたら買い、前のOの列の安値を割ったら売り（か決済）。"""
    m = len(lc)
    out = np.zeros(m)
    x = np.floor(lc / box)
    d = 0
    top = x[0]
    bot = x[0]
    ptop = 1e18
    pbot = -1e18
    cur = 0.0
    for i in range(1, m):
        v = x[i]
        if d == 0:
            if v >= top + 3:
                d, top = 1, v
            elif v <= bot - 3:
                d, bot = -1, v
        elif d == 1:
            if v > top:
                top = v
                if top > ptop:
                    cur = 1.0
            elif v <= top - 3:
                ptop = top
                d, bot = -1, v
                top = top - 1
                if bot < pbot:
                    cur = 0.0 if longonly else -1.0
        else:
            if v < bot:
                bot = v
                if bot < pbot:
                    cur = 0.0 if longonly else -1.0
            elif v >= bot + 3:
                pbot = bot
                d, top = 1, v
                bot = bot + 1
                if top > ptop:
                    cur = 1.0
        out[i] = cur
    return out


@njit
def renko_pos(Ch, b, longonly):
    """練行足（枠＝b）。2本続けて陽線なら買い。陰線が出たら決済（買い＋売りは2本続けて陰線で売り）。"""
    n = len(Ch)
    out = np.zeros(n)
    top = Ch[0]
    bot = Ch[0]
    d1 = 0
    d2 = 0
    cur = 0.0
    for h in range(1, n):
        p = Ch[h]
        bb = b[h]
        if not (bb > 0) or np.isnan(p):
            out[h] = cur
            continue
        while True:
            if d1 >= 0 and p >= top + bb:
                bot, top = top, top + bb
                d2, d1 = d1, 1
            elif d1 <= 0 and p <= bot - bb:
                top, bot = bot, bot - bb
                d2, d1 = d1, -1
            elif d1 == 1 and p <= bot - bb:
                top, bot = bot, bot - bb
                d2, d1 = d1, -1
            elif d1 == -1 and p >= top + bb:
                bot, top = top, top + bb
                d2, d1 = d1, 1
            else:
                break
        if d1 == 1 and d2 == 1:
            cur = 1.0
        elif d1 == -1:
            if longonly:
                cur = 0.0
            elif d2 == -1:
                cur = -1.0
            elif cur > 0:
                cur = 0.0
        out[h] = cur
    return out


def pos_hourly(x, a, pos_h):
    """1時間ごとの建玉（その1時間の終値で決め、次の1時間に持つ）。1R＝その日のATR＝資金の0.5%。毎日の損益と取引一覧。"""
    Ch = x["C"][3::4]
    nh = len(Ch)
    held = np.r_[0.0, np.asarray(pos_h, float)[:-1]]
    dayi = np.arange(nh) // 24
    A = x["atr"][np.clip(dayi, 0, len(x["atr"]) - 1)]
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(A > 0, 0.5 / A, 0.0)
    pc = np.r_[Ch[0], Ch[:-1]]
    change = np.abs(np.diff(np.r_[0.0, held]))
    swd = np.where(held > 0, PF.SPEC[a][1], PF.SPEC[a][2]) / 36500
    roll = (np.arange(nh) % 24) == 21
    pnl = held * (Ch - pc) * scale - change * PF.SPEC[a][0] / 2 * pc * scale - np.abs(held) * roll * swd * pc * scale
    pnl = np.nan_to_num(pnl)
    nd = len(x["dO"])
    daily = np.bincount(np.clip(dayi, 0, nd - 1), weights=pnl, minlength=nd)
    seg = np.cumsum(change > 0)
    on = held != 0
    T = pd.DataFrame({"seg": seg[on], "p": pnl[on], "day": x["idx"][np.clip(dayi[on], 0, nd - 1)]}).groupby("seg").agg(
        R=("p", "sum"), day=("day", "first"), hold=("p", "size"))
    T["R"] = T.R / 0.5
    T["hold"] = T.hold / 24
    return pd.Series(daily, index=x["idx"]), T


def turtle_pos(x, n_in, n_out, umax, longonly):
    h, l, c, N = x["dH"], x["dL"], x["dC"], x["atr"]
    hh_in, ll_in = pd.Series(h).rolling(n_in).max().shift(1).to_numpy(), pd.Series(l).rolling(n_in).min().shift(1).to_numpy()
    ll_out, hh_out = pd.Series(l).rolling(n_out).min().shift(1).to_numpy(), pd.Series(h).rolling(n_out).max().shift(1).to_numpy()
    pos = np.zeros(len(c))
    side, units, last, stop = 0, 0, np.nan, np.nan
    for d in range(len(c)):
        if not (N[d] > 0) or np.isnan(hh_in[d]):
            continue
        if side == 0:
            if c[d] > hh_in[d]:
                side, units, last = 1, 1, c[d]
            elif not longonly and c[d] < ll_in[d]:
                side, units, last = -1, 1, c[d]
            if side:
                stop = last - side * 2 * N[d]
        else:
            out = (c[d] < ll_out[d] or c[d] < stop) if side == 1 else (c[d] > hh_out[d] or c[d] > stop)
            if out:
                side, units = 0, 0
            else:
                while units < umax and side * (c[d] - last) >= 0.5 * N[d]:
                    units += 1
                    last += side * 0.5 * N[d]
                    stop = last - side * 2 * N[d]
        pos[d] = side * units
    return pos


def sec_c(ctx):
    D = ctx["D"]
    pos = lambda pid, lab, f, lo: run_pos(ctx, pid, lab, f, lo)

    print("\n==== C10 タートル：N日高値で入りM日安値で出る。0.5ATRごとに1単位ずつ最大4単位まで足し、最後に足した値段からATR×2で損切り")
    res = {}
    for ni, no in ((20, 10), (55, 20)):
        for um in (1, 4):
            for lo in (False, True):
                res[f"{ni}/{no}日・{um}単位{'・買いだけ' if lo else ''}"] = pos(
                    "C10", f"タートル（{ni}日で入り{no}日で出る・最大{um}単位）", lambda x, a, ni=ni, no=no, um=um, lo=lo: turtle_pos(x, ni, no, um, lo), lo)
    comp(ctx, "C10", res)

    print("\n==== C15 パラボリックSAR：SARの向きで買い・売り")
    res = {}
    for af0, afm in ((0.01, 0.1), (0.02, 0.2), (0.03, 0.3)):
        for lo in (False, True):
            res[f"{af0}/{afm}{'・買いだけ' if lo else ''}"] = pos(
                "C15", f"パラボリックSAR（{af0}/{afm}）", lambda x, a, af0=af0, afm=afm, lo=lo: ls(*(lambda s: (s > 0, s < 0))(psar_dir(x["dH"], x["dL"], af0, afm)), lo), lo)
    comp(ctx, "C15", res)

    print("\n==== C17 アルーン：アルーン上昇が70以上で下降より上なら買い（売りは逆）")

    def aroon(x, n):
        up = pd.Series(x["dH"]).rolling(n + 1).apply(lambda z: 100 * np.argmax(z) / n, raw=True).to_numpy()
        dn = pd.Series(x["dL"]).rolling(n + 1).apply(lambda z: 100 * np.argmin(z) / n, raw=True).to_numpy()
        return up, dn
    res = {}
    for n in (14, 25):
        for lo in (False, True):
            def fn(x, a, n=n, lo=lo):
                up, dn = aroon(x, n)
                with np.errstate(invalid="ignore"):
                    return ls((up >= 70) & (up > dn), (dn >= 70) & (dn > up), lo)
            res[f"{n}日{'・買いだけ' if lo else ''}"] = pos("C17", f"アルーン{n}日", fn, lo)
    comp(ctx, "C17", res)

    print("\n==== C19 ハル移動平均：前日より上がっていれば買い、下がっていれば売り")
    res = {}
    for n in (9, 21, 55):
        for lo in (False, True):
            def fn(x, a, n=n, lo=lo):
                c = x["dC"]
                hm = wma(2 * wma(c, n // 2) - wma(c, n), int(np.sqrt(n)))
                with np.errstate(invalid="ignore"):
                    return ls(hm > prev(hm), hm < prev(hm), lo)
            res[f"{n}日{'・買いだけ' if lo else ''}"] = pos("C19", f"ハル移動平均{n}日", fn, lo)
    comp(ctx, "C19", res)

    print("\n==== C20 KAMA（カウフマンの適応型移動平均）：前日より上がっていれば買い、下がっていれば売り")

    def kama(c, n, fast=2, slow=30):
        s = pd.Series(c)
        er = (s - s.shift(n)).abs() / s.diff().abs().rolling(n).sum()
        sc = ((er * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2).to_numpy()
        k = np.full(len(c), np.nan)
        for i in range(n, len(c)):
            k[i] = c[i] if np.isnan(k[i - 1]) else k[i - 1] + (0 if np.isnan(sc[i]) else sc[i]) * (c[i] - k[i - 1])
        return k
    res = {}
    for n in (10, 20):
        for lo in (False, True):
            def fn(x, a, n=n, lo=lo):
                k = kama(x["dC"], n)
                with np.errstate(invalid="ignore"):
                    return ls(k > prev(k), k < prev(k), lo)
            res[f"{n}/2/30{'・買いだけ' if lo else ''}"] = pos("C20", f"KAMA（{n}/2/30）", fn, lo)
    comp(ctx, "C20", res)

    print("\n==== C21 TRIX：プラスなら買い、マイナスなら売り")
    res = {}
    for n in (9, 15, 30):
        for lo in (False, True):
            def fn(x, a, n=n, lo=lo):
                e3 = ema(ema(ema(x["dC"], n), n), n)
                tr = e3 / prev(e3) - 1
                with np.errstate(invalid="ignore"):
                    return ls(tr > 0, tr < 0, lo)
            res[f"{n}日{'・買いだけ' if lo else ''}"] = pos("C21", f"TRIX{n}日", fn, lo)
    comp(ctx, "C21", res)

    print("\n==== C30 トレンドの強さで量：+DIと−DIで向き、ADXが20未満は0・20〜30は1・30〜40は1.5・40以上は2倍（参考：20以上は一律1）")
    res = {}
    for sized in (True, False):
        for lo in (False, True):
            def fn(x, a, sized=sized, lo=lo):
                pdi, mdi, adx = dmi(x["dH"], x["dL"], x["dC"])
                size = np.where(adx >= 40, 2.0, np.where(adx >= 30, 1.5, np.where(adx >= 20, 1.0, 0.0))) if sized else (adx >= 20).astype(float)
                with np.errstate(invalid="ignore"):
                    return ls(pdi > mdi, mdi > pdi, lo) * size
            res[("ADXで量" if sized else "参考：20以上は一律") + ("・買いだけ" if lo else "")] = pos(
                "C30", "ADXで量を変える" if sized else "参考：ADX20以上で一律の量", fn, lo)
    comp(ctx, "C30", res)

    print("\n==== C31 ストキャスティクスの順張り：%Kが%Dより上で、%Kが一定以上なら買い。%Kが%Dを割ったら決済（売りは逆）")
    res = {}
    for th in (70, 80):
        for lo in (False, True):
            def fn(x, a, th=th, lo=lo):
                k, d = stoch(x["dH"], x["dL"], x["dC"])
                with np.errstate(invalid="ignore"):
                    pl = R3.state((k > d) & (k >= th), k < d, 1)
                    ps = 0 if lo else R3.state((k < d) & (k <= 100 - th), k > d, -1)
                return pl + ps
            res[f"{th}以上{'・買いだけ' if lo else ''}"] = pos("C31", f"ストキャスティクスの順張り（{th}以上）", fn, lo)
    comp(ctx, "C31", res)

    print("\n==== C32 CCI：+100を超えたら買い、0を割ったら決済（売りは逆）")
    res = {}
    for n in (14, 20, 50):
        for lo in (False, True):
            def fn(x, a, n=n, lo=lo):
                tp = pd.Series((x["dH"] + x["dL"] + x["dC"]) / 3)
                md = tp.rolling(n).apply(lambda z: np.abs(z - z.mean()).mean(), raw=True)
                cci = ((tp - tp.rolling(n).mean()) / (0.015 * md)).to_numpy()
                with np.errstate(invalid="ignore"):
                    pl = R3.state(cci > 100, cci < 0, 1)
                    ps = 0 if lo else R3.state(cci < -100, cci > 0, -1)
                return pl + ps
            res[f"{n}日{'・買いだけ' if lo else ''}"] = pos("C32", f"CCI{n}日", fn, lo)
    comp(ctx, "C32", res)

    print("\n==== C33 ROC：N日の変化率がプラスなら買い、マイナスなら売り")
    res = {}
    for n in (10, 20, 30):
        for lo in (False, True):
            def fn(x, a, n=n, lo=lo):
                c = x["dC"]
                r = c / prev(c, n) - 1
                with np.errstate(invalid="ignore"):
                    return ls(r > 0, r < 0, lo)
            res[f"{n}日{'・買いだけ' if lo else ''}"] = pos("C33", f"ROC{n}日", fn, lo)
    comp(ctx, "C33", res)

    print("\n==== C35 新値足：陽転で買い、陰転で決済（売りは逆）")
    res = {}
    for n in (2, 3, 4):
        for lo in (False, True):
            def fn(x, a, n=n, lo=lo):
                d = tlb_dir(x["dC"].astype(np.float64), n)
                return ls(d > 0, d < 0, lo)
            res[f"{n}本{'・買いだけ' if lo else ''}"] = pos("C35", f"新値足（{n}本）", fn, lo)
    comp(ctx, "C35", res)

    print("\n==== C36 練行足（1時間足、枠＝その日のATR×k）：2本続けて陽線なら買い、陰線で決済（売りは2本続けて陰線）")
    res = {}
    for k in (0.5, 1.0, 2.0):
        for lo in (False, True):
            def fn(x, a, k=k, lo=lo):
                Ch = x["C"][3::4].astype(np.float64)
                dayi = np.arange(len(Ch)) // 24
                b = k * x["atr"][np.clip(dayi, 0, len(x["atr"]) - 1)]
                return pos_hourly(x, a, renko_pos(Ch, np.nan_to_num(b).astype(np.float64), lo))
            res[f"k={k}{'・買いだけ' if lo else ''}"] = run_fn(ctx, "C36", f"練行足（ATR×{k}）{'・買いだけ' if lo else '・買い＋売り'}", fn, tf="1時間足")
    comp(ctx, "C36", res)

    print("\n==== C37 ポイント＆フィギュア（終値・3枠転換、1枠＝1〜3%）：前のXの列の高値を抜けたら買い、前のOの列の安値を割ったら売り")
    res = {}
    for bx in (0.01, 0.02, 0.03):
        for lo in (False, True):
            res[f"{bx * 100:.0f}%{'・買いだけ' if lo else ''}"] = pos(
                "C37", f"ポイント＆フィギュア（1枠{bx * 100:.0f}%）", lambda x, a, bx=bx, lo=lo: pnf_pos(np.log(x["dC"]).astype(np.float64), np.log1p(bx), lo), lo)
    comp(ctx, "C37", res)

    print("\n==== C39 トレンドの年齢：50日平均の上（下）で買い（売り）。平均を抜けてからの日数で量を変える")
    res = {}
    for kind in ("若いほど多く", "一定", "古いほど多く"):
        for lo in (False, True):
            def fn(x, a, kind=kind, lo=lo):
                c = x["dC"]
                m = R3.sma(c, 50)
                with np.errstate(invalid="ignore"):
                    s = np.where(c > m, 1.0, np.where(c < m, -1.0, 0.0))
                age = np.zeros(len(c))
                for i in range(1, len(c)):
                    age[i] = age[i - 1] + 1 if s[i] == s[i - 1] and s[i] != 0 else 0
                w = {"若いほど多く": np.where(age <= 20, 1.5, np.where(age <= 60, 1.0, 0.5)), "一定": np.ones(len(c)),
                     "古いほど多く": np.where(age <= 20, 0.5, np.where(age <= 60, 1.0, 1.5))}[kind]
                return np.where(lo, np.clip(s, 0, None), s) * w
            res[f"{kind}{'・買いだけ' if lo else ''}"] = pos("C39", f"トレンドの年齢（{kind}）", fn, lo)
    comp(ctx, "C39", res)


# ---------------------------------------------------------------- D 逆張り
def sec_d(ctx):
    D, days = ctx["D"], ctx["days"]
    swing, limit, two = R3B.swing, R3B.limit, R3B.two_sides

    print("\n==== D12 ウィリアムズ%R：200日平均の上で%Rが一定以下なら翌朝買い。5日平均に戻ったら翌朝決済（最長5日・損切りATR×2）")
    res = {}
    for n in (10, 14):
        for th in (-80, -90):
            def fn(x, a, n=n, th=th):
                h, l, c = x["dH"], x["dL"], x["dC"]
                hh, ll = pd.Series(h).rolling(n).max().to_numpy(), pd.Series(l).rolling(n).min().to_numpy()
                with np.errstate(invalid="ignore", divide="ignore"):
                    wr = -100 * (hh - c) / (hh - ll)
                    sig = (wr <= th) & (c > R3.sma(c, 200))
                    p, T = swing(x, a, sig, 1, 5, c > R3.sma(c, 5), 2.0)
                return pd.Series(p, index=x["idx"]), T
            res[f"{n}日・{th}以下"] = run_fn(ctx, "D12", f"ウィリアムズ%R（{n}日・{th}以下）", fn)
    comp(ctx, "D12", res)

    print("\n==== D13 ストキャスティクスの逆張り：200日平均の上で%Kが一定以下なら翌朝買い（決済はD12と同じ）")
    res = {}
    for th in (10, 20):
        def fn(x, a, th=th):
            k, _ = stoch(x["dH"], x["dL"], x["dC"])
            c = x["dC"]
            with np.errstate(invalid="ignore"):
                p, T = swing(x, a, (k <= th) & (c > R3.sma(c, 200)), 1, 5, c > R3.sma(c, 5), 2.0)
            return pd.Series(p, index=x["idx"]), T
        res[f"{th}以下"] = run_fn(ctx, "D13", f"ストキャスティクスの逆張り（%K{th}以下）", fn)
    comp(ctx, "D13", res)

    print("\n==== D17 ピボットで逆張り：前日の高値・安値・終値からS1・R1（S2・R2）を出し、S で買い・R で売りの指値、翌朝決済")
    res = {}
    for lvl in (1, 2):
        def fn(x, a, lvl=lvl):
            h, l, c = prev(x["dH"]), prev(x["dL"]), prev(x["dC"])
            P = (h + l + c) / 3
            s_, r_ = (2 * P - h, 2 * P - l) if lvl == 1 else (P - (h - l), P + (h - l))
            ok = ~np.isnan(P)
            return two(lambda x_, a_: limit(x_, a_, ok, np.nan_to_num(s_), 1, 1), lambda x_, a_: limit(x_, a_, ok, np.nan_to_num(r_), -1, 1), 0.5)(x, a)
        res[f"S{lvl}・R{lvl}"] = run_fn(ctx, "D17", f"ピボットで逆張り（S{lvl}で買い・R{lvl}で売り）", fn, tf="15分足")
    comp(ctx, "D17", res)

    print("\n==== D18 RSIのダイバージェンス：終値が直近N日の安値を割ったのに、RSI(14)はその時より5以上高い→翌朝買い（売りは逆）。RSIが50を超えたら決済（最長5日・損切りATR×2）")
    res = {}
    for n in (20, 40):
        def fn(x, a, n=n):
            c = x["dC"]
            r = rsi(c, 14)
            sl, ss = np.zeros(len(c), bool), np.zeros(len(c), bool)
            for d in range(n, len(c)):
                w = c[d - n:d]
                j, k = d - n + int(np.argmin(w)), d - n + int(np.argmax(w))
                sl[d] = c[d] < c[j] and r[d] > r[j] + 5
                ss[d] = c[d] > c[k] and r[d] < r[k] - 5
            with np.errstate(invalid="ignore"):
                return two(lambda x_, a_: swing(x_, a_, sl, 1, 5, r > 50, 2.0), lambda x_, a_: swing(x_, a_, ss, -1, 5, r < 50, 2.0), 0.5)(x, a)
        res[f"{n}日"] = run_fn(ctx, "D18", f"RSIのダイバージェンス（直近{n}日）", fn)
    comp(ctx, "D18", res)

    print("\n==== D19 ボリンジャー＋RSI：終値がバンドの外で、RSI(14)も行き過ぎの時だけ逆張り。真ん中に戻ったら決済（最長5日・損切りATR×2）")
    res = {}
    for k, lo_, hi_ in ((2.0, 30, 70), (2.5, 25, 75)):
        def fn(x, a, k=k, lo_=lo_, hi_=hi_):
            c = x["dC"]
            m, sd = R3.sma(c, 20), pd.Series(c).rolling(20).std().to_numpy()
            r = rsi(c, 14)
            with np.errstate(invalid="ignore"):
                return two(lambda x_, a_: swing(x_, a_, (c < m - k * sd) & (r <= lo_), 1, 5, c > m, 2.0),
                           lambda x_, a_: swing(x_, a_, (c > m + k * sd) & (r >= hi_), -1, 5, c < m, 2.0), 0.5)(x, a)
        res[f"{k}σ・RSI{lo_}/{hi_}"] = run_fn(ctx, "D19", f"ボリンジャー{k}σ＋RSI{lo_}/{hi_}", fn)
    comp(ctx, "D19", res)

    print("\n==== D22 夜（日本21時〜翌9時）に一定以上上げたら、9時に売り（下げたら買い）、翌朝9時に決済")
    res = {}
    for th in (0.005, 0.01, 0.02, 0.03):
        sides = {}
        for a in COINS:
            O = D[a]["O"]
            nd = len(D[a]["dO"])
            i0, i1 = np.arange(nd) * DAY, np.arange(nd) * DAY - 48
            night = np.where(i1 >= 0, O[i0] / O[np.clip(i1, 0, None)] - 1, np.nan)
            with np.errstate(invalid="ignore"):
                sides[a] = np.where(night >= th, -1.0, np.where(night <= -th, 1.0, 0.0))
        res[f"{th * 100:.1f}%"] = dt(ctx, "D22", f"夜に{th * 100:.1f}%以上動いた向きの反対に9時〜翌9時", sides)
    comp(ctx, "D22", res)

    print("\n==== D23 はらみ足の逆張り：はらみ足の翌日、その高値で売り・安値で買いの指値、翌朝決済（N日）")
    res = {}
    for md in (1, 2):
        def fn(x, a, md=md):
            h, l = x["dH"], x["dL"]
            ins = prev((h < prev(h)) & (l > prev(l)))
            ok = np.nan_to_num(ins).astype(bool)
            return two(lambda x_, a_: limit(x_, a_, ok, np.nan_to_num(prev(l)), 1, md), lambda x_, a_: limit(x_, a_, ok, np.nan_to_num(prev(h)), -1, md), 0.5)(x, a)
        res[f"{md}日"] = run_fn(ctx, "D23", f"はらみ足の逆張り（{md}日で決済）", fn, tf="15分足")
    comp(ctx, "D23", res)

    print("\n==== D25 週足の逆張り：週足のRSI(14)が一定以下で週が終わったら、翌朝買って4週間持つ")
    res = {}
    for th in (30, 40):
        def fn(x, a, th=th):
            c = pd.Series(x["dC"], index=x["idx"])
            wk = c.resample("W-SUN").last()
            wr = pd.Series(rsi(wk.to_numpy(), 14), index=wk.index)
            sig = pd.Series(False, index=x["idx"])
            hit = wr[wr <= th].index
            sig.loc[sig.index.isin(hit)] = True
            p, T = swing(x, a, sig.to_numpy(), 1, 28)
            return pd.Series(p, index=x["idx"]), T
        res[f"RSI{th}以下"] = run_fn(ctx, "D25", f"週足の逆張り（週足RSI{th}以下で4週持つ）", fn)
    comp(ctx, "D25", res)

    print("\n==== D26 月足の逆張り：前月に一定以上下げたら、翌月の初めに買って1か月持つ")
    res = {}
    for th in (-0.2, -0.3, -0.4):
        def fn(x, a, th=th):
            c = pd.Series(x["dC"], index=x["idx"])
            mo = c.resample("ME").last()
            mr = mo / mo.shift(1) - 1
            sig = pd.Series(False, index=x["idx"])
            sig.loc[sig.index.isin(mr[mr <= th].index)] = True
            p, T = swing(x, a, sig.to_numpy(), 1, 30)
            return pd.Series(p, index=x["idx"]), T
        res[f"{th * 100:.0f}%以下"] = run_fn(ctx, "D26", f"月足の逆張り（前月{th * 100:.0f}%以下で翌月買い）", fn)
    keep("D26", nsig={k: v["_n"] for k, v in res.items()})
    comp(ctx, "D26", res)

    print("\n==== D27 キリ番での逆張り：始値の上下の、一番近いキリ番（BTCは1万・千ドル刻み、ETHは千・百ドル刻み）に指値。上で売り・下で買い、翌朝決済")
    res = {}
    for big in (True, False):
        def fn(x, a, big=big):
            step = {("BTC", True): 10000.0, ("BTC", False): 1000.0, ("ETH", True): 1000.0, ("ETH", False): 100.0}[(a, big)]
            o = x["dO"]
            up = (np.floor(o / step) + 1) * step
            dn = np.floor(o / step) * step
            okl = (o - dn) <= x["atr"]
            oks = (up - o) <= x["atr"]
            return two(lambda x_, a_: limit(x_, a_, np.nan_to_num(okl).astype(bool), dn, 1, 1),
                       lambda x_, a_: limit(x_, a_, np.nan_to_num(oks).astype(bool), up, -1, 1), 0.5)(x, a)
        res["大きい刻み" if big else "小さい刻み"] = run_fn(ctx, "D27", "キリ番での逆張り（" + ("BTC1万・ETH千ドル刻み" if big else "BTC千・ETH百ドル刻み") + "）", fn, tf="15分足")
    comp(ctx, "D27", res)


# ---------------------------------------------------------------- E・H・M
def lead_follow(D, src, dst, th, hh, same=True):
    """src が1時間で th 以上動いたら、dst を同じ向き（same=False なら反対）に次の15分足で入り、hh 時間後に決済（H02 と同じ作り）。"""
    xs, xd = D[src], D[dst]
    r1h = np.r_[np.full(4, np.nan), xs["C"][4:] / xs["C"][:-4] - 1]
    O = xd["O"]
    n = len(O)
    dayi = np.arange(n) // DAY
    A = xd["atr"][np.clip(dayi, 0, len(xd["atr"]) - 1)]
    Rs, ds = [], []
    pnl = np.zeros(len(xd["dO"]))
    j = 4
    while j < n - hh * 4 - 1:
        if abs(r1h[j]) >= th and A[j] > 0:
            side = (1 if r1h[j] > 0 else -1) * (1 if same else -1)
            i1, i2 = j + 1, j + 1 + hh * 4
            e, ex = O[i1], O[i2]
            rolls = sum(1 for r in range(i1 + 1, i2) if r % DAY == 84)
            sw = (PF.SPEC[dst][1] if side == 1 else PF.SPEC[dst][2]) / 36500
            R = side * (ex - e) / A[j] - PF.SPEC[dst][0] * e / A[j] - rolls * sw * e / A[j]
            pnl[dayi[j]] += R * 0.5
            Rs.append(R)
            ds.append(xd["idx"][dayi[j]])
            j = i2
        else:
            j += 1
    other = "BTC" if dst == "ETH" else "ETH"
    per = {dst: pd.Series(pnl, index=xd["idx"]), other: pd.Series(0.0, index=xd["idx"])}
    T = pd.DataFrame({"day": ds, "R": Rs, "hold": hh / 24})
    return per, T[(T.day >= MM.START) & (T.day <= MM.END)]


def sec_eh(ctx):
    D, days, per0, T0 = ctx["D"], ctx["days"], ctx["per0"], ctx["T0"]

    print("\n==== E13 パーキンソン・ボラ（高値と安値から測る10・20日のボラ）が過去1年の中央値より小さい日だけ値幅ブレイク")
    res = {}
    for n in (10, 20):
        m = {}
        for a in COINS:
            x = D[a]
            pk = np.sqrt((pd.Series(np.log(x["dH"] / x["dL"])) ** 2).rolling(n).mean() / (4 * np.log(2))).shift(1)
            med = pk.rolling(365, min_periods=200).median()
            m[a] = np.where(med.isna(), 1.0, (pk < med).astype(float))
        res[f"{n}日"] = scaled(ctx, "E13", f"値幅ブレイク：パーキンソン・ボラ（{n}日）で静かな日だけ", per0, T0, m)
        if n == 10:
            res["S1の形"] = scaled(ctx, "E13", "S1の静かな日を、パーキンソン・ボラ（10日）で判定", ctx["vb5"], ctx["Tb5"], m)
    res["参考：E05"] = scaled(ctx, "E13", "参考：値幅ブレイク：10日の実現ボラで静かな日だけ（E05）", per0, T0, ctx["Mq"])
    keep("E13", rows={k: brief(v) for k, v in res.items()})

    print("\n==== E16 ヒゲの比率：前日の上ヒゲが値幅の一定以上なら、その日の買いを休む（下ヒゲなら売りを休む）")
    res = {}
    for th in (0.4, 0.5, 0.6):
        per, Ts = {}, []
        for a in COINS:
            x = D[a]
            o, h, l, c = x["dO"], x["dH"], x["dL"], x["dC"]
            with np.errstate(invalid="ignore", divide="ignore"):
                upr, dnr = prev((h - np.maximum(o, c)) / (h - l)), prev((np.minimum(o, c) - l) / (h - l))
            bad_l, bad_s = np.nan_to_num(upr) >= th, np.nan_to_num(dnr) >= th
            s, T = R2.run(R2.periods(x, DAY, 0), a, extra_ok=lambda P_, side, bl=bad_l, bs=bad_s: ~(bl if side == 1 else bs))
            per[a] = s
            Ts.append(T)
        res[f"{th}"] = show("E16", f"値幅ブレイク：前日のヒゲが値幅の{th * 100:.0f}%以上なら、その向きを休む", per, pd.concat(Ts, ignore_index=True), tf="15分足")
    keep("E16", rows={k: brief(v) for k, v in res.items()})

    print("\n==== H03 ETHが1時間で一定以上動いたら、BTCを同じ向きに（次の15分足で入り、N時間後に決済）")
    res = {}
    for th in (0.01, 0.02):
        for hh in (1, 4):
            res[f"{th * 100:.0f}%・{hh}時間"] = show("H03", f"ETHの動きでBTC：1時間で{th * 100:.0f}%以上・{hh}時間持つ", *lead_follow(D, "ETH", "BTC", th, hh), tf="15分足")
    comp(ctx, "H03", res)

    print("\n==== H11 BTCが1時間で±2・3%動いた後、ETHを次の1時間だけ同じ向き／反対の向きに")
    res = {}
    for th in (0.02, 0.03):
        for same in (True, False):
            res[f"{th * 100:.0f}%・{'同じ' if same else '反対'}"] = show(
                "H11", f"BTCが1時間で{th * 100:.0f}%動いた後、ETHを{'同じ' if same else '反対の'}向きに1時間", *lead_follow(D, "BTC", "ETH", th, 1, same), tf="15分足")
    comp(ctx, "H11", res)

    print("\n==== H06 BTCとETHの30日の相関が低い日（分散が効く時期）は、値幅ブレイクの量を1.5倍")
    rb, re_ = pd.Series(np.log(D["BTC"]["dC"])).diff(), pd.Series(np.log(D["ETH"]["dC"])).diff()
    cor = rb.rolling(30).corr(re_).shift(1).to_numpy()
    res = {}
    for th in (0.6, 0.7):
        w = (days >= MM.START) & (days <= MM.END)
        share = np.nanmean(cor[w] < th) * 100
        m = np.where(np.nan_to_num(cor, nan=1.0) < th, 1.5, 1.0)
        res[f"{th}未満"] = scaled(ctx, "H06", f"値幅ブレイク：30日の相関が{th}未満の日は1.5倍", per0, T0, {a: m for a in COINS}, memo=f"相関が{th}未満の日は{share:.0f}%")
        print(f"      相関が{th}未満の日は{share:.0f}%")
    keep("H06", rows={k: brief(v) for k, v in res.items()})

    print("\n==== H07 ETHの出遅れを買う：直近7日でETHがBTCよりX%以上出遅れたら、翌朝ETHを買って7日持つ")
    res = {}
    lb, le = np.log(D["BTC"]["dC"]), np.log(D["ETH"]["dC"])
    gap = (le - prev(le, 7)) - (lb - prev(lb, 7))
    for th in (0.03, 0.05, 0.10):
        def fn(x, a, th=th):
            if a == "BTC":
                return pd.Series(0.0, index=x["idx"]), pd.DataFrame({"day": [], "R": [], "hold": []})
            with np.errstate(invalid="ignore"):
                p, T = R3B.swing(x, a, gap <= -th, 1, 7)
            return pd.Series(p, index=x["idx"]), T
        res[f"{th * 100:.0f}%"] = run_fn(ctx, "H07", f"ETHの出遅れを買う（7日で{th * 100:.0f}%以上出遅れたら7日持つ）", fn)
    comp(ctx, "H07", res)

    print("\n==== H08 アルトの先行：前日にアルト（XRP・ADA・BNB・DOGE・LTC・SOL）の平均がBTCよりX%以上上げたら、BTCを9時〜翌9時に買い")
    alts = []
    for nm in ("XRP", "ADA", "BNB", "DOGE", "LTC", "SOL"):
        s = pd.read_csv(os.path.join(ctx["cache"], f"{nm}.csv"), index_col=0, parse_dates=True)["close"].astype(float)
        s.index = pd.DatetimeIndex(s.index).normalize()
        alts.append(np.log(s[~s.index.duplicated()]).diff().rename(nm))
    am = pd.concat(alts, axis=1).mean(axis=1)
    am_prev = am.reindex(days).shift(1).to_numpy()
    bt_prev = prev(np.log(D["BTC"]["dC"]) - prev(np.log(D["BTC"]["dC"])))
    res = {}
    for th in (0.02, 0.04):
        with np.errstate(invalid="ignore"):
            sig = np.nan_to_num(am_prev - bt_prev) >= th
        res[f"{th * 100:.0f}%"] = dt(ctx, "H08", f"アルトの平均が前日にBTCより{th * 100:.0f}%以上上げたらBTCを買い", {"BTC": sig.astype(float), "ETH": np.zeros(len(days))})
    comp(ctx, "H08", res)

    print("\n==== M10 金も加える：PAXG（1時間足、2020年8月〜）で同じ値幅ブレイク（往復0.03%、買いスワップ年6.5%）を、A01に×1で足す")
    xg = EX.day_matrix_n(os.path.join(ctx["cache"], "PAXG_1h.csv"), 24)
    M = xg["M"]
    xg.update(O=np.ascontiguousarray(M["open"].ravel()), H=np.ascontiguousarray(M["high"].ravel()),
              L=np.ascontiguousarray(M["low"].ravel()), C=np.ascontiguousarray(M["close"].ravel()),
              dH=M["high"].max(1), dL=M["low"].min(1), dC=M["close"][:, -1], idx=pd.DatetimeIndex(xg["days"]))
    Pd = R2.periods(xg, 24, 0)
    Pd["roll"] = (np.arange(Pd["n"] * 24) % 24) == 21
    sg, Tg = R2.run(Pd, "Gold")
    g = MM.clip(sg)
    rg = metrics_row(g, len(Tg) / ((g.index[-1] - g.index[0]).days / 365.25))
    print(f"    金だけ（{g.index[0]:%Y-%m}〜）：年{rg['年平均%']:+.2f}%・落ち込み{rg['最大の落ち込み%']:.1f}%・安定度{rg['安定度']:.2f}")
    tot = ctx["v0"].add(g, fill_value=0)
    r = R1.stats(tot)
    R1.line("A01＋金の値幅ブレイク", r)
    record(グループ=GROUP, 計画ID="M10", 名前="M10 A01＋金（PAXG）の値幅ブレイク", 銘柄="BTC,ETH,PAXG", 時間足="15分足・1時間足",
           メモ=f"金だけ（2020年8月〜）は年{rg['年平均%']:+.2f}%・安定度{rg['安定度']:.2f}", **{k: v for k, v in r.items() if not k.startswith("_")})
    corr = ctx["v0"].resample("ME").sum().corr(g.resample("ME").sum())
    sub = ctx["v0"][ctx["v0"].index >= g.index[0]]
    r_sub, r_sub2 = metrics_row(sub), metrics_row(sub.add(g, fill_value=0))
    print(f"    同じ期間（2020年8月〜）で比べると、A01の安定度{r_sub['安定度']:.2f} → 金を足して{r_sub2['安定度']:.2f}。月の相関{corr:+.2f}")
    keep("M10", gold=rg, mix=brief(r), corr=corr, sub=(r_sub["安定度"], r_sub2["安定度"]))


# ---------------------------------------------------------------- 本体
SECTIONS = {"a": sec_a, "c": sec_c, "d": sec_d, "eh": sec_eh}


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
    vb5, Tb5 = R4.vbo_var(D, "btc5", False)
    Mq = {a: R4.mult(D[a], "静かな日だけ") for a in COINS}
    ctx = dict(cache=cache, D=D, days=days, per0=per0, T0=T0, v0=v0, base=base, vb5=vb5, Tb5=Tb5, Mq=Mq)
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
