#!/usr/bin/env python3
"""デイトレードの計画書（project/research/daytrade_plan.md）第3回：逆張り・ボラ・出来高・2銘柄・暗号資産ならではの残り（13件）。

  TD02 日中のVWAPからの離れの戻り／TD03 RSIの極端（15分足・1時間足）／TD06 投げ売りの後の買い／TD09 夜中（1〜5時）の逆張り
  TE02 直近24時間・10日の値動きの大きさで TA03 を絞る／TE04 その日の値幅が大きい日は TA03 を見送る／TE05 前日の値幅が小さい日（NR4・NR7）だけ
  TG03 日中のVWAPの上下／TG05 出来高を伴う日中の高値・安値の更新
  TH02 ETH/BTCの比の日中の戻り（2銘柄の組）／TH03 もう一方の銘柄が先に同じ向きに抜けた時だけ
  TJ03 オプションの満期（金曜17時）の前後／TJ06 米国の雇用統計（第1金曜で近い日を使う。CPI・FOMCは日程のデータがないので未）
共通の決まりは第1回（dt1_lab.py）と同じ。TA03 は5分足で計算する。結果は results.csv（グループ「計画・デイトレ第3回」）と daytrade_plan.csv。
使い方: python3 project/tools/dt3_lab.py <キャッシュ（BTC_15m_vol.csv・ETH_15m_vol.csv・BTC_5m.csv・ETH_5m.csv・BTC_15m.csv・ETH_15m.csv）> [項目]
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dt1_lab as DT  # noqa: E402
import dt2_lab as D2  # noqa: E402
import round1_lab as R1  # noqa: E402
import round7_lab as R7  # noqa: E402
import vantage_exit_time as VX  # noqa: E402
from registry import drop  # noqa: E402

warnings.filterwarnings("ignore")
GROUP = "計画・デイトレ第3回"
DT.GROUP = GROUP
COINS, DAY, END = DT.COINS, DT.DAY, DT.END
run = D2.run


# ---------------------------------------------------------------- 共通
def vwap(x):
    """その日（日本時間9時から）のVWAP。15分足ごと、その足の終値までで計算。"""
    tp = (x["H"] + x["L"] + x["C"]) / 3
    v = x["V"]
    nd = x["nd"]
    num = np.cumsum((tp * v).reshape(nd, DAY), 1).ravel()
    den = np.cumsum(v.reshape(nd, DAY), 1).ravel()
    return np.where(den > 0, num / np.where(den > 0, den, 1), x["C"])


def rsi(c, n):
    d = np.diff(c, prepend=np.nan)
    up = pd.Series(np.clip(d, 0, None)).ewm(alpha=1 / n, adjust=False).mean()
    dn = pd.Series(np.clip(-d, 0, None)).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).to_numpy()


def sig_trades(ctx, a, sig, H, mask=None, slk=0.0, tpk=0.0, win=None):
    x = ctx["X"][a]
    w = (DT.window(x) if win is None else win) & DT.day_mask(x, None if mask is None else mask[a])
    A = np.nan_to_num(x["atr"][x["bd"]])
    d, s, e, xp, eb, xb = D2.sig_engine(x["O"], x["H"], x["L"], x["C"], np.nan_to_num(sig), A, x["bd"], H, slk, tpk, 0.0, 1e9, w)
    return DT.res_trades(x, a, d, s, e, xp, np.ones(len(d)), eb, xb)


def ta03_raw(ctx, rl15=2):
    """TA03（5分足・米国株の始まりから rl15×15分のレンジ）の取引の一覧（銘柄ごと）。"""
    X, X5, us = ctx["X"], ctx["X5"], ctx["us"]
    nd = len(ctx["idx"])
    rl = rl15 * 3
    s0, xb = np.where(us, 162, 174), np.where(us, 240, 252)
    out = {}
    for a in COINS:
        x5, x = X5[a], X[a]
        Hm, Lm = x5["H"].reshape(nd, 288), x5["L"].reshape(nd, 288)
        cols = s0[:, None] + np.arange(rl)[None, :]
        hi, lo = np.take_along_axis(Hm, cols, 1).max(1), np.take_along_axis(Lm, cols, 1).min(1)
        nan = np.full(nd, np.nan)
        d, s, e, xp, eb, xbb = D2.brk_p(x5["O"], x5["H"], x5["L"], x5["C"], 288, (s0 + rl).astype(np.int64), (xb - 1).astype(np.int64),
                                        xb.astype(np.int64), hi, lo, lo, hi, nan, nan, nan, np.zeros(nd, np.int64), np.isfinite(x["atr"]))
        out[a] = dict(d=d, s=s, e=e, xp=xp, eb=eb, xb=xbb)
    return out


def from_raw(ctx, raw, keep=None):
    out = {}
    for a in COINS:
        t = raw[a]
        k = np.ones(len(t["d"]), bool) if keep is None else keep[a]
        out[a] = DT.res_trades(ctx["X"][a], a, t["d"][k], t["s"][k], t["e"][k], t["xp"][k], np.ones(int(k.sum())), t["eb"][k] // 3, t["xb"][k] // 3)
    return out


# ---------------------------------------------------------------- TD 逆張り
def sec_td(ctx):
    X = ctx["X"]
    vw = {a: vwap(X[a]) for a in COINS}
    v = {}
    for k in (0.5, 1.0, 1.5):
        for H, hl in ((16, "4時間"), (96, "翌5時まで")):
            for rev in (True, False):
                def fn(mask, k=k, H=H, rev=rev):
                    out = {}
                    for a in COINS:
                        x = X[a]
                        dev = (x["C"] - vw[a]) / x["atr"][x["bd"]]
                        sig = np.where(dev > k, -1.0, np.where(dev < -k, 1.0, 0.0)) * (1 if rev else -1)
                        out[a] = sig_trades(ctx, a, sig, H, mask, tpk=k if rev else 0.0)
                    return out
                v[f"VWAPからATR×{k:g}・{hl}・{'戻る向き（VWAPで利確）' if rev else '離れる向き'}"] = fn
    run(ctx, "TD02", "その日のVWAPから離れたら、戻る向き（と離れる向き）に", v, family=lambda k: k.split("・")[-1])
    v = {}
    for kind in ("RSI(2)・15分足", "RSI(14)・15分足", "RSI(2)・1時間足"):
        for H in (4, 16):
            for rev in (True, False):
                def fn(mask, kind=kind, H=H, rev=rev):
                    out = {}
                    for a in COINS:
                        x = X[a]
                        C = x["C"]
                        if kind.endswith("1時間足"):
                            r = rsi(C[3::4], 2)
                            lo_, hi_ = 5, 95
                            sig = np.zeros(len(C))
                            s_ = np.where(r < lo_, 1.0, np.where(r > hi_, -1.0, 0.0))
                            sig[3::4][:len(s_)] = s_
                        else:
                            n = 2 if kind.startswith("RSI(2)") else 14
                            lo_, hi_ = (5, 95) if n == 2 else (20, 80)
                            r = rsi(C, n)
                            sig = np.where(r < lo_, 1.0, np.where(r > hi_, -1.0, 0.0))
                        out[a] = sig_trades(ctx, a, sig * (1 if rev else -1), H, mask)
                    return out
                v[f"{kind}・{H * 0.25:g}時間・{'逆張り' if rev else '順張り'}"] = fn
    run(ctx, "TD03", "短い足のRSIが極端な時（逆張りと順張り）", v, family=lambda k: k.split("・")[-1])
    v = {}
    for zk in (2.0, 3.0):
        for vm in (3.0, 5.0):
            for H in (4, 16):
                def fn(mask, zk=zk, vm=vm, H=H):
                    out = {}
                    for a in COINS:
                        x = X[a]
                        lr, z = DT.zscore(x)
                        vr = x["V"] / pd.Series(x["V"]).rolling(672, min_periods=200).mean().shift(1).to_numpy()
                        sig = np.where((np.nan_to_num(z) < -zk) & (np.nan_to_num(vr) > vm), 1.0, 0.0)
                        out[a] = sig_trades(ctx, a, sig, H, mask)
                    return out
                v[f"{zk:g}倍の下げ・出来高{vm:g}倍・{H * 0.25:g}時間"] = fn
    run(ctx, "TD06", "投げ売り（出来高が急に増えた大きな下げ）の後に買い", v)
    v = {}
    for n in (20, 40):
        for kk in (2.0, 2.5):
            for rev in (True, False):
                def fn(mask, n=n, kk=kk, rev=rev):
                    out = {}
                    for a in COINS:
                        x = X[a]
                        Cs = pd.Series(x["C"])
                        mid, sd = Cs.rolling(n).mean().to_numpy(), Cs.rolling(n).std().to_numpy()
                        sig = np.where(x["C"] > mid + kk * sd, -1.0, np.where(x["C"] < mid - kk * sd, 1.0, 0.0)) * (1 if rev else -1)
                        win = (x["bod"] >= 64) & (x["bod"] < END) & np.isfinite(x["atr"][x["bd"]])
                        out[a] = sig_trades(ctx, a, sig, 4, mask, win=win)
                    return out
                v[f"{n}本・{kk:g}σ・{'逆張り' if rev else '順張り'}"] = fn
    run(ctx, "TD09", "夜中（日本時間1〜5時）に、バンドの外に出たら逆に（と同じ向きに）。1時間か5時で決済", v, family=lambda k: k.split("・")[-1])


# ---------------------------------------------------------------- TE ボラ
def sec_te(ctx):
    X, X5, idx = ctx["X"], ctx["X5"], ctx["idx"]
    nd = len(idx)
    us = ctx["us"]
    raw = ta03_raw(ctx)
    # TE02：直近24時間の値動き（米国株の始まりの前まで）と、10日の値動き（S1の静かな日）で TA03 を絞る
    v = {"基準（絞りなし）": lambda mask: from_raw(ctx, raw)}
    s0 = np.where(us, 162, 174)
    rv24 = {}
    for a in COINS:
        c5 = X5[a]["C"]
        lr = np.r_[0.0, np.diff(np.log(c5))]
        cs = np.r_[0.0, np.cumsum(lr ** 2)]
        endb = np.arange(nd) * 288 + s0
        rv = np.sqrt((cs[endb] - cs[np.maximum(endb - 288, 0)]))
        med = pd.Series(rv).rolling(30, min_periods=10).median().shift(1).to_numpy()
        rv24[a] = (rv, med)
    for lab, cond in (("直近24時間が静か", lambda a: rv24[a][0] < rv24[a][1]), ("直近24時間が荒い", lambda a: rv24[a][0] >= rv24[a][1]),
                      ("10日が静か（S1の静かな日）", lambda a: X[a]["q"]), ("10日が荒い", lambda a: ~X[a]["q"])):
        v[lab] = (lambda mask, cond=cond: from_raw(ctx, raw, {a: cond(a)[raw[a]["d"]] for a in COINS}))
    run(ctx, "TE02", "TA03（5分足）を、直前の値動きの大きさで絞る", v)
    # TE04：その日（9時〜米国株の始まり）の値幅が ATR×k を超えていたら、TA03 を見送る
    v = {"基準": lambda mask: from_raw(ctx, raw)}
    rng_so_far = {}
    for a in COINS:
        Hm, Lm = X5[a]["H"].reshape(nd, 288), X5[a]["L"].reshape(nd, 288)
        cols = np.arange(288)[None, :] < (s0 + 6)[:, None]
        rng_so_far[a] = np.where(cols, Hm, -np.inf).max(1) - np.where(cols, Lm, np.inf).min(1)
    for k in (0.75, 1.0, 1.5):
        v[f"その日の値幅がATR×{k:g}を超えたら見送る"] = \
            (lambda mask, k=k: from_raw(ctx, raw, {a: (rng_so_far[a] <= k * X[a]["atr"])[raw[a]["d"]] for a in COINS}))
    run(ctx, "TE04", "TA03（5分足）で、米国株の始まりまでの値幅が大きい日は見送る", v)
    # TE05：前日の値幅が直近4日・7日で一番小さい日（NR4・NR7）だけ
    v = {}
    for n in (4, 7):
        for base in ("TA05", "値幅ブレイク"):
            for xb, xl in ((80, "翌5時"), (93, "翌朝8時15分")):
                if base == "値幅ブレイク" and xb == 93:
                    continue

                def fn(mask, n=n, base=base, xb=xb):
                    out = {}
                    for a in COINS:
                        x = X[a]
                        rg = x["dH"] - x["dL"]
                        nr = np.r_[False, (pd.Series(rg).rolling(n).min() == rg).to_numpy()[:-1]]
                        m = nr & (True if mask is None else np.asarray(mask[a]))
                        if base == "TA05":
                            ph, pl = np.r_[np.nan, x["dH"][:-1]], np.r_[np.nan, x["dL"][:-1]]
                            mid = (ph + pl) / 2
                            out[a] = DT.brk(x, a, 0, xb - 1, xb, ph, pl, mid, mid, ph + 2 * x["atr"], pl - 2 * x["atr"], mask=m)
                        else:
                            up = np.where(x["sdir"] > 0, x["dO"] + 0.8 * x["rng"], np.nan)
                            dn = np.where(x["sdir"] < 0, x["dO"] - 0.8 * x["rng"], np.nan)
                            out[a] = DT.brk(x, a, 0, xb - 1, xb, up, dn, ws=0.5, mask=m)
                    return out
                v[f"{base}・NR{n}の翌日だけ・決済{xl}"] = fn
    run(ctx, "TE05", "前日の値幅が小さい日（NR4・NR7）の翌日だけ、日中のブレイク", v, family=lambda k: k.split("・")[0])


# ---------------------------------------------------------------- TG 出来高
def sec_tg(ctx):
    X = ctx["X"]
    vw = {a: vwap(X[a]) for a in COINS}
    v = {}
    for band in (0.0, 0.25):
        for filt in (False, True):
            for lo in (False, True):
                def fn(mask, band=band, filt=filt, lo=lo):
                    out = {}
                    for a in COINS:
                        x = X[a]
                        dev = (x["C"] - vw[a]) / x["atr"][x["bd"]]
                        s = np.where(dev > band, 1.0, np.where(dev < -band, -1.0, 0.0))
                        pos = np.r_[0.0, s[:-1]]
                        if filt:
                            sd = x["sdir"][x["bd"]]
                            pos = np.where(pos * sd > 0, pos, 0.0)
                        if lo:
                            pos = np.clip(pos, 0, None)
                        out[a] = DT.res_pos(x, a, pos * DT.window(x) * DT.day_mask(x, None if mask is None else mask[a]))
                    return out
                v[f"幅ATR×{band:g}・{'S1の向き' if filt else '絞りなし'}・{'買いだけ' if lo else '買い＋売り'}"] = fn
    run(ctx, "TG03", "その日のVWAPより上なら買い・下なら売り（9時〜翌5時）", v)
    v = {}
    for vm in (1.5, 2.0, 3.0):
        for H, hl in ((16, "4時間"), (96, "翌5時まで")):
            def fn(mask, vm=vm, H=H):
                out = {}
                for a in COINS:
                    x = X[a]
                    nd = x["nd"]
                    Hm, Lm = x["H"].reshape(nd, DAY), x["L"].reshape(nd, DAY)
                    hi_prev = np.maximum.accumulate(Hm, 1)
                    lo_prev = np.minimum.accumulate(Lm, 1)
                    hi_prev = np.c_[np.full(nd, np.inf), hi_prev[:, :-1]].ravel()
                    lo_prev = np.c_[np.full(nd, -np.inf), lo_prev[:, :-1]].ravel()
                    vr = x["V"] / pd.Series(x["V"]).rolling(672, min_periods=200).mean().shift(1).to_numpy()
                    big = np.nan_to_num(vr) > vm
                    sig = np.where((x["H"] > hi_prev) & big & (x["C"] > x["O"]), 1.0, np.where((x["L"] < lo_prev) & big & (x["C"] < x["O"]), -1.0, 0.0))
                    out[a] = sig_trades(ctx, a, sig, H, mask)
                return out
            v[f"出来高{vm:g}倍・{hl}"] = fn
    run(ctx, "TG05", "出来高が増えた足で、その日の高値（安値）を更新したら、その向きに", v)


# ---------------------------------------------------------------- TH 2銘柄
def sec_th(ctx):
    X = ctx["X"]
    b, e = X["BTC"], X["ETH"]
    lr = np.log(e["C"]) - np.log(b["C"])
    m, sd = pd.Series(lr).rolling(96).mean().shift(1).to_numpy(), pd.Series(lr).rolling(96).std().shift(1).to_numpy()
    z = (lr - m) / sd
    v = {}
    for k in (2.0, 3.0):
        for H in (4, 16):
            for rev in (True, False):
                def fn(mask, k=k, H=H, rev=rev):
                    sig = np.where(np.nan_to_num(z) > k, -1.0, np.where(np.nan_to_num(z) < -k, 1.0, 0.0)) * (1 if rev else -1)
                    w = DT.window(e) & DT.window(b)
                    if mask is not None:
                        w &= DT.day_mask(e, mask["ETH"]) & DT.day_mask(b, mask["BTC"])
                    p = DT.hold_after(sig, H, w)
                    return {"ETH": DT.res_pos(e, "ETH", p), "BTC": DT.res_pos(b, "BTC", -p)}
                v[f"比が{k:g}σ・{H * 0.25:g}時間・{'戻る向き' if rev else '離れる向き'}"] = fn
    run(ctx, "TH02", "ETH/BTCの比が直近1日の平均から離れたら、ETHとBTCを逆向きに持つ（戻る向きと離れる向き）", v, family=lambda k: k.split("・")[-1])
    # TH03：もう一方の銘柄が、先に（同じ足までに）同じ向きに抜けていた時だけ入る
    raw = ta03_raw(ctx)
    nd = len(ctx["idx"])

    def confirmed(r_me, r_ot):
        first = np.full(nd, 10 ** 9)
        side = np.zeros(nd)
        first[r_ot["d"]] = r_ot["eb"] - r_ot["d"] * 288
        side[r_ot["d"]] = r_ot["s"]
        my = r_me["eb"] - r_me["d"] * 288
        return (side[r_me["d"]] == r_me["s"]) & (first[r_me["d"]] <= my)
    keep = {"BTC": confirmed(raw["BTC"], raw["ETH"]), "ETH": confirmed(raw["ETH"], raw["BTC"])}
    v = {"TA03・基準": lambda mask: from_raw(ctx, raw), "TA03・もう一方が先に同じ向きに抜けた時だけ": lambda mask: from_raw(ctx, raw, keep)}
    # TC07（最初の2時間・S1の向き）も
    tc = {}
    for a in COINS:
        x = X[a]
        hi, lo = x["M"]["H"][:, :8].max(1), x["M"]["L"][:, :8].min(1)
        up = np.where(x["sdir"] > 0, hi, np.nan)
        dn = np.where(x["sdir"] < 0, lo, np.nan)
        nan = np.full(nd, np.nan)
        d, s, e_, xp, eb, xb = D2.brk_p(x["O"], x["H"], x["L"], x["C"], 96, np.full(nd, 8, np.int64), np.full(nd, END - 1, np.int64),
                                        np.full(nd, END, np.int64), up, dn, lo, hi, nan, nan, nan, np.zeros(nd, np.int64), np.isfinite(x["atr"]))
        tc[a] = dict(d=d, s=s, e=e_, xp=xp, eb=eb * 3, xb=xb * 3)

    def conf15(r_me, r_ot):
        first = np.full(nd, 10 ** 9)
        side = np.zeros(nd)
        first[r_ot["d"]] = r_ot["eb"] - r_ot["d"] * 288
        side[r_ot["d"]] = r_ot["s"]
        return (side[r_me["d"]] == r_me["s"]) & (first[r_me["d"]] <= r_me["eb"] - r_me["d"] * 288)
    keep2 = {"BTC": conf15(tc["BTC"], tc["ETH"]), "ETH": conf15(tc["ETH"], tc["BTC"])}
    v["TC07・基準"] = lambda mask: from_raw(ctx, tc)
    v["TC07・もう一方が先に同じ向きに抜けた時だけ"] = lambda mask: from_raw(ctx, tc, keep2)
    run(ctx, "TH03", "もう一方の銘柄が先に同じ向きに抜けていた時だけ入る（TA03・TC07）", v, family=lambda k: k.split("・")[0])


# ---------------------------------------------------------------- TJ 暗号資産ならでは
def sec_tj(ctx):
    X, idx = ctx["X"], ctx["idx"]
    wd = idx.dayofweek.to_numpy()
    fri = wd == 4
    nxt = idx + pd.Timedelta(days=7)
    lastfri = fri & np.asarray(nxt.month != idx.month)
    v = {}
    for lab, kind in (("金曜9〜17時・買い", ("pre", 1)), ("金曜9〜17時・売り", ("pre", -1)), ("満期の後17時〜翌5時・9〜17時の向き", ("post", 1)),
                      ("満期の後17時〜翌5時・9〜17時の逆", ("post", -1)), ("月末の満期の後・9〜17時の向き", ("mpost", 1)), ("月末の満期の後・9〜17時の逆", ("mpost", -1))):
        def fn(mask, kind=kind):
            out = {}
            for a in COINS:
                x = X[a]
                P = np.zeros((x["nd"], DAY))
                days = (fri if kind[0] != "mpost" else lastfri) & (True if mask is None else np.asarray(mask[a]))
                if kind[0] == "pre":
                    P[days, 0:32] = kind[1]
                else:
                    ra = np.sign(x["M"]["C"][:, 31] - x["prevC"])
                    P[days, 32:END] = (ra * kind[1])[days][:, None]
                out[a] = DT.res_pos(x, a, np.nan_to_num(P).ravel() * np.isfinite(x["atr"][x["bd"]]))
            return out
        v[lab] = fn
    run(ctx, "TJ03", "オプションの満期（毎週金曜17時）の前後", v, family=lambda k: k.split("・")[0])
    # TJ06：雇用統計（第1金曜の米国8時半＝UTC 12時半、冬は13時半）。発表から15・30分の向きに、2・4時間持つ
    us = ctx["us"]
    nfp = fri & np.asarray(idx.day <= 7)
    v = {}
    for kb in (1, 2):
        for H in (8, 16):
            def fn(mask, kb=kb, H=H):
                out = {}
                for a in COINS:
                    x = X[a]
                    P = np.zeros((x["nd"], DAY))
                    rb = np.where(us, 50, 54)
                    for d in np.flatnonzero(nfp & (True if mask is None else np.asarray(mask[a]))):
                        o = x["M"]["O"][d, rb[d]]
                        c = x["M"]["C"][d, rb[d] + kb - 1]
                        P[d, rb[d] + kb:min(rb[d] + kb + H, END)] = np.sign(c - o)
                    out[a] = DT.res_pos(x, a, P.ravel() * np.isfinite(x["atr"][x["bd"]]))
                return out
            v[f"発表から{kb * 15}分の向き・{H * 0.25:g}時間"] = fn
    run(ctx, "TJ06", "米国の雇用統計（第1金曜）の発表直後の向きに乗る", v)


SECTIONS = {"td": sec_td, "te": sec_te, "tg": sec_tg, "th": sec_th, "tj": sec_tj}


def main(cache, only=None):
    names = list(SECTIONS) if not only else only.split(",")
    if not only:
        print(f"（古い「{GROUP}」の行を {drop(GROUP)} 行消して、計算し直す）")
    X = DT.load(cache)
    idx = X["BTC"]["idx"]
    X5 = D2.C5.load5(cache, idx)
    D = R1.prep(cache)
    parts = R7.s1_parts(D, COINS, D["BTC"])
    per, _ = R7.build(parts, R7.size_rules(parts, COINS)["S4案"])
    s4 = R1.total(per)
    s4r = R1.stats(s4)
    R1.line("S4案（比べる相手）", s4r)
    ctx = dict(X=X, X5=X5, idx=idx, us=VX.summer(idx), s4=s4, s4r=s4r)
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
