#!/usr/bin/env python3
"""計画書（project/research/plan.md）第3回・前半：トレンドに乗る（C）と BTC・ETHの関係（H）の優先「中」16件（2018年〜）。

  C02 ゴールデンクロス／C03 3本の平均の並び／C16 ADX＋DMI／C18 回帰直線の傾き／C22 バンドウォーク／C23 1年の高値／
  C26 週足のトレンド／C27 日足の上昇中に4時間足の押し目／C28 ブレイク後の押し目／C29 値動きで割ったモメンタム／
  C34 ダウ理論（高値・安値の切り上げ）／C38 移動平均の傾きだけ
  H02 BTCが先に動いたらETHを追う／H04 BTCとETHの向きがそろう時だけ値幅ブレイク／H05 ETHの値幅ブレイクの向きをBTCで決める／
  H10 BTCとETHの配分
共通: Vantage CFD（往復0.05%、買いのスワップ年25〜35%、売り0%）、1R＝ATR（日足）1本分＝資金の0.5%（単利）、含み損込み。
      トレンドの部品は「単独の成績」と「値幅ブレイク（A01）に×0.5で足した時」を見る（計画書の1〜3次）。
      平均保有が10日以上の形は「長期の形」として、取引回数（年20回以上）の条件は外す。
結果は results.csv（グループ「計画・第3回」、計画IDつき）と plan.csv。
使い方: python3 project/tools/round3_lab.py <キャッシュ（BTC_15m.csv, ETH_15m.csv）> [followup]
        followup: C29を足す量と、H05の日数を変えて確かめる
"""
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
import more_methods_lab as MM  # noqa: E402
import portfolio_lab as PF  # noqa: E402
import round1_lab as R1  # noqa: E402
import round2_lab as R2  # noqa: E402
from registry import drop, metrics_row, record  # noqa: E402

COINS = ("BTC", "ETH")
SPLIT = PF.SPLIT
GROUP = "計画・第3回"
DAY = 96
OUT = {}


# ---------------------------------------------------------------- エンジン
@njit
def swing_p(O, H, L, C, P, A, sig, xsig, side, stop_k, maxd, cost, swd, roll_flag):
    """期間（P本）の終わりに sig なら次の期間の始値で入る。損切りは A×stop_k、xsig（期間の終わりで判定）か maxd 期間で次の始値で決済。
    1R＝A（日足のATR）＝資金の0.5%。期間ごとの損益と、取引ごとの期間・R・持った期間。"""
    nd = sig.shape[0]
    N = nd * P
    pnl = np.zeros(nd)
    te = np.empty(nd, np.int64)
    tr = np.empty(nd)
    th = np.empty(nd, np.int64)
    nt = 0
    d = 1
    while d < nd:
        Ad = A[d]
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
            if j > k and roll_flag[j]:
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
        th[nt] = out - d
        nt += 1
        d = out + 1
    return pnl, te[:nt], tr[:nt], th[:nt]


@njit
def limit_swing(O, H, L, C, P, A, ok, lvl, xsig, side, stop_k, maxd, cost, swd, roll_flag):
    """ok[d] の日に lvl[d] の指値（買いは下で待つ）。約定したら損切り A×stop_k、xsig か maxd 日で翌朝の始値で決済。"""
    nd = ok.shape[0]
    N = nd * P
    pnl = np.zeros(nd)
    te = np.empty(nd, np.int64)
    tr = np.empty(nd)
    th = np.empty(nd, np.int64)
    nt = 0
    d = 0
    while d < nd:
        Ad = A[d]
        if not ok[d] or not (Ad > 0) or np.isnan(lvl[d]):
            d += 1
            continue
        b0 = d * P
        k = -1
        e = 0.0
        for j in range(b0, b0 + P - 4):
            if (side == 1 and L[j] <= lvl[d]) or (side == -1 and H[j] >= lvl[d]):
                k = j
                e = min(O[j], lvl[d]) if side == 1 else max(O[j], lvl[d])
                break
        if k < 0:
            d += 1
            continue
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
            if j > k and roll_flag[j]:
                nroll += 1
                pnl[dd] -= sR * 0.5
            adv = L[j] if side == 1 else H[j]
            if j > k and side * (adv - stop) <= 0:
                x = min(O[j], stop) if side == 1 else max(O[j], stop)
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
        th[nt] = out - d
        nt += 1
        d = out + 1
    return pnl, te[:nt], tr[:nt], th[:nt]


def pos_daily(x, a, pos):
    """日足の建玉（その日の終値で決め、翌日に持つ）。1R＝ATR＝資金の0.5%。毎日の損益と取引一覧（保有日数つき）。"""
    held = np.r_[0.0, np.nan_to_num(pos)[:-1]]
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.nan_to_num(0.5 / x["atr"])
    change = np.abs(np.diff(np.r_[0.0, held]))
    swd = np.where(held > 0, PF.SPEC[a][1], PF.SPEC[a][2]) / 36500
    pnl = held * np.nan_to_num(x["dC"] - x["prevC"]) * scale - change * PF.SPEC[a][0] / 2 * x["dO"] * scale \
        - np.abs(held) * swd * x["dO"] * scale
    seg = np.cumsum(change > 0)
    on = held != 0
    T = pd.DataFrame({"seg": seg[on], "p": pnl[on], "day": x["idx"][on]}).groupby("seg").agg(
        R=("p", "sum"), day=("day", "first"), hold=("day", "size"))
    T["R"] = T.R / 0.5
    return pd.Series(pnl, index=x["idx"]), T


def state(enter, leave, side=1):
    """入る条件と出る条件（どちらも日足の終値で判定）から、建玉（0か side）を作る。"""
    pos = np.zeros(len(enter))
    on = False
    for i in range(len(enter)):
        if on and leave[i]:
            on = False
        elif not on and enter[i]:
            on = True
        pos[i] = side if on else 0.0
    return pos


def both(D, fn):
    per, Ts = {}, []
    for a in COINS:
        s, T = fn(D[a], a)
        per[a] = s
        Ts.append(T)
    T = pd.concat(Ts, ignore_index=True)
    return per, T[(T.day >= MM.START) & (T.day <= MM.END)]


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


def component(pid, res, v0, base):
    """部品の判定（1〜3次）。前半の安定度で一番良い形を選び、値幅ブレイクに×0.5で足す。"""
    bk = max(res, key=lambda k: res[k]["_h1s"])
    r = res[bk]
    long_term = r["_hold"] == r["_hold"] and r["_hold"] >= 10
    s1 = r["年平均%"] > 0 and r["前半の年平均%"] > 0 and r["後半の年平均%"] > 0 and (long_term or r["_n"] != r["_n"] or r["_n"] >= 20)
    neg = 0 if r["マイナスの年"] == "なし" else len(str(r["マイナスの年"]).split(","))
    pos_frac = np.mean([v["年平均%"] > 0 for v in res.values()])
    s2 = s1 and pos_frac >= 0.7 and all(c["年平均%"] > 0 for c in r["_coins"]) and r["安定度"] >= 0.8 and neg <= 2
    x = r["_x"]
    corr = v0.resample("ME").sum().corr(x.resample("ME").sum())
    mix = R1.stats(v0.add(0.5 * x, fill_value=0))
    s3 = s2 and corr < 0.5 and mix["安定度"] > base["安定度"]
    R1.line(f"→ 値幅ブレイク＋「{bk}」×0.5", mix, f" ｜月の相関{corr:+.2f}")
    record(グループ=GROUP, 計画ID=pid, 名前=f"{pid} 値幅ブレイク＋{bk}×0.5", 銘柄="BTC,ETH", 時間足="日足",
           メモ=f"月の相関{corr:+.2f}", **{k: v for k, v in mix.items() if not k.startswith("_")})
    verdict = "3次合格" if s3 else ("2次合格・3次不合格" if s2 else ("1次合格・2次不合格" if s1 else "1次不合格"))
    OUT[pid] = dict(best=bk, r=r, n=len(res), npos=sum(v["年平均%"] > 0 for v in res.values()), corr=corr, mix=mix, verdict=verdict,
                    rng=(min(v["安定度"] for v in res.values()), max(v["安定度"] for v in res.values())))
    print(f"    判定：{verdict}（{len(res)}通り中{OUT[pid]['npos']}通りがプラス、前半で一番良い形は「{bk}」）")


def sma(a, n):
    return pd.Series(a).rolling(n).mean().to_numpy()


def ls(cond_long, cond_short, longonly):
    """買い・売りの条件から建玉（1・-1・0）。"""
    with np.errstate(invalid="ignore"):
        pos = np.where(cond_long, 1.0, np.where(cond_short, 0.0 if longonly else -1.0, 0.0))
    return pos


# ---------------------------------------------------------------- 本体
def main(cache):
    print(f"（古い「{GROUP}」の行を {drop(GROUP)} 行消して、計算し直す）")
    D = R1.prep(cache)
    assert (D["BTC"]["idx"] == D["ETH"]["idx"]).all()
    per0, T0 = R2.combo(D)
    v0 = R1.total(per0)
    base = R1.stats(v0, T0)
    R1.line("A01（基準）", base)

    def run_pos(pid, label, fn, longonly):
        per, T = both(D, lambda x, a: pos_daily(x, a, fn(x, a)))
        return show(pid, f"{label}・{'買いだけ' if longonly else '買い＋売り'}", per, T)

    # ---- C02 ゴールデンクロス
    print("\n==== C02 短期平均が長期平均の上なら買い（売りは逆）")
    res = {}
    for s_, l_ in ((5, 25), (10, 30), (20, 50), (50, 200)):
        for lo in (False, True):
            res[f"{s_}/{l_}日{'・買いだけ' if lo else ''}"] = run_pos(
                "C02", f"{s_}日と{l_}日", lambda x, a, s_=s_, l_=l_, lo=lo: ls(sma(x["dC"], s_) > sma(x["dC"], l_), sma(x["dC"], s_) < sma(x["dC"], l_), lo), lo)
    component("C02", res, v0, base)

    # ---- C03 3本の平均の並び
    print("\n==== C03 短＞中＞長の時だけ買い（短＜中＜長なら売り、それ以外は持たない）")
    res = {}
    for s_, m_, l_ in ((5, 20, 60), (10, 30, 90)):
        for lo in (False, True):
            def fn(x, a, s_=s_, m_=m_, l_=l_, lo=lo):
                a1, a2, a3 = sma(x["dC"], s_), sma(x["dC"], m_), sma(x["dC"], l_)
                return ls((a1 > a2) & (a2 > a3), (a1 < a2) & (a2 < a3), lo)
            res[f"{s_}/{m_}/{l_}日{'・買いだけ' if lo else ''}"] = run_pos("C03", f"{s_}・{m_}・{l_}日", fn, lo)
    component("C03", res, v0, base)

    # ---- C16 ADX＋DMI
    print("\n==== C16 +DIが−DIより上で、ADXが一定以上なら買い（売りは逆）")
    res = {}
    for th in (20, 25, 30):
        for lo in (False, True):
            def fn(x, a, th=th, lo=lo):
                h, l, c = x["dH"], x["dL"], x["dC"]
                ph, pl, pc = np.r_[np.nan, h[:-1]], np.r_[np.nan, l[:-1]], np.r_[np.nan, c[:-1]]
                up, dn = h - ph, pl - l
                pdm = np.where((up > dn) & (up > 0), up, 0.0)
                mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
                tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
                w = lambda v: pd.Series(np.nan_to_num(v)).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
                at = w(tr)
                with np.errstate(invalid="ignore", divide="ignore"):
                    pdi, mdi = 100 * w(pdm) / at, 100 * w(mdm) / at
                    adx = w(100 * np.abs(pdi - mdi) / (pdi + mdi))
                return ls((pdi > mdi) & (adx >= th), (mdi > pdi) & (adx >= th), lo)
            res[f"ADX{th}以上{'・買いだけ' if lo else ''}"] = run_pos("C16", f"ADX{th}以上", fn, lo)
    component("C16", res, v0, base)

    # ---- C18 回帰直線の傾き
    print("\n==== C18 N日の回帰直線の傾きがプラスなら買い（マイナスなら売り）")
    res = {}
    for n in (10, 20, 40, 60):
        for lo in (False, True):
            def fn(x, a, n=n, lo=lo):
                t = np.arange(n) - (n - 1) / 2
                slope = pd.Series(x["dC"]).rolling(n).apply(lambda v: np.dot(t, v) / np.dot(t, t), raw=True).to_numpy()
                return ls(slope > 0, slope < 0, lo)
            res[f"{n}日{'・買いだけ' if lo else ''}"] = run_pos("C18", f"{n}日の傾き", fn, lo)
    component("C18", res, v0, base)

    # ---- C22 バンドウォーク
    print("\n==== C22 終値がボリンジャー＋2σを超えたら買い、真ん中（20日平均）を割ったら決済（売りは逆）")
    res = {}
    for k_ in (1.5, 2.0, 2.5):
        for lo in (False, True):
            def fn(x, a, k_=k_, lo=lo):
                s = pd.Series(x["dC"])
                m, sd = s.rolling(20).mean().to_numpy(), s.rolling(20).std().to_numpy()
                c = x["dC"]
                with np.errstate(invalid="ignore"):
                    pl = state(c > m + k_ * sd, c < m, 1)
                    ps = state(c < m - k_ * sd, c > m, -1)
                return pl if lo else pl + ps
            res[f"{k_}σ{'・買いだけ' if lo else ''}"] = run_pos("C22", f"{k_}σ", fn, lo)
    component("C22", res, v0, base)

    # ---- C23 1年の高値
    print("\n==== C23 過去N日の終値の高値を更新したら買い（50日平均を割るか、M日で決済）")
    res = {}
    for n in (180, 365):
        for ex, lab in (("ma", "50日平均を割ったら決済"), (20, "20日で決済"), (60, "60日で決済")):
            def fn(x, a, n=n, ex=ex):
                c = x["dC"]
                hi = pd.Series(c).rolling(n).max().shift(1).to_numpy()
                with np.errstate(invalid="ignore"):
                    ent = c > hi
                if ex == "ma":
                    with np.errstate(invalid="ignore"):
                        return state(ent, c < sma(c, 50), 1)
                pos = np.zeros(len(c))
                i = 0
                while i < len(c):
                    if ent[i]:
                        pos[i:i + ex] = 1.0
                        i += ex
                    else:
                        i += 1
                return pos
            res[f"{n}日の高値・{lab}"] = run_pos("C23", f"{n}日の高値・{lab}", fn, True)
    component("C23", res, v0, base)

    # ---- C26 週足のトレンド
    print("\n==== C26 週足の終値が週足のN本平均の上なら、次の週は買い（下なら売り）")
    res = {}
    for n in (5, 10, 20):
        for lo in (False, True):
            def fn(x, a, n=n, lo=lo):
                s = pd.Series(x["dC"], index=x["idx"])
                wk = s.resample("W-SUN").last()
                sig = np.sign(wk - wk.rolling(n).mean())
                if lo:
                    sig = sig.clip(lower=0)
                d = sig.reindex(x["idx"], method="ffill").shift(1)          # その週の終わりで決め、次の日から持つ
                d[x["idx"].weekday != 0] = np.nan
                return d.ffill().fillna(0).to_numpy()
            res[f"{n}週{'・買いだけ' if lo else ''}"] = run_pos("C26", f"{n}週平均", fn, lo)
    component("C26", res, v0, base)

    # ---- C38 移動平均の傾きだけ
    print("\n==== C38 N日平均が前日より上がっていれば買い（下がっていれば売り）")
    res = {}
    for n in (10, 20, 50, 100):
        for lo in (False, True):
            def fn(x, a, n=n, lo=lo):
                m = sma(x["dC"], n)
                pm = np.r_[np.nan, m[:-1]]
                return ls(m > pm, m < pm, lo)
            res[f"{n}日{'・買いだけ' if lo else ''}"] = run_pos("C38", f"{n}日平均の傾き", fn, lo)
    component("C38", res, v0, base)

    # ---- C29 値動きで割ったモメンタム
    print("\n==== C29 過去30日の騰落÷その間の値動きの大きさ（z）が一定以上なら買い（以下なら売り）")
    res = {}
    for th in (0.5, 1.0, 2.0):
        for lo in (False, True):
            def fn(x, a, th=th, lo=lo):
                lr = np.log(pd.Series(x["dC"])).diff()
                z = (lr.rolling(30).sum() / (lr.rolling(30).std() * np.sqrt(30))).to_numpy()
                return ls(z > th, z < -th, lo)
            res[f"z{th}{'・買いだけ' if lo else ''}"] = run_pos("C29", f"z>{th}", fn, lo)
    component("C29", res, v0, base)

    # ---- C34 ダウ理論
    print("\n==== C34 直近N日の高値・安値が、その前のN日より両方切り上がったら買い（両方切り下がったら売り）")
    res = {}
    for n in (5, 10, 20):
        for lo in (False, True):
            def fn(x, a, n=n, lo=lo):
                hh = pd.Series(x["dH"]).rolling(n).max()
                ll = pd.Series(x["dL"]).rolling(n).min()
                up = ((hh > hh.shift(n)) & (ll > ll.shift(n))).to_numpy()
                dn = ((hh < hh.shift(n)) & (ll < ll.shift(n))).to_numpy()
                return ls(up, dn, lo)
            res[f"{n}日{'・買いだけ' if lo else ''}"] = run_pos("C34", f"{n}日ずつの高値・安値", fn, lo)
    component("C34", res, v0, base)

    # ---- C27 日足の上昇中に4時間足の押し目
    print("\n==== C27 日足が50日平均の上で、4時間足のRSI(14)が30以下なら次の4時間足で買い（RSI50以上か2日で決済、損切りATR×1）")
    res = {}
    for lo_th in (30, 35):
        def fn(x, a, lo_th=lo_th):
            Pd = R2.periods(x, 16, 0)
            c4 = Pd["pC"]
            d4 = np.diff(c4, prepend=c4[0])
            up = pd.Series(np.clip(d4, 0, None)).ewm(alpha=1 / 14, adjust=False).mean()
            dn = pd.Series(np.clip(-d4, 0, None)).ewm(alpha=1 / 14, adjust=False).mean()
            rsi4 = (100 - 100 / (1 + up / dn)).to_numpy()
            dayi = np.arange(Pd["n"]) // 6
            trend = np.r_[np.nan, (x["dC"] > sma(x["dC"], 50)).astype(float)[:-1]]     # 前日までの日足
            tr4 = trend[np.clip(dayi, 0, len(trend) - 1)] == 1
            A4 = x["atr"][np.clip(dayi, 0, len(trend) - 1)]
            with np.errstate(invalid="ignore"):
                sig = tr4 & (rsi4 <= lo_th)
                xs = rsi4 >= 50
            fl = Pd["fl"]
            pnl, te, tr, th = swing_p(fl["O"], fl["H"], fl["L"], fl["C"], 16, A4, sig, xs, 1, 1.0, 12, PF.SPEC[a][0],
                                      PF.SPEC[a][1] / 36500, Pd["roll"])
            s = pd.Series(pnl, index=Pd["idx"])
            T = pd.DataFrame({"day": Pd["idx"][te], "R": tr, "hold": th / 6})
            return s.groupby(s.index.normalize()).sum(), T
        per, T = both(D, fn)
        res[f"RSI{lo_th}以下"] = show("C27", f"日足の上昇中・4時間足RSI{lo_th}以下で買い", per, T, tf="4時間足")
    component("C27", res, v0, base)

    # ---- C28 ブレイク後の押し目
    print("\n==== C28 20日高値を更新してからN日以内に、前日安値まで下げたら買い（指値。5日か20日平均割れで決済、損切りATR×1.5）")
    res = {}
    for n in (3, 5):
        def fn(x, a, n=n):
            c = x["dC"]
            hi = pd.Series(x["dH"]).rolling(20).max().shift(1).to_numpy()
            with np.errstate(invalid="ignore"):
                brk = (c > hi).astype(float)
            recent = pd.Series(brk).rolling(n, min_periods=1).max().shift(1).fillna(0).to_numpy() > 0   # 前日までのN日に更新があった
            lvl = np.r_[np.nan, x["dL"][:-1]]
            with np.errstate(invalid="ignore"):
                xs = c < sma(c, 20)
            roll = (np.arange(len(x["O"])) % DAY == 84)
            pnl, te, tr, th = limit_swing(x["O"], x["H"], x["L"], x["C"], DAY, x["atr"], recent, lvl, xs, 1, 1.5, 5,
                                          PF.SPEC[a][0], PF.SPEC[a][1] / 36500, roll)
            return pd.Series(pnl, index=x["idx"]), pd.DataFrame({"day": x["idx"][te], "R": tr, "hold": th})
        per, T = both(D, fn)
        res[f"{n}日以内"] = show("C28", f"更新から{n}日以内の押し目", per, T, tf="日足（約定は15分足）")
    component("C28", res, v0, base)

    # ---- H02 BTCが先、ETHが後
    print("\n==== H02 BTCが1時間で一定以上動いたら、ETHを同じ向きに（次の15分足で入り、N時間後に決済）")
    res = {}
    xb, xe = D["BTC"], D["ETH"]
    for lead, lab in (("BTC", "BTCの動きでETH"), ("ETH", "参考：ETH自身の動きでETH")):
        src = xb if lead == "BTC" else xe
        r1h = np.r_[np.full(4, np.nan), src["C"][4:] / src["C"][:-4] - 1]
        for th_ in (0.01, 0.02):
            for hh in (1, 4):
                O = xe["O"]
                n = len(O)
                dayi = np.arange(n) // DAY
                A = xe["atr"][np.clip(dayi, 0, len(xe["atr"]) - 1)]
                Rs, days = [], []
                pnl = np.zeros(len(xe["dO"]))
                j = 4
                while j < n - hh * 4 - 1:
                    if abs(r1h[j]) >= th_ and A[j] > 0:
                        side = 1 if r1h[j] > 0 else -1
                        i1, i2 = j + 1, j + 1 + hh * 4
                        e, ex = O[i1], O[i2]
                        rolls = sum(1 for r in range(i1 + 1, i2) if r % DAY == 84)
                        sw = (PF.SPEC["ETH"][1] if side == 1 else PF.SPEC["ETH"][2]) / 36500
                        R = side * (ex - e) / A[j] - PF.SPEC["ETH"][0] * e / A[j] - rolls * sw * e / A[j]
                        pnl[dayi[j]] += R * 0.5
                        Rs.append(R)
                        days.append(xe["idx"][dayi[j]])
                        j = i2
                    else:
                        j += 1
                per = {"BTC": pd.Series(0.0, index=xe["idx"]), "ETH": pd.Series(pnl, index=xe["idx"])}
                T = pd.DataFrame({"day": days, "R": Rs})
                T = T[(T.day >= MM.START) & (T.day <= MM.END)]
                res[(lead, th_, hh)] = show("H02", f"{lab}：1時間で{th_ * 100:.0f}%以上・{hh}時間持つ", per, T, tf="15分足")
    OUT["H02"] = res

    # ---- H04・H05 値幅ブレイクの向きを2銘柄で決める
    print("\n==== H04 BTCとETHの両方が5日平均の同じ側の時だけ、値幅ブレイク（A01）")
    sides = {a: np.sign(D[a]["dO"] - R1.sma_prev(D[a], 5)) for a in COINS}
    agree = pd.Series(np.nan_to_num(sides["BTC"]) == np.nan_to_num(sides["ETH"]), index=D["BTC"]["idx"])
    T = T0.copy()
    T["d"] = T.day.dt.normalize()
    keep = agree.reindex(T.d).fillna(False).to_numpy()
    per, U = R2.from_trades(T, keep)
    OUT["H04"] = (show("H04", "両方の向きがそろう日だけ", per, U, tf="15分足"),
                  show("H04", "参考：向きが分かれる日だけ", *R2.from_trades(T, ~keep), tf="15分足"), agree.mean())
    print(f"    向きがそろう日の割合 {agree[(agree.index >= MM.START)].mean() * 100:.0f}%")

    print("\n==== H05 ETHの値幅ブレイクの向きを、BTCの5日平均の向きで決める（BTCはそのまま）")
    per, Ts = {}, []
    for a in COINS:
        Pd = R2.periods(D[a], DAY, 0)
        if a == "BTC":
            s, T_ = R2.run(Pd, a)
        else:
            bs = sides["BTC"]
            s, T_ = R2.run(Pd, a, ma=False, extra_ok=lambda P_, side, bs=bs: (side * np.nan_to_num(bs)) > 0)
        per[a] = s
        Ts.append(T_)
    OUT["H05"] = show("H05", "ETHの向きをBTCの5日平均で決める", per, pd.concat(Ts), tf="15分足")

    # ---- H10 2銘柄の配分
    print("\n==== H10 BTCとETHの配分（合計の量は今と同じ2）")
    res = {}
    for wb, lab in ((1.0, "1:1（今の形）"), (1.2, "BTC多め 1.2:0.8"), (0.8, "ETH多め 0.8:1.2"), (1.4, "BTC多め 1.4:0.6"), (0.6, "ETH多め 0.6:1.4")):
        per = {"BTC": per0["BTC"] * wb, "ETH": per0["ETH"] * (2 - wb)}
        res[lab] = show("H10", lab, per, tf="15分足")
    vb, ve = per0["BTC"].rolling(60).std().shift(1), per0["ETH"].rolling(60).std().shift(1)
    wbv = (2 * (1 / vb) / (1 / vb + 1 / ve)).fillna(1.0)
    per = {"BTC": per0["BTC"] * wbv, "ETH": per0["ETH"] * (2 - wbv)}
    res["振れの逆数（過去60日）"] = show("H10", "振れの逆数で配分（過去60日、毎日見直す）", per, tf="15分足")
    OUT["H10"] = res
    OUT["_base"] = base
    return OUT


if __name__ == "__main__" and len(sys.argv) == 2:
    main(sys.argv[1])


def followup(cache):
    """第3回・前半の続き：C29（値動きで割ったモメンタム）を足す量と、H05（ETHの向きをBTCで決める）の日数を変えて確かめる。"""
    D = R1.prep(cache)
    per0, T0 = R2.combo(D)
    v0 = R1.total(per0)
    base = R1.stats(v0, T0)
    R1.line("A01（基準）", base, f" ｜効率{base['年平均%'] / base['最大の落ち込み%']:.2f}")
    print("\n==== C29 を足す量（年平均と落ち込みの比＝効率も見る）")
    for th in (0.5, 1.0):
        def fn(x, a, th=th):
            lr = np.log(pd.Series(x["dC"])).diff()
            z = (lr.rolling(30).sum() / (lr.rolling(30).std() * np.sqrt(30))).to_numpy()
            return ls(z > th, z < -th, False)
        per, T = both(D, lambda x, a: pos_daily(x, a, fn(x, a)))
        xm = R1.total(per)
        for wgt in (0.25, 0.5, 0.75, 1.0):
            r = R1.stats(v0.add(wgt * xm, fill_value=0))
            R1.line(f"値幅ブレイク＋z>{th}×{wgt}", r, f" ｜効率{r['年平均%'] / r['最大の落ち込み%']:.2f}")
            record(グループ=GROUP, 計画ID="C29", 名前=f"C29 値幅ブレイク＋z>{th}（買い＋売り）×{wgt}", 銘柄="BTC,ETH", 時間足="日足",
                   メモ=f"効率{r['年平均%'] / r['最大の落ち込み%']:.2f}", **{k: v for k, v in r.items() if not k.startswith("_")})
    r = R1.stats(v0 * 1.44)
    R1.line("参考：値幅ブレイクの量を1.44倍", r, f" ｜効率{r['年平均%'] / r['最大の落ち込み%']:.2f}")

    print("\n==== H05 ETHの向きを、BTCのN日平均で決める")
    for n in (3, 5, 10):
        bs = np.sign(D["BTC"]["dO"] - R1.sma_prev(D["BTC"], n))
        per, Ts = {"BTC": per0["BTC"]}, []
        Pd = R2.periods(D["ETH"], DAY, 0)
        s, T_ = R2.run(Pd, "ETH", ma=False, extra_ok=lambda P_, side, bs=bs: (side * np.nan_to_num(bs)) > 0)
        per["ETH"] = s
        r = show("H05", f"ETHの向きをBTCの{n}日平均で決める", per, T_, tf="15分足")
    for n in (3, 10):
        Pd = R2.periods(D["ETH"], DAY, 0)
        es = np.sign(D["ETH"]["dO"] - R1.sma_prev(D["ETH"], n))
        s, T_ = R2.run(Pd, "ETH", ma=False, extra_ok=lambda P_, side, es=es: (side * np.nan_to_num(es)) > 0)
        print(f"    参考：ETH自身の{n}日平均の向き → ETHだけ年{metrics_row(MM.clip(s))['年平均%']:+.2f}%・安定度{metrics_row(MM.clip(s))['安定度']:.2f}")
    s, _ = R2.run(R2.periods(D["ETH"], DAY, 0), "ETH")
    print(f"    参考：ETH自身の5日平均の向き（今の形） → ETHだけ年{metrics_row(MM.clip(s))['年平均%']:+.2f}%・安定度{metrics_row(MM.clip(s))['安定度']:.2f}")


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "followup":
    followup(sys.argv[1])
