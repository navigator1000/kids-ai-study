#!/usr/bin/env python3
"""計画書（project/research/plan.md）第5回・後半：奇抜なアイデア（O）と、実際の運用に近づける確認（P）（BTC・ETH、2018年〜）。

  O12 コイン投げ（ランダムな時刻・向きで入り、値幅ブレイクと同じ翌朝9時に決済）
  O19 負けた戦略の逆（RSI(2)の逆張り＝D02 の反対）／O20 全敗の後は休む／O21 円で見るビットコイン（ドル円で円建てに直す）
  P02 買いのスワップが上がったら／P05 国内現物の買いだけ（SBI VCトレード：往復0.10〜0.20%・スワップなし）
  P07 最近1年（2025年11月〜2026年11月）／P09 1回の量を0.25%・0.5%・1%にした時の落ち込みの分布
  おまけ（データの取得なしで試せるOの「低」18件）：O01 満月・新月／O02 13日の金曜日／O06 ビットコインの記念日／O07 価格のゾロ目／
  O08 旧正月／O09 米国の確定申告の期限／O10 ハロウィン効果／O11 素数の日／O13 フィボナッチ数の日／O14 終値の下1桁／
  O15 半減期の最適日探し／O16 日本の祝日／O17 米大統領選の年／O18 オリンピック・W杯／O25 桁が増える時／O26 月曜は休む／
  O27 午前と午後／O28 サンタクロース・ラリー
共通: Vantage CFD（往復0.05%、買いのスワップ年25〜35%、売り0%）、1R＝ATR（日足）1本分＝資金の0.5%（単利）。
1日は日本時間9時〜翌9時（UTCの日付と同じ）。日付の条件は、その朝にわかるものだけを使う。
「休む・だけ」の条件は、同じ割合の日をランダムに選んだ1,000回と比べる（偶然でも出る程度かを見る）。
必要なもの: pip install holidays（日本の祝日）
結果は results.csv（グループ「計画・第5回後半」、計画IDつき）と plan.csv。
使い方: python3 project/tools/round5b_lab.py <キャッシュ> [o12,o19,o20,o21,p02,p07,p09,cal のどれか（カンマ区切り）]
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
import more_methods_lab as MM  # noqa: E402
import portfolio_lab as PF  # noqa: E402
import round1_lab as R1  # noqa: E402
import round2_lab as R2  # noqa: E402
import round2b_lab as R2B  # noqa: E402
import round3_lab as R3  # noqa: E402
import round3b_lab as R3B  # noqa: E402
import round4_lab as R4  # noqa: E402
import round4b_lab as R4B  # noqa: E402
from registry import SPLIT, drop, metrics_row, record  # noqa: E402

COINS = ("BTC", "ETH")
GROUP = "計画・第5回後半"
DAY = 96
R3.GROUP = GROUP
warnings.filterwarnings("ignore")
SUM = {}


# ---------------------------------------------------------------- 共通
def keep(pid, **kw):
    SUM.setdefault(pid, {}).update(kw)


def brief(r):
    return dict(a=r["年平均%"], dd=r["最大の落ち込み%"], s=r["安定度"], s1=r.get("_h1s"), s2=r.get("_h2s"), a1=r["前半の年平均%"],
                a2=r["後半の年平均%"], neg=r["マイナスの年"], n=r.get("_n"), cal=r.get("_cal"))


def show(pid, name, per, T=None, memo="", tf="15分足", cost_note=None):
    """R3.show と同じ。コストの前提を変えた時は cost_note を記録する。"""
    x = R1.total(per)
    r = R1.stats(x, T)
    cb, ce = (metrics_row(MM.clip(per[a])) for a in COINS)
    r["_coins"] = (cb, ce)
    r["_hold"] = T["hold"].mean() if T is not None and "hold" in T and len(T) else np.nan
    r["_x"] = x
    R1.line(name, r, f" ｜BTC{cb['年平均%']:+.1f}% ETH{ce['年平均%']:+.1f}%")
    extra = {"コストの前提": cost_note} if cost_note else {}
    record(グループ=GROUP, 計画ID=pid, 名前=f"{pid} {name}", 銘柄="BTC,ETH", 時間足=tf,
           メモ=(memo + "。" if memo else "") + f"BTCだけ年{cb['年平均%']:+.1f}%・ETHだけ年{ce['年平均%']:+.1f}%",
           **extra, **{k: v for k, v in r.items() if not k.startswith("_")})
    return r


def scaled(ctx, pid, name, per, T, m, memo="", cost_note=None):
    """値幅ブレイクの毎日の損益に、日ごとの倍率 m（0なら休む）をかける。"""
    per2 = {a: per[a] * m[a] for a in COINS}
    T = T.reset_index(drop=True)
    keepm = np.zeros(len(T), bool)
    for a in COINS:
        sel = (T["coin"] == a).to_numpy()
        pos = ctx["D"][a]["idx"].get_indexer(pd.DatetimeIndex(T.loc[sel, "day"]))
        keepm[sel] = np.asarray(m[a])[pos] > 0
    return show(pid, name, per2, T[keepm], memo=memo, cost_note=cost_note)


def dt(ctx, pid, name, sides, memo=""):
    """日ごとの向き（その朝にわかる値）で9時〜翌9時を持つ。買い×1・売り×0.5。"""
    f = (lambda x, a: sides[a]) if isinstance(sides, dict) else (lambda x, a: sides)
    return show(pid, name, *R3.both(ctx["D"], R4B.daytrade(f)), memo=memo, tf="日足")


def hold(ctx, pid, name, cond, memo=""):
    """条件（その朝にわかる値）が成り立つ日だけ買いで持つ（続く間は持ち続け、出入りの時だけコスト）。"""
    f = (lambda x, a: cond[a]) if isinstance(cond, dict) else (lambda x, a: cond)
    return show(pid, name, *R3.both(ctx["D"], R4B.holdwhile(f)), memo=memo, tf="日足")


@contextmanager
def spec(cost=None, swl=None, sws=None):
    """コスト・スワップ（年%）を一時的に変える。None はそのまま。"""
    saved = {a: PF.SPEC[a] for a in COINS}
    try:
        for a in COINS:
            c, sl, ss = PF.SPEC[a]
            PF.SPEC[a] = (c if cost is None else cost, sl if swl is None else swl, ss if sws is None else sws)
        yield
    finally:
        for a in COINS:
            PF.SPEC[a] = saved[a]


def calib(ctx, daily, keepm, runs=1000, seed=0):
    """日ごとの損益（2銘柄の合計）を keepm の日だけにした結果を、同じ割合の日をランダムに選んだ1,000回と比べる。"""
    w = (ctx["days"] >= MM.START) & (ctx["days"] <= MM.END)
    v, k, mid = daily[w], np.asarray(keepm)[w].astype(bool), ctx["mid"][w]

    def st(z):
        m = np.bincount(mid, weights=z)
        return m.mean() / m.std() * np.sqrt(12) if m.std() > 0 else 0.0
    act_s, act_a = st(v * k), (v * k).sum() / R1.YEARS
    g = np.random.default_rng(seed)
    f = k.mean()
    S, A = np.empty(runs), np.empty(runs)
    for i in range(runs):
        r = g.random(len(v)) < f
        S[i], A[i] = st(v * r), (v * r).sum() / R1.YEARS
    return dict(frac=f * 100, s=act_s, a=act_a, s_rank=(S < act_s).mean() * 100, a_rank=(A < act_a).mean() * 100,
                s_med=float(np.median(S)), a_med=float(np.median(A)))


def vf(ctx, pid, name, keepm, memo=""):
    """値幅ブレイク（A01）を keepm の日だけにする（それ以外は休む）。ランダムに同じ割合の日を選んだ場合と比べる。"""
    r = scaled(ctx, pid, f"値幅ブレイク：{name}", ctx["per0"], ctx["T0"], {a: np.asarray(keepm, float) for a in COINS}, memo)
    c = calib(ctx, ctx["v0d"], keepm)
    print(f"      残した日{c['frac']:.0f}%｜ランダムに同じ割合の日を選んだ1,000回の中で、安定度は下から{c['s_rank']:.0f}%"
          f"（ランダムの中央値{c['s_med']:.2f}）")
    r["_cal"] = c
    return r


def hf(ctx, pid, name, keepm, memo=""):
    """keepm の日だけ買いで持つ。ランダムに同じ割合の日を選んだ場合と比べる。"""
    r = hold(ctx, pid, f"{name}だけ買い持ち", {a: np.asarray(keepm, float) for a in COINS}, memo)
    c = calib(ctx, ctx["longd"], keepm)
    print(f"      持った日{c['frac']:.0f}%｜ランダムに同じ割合の日を選んだ1,000回の中で、年平均は下から{c['a_rank']:.0f}%"
          f"（ランダムの中央値{c['a_med']:+.2f}%）")
    r["_cal"] = c
    return r


# ---------------------------------------------------------------- O12 コイン投げ
def sec_o12(ctx, runs=1000):
    pid = "O12"
    D, days, T0 = ctx["D"], ctx["days"], ctx["T0"]
    print("\n==== O12 コイン投げ：A01と同じ回数だけ、ランダムな時刻・ランダムな向きで入り、翌朝9時に決済（1,000回）")
    iw = np.nonzero((days >= MM.START) & (days <= MM.END))[0]
    mid = ctx["mid"][iw]
    nb = DAY - 4
    ar = np.arange(len(iw))
    RL, RS, q = {}, {}, {}
    for a in COINS:
        x = D[a]
        O, Cx = x["M"]["open"][:, :nb], x["M"]["close"][:, -1]
        Ai = np.where(x["atr"] > 0, x["atr"], np.nan)[:, None]
        with np.errstate(invalid="ignore"):
            mv = (Cx[:, None] - O) / Ai
            c = PF.SPEC[a][0] * O / Ai
            sw = (np.arange(nb) < 84)[None, :] * PF.SPEC[a][1] / 36500 * O / Ai
        RL[a], RS[a] = np.nan_to_num(mv - c - sw)[iw], np.nan_to_num(-mv - c)[iw]
        q[a] = (T0.coin == a).sum() / len(iw)
    g = np.random.default_rng(0)
    out = {}
    for mode in ("コイン投げ（向きもランダム）", "いつも買い（時刻だけランダム）"):
        A_, S_, DD_ = np.empty(runs), np.empty(runs), np.empty(runs)
        for i in range(runs):
            tot = np.zeros(len(iw))
            for a in COINS:
                take = g.random(len(iw)) < q[a]
                b = g.integers(0, nb, len(iw))
                lng = (g.random(len(iw)) < 0.5) if mode.startswith("コイン") else np.ones(len(iw), bool)
                tot += take * np.where(lng, RL[a][ar, b], 0.5 * RS[a][ar, b]) * 0.5
            m = np.bincount(mid, weights=tot)
            eq = np.cumsum(tot)
            A_[i], S_[i], DD_[i] = tot.sum() / R1.YEARS, m.mean() / m.std() * np.sqrt(12), (np.maximum.accumulate(eq) - eq).max()
        o = dict(a_med=np.median(A_), a95=np.percentile(A_, 95), a_max=A_.max(), s_med=np.median(S_), s95=np.percentile(S_, 95),
                 s_max=S_.max(), dd_med=np.median(DD_), beat=(S_ >= ctx["base"]["安定度"]).mean() * 100, pos=(A_ > 0).mean() * 100)
        out[mode] = o
        print(f"    {mode}：年平均の中央値{o['a_med']:+.2f}%（上位5%{o['a95']:+.2f}%・一番良い回{o['a_max']:+.2f}%、プラスの回{o['pos']:.0f}%）"
              f"／安定度の中央値{o['s_med']:.2f}（一番良い回{o['s_max']:.2f}）／落ち込みの中央値{o['dd_med']:.1f}%"
              f"／A01の安定度{ctx['base']['安定度']:.2f}以上の回{o['beat']:.1f}%")
    keep(pid, out=out, q=q)


# ---------------------------------------------------------------- O19 負けた戦略の逆
def rsi2(c):
    d = np.diff(c, prepend=c[0])
    up = pd.Series(np.clip(d, 0, None)).ewm(alpha=0.5, adjust=False).mean()
    dn = pd.Series(np.clip(-d, 0, None)).ewm(alpha=0.5, adjust=False).mean()
    return (100 - 100 / (1 + up / dn)).to_numpy()


def sec_o19(ctx):
    pid = "O19"
    print("\n==== O19 負けた戦略の逆：RSI(2)の逆張り（D02：10未満で買い・90超で売り、5日平均に戻ったら翌朝決済、最長5日）の向きを反対にする")

    def make(lo, hi, reverse, stop):
        def fn(x, a):
            c = x["dC"]
            r2 = rsi2(c)
            m5 = R3.sma(c, 5)
            with np.errstate(invalid="ignore"):
                if not reverse:
                    fl = lambda x_, a_: R3B.swing(x_, a_, r2 < lo, 1, 5, c > m5, stop)
                    fs = lambda x_, a_: R3B.swing(x_, a_, r2 > hi, -1, 5, c < m5, stop)
                else:                                  # 同じ合図・同じ決済の時期で、向きだけ反対
                    fl = lambda x_, a_: R3B.swing(x_, a_, r2 > hi, 1, 5, c < m5, stop)
                    fs = lambda x_, a_: R3B.swing(x_, a_, r2 < lo, -1, 5, c > m5, stop)
                return R3B.two_sides(fl, fs, 0.5)(x, a)
        return fn
    run = lambda name, fn: show(pid, name, *R3.both(ctx["D"], fn), tf="日足（約定は15分足）")
    ref = run("参考：D02（元の逆張り・10未満で買い・90超で売り・損切りATR×2）", make(10, 90, False, 2.0))
    res = {}
    for lo, hi in ((10, 90), (5, 95), (20, 80)):
        for stop in (2.0, 99.0):
            lab = f"{lo}未満で売り・{hi}超で買い・" + ("損切りATR×2" if stop < 99 else "損切りなし")
            res[lab] = run(f"逆（{lab}）", make(lo, hi, True, stop))
    R3.component(pid, res, ctx["v0"], ctx["base"])
    o = R3.OUT[pid]
    keep(pid, ref=brief(ref), rows={k: brief(v) for k, v in res.items()}, verdict=o["verdict"], best=o["best"], mix=brief(o["mix"]),
         corr=o["corr"])


# ---------------------------------------------------------------- O20 全敗の後は休む
def sec_o20(ctx):
    pid = "O20"
    D, days, per0, T0 = ctx["D"], ctx["days"], ctx["per0"], ctx["T0"]
    print("\n==== O20 全敗の後は休む：直近N日の値幅ブレイク（休んだ日も、取ったはずの結果で数える）が全部負けなら、M日休む")
    T = T0.reset_index(drop=True)
    di = days.get_indexer(pd.DatetimeIndex(T.day))
    won = (T.R.to_numpy() > 0).astype(float)

    def rest(coins, look, rn, minn):
        sel = T.coin.isin(coins).to_numpy()
        n = np.bincount(di[sel], minlength=len(days)).astype(float)
        wv = np.bincount(di[sel], weights=won[sel], minlength=len(days))
        cn, cw = np.r_[0, np.cumsum(n)], np.r_[0, np.cumsum(wv)]
        flag = np.zeros(len(days), bool)
        for d in range(len(days)):
            lo = max(0, d - look)
            if cn[d] - cn[lo] >= minn and cw[d] - cw[lo] == 0:
                flag[d:d + rn] = True
        return (~flag).astype(float)
    res, info = {}, {}
    for look, rn, minn in ((7, 3, 2), (5, 3, 2), (10, 3, 2), (7, 1, 2), (7, 5, 2), (7, 3, 3)):
        m = {a: rest([a], look, rn, minn) for a in COINS}
        lab = f"直近{look}日の{minn}回以上が全敗なら{rn}日休む"
        res[lab] = scaled(ctx, pid, f"値幅ブレイク：{lab}（銘柄ごと）", per0, T0, m)
        skipped = np.concatenate([T.R.to_numpy()[(T.coin == a).to_numpy() & (m[a][np.clip(di, 0, None)] == 0)] for a in COINS])
        info[lab] = dict(skip_n=len(skipped) / R1.YEARS, skip_R=float(skipped.mean()) if len(skipped) else np.nan)
        print(f"      休んだ取引 年{info[lab]['skip_n']:.1f}回・1回平均{info[lab]['skip_R']:+.3f}R（A01全体は{T.R.mean():+.3f}R）")
    m = rest(list(COINS), 7, 3, 2)
    res["2銘柄合わせて"] = scaled(ctx, pid, "値幅ブレイク：2銘柄合わせて直近7日の2回以上が全敗なら3日休む", per0, T0, {a: m for a in COINS})
    keep(pid, rows={k: brief(v) for k, v in res.items()}, info=info, base=brief(ctx["base"]), allR=float(T.R.mean()))


# ---------------------------------------------------------------- O21 円で見るビットコイン・P05 国内現物
def fx_path(cache, days):
    """ドル円。その日の朝に確定している前日の終値（f0）と、その日の終わりの値（f1：翌日の f0）。"""
    s = R4B.macro(cache, "USDJPY")
    st = s.copy()
    st.index = st.index + pd.Timedelta(days=1)
    f0 = st.reindex(days, method="ffill").to_numpy()
    return f0, np.r_[f0[1:], f0[-1]]


def to_jpy(x, f0, f1):
    """円建ての15分足。1日の中はその朝のドル円（f0）で換算し、その日の最後の終値だけ、その日の終わりのドル円（f1）で換算する。
    1日の中のドル円の動きは、先読みにならないよう信号には使わない（損益には、その日のドル円の動きが入る）。"""
    M = {k: x["M"][k] * f0[:, None] for k in ("open", "high", "low", "close")}
    M["close"] = M["close"].copy()
    M["close"][:, -1] = x["M"]["close"][:, -1] * f1
    y = dict(x)
    y["M"] = M
    y.update(O=np.ascontiguousarray(M["open"].ravel()), H=np.ascontiguousarray(M["high"].ravel()),
             L=np.ascontiguousarray(M["low"].ravel()), C=np.ascontiguousarray(M["close"].ravel()),
             dO=M["open"][:, 0], dH=M["high"].max(1), dL=M["low"].min(1), dC=M["close"][:, -1])
    y["prevC"] = np.r_[np.nan, y["dC"][:-1]]
    tr = np.nanmax(np.vstack([y["dH"] - y["dL"], np.abs(y["dH"] - y["prevC"]), np.abs(y["dL"] - y["prevC"])]), axis=0)
    y["atr"] = pd.Series(tr).rolling(14).mean().shift(1).to_numpy()
    y["rng"] = np.r_[np.nan, (y["dH"] - y["dL"])[:-1]]
    y["sma5"] = pd.Series(y["dC"]).rolling(5).mean().shift(1).to_numpy()
    return y


def s1_long(Dx):
    """S1（ETHはBTCの5日平均の向き）の買いだけ。量の倍率は別にかける。"""
    bs = np.sign(Dx["BTC"]["dO"] - R1.sma_prev(Dx["BTC"], 5))
    per, Ts = {}, []
    for a in COINS:
        Pd = R2.periods(Dx[a], DAY, 0)
        if a == "ETH":
            s, T = R2.run(Pd, a, ma=False, extra_ok=lambda P_, side: (side * np.nan_to_num(bs)) > 0, short_w=0)
        else:
            s, T = R2.run(Pd, a, short_w=0)
        per[a] = s
        Ts.append(T)
    return per, pd.concat(Ts, ignore_index=True)


def sec_o21(ctx):
    D = ctx["D"]
    print("\n==== O21 円で見るビットコイン：ドル円で円建ての値動きに直し（1日の中はその朝のドル円で換算）、同じ値幅ブレイクをする")
    f0, f1 = fx_path(ctx["cache"], ctx["days"])
    Dj = {a: to_jpy(D[a], f0, f1) for a in COINS}
    ctx["Dj"] = Dj
    perj, Tj = R2.combo(Dj)
    a01j = show("O21", "値幅ブレイク（A01）を円建ての値動きで（損益も円建て）", perj, Tj)
    vb5j, Tb5j = R4.vbo_var(Dj, "btc5", False)
    Mqj = {a: R4.mult(Dj[a], "静かな日だけ") for a in COINS}
    s1j = scaled(ctx, "O21", "S1を円建ての値動きで", vb5j, Tb5j, Mqj)
    # 同じ日に同じ向きで約定したか
    k0 = set(zip(ctx["T0"].coin, ctx["T0"].day, ctx["T0"].side))
    kj = set(zip(Tj.coin, Tj.day, Tj.side))
    same = len(k0 & kj) / max(len(k0 | kj), 1) * 100
    cr = ctx["v0"].resample("ME").sum().corr(a01j["_x"].resample("ME").sum())
    print(f"    ドル建てと円建てで、同じ日・同じ向きの取引は{same:.0f}%。月の損益の相関{cr:+.2f}")
    keep("O21", a01=brief(a01j), s1=brief(s1j), same=same, corr=cr)

    print("\n==== P05 国内現物（SBI VCトレード）：値幅ブレイクの買いだけ。往復0.10・0.15・0.20%、スワップなし")
    rows = {}
    with spec(swl=None):
        per, T = R2.combo(D, short_w=0)
        rows["参考：Vantageで買いだけ（往復0.05%・スワップあり）"] = show("P05", "参考：Vantageで値幅ブレイクの買いだけ", per, T)
    for cost in (0.0010, 0.0015, 0.0020):
        note = f"国内現物（往復{cost * 100:.2f}%・スワップなし・買いだけ）"
        with spec(cost=cost, swl=0.0, sws=0.0):
            per, T = R2.combo(D, short_w=0)
            rows[f"A01の買い・{cost * 100:.2f}%"] = show("P05", f"国内現物：値幅ブレイク（A01）の買いだけ（往復{cost * 100:.2f}%）", per, T, cost_note=note)
            per, T = R2.combo(Dj, short_w=0)
            rows[f"A01の買い・円建てで判定・{cost * 100:.2f}%"] = show("P05", f"国内現物：A01の買いだけ・円建ての値動きで判定（往復{cost * 100:.2f}%）",
                                                              per, T, cost_note=note)
            per, T = s1_long(D)
            rows[f"S1の買い・{cost * 100:.2f}%"] = scaled(ctx, "P05", f"国内現物：S1の買いだけ（往復{cost * 100:.2f}%）", per, T, ctx["Mq"],
                                                       cost_note=note)
    keep("P05", rows={k: brief(v) for k, v in rows.items()})


# ---------------------------------------------------------------- P02 スワップが上がる
@njit
def vbo_until(O, H, L, C, P, trig, ok, A, side, cost, swd, last, xbar, rollbar):
    """値幅ブレイクを、その日の xbar 本目の前（xbar−1本目の終値）で決済する。rollbar をまたいだらスワップ1回。"""
    nd = trig.shape[0]
    pnl = np.zeros(nd)
    te = np.empty(nd, np.int64)
    tr = np.empty(nd)
    nt = 0
    lastb = min(last, xbar - 4)
    for d in range(nd):
        if not ok[d] or not (A[d] > 0) or np.isnan(trig[d]):
            continue
        b0 = d * P
        k = -1
        e = 0.0
        for j in range(b0, b0 + lastb):
            if j == b0 and side * (O[j] - trig[d]) > 0:
                break
            if (side == 1 and H[j] >= trig[d]) or (side == -1 and L[j] <= trig[d]):
                k = j
                e = max(O[j], trig[d]) if side == 1 else min(O[j], trig[d])
                break
        if k < 0:
            continue
        x = C[b0 + xbar - 1]
        nroll = 1 if (k - b0) < rollbar and xbar > rollbar else 0
        R = side * (x - e) / A[d] - cost * e / A[d] - nroll * swd * e / A[d]
        pnl[d] = R * 0.5
        te[nt] = d
        tr[nt] = R
        nt += 1
    return pnl, te[:nt], tr[:nt]


def vbo_until_all(D, xbar, dirmode="own5", k=0.8):
    """A01（dirmode="own5"）か、ETHはBTCの向き（"btc5"）の値幅ブレイクを、xbar 本目の前で決済する。"""
    bs = np.sign(D["BTC"]["dO"] - R1.sma_prev(D["BTC"], 5))
    per, Ts = {}, []
    for a in COINS:
        Pd = R2.periods(D[a], DAY, 0)
        fl = Pd["fl"]
        tot = np.zeros(Pd["n"])
        ts = []
        for side, w in ((1, 1.0), (-1, 0.5)):
            trig = (Pd["pO"] + side * k * Pd["rng"]).astype(np.float64)
            dirv = bs if (dirmode == "btc5" and a == "ETH") else np.sign(Pd["pO"] - Pd["sma5"])
            with np.errstate(invalid="ignore"):
                ok = ~np.isnan(Pd["atr"]) & ~np.isnan(trig) & (side * np.nan_to_num(dirv) > 0)
            swd = (PF.SPEC[a][1] if side == 1 else PF.SPEC[a][2]) / 36500
            pnl, te, tr = vbo_until(fl["O"], fl["H"], fl["L"], fl["C"], DAY, trig, ok, Pd["atr"], side, PF.SPEC[a][0], swd, DAY - 4, xbar, 84)
            tot += w * pnl
            ts.append(pd.DataFrame({"coin": a, "day": Pd["idx"][te], "side": side, "w": w, "R": tr}))
        per[a] = pd.Series(tot, index=Pd["idx"])
        T = pd.concat(ts, ignore_index=True)
        Ts.append(T[(T.day >= MM.START) & (T.day <= MM.END)])
    return per, pd.concat(Ts, ignore_index=True)


def sec_p02(ctx):
    pid = "P02"
    D = ctx["D"]
    print("\n==== P02 買いのスワップが上がったら（今：BTC年25.35%・ETH年35.49%）。A01・S1と、日本時間6時（スワップの時刻）前に決済する形")
    chk = R1.total(vbo_until_all(D, DAY)[0])
    print(f"    （確かめ：決済を翌朝9時にすると A01 と同じか → 年{metrics_row(chk)['年平均%']:+.2f}%・A01は{ctx['base']['年平均%']:+.2f}%）")
    rows = {}
    for lab, sw in (("今", None), ("年40%", 40.0), ("年60%", 60.0), ("年80%", 80.0)):
        note = "Vantage CFD（往復0.05%・売り0%・買いスワップ" + ("年25〜35%）" if sw is None else f"{lab}）")
        with spec(swl=sw):
            per, T = R2.combo(D)
            rows[f"A01・{lab}"] = show(pid, f"A01：買いのスワップ{lab}", per, T, cost_note=note)
            vb5, Tb5 = R4.vbo_var(D, "btc5", False)
            rows[f"S1・{lab}"] = scaled(ctx, pid, f"S1：買いのスワップ{lab}", vb5, Tb5, ctx["Mq"], cost_note=note)
            per, T = vbo_until_all(D, 84)
            rows[f"A01・6時前に決済・{lab}"] = show(pid, f"A01を日本時間6時前に決済：買いのスワップ{lab}", per, T, cost_note=note)
            per, T = vbo_until_all(D, 84, "btc5")
            rows[f"S1・6時前に決済・{lab}"] = scaled(ctx, pid, f"S1を日本時間6時前に決済：買いのスワップ{lab}", per, T, ctx["Mq"], cost_note=note)
    keep(pid, rows={k: brief(v) for k, v in rows.items()})


# ---------------------------------------------------------------- P07 最近1年
def sec_p07(ctx):
    pid = "P07"
    D = ctx["D"]
    L0, L1 = MM.END - pd.Timedelta(days=364), MM.END
    print(f"\n==== P07 最近1年（{L0:%Y-%m-%d}〜{L1:%Y-%m-%d}）の成績")
    c29 = R4.tot(R4.comp_c29(D))
    S = {"A01": (R4.tot(ctx["per0"]), ctx["T0"]),
         "S1": (R4.tot(ctx["vb5"], ctx["Mq"]), None),
         "S2": (R4.tot(ctx["vb5"], ctx["M3"]).add(0.25 * c29, fill_value=0), None),
         "E05（A01の静かな日だけ）": (R4.tot(ctx["per0"], ctx["Mq"]), None)}
    with spec(cost=0.0015, swl=0.0, sws=0.0):
        per, T = s1_long(D)
        S["国内現物：S1の買いだけ（往復0.15%）"] = (R4.tot(per, ctx["Mq"]), None)
    tab, out = {}, {}
    for k, (s, T) in S.items():
        z = s[(s.index >= L0) & (s.index <= L1)]
        full = metrics_row(s)
        r = metrics_row(z)
        n = len(T[(T.day >= L0) & (T.day <= L1)]) if T is not None else np.nan
        record(グループ=GROUP, 計画ID=pid, 名前=f"{pid} 最近1年：{k}", 銘柄="BTC,ETH", 時間足="15分足・日足",
               メモ=f"2018年からの年平均は{full['年平均%']:+.2f}%・安定度{full['安定度']:.2f}", **{**r, "年の取引回数": n if n == n else ""})
        mon = z.resample("ME").sum()
        tab[k] = mon
        out[k] = dict(tot=z.sum(), dd=r["最大の落ち込み%"], s=r["安定度"], posm=r["プラスの月%"], full=full["年平均%"], fulls=full["安定度"],
                      worst=mon.min(), n=n)
        print(f"    {k:28s} 1年で{z.sum():+6.2f}%（2018年からの年平均{full['年平均%']:+.2f}%）・落ち込み{r['最大の落ち込み%']:.1f}%・"
              f"安定度{r['安定度']:.2f}・プラスの月{r['プラスの月%']:.0f}%・一番悪い月{mon.min():+.2f}%")
    t = pd.DataFrame(tab).round(2)
    t.index = t.index.strftime("%Y-%m")
    print(t.to_string())
    keep(pid, out=out, table=t)


# ---------------------------------------------------------------- P09 量と落ち込み
def sec_p09(ctx, n=10000, block=20):
    pid = "P09"
    D = ctx["D"]
    print(f"\n==== P09 1回の量（1R＝資金の0.25%・0.5%・1%）と落ち込みの分布：20日ずつの塊を並べ替えた1年・5年を{n:,}回")
    c29 = R4.tot(R4.comp_c29(D))
    S = {"A01": R4.tot(ctx["per0"]), "S1": R4.tot(ctx["vb5"], ctx["Mq"]),
         "S2": R4.tot(ctx["vb5"], ctx["M3"]).add(0.25 * c29, fill_value=0)}
    g = np.random.default_rng(0)
    out = {}
    for k, s in S.items():
        x = s.to_numpy()
        res = {}
        for L in (365, 1826):
            dd, fin = [], []
            for _ in range(n // 2000):
                nbk = -(-L // block)
                st = g.integers(0, len(x) - block, (2000, nbk))
                p = x[st[..., None] + np.arange(block)].reshape(2000, -1)[:, :L]
                eq = p.cumsum(1)
                dd.append((np.maximum.accumulate(eq, 1) - eq).max(1))
                fin.append(eq[:, -1])
            res[L] = (np.concatenate(dd), np.concatenate(fin))
        dd1, fin1 = res[365]
        dd5, _ = res[1826]
        out[k] = {}
        for risk in (0.25, 0.5, 1.0):
            f = risk / 0.5
            o = dict(dd1_50=np.percentile(dd1, 50) * f, dd1_90=np.percentile(dd1, 90) * f, dd1_99=np.percentile(dd1, 99) * f,
                     dd5_50=np.percentile(dd5, 50) * f, dd5_99=np.percentile(dd5, 99) * f, y_med=np.median(fin1) * f,
                     y_05=np.percentile(fin1, 5) * f, p_neg=(fin1 < 0).mean() * 100)
            out[k][risk] = o
            print(f"    {k} 1回{risk}%：1年の年間損益 中央値{o['y_med']:+.1f}%（悪い方から5%で{o['y_05']:+.1f}%）・マイナスの年になる確率{o['p_neg']:.1f}%"
                  f"／1年の中の最大の落ち込み 中央値{o['dd1_50']:.1f}%・90%で{o['dd1_90']:.1f}%・99%で{o['dd1_99']:.1f}%"
                  f"／5年の中 中央値{o['dd5_50']:.1f}%・99%で{o['dd5_99']:.1f}%")
    # 最小ロット（0.01枚）で足りる資金
    fx = float(R4B.macro(ctx["cache"], "USDJPY").iloc[-1])
    lot = {}
    for a in COINS:
        atr, px = float(D[a]["atr"][-1]), float(D[a]["dC"][-1])
        lot[a] = dict(atr=atr, px=px, need={r: 0.01 * atr * fx / (r / 100) for r in (0.25, 0.5, 1.0)})
        print(f"    {a}：今のATR{atr:,.0f}ドル（価格{px:,.0f}ドル・ドル円{fx:.1f}円）→ 0.01枚がちょうど1Rになる資金は "
              + "・".join(f"1回{r}%なら{v / 1e4:,.0f}万円" for r, v in lot[a]["need"].items()))
    keep(pid, out=out, lot=lot, fx=fx)


# ---------------------------------------------------------------- おまけ：Oの「低」18件
def sec_cal(ctx):
    D, days, per0, T0 = ctx["D"], ctx["days"], ctx["per0"], ctx["T0"]
    wd, dom, mon, yr = (days.dayofweek.to_numpy(), days.day.to_numpy(), days.month.to_numpy(), days.year.to_numpy())
    T = T0.reset_index(drop=True)
    tdi = days.get_indexer(pd.DatetimeIndex(T.day))

    def vbo_on(mask):
        sel = np.asarray(mask)[tdi]
        return int(sel.sum()), float(T.R[sel].mean()) if sel.any() else np.nan

    def near(dates, before, after):
        m = np.zeros(len(days), bool)
        for c in pd.DatetimeIndex(dates):
            m |= (days >= c - pd.Timedelta(days=before)) & (days <= c + pd.Timedelta(days=after))
        return m

    out = {}
    # ---- O01 満月・新月
    print("\n==== O01 満月・新月（月齢は29.53日の周期で計算）")
    syn = 29.530588853
    age = ((days + pd.Timedelta(hours=12) - pd.Timestamp("2000-01-06 18:14")) / pd.Timedelta(days=1)).to_numpy() % syn
    full2, new2, waxing = np.abs(age - syn / 2) <= 2, np.minimum(age, syn - age) <= 2, age < syn / 2
    out["O01"] = {"満月の前後2日は休む": vf(ctx, "O01", "満月の前後2日は休む", ~full2),
                  "新月の前後2日は休む": vf(ctx, "O01", "新月の前後2日は休む", ~new2),
                  "新月から満月まで": hf(ctx, "O01", "新月から満月まで", waxing),
                  "満月から新月まで": hf(ctx, "O01", "満月から新月まで", ~waxing)}
    # ---- O02 13日の金曜日
    print("\n==== O02 13日の金曜日")
    f13 = (dom == 13) & (wd == 4)
    mv = {a: np.abs(np.log(D[a]["dC"] / D[a]["prevC"])) for a in COINS}
    w = (days >= MM.START) & (days <= MM.END)
    n13, r13 = vbo_on(f13 & w)
    print(f"    {int((f13 & w).sum())}日。値動きの大きさ（絶対値）の平均：BTC {np.nanmean(mv['BTC'][f13 & w]) * 100:.2f}%（ふだん"
          f"{np.nanmean(mv['BTC'][w]) * 100:.2f}%）・ETH {np.nanmean(mv['ETH'][f13 & w]) * 100:.2f}%（{np.nanmean(mv['ETH'][w]) * 100:.2f}%）。"
          f"その日の値幅ブレイク{n13}回・1回平均{r13:+.3f}R（全体{T.R.mean():+.3f}R）")
    out["O02"] = {"休む": vf(ctx, "O02", "13日の金曜日は休む", ~f13), "_info": (int((f13 & w).sum()), n13, r13)}
    # ---- O06 ビットコインの記念日
    print("\n==== O06 ビットコインの記念日（1月3日：最初のブロック、10月31日：論文の公開日）の前後3日")
    ann = near([f"{y}-01-03" for y in range(2018, 2027)] + [f"{y}-10-31" for y in range(2017, 2026)], 3, 3)
    out["O06"] = {"前後3日": hf(ctx, "O06", "記念日の前後3日", ann)}
    # ---- O07 価格のゾロ目
    print("\n==== O07 価格のゾロ目（例：77,777ドル・3,333ドル）を抜けたら順張り。始値から1ATR以内のゾロ目だけ、翌朝9時に決済")
    lv = np.array(sorted(float(dg * (10 ** n - 1) // 9) for n in range(2, 7) for dg in range(1, 10)))

    def zoro(P_, side):
        o = P_["pO"]
        i = np.searchsorted(lv, o, side="right")
        t = lv[np.clip(i, 0, len(lv) - 1)] if side == 1 else lv[np.clip(i - 1, 0, len(lv) - 1)]
        with np.errstate(invalid="ignore"):
            good = (side * (t - o) > 0) & (np.abs(t - o) <= P_["atr"])
        return np.where(good, t, np.nan)
    out["O07"] = {}
    for ma in (False, True):
        per, Ts = {}, []
        for a in COINS:
            s, T_ = R2.run(R2.periods(D[a], DAY, 0), a, ma=ma, trig_fn=zoro)
            per[a] = s
            Ts.append(T_)
        lab = "5日平均の向きだけ" if ma else "向きの条件なし"
        out["O07"][lab] = show("O07", f"ゾロ目の価格を抜けたら順張り・翌朝決済（{lab}）", per, pd.concat(Ts, ignore_index=True))
    # ---- O08 旧正月
    print("\n==== O08 旧正月（中国の春節）の前の2週間・後の2週間")
    cny = pd.to_datetime(["2018-02-16", "2019-02-05", "2020-01-25", "2021-02-12", "2022-02-01", "2023-01-22", "2024-02-10", "2025-01-29",
                          "2026-02-17"])
    out["O08"] = {"前の2週間": hf(ctx, "O08", "旧正月の前の14日", near(cny, 14, -1)),
                  "後の2週間": hf(ctx, "O08", "旧正月から14日", near(cny, 0, 13))}
    # ---- O09 米国の確定申告の期限
    print("\n==== O09 米国の確定申告の期限（年ごとの実際の期限）の前10日に売り・後10日に買い（9時〜翌9時）")
    tax = pd.to_datetime(["2018-04-17", "2019-04-15", "2020-07-15", "2021-05-17", "2022-04-18", "2023-04-18", "2024-04-15", "2025-04-15",
                          "2026-04-15"])
    bef, aft = near(tax, 10, -1), near(tax, -1, 10)
    r = dt(ctx, "O09", "確定申告の期限の前10日に売り・後10日に買い", {a: np.where(bef, -1.0, np.where(aft, 1.0, 0.0)) for a in COINS})
    yrs = r["_x"].groupby(r["_x"].index.year).sum()
    print("    年ごと：" + "・".join(f"{y}年{v:+.2f}%" for y, v in yrs.items()))
    out["O09"] = {"前に売り・後に買い": r, "_years": yrs}
    # ---- O10 ハロウィン効果
    print("\n==== O10 ハロウィン効果：11〜4月だけ買い持ち／5〜10月だけ買い持ち")
    winter = np.isin(mon, [11, 12, 1, 2, 3, 4])
    allr = hold(ctx, "O10", "参考：ずっと買い持ち", np.ones(len(days)))
    out["O10"] = {"11〜4月": hf(ctx, "O10", "11〜4月", winter), "5〜10月": hf(ctx, "O10", "5〜10月", ~winter), "参考：ずっと": allr}
    # ---- O11・O13 素数の日・フィボナッチ数の日
    print("\n==== O11 日付が素数の日だけ値幅ブレイク／O13 日付がフィボナッチ数の日だけ（どちらも偶然の物差し）")
    out["O11"] = {"素数の日だけ": vf(ctx, "O11", "日付が素数（2・3・5・7・11・13・17・19・23・29・31日）の日だけ",
                                np.isin(dom, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]))}
    out["O13"] = {"フィボナッチ数の日だけ": vf(ctx, "O13", "日付がフィボナッチ数（1・2・3・5・8・13・21日）の日だけ", np.isin(dom, [1, 2, 3, 5, 8, 13, 21]))}
    # ---- O14 終値の下1桁
    print("\n==== O14 前日の終値（ドル）の下1桁が奇数なら買い（偶然の物差し）")
    odd = {a: (np.nan_to_num(np.floor(D[a]["prevC"])) % 2 == 1) for a in COINS}
    ref = dt(ctx, "O14", "参考：毎日買い（9時〜翌9時）", {a: np.ones(len(days)) for a in COINS})
    out["O14"] = {"奇数なら買い": dt(ctx, "O14", "前日の終値の下1桁が奇数なら買い（9時〜翌9時）", {a: odd[a].astype(float) for a in COINS}),
                  "奇数なら買い・偶数なら売り": dt(ctx, "O14", "前日の終値の下1桁が奇数なら買い・偶数なら売り（9時〜翌9時）",
                                         {a: np.where(odd[a], 1.0, -1.0) for a in COINS}),
                  "参考：毎日買い": ref}
    # ---- O15 半減期の最適日探し
    print("\n==== O15 半減期の最適日探し：2016・2020年の半減期で「何日前に買って何日持つと一番良いか」を探し、2024年に当てはめる")
    yb = pd.read_csv(os.path.join(ctx["cache"], "BTC-USD_yahoo.csv"), index_col=0, parse_dates=True)["close"].astype(float)
    yb.index = pd.DatetimeIndex(yb.index).normalize()
    halv = pd.to_datetime(["2016-07-09", "2020-05-11", "2024-04-20"])
    Ns, Ms = np.arange(0, 361, 15), np.arange(30, 541, 30)

    def hret(h, nb, mh):
        a0 = yb.index.searchsorted(h - pd.Timedelta(days=int(nb)))
        a1 = a0 + int(mh)
        return np.log(yb.iloc[a1] / yb.iloc[a0]) * 100 if a1 < len(yb) else np.nan
    G = np.array([[[hret(h, nb, mh) for mh in Ms] for nb in Ns] for h in halv])
    train = np.nanmean(G[:2], axis=0)
    bi = np.unravel_index(np.nanargmax(train), train.shape)
    test = G[2][bi]
    rank = (G[2] < test).sum() / np.isfinite(G[2]).sum() * 100
    lr = np.log(yb[(yb.index >= "2018-01-01")]).to_numpy()
    mh = int(Ms[bi[1]])
    rnd = (lr[mh:] - lr[:-mh]) * 100
    print(f"    2016・2020年で一番良かったのは「半減期の{Ns[bi[0]]}日前に買い、{mh}日持つ」：2回の平均{train[bi]:+.0f}%（対数）。"
          f"2024年に当てはめると{test:+.0f}%で、2024年の{len(Ns) * len(Ms)}通りの中では下から{rank:.0f}%。"
          f"2018年からの、いつ買っても{mh}日持った時の中央値は{np.median(rnd):+.0f}%")
    out["O15"] = dict(nb=int(Ns[bi[0]]), mh=mh, train=float(train[bi]), test=float(test), rank=float(rank), rnd_med=float(np.median(rnd)),
                      g2016=float(G[0][bi]), g2020=float(G[1][bi]))
    # ---- O16 日本の祝日
    print("\n==== O16 日本の祝日")
    import holidays
    jp = holidays.Japan(years=range(2017, 2027))
    hol = np.array([d in jp for d in days.date])
    nh, rh = vbo_on(hol & w)
    print(f"    祝日{int((hol & w).sum())}日。その日の値幅ブレイク{nh}回・1回平均{rh:+.3f}R（全体{T.R.mean():+.3f}R）")
    out["O16"] = {"祝日は休む": vf(ctx, "O16", "日本の祝日は休む", ~hol), "祝日だけ": vf(ctx, "O16", "日本の祝日だけ", hol), "_info": (nh, rh)}
    # ---- O17 米大統領選の年
    print("\n==== O17 米大統領選の年（2020・2024年）だけ買い持ち")
    ys = allr["_x"].groupby(allr["_x"].index.year).sum()
    print("    参考：ずっと買い持ちの年ごと：" + "・".join(f"{y}年{v:+.1f}%" for y, v in ys.items()))
    out["O17"] = {"選挙の年だけ": hf(ctx, "O17", "米大統領選の年（2020・2024年）", np.isin(yr, [2020, 2024])), "_years": ys}
    # ---- O18 オリンピック・W杯
    print("\n==== O18 オリンピック・サッカーW杯の開催中")
    ev = [("2018-02-09", "2018-02-25"), ("2018-06-14", "2018-07-15"), ("2021-07-23", "2021-08-08"), ("2022-02-04", "2022-02-20"),
          ("2022-11-20", "2022-12-18"), ("2024-07-26", "2024-08-11"), ("2026-02-06", "2026-02-22"), ("2026-06-11", "2026-07-19")]
    evm = np.zeros(len(days), bool)
    for s0, s1 in ev:
        evm |= (days >= s0) & (days <= s1)
    out["O18"] = {"開催中だけ買い持ち": hf(ctx, "O18", "オリンピック・W杯の開催中", evm),
                  "開催中は休む": vf(ctx, "O18", "オリンピック・W杯の開催中は休む", ~evm)}
    # ---- O25 桁が増える時
    print("\n==== O25 桁が増える時：終値が節目（BTC 1万・10万ドル、ETH 100・1,000ドル）を、30日ぶりに上に抜けた日から")
    ev25 = []
    for a, levels in (("BTC", (1e4, 1e5)), ("ETH", (100.0, 1000.0))):
        c = D[a]["dC"]
        for L in levels:
            for d in range(30, len(c) - 30):
                if days[d] < MM.START or days[d] > MM.END:
                    continue
                if c[d] > L and np.nanmax(c[d - 30:d]) < L:
                    r1, r10, r30 = (np.log(c[d + k] / c[d]) * 100 if d + k < len(c) else np.nan for k in (1, 10, 30))
                    ev25.append((a, int(L), days[d].strftime("%Y-%m-%d"), r1, r10, r30))
                    print(f"    {a} {int(L):,}ドル {days[d]:%Y-%m-%d}：翌日{r1:+.1f}%・10日後{r10:+.1f}%・30日後{r30:+.1f}%")
    out["O25"] = ev25
    # ---- O26 月曜は休む
    print("\n==== O26 月曜（日本時間の月曜9時〜火曜9時）は値幅ブレイクを休む")
    out["O26"] = {"月曜は休む": vf(ctx, "O26", "月曜は休む", wd != 0)}
    # ---- O27 午前と午後
    print("\n==== O27 日本の午前（0〜12時）だけ・午後（12〜24時）だけ買いで持つ")

    def seg(h0, h1):
        def fn(x, a):
            d0 = np.arange(len(x["dO"])) * DAY
            i1 = d0 + ((h0 - 9) % 24) * 4
            s, T_ = R2B.window(x, a, i1, i1 + ((h1 - h0) % 24) * 4, np.ones(len(d0)), np.ones(len(d0)))
            return s, T_.assign(hold=0)
        return fn
    out["O27"] = {}
    for h0, h1, lab in ((0, 12, "午前（0〜12時）"), (12, 24, "午後（12〜24時）")):
        out["O27"][lab] = show("O27", f"日本の{lab}だけ毎日買い", *R3.both(D, seg(h0, h1 % 24)), tf="15分足")
        with spec(cost=0.0, swl=0.0, sws=0.0):
            g_ = metrics_row(R1.total(R3.both(D, seg(h0, h1 % 24))[0]))["年平均%"]
        out["O27"][lab]["_gross"] = g_
        print(f"      コストとスワップなしなら年{g_:+.2f}%")
    # ---- O28 サンタクロース・ラリー
    print("\n==== O28 サンタクロース・ラリー：12月27日〜1月2日だけ買い持ち")
    santa = ((mon == 12) & (dom >= 27)) | ((mon == 1) & (dom <= 2))
    r = hf(ctx, "O28", "12月27日〜1月2日", santa)
    ys = r["_x"].groupby(np.where(r["_x"].index.month == 1, r["_x"].index.year - 1, r["_x"].index.year)).sum()
    ys = ys[ys != 0]
    print("    年ごと（12月の年）：" + "・".join(f"{y}年{v:+.2f}%" for y, v in ys.items()))
    out["O28"] = {"12月27日〜1月2日": r, "_years": ys}
    keep("cal", out=out)


# ---------------------------------------------------------------- 本体
SECTIONS = {"o12": sec_o12, "o19": sec_o19, "o20": sec_o20, "o21": sec_o21, "p02": sec_p02, "p07": sec_p07, "p09": sec_p09, "cal": sec_cal}


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
    M3 = {a: R4.mult(D[a], "3段階") for a in COINS}
    v0d = (per0["BTC"] + per0["ETH"]).reindex(days).fillna(0).to_numpy()
    longd = sum(R3.pos_daily(D[a], a, np.ones(len(days)))[0].reindex(days).fillna(0).to_numpy() for a in COINS)
    mid = ((days.year - 2018) * 12 + days.month - 1).to_numpy()
    ctx = dict(cache=cache, D=D, days=days, per0=per0, T0=T0, v0=v0, base=base, vb5=vb5, Tb5=Tb5, Mq=Mq, M3=M3, v0d=v0d, longd=longd, mid=mid)
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
