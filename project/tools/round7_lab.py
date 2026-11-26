#!/usr/bin/env python3
"""計画書（project/research/plan.md）第7回：よりよい形を探す（R01〜R18 と、出来高の G01〜G11）。

  R01〜R06 S4案（S1＋直近2日が全敗なら翌日は2倍）の量の決め方を詰める
           R01 連敗の日数で段階的に（1日1.25倍・2日1.5倍・3日2倍）／R02 勝った翌日は0.75倍／R03 2銘柄の負けを合わせて数える／
           R04 直近2日の損益が−1R以下の時だけ2倍／R05 量を3段階（S2の形）と組み合わせる／R06 2銘柄とも2倍の日は1.5倍ずつに抑える
  R07〜R10 取引を選ぶ機械学習（メタ・ラベリング）：値幅ブレイクの合図ごとに「勝ちやすさ」を予想して量を変える（毎月学び直し、2019年から評価）
  R11〜R13 ほかの銘柄（SOL・XRP・BNB・LTC・ADA・DOGE）で A01・S1・S4案と量の決め方を確かめる（売買のためでなく確かめのため）
  R14〜R18 新しい組み合わせ：R14 前日の引けの位置／R15 ETHの静かさをBTCで判定／R16 2銘柄とも静かな日だけ／
           R17 S4案＋50日平均の向き（C39）／R18 S1・S4案を12時・12時半区切りで
  G01〜G11 出来高（Binanceの15分足の出来高・売買代金・取引回数・成行の買い）。G11 は出来高で「静かな日」を判定する新しい項目
共通: Vantage CFD（往復0.05%、買いのスワップ年25〜35%、売り0%）、1R＝ATR（日足）1本分＝資金の0.5%（単利）。
ほかの銘柄のスワップは ETH と同じ年35.49%と仮定する。前半（2018〜2022年）で選び、後半（2023年〜）で確かめる。
結果は results.csv（グループ「計画・第7回」、計画IDつき）と plan.csv。
必要なデータ: キャッシュに BTC_15m.csv・ETH_15m.csv（今まで通り）、BTC_15m_vol.csv・ETH_15m_vol.csv（出来高つき）、
             SOL・XRP・BNB・LTC・ADA・DOGE の _15m.csv（marketdata.py klines で取得）
使い方: python3 project/tools/round7_lab.py <キャッシュ> [size,meta,alt,combo,vol のどれか（カンマ区切り）]
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
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
import round5b_lab as R5B  # noqa: E402
import round6_lab as R6  # noqa: E402
from registry import drop, metrics_row, record  # noqa: E402

COINS = ("BTC", "ETH")
ALTS = ("SOL", "XRP", "BNB", "LTC", "ADA", "DOGE")
GROUP = "計画・第7回"
DAY = 96
R3.GROUP = R5B.GROUP = R6.GROUP = GROUP
for _a in ALTS:
    PF.SPEC.setdefault(_a, (0.0005, 35.49, 0.0))
warnings.filterwarnings("ignore")
SUM = {}
H2 = pd.Timestamp("2023-01-01")


def keep(pid, **kw):
    SUM.setdefault(pid, {}).update(kw)


def brief(r):
    return dict(a=r["年平均%"], dd=r["最大の落ち込み%"], s=r["安定度"], s1=r.get("_h1s"), s2=r.get("_h2s"), neg=r["マイナスの年"], n=r.get("_n"))


# ---------------------------------------------------------------- 共通
def prep1(path):
    """1銘柄の15分足を日ごとの行列にする（R1.prep と同じ形）。"""
    x = EX.day_matrix_n(path, DAY)
    M = x["M"]
    x.update(O=np.ascontiguousarray(M["open"].ravel()), H=np.ascontiguousarray(M["high"].ravel()),
             L=np.ascontiguousarray(M["low"].ravel()), C=np.ascontiguousarray(M["close"].ravel()),
             dH=M["high"].max(1), dL=M["low"].min(1), dC=M["close"][:, -1], idx=pd.DatetimeIndex(x["days"]))
    x["prevC"] = np.r_[np.nan, x["dC"][:-1]]
    return x


def show_any(pid, name, per, T=None, memo="", tf="15分足", norm=False):
    """銘柄がいくつでも記録できる show。合計の数字と、銘柄ごとの年平均をメモに残す。
    norm=True なら、その日にデータがある銘柄の数で割って「2銘柄分の量」にそろえる（上場の時期が違うほかの銘柄用）。"""
    if norm:
        cnt = None
        for s in per.values():
            one = pd.Series(1.0, index=s.index)
            cnt = one if cnt is None else cnt.add(one, fill_value=0)
        per_w = {a: s * (2.0 / cnt.reindex(s.index)) for a, s in per.items()}
    else:
        per_w = per
    tot = None
    for a, s in per_w.items():
        tot = s if tot is None else tot.add(s, fill_value=0)
    x = MM.clip(tot)
    if T is not None:
        T = T[(T.day >= MM.START) & (T.day <= MM.END)]
    r = R1.stats(x, T)
    if T is not None and len(x):
        yrs = (x.index[-1] - x.index[0]).days / 365.25
        r["_n"] = len(T) / yrs if yrs > 0 else np.nan
        r["年の取引回数"] = round(r["_n"], 1)
    coins = {a: metrics_row(MM.clip(s))["年平均%"] for a, s in per.items() if len(MM.clip(s))}
    r["_x"] = x
    r["_hold"] = np.nan
    r["_coins"] = tuple(metrics_row(MM.clip(per[a])) for a in per)
    R1.line(name, r, " ｜" + " ".join(f"{a}{v:+.1f}%" for a, v in coins.items()))
    record(グループ=GROUP, 計画ID=pid, 名前=f"{pid} {name}", 銘柄=",".join(per), 時間足=tf,
           メモ=(memo + "。" if memo else "") + "・".join(f"{a}だけ年{v:+.1f}%" for a, v in coins.items()),
           **{k: v for k, v in r.items() if not k.startswith("_")})
    return r


def day_stats(T, coin, idx):
    """銘柄ごと・期間ごとの取引の数・勝ちの数・Rの合計。"""
    t = T[T.coin == coin]
    di = pd.DatetimeIndex(idx).get_indexer(pd.DatetimeIndex(t.day))
    ok = di >= 0
    R = t.R.to_numpy()[ok]
    di = di[ok]
    n = np.bincount(di, minlength=len(idx)).astype(float)
    w = np.bincount(di, weights=(R > 0).astype(float), minlength=len(idx))
    s = np.bincount(di, weights=R, minlength=len(idx))
    return n, w, s


def last_window(v, look):
    """前の look 期間（今日は入れない）の合計。"""
    c = np.r_[0.0, np.cumsum(v)]
    d = np.arange(len(v))
    return c[d] - c[np.maximum(0, d - look)]


def lost_last(n, w, look):
    nn, ww = last_window(n, look), last_window(w, look)
    return (nn >= 1) & (ww == 0)


def won_last(n, w, look):
    nn, ww = last_window(n, look), last_window(w, look)
    return (nn >= 1) & (ww == nn)


def s1_parts(Dc, coins, btc, off=0, own=False):
    """S1 の材料（銘柄ごと）：毎期間の損益 s（静かな日の判定の前）、取引一覧 T、静かな日の倍率 mq、期間の並び idx。
    向きは BTC の5本平均（BTC 自身も同じ。own=True なら銘柄ごとに自分の5本平均）。静かな日は、その期間が始まった日（UTC）の判定を使う。"""
    Pb = R2.periods(btc, DAY, off)
    bdir = pd.Series(np.sign(Pb["pO"] - Pb["sma5"]), index=Pb["idx"])
    out = {}
    for a in coins:
        x = Dc[a]
        Pd = R2.periods(x, DAY, off)
        bd = np.sign(Pd["pO"] - Pd["sma5"]) if own else bdir.reindex(Pd["idx"]).to_numpy()
        s, T = R2.run(Pd, a, ma=False, extra_ok=lambda P_, side, bd=bd: side * np.nan_to_num(bd) > 0)
        mq = pd.Series(R4.mult(x, "静かな日だけ"), index=x["idx"]).reindex(Pd["idx"].normalize()).fillna(1.0).to_numpy()
        m3 = pd.Series(R4.mult(x, "3段階"), index=x["idx"]).reindex(Pd["idx"].normalize()).fillna(1.0).to_numpy()
        out[a] = dict(s=s, T=T, mq=mq, m3=m3, idx=Pd["idx"])
    return out


def taken(part, base="mq"):
    """実際に取った取引だけの一覧（base の倍率が0の日は取っていない。S1なら静かな日だけ）。"""
    T = part["T"].reset_index(drop=True)
    pos = pd.DatetimeIndex(part["idx"]).get_indexer(pd.DatetimeIndex(T.day))
    return T[(pos >= 0) & (part[base][np.clip(pos, 0, None)] > 0)]


def s4_mult(part, coin, look=2, mlt=2.0, base="mq"):
    n, w, _ = day_stats(taken(part, base), coin, part["idx"])
    return 1 + (mlt - 1) * lost_last(n, w, look)


def build(parts, mfun):
    """銘柄ごとに s × 倍率（mfun(a, part)）。取引一覧は倍率が0の日を除く。"""
    per, Ts = {}, []
    for a, p in parts.items():
        m = np.asarray(mfun(a, p), float)
        per[a] = pd.Series(p["s"].to_numpy() * m, index=p["s"].index)
        T = p["T"].reset_index(drop=True)
        pos = pd.DatetimeIndex(p["idx"]).get_indexer(pd.DatetimeIndex(T.day))
        Ts.append(T[(pos >= 0) & (m[np.clip(pos, 0, None)] > 0)])
    return per, pd.concat(Ts, ignore_index=True)


# ---------------------------------------------------------------- R01〜R06 S4案の量
def size_rules(parts, coins):
    """S1（静かな日だけ）を土台に、量の決め方を変える関数の一覧。"""
    def st(a, p):
        return day_stats(taken(p), a, p["idx"])
    rules = {}
    rules["S1"] = lambda a, p: p["mq"]
    rules["S4案"] = lambda a, p: p["mq"] * s4_mult(p, a)

    def r01(a, p):
        n, w, _ = st(a, p)
        m = np.ones(len(n))
        m = np.where(lost_last(n, w, 1), 1.25, m)
        m = np.where(lost_last(n, w, 2), 1.5, m)
        m = np.where(lost_last(n, w, 3), 2.0, m)
        return p["mq"] * m
    rules["R01"] = r01

    def r02(a, p):
        n, w, _ = st(a, p)
        return p["mq"] * np.where(lost_last(n, w, 2), 2.0, np.where(won_last(n, w, 1), 0.75, 1.0))
    rules["R02"] = r02

    def r03(a, p):
        ns, ws = 0, 0
        for b in coins:
            n, w, _ = st(b, parts[b])
            ns, ws = ns + n, ws + w
        return p["mq"] * np.where(lost_last(ns, ws, 2), 2.0, 1.0)
    rules["R03"] = r03

    def r04(a, p):
        n, w, s = st(a, p)
        return p["mq"] * np.where((last_window(n, 2) >= 1) & (last_window(s, 2) <= -1.0), 2.0, 1.0)
    rules["R04"] = r04
    rules["R05"] = lambda a, p: p["m3"] * s4_mult(p, a, base="m3")

    def r06(a, p):
        both = np.ones(len(p["mq"]), bool)
        for b in coins:
            pb = parts[b]
            both &= (pb["mq"] > 0) & (s4_mult(pb, b) > 1)
        return p["mq"] * np.where(both, 1.5, s4_mult(p, a))
    rules["R06"] = r06
    return rules


NAMES = {"S1": "S1（基準）", "S4案": "S4案（直近2日が全敗なら翌日2倍）", "R01": "連敗の日数で段階的に（1日1.25倍・2日1.5倍・3日2倍）",
         "R02": "S4案＋勝った翌日は0.75倍", "R03": "2銘柄の負けを合わせて数える", "R04": "直近2日の損益が−1R以下の時だけ2倍",
         "R05": "S4案＋量を3段階（S2の形）", "R06": "S4案で2銘柄とも2倍の日は1.5倍ずつに抑える"}


def sec_size(ctx):
    print("\n==== R01〜R06 S4案の量の決め方（BTC・ETH）")
    parts = ctx["parts"]
    rules = size_rules(parts, COINS)
    out = {}
    for k, fn in rules.items():
        per, T = build(parts, fn)
        out[k] = show_any(k if k.startswith("R") else "R01", NAMES[k] if k.startswith("R") else "参考：" + NAMES[k], per, T)
    c29 = R4.tot(R4.comp_c29(ctx["D"]))
    per, T = build(parts, rules["R05"])
    tot = R1.total(per).add(0.25 * c29, fill_value=0)
    r = R1.stats(tot)
    R1.line("R05 S4案＋3段階＋C29×0.25（S2＋S4案）", r)
    record(グループ=GROUP, 計画ID="R05", 名前="R05 S4案＋量を3段階＋C29×0.25（S2にS4案を足した形）", 銘柄="BTC,ETH", 時間足="15分足・日足",
           **{k: v for k, v in r.items() if not k.startswith("_")})
    out["R05+C29"] = r
    keep("size", rows={k: brief(v) for k, v in out.items()})
    ctx["size_rules"] = rules

    # ---- 確かめ：R03（2銘柄の負けを合わせて数える）と R02（勝った翌日は控えめ）が、設定を変えても良いか
    print("    ---- 確かめ：R03 を直近1〜3日・1.5倍／2倍で、R02 を勝った翌日0.5倍／0.75倍で")

    def pooled(n_look, mlt, win_m=1.0):
        ns, ws = 0, 0
        for b in COINS:
            n, w, _ = day_stats(taken(parts[b]), b, parts[b]["idx"])
            ns, ws = ns + n, ws + w
        up = lost_last(ns, ws, n_look)
        dn = won_last(ns, ws, 1)
        return lambda a, p: p["mq"] * np.where(up, mlt, np.where(dn, win_m, 1.0))
    s1r, s4r = out["S1"], out["S4案"]
    grid = {}
    for n_look in (1, 2, 3):
        for mlt in (1.5, 2.0):
            if (n_look, mlt) == (2, 2.0):
                grid[(n_look, mlt)] = out["R03"]
                continue
            per, T = build(parts, pooled(n_look, mlt))
            grid[(n_look, mlt)] = show_any("R03", f"2銘柄の負けを合わせて数える：直近{n_look}日が全敗なら翌日は{mlt}倍", per, T)
    b1 = sum(v["_h1s"] > s1r["_h1s"] and v["_h2s"] > s1r["_h2s"] for v in grid.values())
    b4 = sum(v["_h1s"] > s4r["_h1s"] and v["_h2s"] > s4r["_h2s"] for v in grid.values())
    print(f"      R03の6通り中、前半・後半ともS1より上：{b1}通り、S4案より上：{b4}通り")
    g2 = {}
    for wm in (0.5, 0.75):
        if wm != 0.75:
            def fn(a, p, wm=wm):
                n, w, _ = day_stats(taken(p), a, p["idx"])
                return p["mq"] * np.where(lost_last(n, w, 2), 2.0, np.where(won_last(n, w, 1), wm, 1.0))
            per, T = build(parts, fn)
            g2[wm] = show_any("R02", f"S4案＋勝った翌日は{wm}倍", per, T)
        else:
            g2[wm] = out["R02"]

    def win_only(a, p):
        n, w, _ = day_stats(taken(p), a, p["idx"])
        return p["mq"] * np.where(won_last(n, w, 1), 0.75, 1.0)
    per, T = build(parts, win_only)
    g2["S1＋勝った翌日0.75倍"] = show_any("R02", "参考：S1＋勝った翌日は0.75倍（負けの後に増やさない）", per, T)
    per, T = build(parts, pooled(2, 2.0, 0.75))
    both = show_any("R03", "2銘柄の負けを合わせて数える＋勝った翌日は0.75倍（R02とR03を合わせた形）", per, T)
    # 年ごとの損益
    yt = pd.DataFrame({k: v["_x"].groupby(v["_x"].index.year).sum() for k, v in
                       (("S1", s1r), ("S4案", s4r), ("R02", out["R02"]), ("R03", out["R03"]), ("R02+R03", both))}).round(1)
    print("      年ごとの損益（%）：\n" + "\n".join("        " + ln for ln in yt.T.to_string().splitlines()))
    # 量を増やす日を、同じ割合でランダムに選んだ場合と比べる（1,000回）
    days = ctx["days"]
    w = ((days >= MM.START) & (days <= MM.END))
    mid = ((days.year - 2018) * 12 + days.month - 1).to_numpy()[w]
    sq = {a: (parts[a]["s"].reindex(days).fillna(0).to_numpy() * parts[a]["mq"])[w] for a in COINS}

    def st(z):
        m = np.bincount(mid, weights=z)
        return m.mean() / m.std(ddof=1) * np.sqrt(12)
    g = np.random.default_rng(0)
    cal = {}
    up_s4 = {a: (s4_mult(parts[a], a) > 1)[w] for a in COINS}
    ns, ws = 0, 0
    for b in COINS:
        n, w_, _ = day_stats(taken(parts[b]), b, parts[b]["idx"])
        ns, ws = ns + n, ws + w_
    up_r3 = lost_last(ns, ws, 2)[w]
    for lab, ups, common in (("S4案（銘柄ごと）", up_s4, False), ("R03（2銘柄を合わせて）", {a: up_r3 for a in COINS}, True)):
        act = st(sum(sq[a] * (1 + ups[a]) for a in COINS))
        fr = {a: ups[a].mean() for a in COINS}
        rnd = []
        for _ in range(1000):
            if common:
                u = g.random(len(mid)) < fr["BTC"]
                rnd.append(st(sum(sq[a] * (1 + u) for a in COINS)))
            else:
                rnd.append(st(sum(sq[a] * (1 + (g.random(len(mid)) < fr[a])) for a in COINS)))
        rnd = np.array(rnd)
        cal[lab] = dict(frac=np.mean(list(fr.values())) * 100, s=act, rank=(rnd < act).mean() * 100, med=float(np.median(rnd)),
                        p95=float(np.quantile(rnd, 0.95)))
        print(f"      {lab}：2倍にする日{cal[lab]['frac']:.0f}%。同じ割合の日をランダムに2倍にした1,000回の中で、安定度{act:.2f}は"
              f"下から{cal[lab]['rank']:.1f}%（中央値{cal[lab]['med']:.2f}・上位5%の境目{cal[lab]['p95']:.2f}）")
    keep("size", grid={f"{k[0]}日・{k[1]}倍": brief(v) for k, v in grid.items()}, b1=b1, b4=b4,
         g2={str(k): brief(v) for k, v in g2.items()}, both=brief(both), years=yt, cal=cal)


# ---------------------------------------------------------------- R11〜R13 ほかの銘柄
def sec_alt(ctx):
    cache, D = ctx["cache"], ctx["D"]
    print("\n==== R11〜R13 ほかの銘柄で確かめる（SOL・XRP・BNB・LTC・ADA・DOGE、上場してから。売買のためでなく確かめのため）")
    Da = {}
    for a in ALTS:
        p = os.path.join(cache, f"{a}_15m.csv")
        if os.path.exists(p):
            Da[a] = prep1(p)
    if not Da:
        print("    データがない")
        return
    for a, x in Da.items():
        print(f"    {a}：{x['idx'][0]:%Y-%m-%d}〜{x['idx'][-1]:%Y-%m-%d}")
    alts = tuple(Da)
    tag = f"ほかの{len(alts)}銘柄で"
    note = "その日にデータがある銘柄で割って、2銘柄分の量にそろえる"
    res, single = {}, {}
    # R11 A01（自分の5日平均の向き）
    per, Ts = {}, []
    for a in alts:
        s, T = R2.run(R2.periods(Da[a], DAY, 0), a)
        per[a] = s
        Ts.append(T)
    res["A01"] = show_any("R11", f"{tag}A01（{note}）", per, pd.concat(Ts, ignore_index=True), norm=True)
    for a in alts:
        single[("A01", a)] = show_any("R11", f"A01：{a}だけ", {a: per[a]}, Ts[alts.index(a)])
    # R12 S1（向きはBTCの5日平均、静かな日は銘柄ごと）／R13 S4案と量の決め方
    parts = s1_parts(Da, alts, D["BTC"])
    rules = size_rules(parts, alts)
    for k in ("S1", "S4案", "R01", "R02", "R04", "R05"):
        per, T = build(parts, rules[k])
        pid = "R12" if k == "S1" else "R13"
        lab = {"S1": "S1", "S4案": "S4案"}.get(k, k + " " + NAMES[k])
        res[k] = show_any(pid, f"{tag}{lab}（{note}）", per, T, norm=True)
        if k in ("S1", "S4案"):
            for a in alts:
                single[(k, a)] = show_any(pid, f"{lab}：{a}だけ", {a: per[a]}, T[T.coin == a])
    # 向きを自分の5日平均にした形（BTCを使わない）：静かな日だけ（E05の形）と、それに負けの後の2倍を足した形
    po = s1_parts(Da, alts, D["BTC"], own=True)
    ro = size_rules(po, alts)
    per, T = build(po, ro["S1"])
    res["E05"] = show_any("R12", f"参考：{tag}自分の5日平均の向き＋静かな日だけ（E05の形。{note}）", per, T, norm=True)
    per, T = build(po, ro["S4案"])
    res["E05+S4"] = show_any("R13", f"{tag}自分の5日平均の向き＋静かな日だけ＋直近2日が全敗なら2倍（{note}）", per, T, norm=True)
    for a in alts:
        single[("E05+S4", a)] = show_any("R13", f"自分の向き＋静かな日だけ＋直近2日が全敗なら2倍：{a}だけ", {a: per[a]}, T[T.coin == a])
    # 静かな日の判定なし（A01）に、負けの後の2倍だけを足した形
    pa = s1_parts(Da, alts, D["BTC"], own=True)
    for a in alts:
        pa[a]["one"] = np.ones(len(pa[a]["mq"]))
    per, T = build(pa, lambda a, p: s4_mult(p, a, base="one"))
    res["A01+S4"] = show_any("R13", f"{tag}A01＋直近2日が全敗なら2倍（{note}）", per, T, norm=True)
    keep("alt", rows={k: brief(v) for k, v in res.items()}, span={a: f"{x['idx'][0]:%Y-%m}" for a, x in Da.items()},
         single={f"{k}・{a}": brief(v) for (k, a), v in single.items()})


# ---------------------------------------------------------------- R07〜R10 取引を選ぶ機械学習
def trade_features(D, parts):
    """S1 の向き（BTCの5日平均）で出た値幅ブレイクの取引ごとに、その朝にわかる材料を作る（静かな日の判定の前の、全部の取引）。"""
    rows = []
    bx = D["BTC"]
    bdist_s = pd.Series((bx["dO"] - R1.sma_prev(bx, 5)) / bx["atr"], index=bx["idx"])
    for a in COINS:
        x, p = D[a], parts[a]
        idx = x["idx"]
        bdist = bdist_s.reindex(idx).to_numpy()
        lr = pd.Series(np.log(x["dC"]), index=idx).diff()
        rv10, rv5, rv60 = (lr.rolling(n).std().shift(1) for n in (10, 5, 60))
        qrank = rv10.rolling(365, min_periods=120).apply(lambda v: (v[:-1] < v[-1]).mean(), raw=True).to_numpy()
        atrp = pd.Series(x["atr"] / x["dO"], index=idx)
        arank = atrp.rolling(365, min_periods=120).apply(lambda v: (v[:-1] < v[-1]).mean(), raw=True).to_numpy()
        vr = (rv5 / rv60).to_numpy()
        h, l, c = R6.prev(x["dH"]), R6.prev(x["dL"]), R6.prev(x["dC"])
        with np.errstate(invalid="ignore", divide="ignore"):
            clv = (c - l) / (h - l)
            trig = 0.8 * x["rng"] / x["atr"]
            r1 = (lr.shift(1) / rv10).to_numpy()
            r5 = (lr.rolling(5).sum().shift(1) / (rv10 * np.sqrt(5))).to_numpy()
        n, w, s = day_stats(p["T"], a, idx)
        l1, l2, w1 = lost_last(n, w, 1).astype(float), lost_last(n, w, 2).astype(float), won_last(n, w, 1).astype(float)
        rprev = last_window(s, 1)
        T = p["T"].sort_values("day", kind="stable").reset_index(drop=True)
        di = idx.get_indexer(pd.DatetimeIndex(T.day))
        # 直前5回の取引（どれも前日の朝までに決済ずみ）の平均のR
        last5 = pd.Series(T.R.to_numpy()).shift(1).rolling(5, min_periods=1).mean().to_numpy()
        for j in range(len(T)):
            d = di[j]
            if d < 0:
                continue
            sd = T.side[j]
            rows.append(dict(coin=a, day=idx[d], side=sd, w=T.w[j], R=T.R[j], qrank=qrank[d], arank=arank[d], vr=vr[d],
                             clv=clv[d] if sd == 1 else 1 - clv[d], trig=trig[d], r1=sd * r1[d], r5=sd * r5[d], bstr=sd * bdist[d],
                             l1=l1[d], l2=l2[d], w1=w1[d], rprev=rprev[d], last5=last5[j], dow=idx[d].dayofweek,
                             eth=float(a == "ETH"), long=float(sd == 1), quiet=float(p["mq"][d] > 0)))
    F = pd.DataFrame(rows)
    for k in range(6):
        F[f"dow{k}"] = (F.dow == k).astype(float)
    return F


FEATS = ["qrank", "arank", "vr", "clv", "trig", "r1", "r5", "bstr", "l1", "l2", "w1", "rprev", "last5", "eth", "long"] + [f"dow{k}" for k in range(6)]


def walk_trades(F, make, target="win", start=pd.Timestamp("2019-01-01"), min_train=150):
    """毎月はじめに、前の月の末日より前の取引（どれも月はじめの朝までに決済ずみ）で学び直し、その月の取引の予想を出す。
    量の区切り（学んだ取引の予想の下から3割・7割、プラスの予想の中央値）も、学んだ取引だけから決める。"""
    out = np.full(len(F), np.nan)
    for c in ("q30", "q70", "qmed"):
        F[c] = np.nan
    X = F[FEATS].to_numpy(float)
    ok = ~np.isnan(X).any(1)
    y = (F.R.to_numpy() > 0).astype(int) if target == "win" else F.R.to_numpy()
    mons = pd.date_range(start, MM.END, freq="MS")
    for i, m0 in enumerate(mons):
        m1 = mons[i + 1] if i + 1 < len(mons) else MM.END + pd.Timedelta(days=1)
        tr = ok & (F.day < m0 - pd.Timedelta(days=1)).to_numpy()
        te = ok & (F.day >= m0).to_numpy() & (F.day < m1).to_numpy()
        if tr.sum() < min_train or not te.any():
            continue
        mdl = make().fit(X[tr], y[tr])
        pred = (lambda Z: mdl.predict_proba(Z)[:, 1]) if target == "win" else mdl.predict
        out[te] = pred(X[te])
        ptr = pred(X[tr])
        F.loc[te, "q30"] = np.quantile(ptr, 0.3)
        F.loc[te, "q70"] = np.quantile(ptr, 0.7)
        F.loc[te, "qmed"] = np.median(ptr[ptr > 0]) if (ptr > 0).any() else np.nan
    return out


def per_from_trades(F, D, size):
    """取引ごとの量（size）から、銘柄ごとの毎日の損益を作る（値幅ブレイクは翌朝決済なので、その日の損益になる）。"""
    per, Ts = {}, []
    for a in COINS:
        idx = D[a]["idx"]
        f = F[(F.coin == a) & (size > 0)]
        sz = size[(F.coin == a).to_numpy() & (size > 0)]
        di = idx.get_indexer(pd.DatetimeIndex(f.day))
        pnl = np.bincount(di, weights=f.R.to_numpy() * 0.5 * f.w.to_numpy() * sz, minlength=len(idx))
        per[a] = pd.Series(pnl, index=idx)
        Ts.append(pd.DataFrame({"coin": a, "day": f.day.to_numpy(), "R": f.R.to_numpy()}))
    T = pd.concat(Ts, ignore_index=True)
    return per, T[(T.day >= MM.START) & (T.day <= MM.END)]


def sec_meta(ctx):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    D, parts = ctx["D"], ctx["parts"]
    print("\n==== R07〜R10 取引を選ぶ機械学習：S1の向きで出た値幅ブレイクの合図ごとに、勝ちやすさを予想して量を変える（毎月学び直し・2019年から）")
    F = trade_features(D, parts)
    old = (MM.START, R1.YEARS)
    MM.START = pd.Timestamp("2019-01-01")
    R1.YEARS = (MM.END - MM.START).days / 365.25
    try:
        rules = size_rules(parts, COINS)
        out = {}
        for k in ("S1", "S4案"):
            per, T = build(parts, rules[k])
            out[k] = show_any("R07", f"参考：{k}（2019年から）", per, T)
        lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        rf = lambda: RandomForestClassifier(n_estimators=150, max_depth=4, min_samples_leaf=50, n_jobs=4, random_state=0)
        quiet = F.quiet.to_numpy() > 0
        ev = (F.day >= MM.START).to_numpy()
        print(f"    合図の数：{len(F)}回（2019年から{ev.sum()}回）・勝率{(F.R[ev] > 0).mean() * 100:.0f}%。静かな日の合図：{(quiet & ev).sum()}回")
        for pid, lab, make, only_quiet in (("R07", "ロジスティック回帰・全部の合図から選ぶ", lr, False),
                                           ("R08", "ロジスティック回帰・静かな日の合図の中で量を変える", lr, True),
                                           ("R09", "ランダムフォレスト・全部の合図から選ぶ", rf, False)):
            G = F[quiet].reset_index(drop=True) if only_quiet else F.copy()
            p = walk_trades(G, make)
            q30, q70 = G.q30.to_numpy(), G.q70.to_numpy()
            low = 0.5 if only_quiet else 0.0
            # 予想がまだない月（学ぶ取引が足りない最初の数か月）は、いつも通り1倍
            size = np.where(np.isnan(p), 1.0, np.where(p >= q70, 1.5, np.where(p <= q30, low, 1.0)))
            per, T = per_from_trades(G, D, size)
            halves = (((G.day >= MM.START) & (G.day < H2)).to_numpy(), (G.day >= H2).to_numpy())
            hit = [(G.R[(size > 0) & sel] > 0).mean() * 100 for sel in halves]
            allw = [(G.R[sel] > 0).mean() * 100 for sel in halves]
            out[pid] = show_any(pid, f"{lab}（上位3割は1.5倍・下位3割は{'0.5倍' if only_quiet else '見送り'}）", per, T,
                                memo=f"取った取引の勝率：前半{hit[0]:.0f}%・後半{hit[1]:.0f}%（合図全体は{allw[0]:.0f}%・{allw[1]:.0f}%）")
            print(f"      取った取引の勝率：前半{hit[0]:.0f}%・後半{hit[1]:.0f}%（合図全体は前半{allw[0]:.0f}%・後半{allw[1]:.0f}%）")
        G = F.copy()
        pr = walk_trades(G, lambda: make_pipeline(StandardScaler(), LinearRegression()), target="R")
        size = np.where(np.isnan(pr), 1.0, np.clip(np.nan_to_num(pr) / G.qmed.to_numpy(), 0, 2))
        size = np.nan_to_num(size, nan=1.0)
        per, T = per_from_trades(G, D, size)
        out["R10"] = show_any("R10", "予想したRの大きさに合わせて量を0〜2倍に（線形回帰・全部の合図。予想がマイナスなら見送り）", per, T)
        # 効いた材料（2022年末まで）
        X = F[FEATS].to_numpy(float)
        ok = ~np.isnan(X).any(1) & (F.day < H2).to_numpy()
        m = lr().fit(X[ok], (F.R.to_numpy()[ok] > 0).astype(int))
        coef = pd.Series(m[-1].coef_[0], index=FEATS).sort_values(key=np.abs, ascending=False)[:6]
        print("    2022年末までで学んだ時の、効きの大きい材料：" + "・".join(f"{k}{v:+.2f}" for k, v in coef.items()))
        keep("meta", rows={k: brief(v) for k, v in out.items()}, coef=coef.round(3).to_dict(), n=len(F))
    finally:
        MM.START, R1.YEARS = old


# ---------------------------------------------------------------- R14〜R18 新しい組み合わせ
def sec_combo(ctx):
    D, parts = ctx["D"], ctx["parts"]
    rules = size_rules(parts, COINS)
    out = {}
    print("\n==== R14 前日の引けの位置（高値寄り・安値寄り）で値幅ブレイクの向きを絞る（S1の形）")
    for lab, good in (("買いは前日に高値寄りで引けた日・売りは安値寄りの日だけ", True), ("買いは安値寄り・売りは高値寄りの日だけ（逆）", False)):
        p2 = {}
        for a in COINS:
            x, p = D[a], parts[a]
            with np.errstate(invalid="ignore", divide="ignore"):
                clv = R6.prev((x["dC"] - x["dL"]) / (x["dH"] - x["dL"]))
            T = p["T"].reset_index(drop=True)
            di = x["idx"].get_indexer(pd.DatetimeIndex(T.day))
            c = clv[np.clip(di, 0, None)]
            okt = np.where(T.side.to_numpy() == 1, c >= 0.5, c <= 0.5)
            if not good:
                okt = ~okt
            okt &= ~np.isnan(c)
            pnl = np.zeros(len(x["idx"]))
            np.add.at(pnl, di[okt], T.R.to_numpy()[okt] * 0.5 * T.w.to_numpy()[okt])
            p2[a] = dict(s=pd.Series(pnl, index=x["idx"]), T=T[okt], mq=p["mq"], m3=p["m3"], idx=p["idx"])
        per, T = build(p2, lambda a, p: p["mq"])
        out["R14" + ("" if good else "逆")] = show_any("R14", f"S1で{lab}", per, T)
    print("\n==== R15・R16 静かな日の判定を変える（S1の形）")
    mb = parts["BTC"]["mq"]
    per, T = build(parts, lambda a, p: mb)
    out["R15"] = show_any("R15", "S1で、ETHの静かな日もBTCの値動きで判定", per, T)
    both = parts["BTC"]["mq"] * parts["ETH"]["mq"]
    per, T = build(parts, lambda a, p: both)
    out["R16"] = show_any("R16", "S1で、2銘柄とも静かな日だけ取引", per, T)
    print("\n==== R17 S4案＋50日平均の向き（C39の一定の量）を少しだけ")
    per, T = build(parts, rules["S4案"])
    s4 = R1.total(per)

    def c39(x, a):
        c = x["dC"]
        m = R3.sma(c, 50)
        with np.errstate(invalid="ignore"):
            return np.where(c > m, 1.0, np.where(c < m, -1.0, 0.0))
    trend = R1.total(R3.both(D, lambda x, a: R3.pos_daily(x, a, c39(x, a)))[0])
    corr = s4.resample("ME").sum().corr(trend.resample("ME").sum())
    for wgt in (0.1, 0.25, 0.5):
        r = R1.stats(s4.add(wgt * trend, fill_value=0))
        R1.line(f"S4案＋C39×{wgt}", r, f" ｜月の相関{corr:+.2f}")
        record(グループ=GROUP, 計画ID="R17", 名前=f"R17 S4案＋50日平均の向き×{wgt}", 銘柄="BTC,ETH", 時間足="15分足・日足",
               メモ=f"月の相関{corr:+.2f}", **{k: v for k, v in r.items() if not k.startswith("_")})
        out[f"R17×{wgt}"] = r
    print("\n==== R18 S1・S4案を、日本時間12時・12時半で区切る（手で操作しやすい時刻）")
    for off, lab in ((12, "12時"), (14, "12時半")):
        po = s1_parts(D, COINS, D["BTC"], off)
        ro = size_rules(po, COINS)
        for k in ("S1", "S4案"):
            per, T = build(po, ro[k])
            out[f"R18 {lab} {k}"] = show_any("R18", f"{k}を日本時間{lab}区切りで（{lab}に注文、翌日の{lab}に決済）", per, T)
    keep("combo", rows={k: brief(v) for k, v in out.items()}, corr17=corr)


# ---------------------------------------------------------------- G01〜G11 出来高
def vol_mats(path, idx):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df[~df.index.duplicated()].sort_index()
    step = pd.Timedelta(minutes=15)
    full = pd.date_range(idx[0], idx[-1] + pd.Timedelta(days=1) - step, freq=step)
    v = df.reindex(full)
    return {k: v[k].fillna(0.0).to_numpy().reshape(len(idx), DAY) for k in ("volume", "quote_volume", "trades", "taker_buy")}


def sec_vol(ctx):
    cache, D = ctx["cache"], ctx["D"]
    per0, T0 = ctx["per0"], ctx["T0"]
    V = {}
    for a in COINS:
        p = os.path.join(cache, f"{a}_15m_vol.csv")
        if not os.path.exists(p):
            print("    出来高のデータがない")
            return
        V[a] = vol_mats(p, D[a]["idx"])
    out = {}
    days = D["BTC"]["idx"]
    dq = {a: V[a]["quote_volume"].sum(1) for a in COINS}
    avg20 = {a: pd.Series(dq[a]).rolling(20).mean().shift(1).to_numpy() for a in COINS}
    spike = {a: R6.prev(dq[a]) >= 2 * R6.prev(avg20[a]) for a in COINS}

    print("\n==== G01 出来高の急増（20日平均の2倍・3倍）の翌日、その日の終値の向きに9時〜翌9時")
    res = {}
    for kx in (2, 3):
        sides = {}
        for a in COINS:
            x = D[a]
            with np.errstate(invalid="ignore"):
                sp = R6.prev(dq[a]) >= kx * R6.prev(avg20[a])
                sides[a] = np.where(sp, np.sign(R6.prev(x["dC"] - x["dO"])), 0.0)
        res[f"{kx}倍"] = R6.dt(ctx, "G01", f"出来高が20日平均の{kx}倍以上の翌日、終値の向きに", {a: np.nan_to_num(sides[a]) for a in COINS})
    R6.comp(ctx, "G01", res)
    out["G01"] = res

    print("\n==== G02・G09 値幅ブレイクの約定した時間の出来高で選ぶ（A01の取引を後から選ぶ）")
    T = T0.reset_index(drop=True)
    hv = {a: V[a]["quote_volume"].reshape(len(days), 24, 4).sum(2) for a in COINS}
    hshare = {a: pd.DataFrame(hv[a] / hv[a].sum(1, keepdims=True).clip(1e-9)).rolling(60, min_periods=20).mean().shift(1).to_numpy() for a in COINS}
    # 約定した足の直前1時間（4本）の売買代金。約定した足そのものは、その時点では出来高が決まっていないので入れない
    v4 = {}
    for a in COINS:
        flat = V[a]["quote_volume"].ravel()
        c = np.r_[0.0, np.cumsum(flat)]
        b = np.arange(len(flat))
        v4[a] = (c[b] - c[np.maximum(0, b - 4)]).reshape(len(days), DAY)
    v4avg = {a: pd.DataFrame(v4[a]).rolling(20, min_periods=10).mean().shift(1).to_numpy() for a in COINS}
    keepG02, keepG09 = np.zeros(len(T), bool), np.zeros(len(T), bool)
    for a in COINS:
        sel = (T.coin == a).to_numpy()
        di = days.get_indexer(pd.DatetimeIndex(T.day[sel]))
        bar = T.pos.to_numpy()[sel].astype(int)
        keepG02[sel] = v4[a][di, bar] >= v4avg[a][di, bar]
        sh = hshare[a][di]
        thr = np.nanquantile(sh, 1 / 3, axis=1)
        keepG09[sel] = sh[np.arange(len(di)), bar // 4] > thr
    for pid, kp, lab in (("G02", keepG02, "約定の直前1時間の売買代金が、同じ時間の20日平均以上の時だけ"),
                         ("G09", keepG09, "出来高の少ない時間（1日の中で下から3分の1）の約定は見送る")):
        per = {}
        for a in COINS:
            sel = (T.coin == a).to_numpy() & kp
            di = days.get_indexer(pd.DatetimeIndex(T.day[sel]))
            per[a] = pd.Series(np.bincount(di, weights=T.R.to_numpy()[sel] * 0.5 * T.w.to_numpy()[sel], minlength=len(days)), index=days)
        out[pid] = R6.show(pid, f"値幅ブレイク：{lab}", per, T[kp], tf="15分足")

    print("\n==== G03 出来高が減りながら5日上げたら、9時〜翌9時に売り（×0.5）")
    sides = {}
    for a in COINS:
        x = D[a]
        c = x["dC"]
        v5, v5p = pd.Series(dq[a]).rolling(5).mean().to_numpy(), pd.Series(dq[a]).rolling(5).mean().shift(5).to_numpy()
        with np.errstate(invalid="ignore"):
            cond = R6.prev((c / R6.prev(c, 5) - 1 > 0.05) & (v5 < v5p))
        sides[a] = np.where(np.nan_to_num(cond).astype(bool), -1.0, 0.0)
    out["G03"] = R6.dt(ctx, "G03", "出来高が減りながら5日で5%以上上げたら売り", sides)

    print("\n==== G04 OBV が20日平均の上なら買い、下なら売り")
    res = {}
    for lo in (False, True):
        def fn(x, a, lo=lo):
            c = x["dC"]
            obv = np.cumsum(np.sign(np.nan_to_num(np.diff(c, prepend=c[0]))) * dq[a])
            m = R3.sma(obv, 20)
            with np.errstate(invalid="ignore"):
                return R3.ls(obv > m, obv < m, lo)
        res["買いだけ" if lo else "買い＋売り"] = R6.run_pos(ctx, "G04", "OBVが20日平均の上（下）", fn, lo)
    R6.comp(ctx, "G04", res)
    out["G04"] = res

    print("\n==== G05 前日のVWAP（出来高で重みをつけた平均の値段）より上で始まった日は買いだけ、下なら売りだけ（値幅ブレイク）")
    per, Ts = {}, []
    for a in COINS:
        x = D[a]
        M = x["M"]
        tp = (M["high"] + M["low"] + M["close"]) / 3
        vw = (tp * V[a]["volume"]).sum(1) / np.clip(V[a]["volume"].sum(1), 1e-9, None)
        dirv = np.sign(x["dO"] - R6.prev(vw))
        s, T_ = R2.run(R2.periods(x, DAY, 0), a, ma=False, extra_ok=lambda P_, side, dv=dirv: side * np.nan_to_num(dv) > 0)
        per[a] = s
        Ts.append(T_)
    out["G05"] = R6.show("G05", "値幅ブレイク：向きを5日平均の代わりに前日のVWAPで決める", per, pd.concat(Ts, ignore_index=True), tf="15分足")

    print("\n==== G06 直近30日でいちばん出来高の多い値段（1%刻み）に指値、翌朝決済（上なら売り×0.5・下なら買い）")

    def poc(a):
        x = D[a]
        M = x["M"]
        lp = np.log((M["high"] + M["low"]) / 2)
        out_ = np.full(len(x["idx"]), np.nan)
        b = np.log(1.01)
        for d in range(31, len(x["idx"])):
            bins = np.floor(lp[d - 30:d].ravel() / b)
            vols = V[a]["quote_volume"][d - 30:d].ravel()
            u, inv = np.unique(bins, return_inverse=True)
            out_[d] = np.exp((u[np.argmax(np.bincount(inv, weights=vols))] + 0.5) * b)
        return out_
    pocs = {a: poc(a) for a in COINS}

    def g06(x, a):
        o, A, pc = x["dO"], x["atr"], pocs[a]
        with np.errstate(invalid="ignore"):
            okl = (pc < o) & (o - pc <= A)
            oks = (pc > o) & (pc - o <= A)
        return R3B.two_sides(lambda x_, a_: R3B.limit(x_, a_, np.nan_to_num(okl).astype(bool), np.nan_to_num(pc), 1, 1),
                             lambda x_, a_: R3B.limit(x_, a_, np.nan_to_num(oks).astype(bool), np.nan_to_num(pc), -1, 1), 0.5)(x, a)
    out["G06"] = R6.run_fn(ctx, "G06", "出来高の一番多い値段に逆張りの指値（1ATR以内）", g06, tf="15分足")

    print("\n==== G07 前日の成行の買いの比率（成行の買いの出来高÷出来高）")
    res = {}
    for th in (0.52, 0.55):
        sides = {}
        for a in COINS:
            with np.errstate(invalid="ignore", divide="ignore"):
                tbr = R6.prev(V[a]["taker_buy"].sum(1) / V[a]["volume"].sum(1))
            sides[a] = np.where(tbr >= th, 1.0, np.where(tbr <= 1 - th, -1.0, 0.0))
        res[f"{th}"] = R6.dt(ctx, "G07", f"前日の成行の買いの比率が{th * 100:.0f}%以上なら買い・{(1 - th) * 100:.0f}%以下なら売り", sides)
    R6.comp(ctx, "G07", res)
    out["G07"] = res

    print("\n==== G08 取引回数の急増（20日平均の2倍以上）の翌日は値幅ブレイクを休む／その翌日だけ")
    tr = {a: V[a]["trades"].sum(1) for a in COINS}
    m = {}
    for a in COINS:
        with np.errstate(invalid="ignore"):
            m[a] = R6.prev(tr[a]) >= 2 * R6.prev(pd.Series(tr[a]).rolling(20).mean().shift(1).to_numpy())
    out["G08"] = {"休む": R6.scaled(ctx, "G08", "値幅ブレイク：取引回数が20日平均の2倍以上の翌日は休む", per0, T0, {a: (~np.nan_to_num(m[a]).astype(bool)).astype(float) for a in COINS}),
                  "だけ": R6.scaled(ctx, "G08", "値幅ブレイク：取引回数が20日平均の2倍以上の翌日だけ", per0, T0, {a: np.nan_to_num(m[a]).astype(float) for a in COINS})}

    print("\n==== G10 出来高の多い日（20日平均の2倍以上）の高値を、次の日に超えたら買い（翌朝決済）")
    per, Ts = {}, []
    for a in COINS:
        x = D[a]
        sp = np.nan_to_num(spike[a]).astype(bool)
        s, T_ = R2.run(R2.periods(x, DAY, 0), a, ma=False, short_w=0, trig_fn=lambda P_, side: R6.prev(P_["pH"]),
                       extra_ok=lambda P_, side, sp=sp: sp)
        per[a] = s
        Ts.append(T_)
    out["G10"] = R6.show("G10", "出来高の多い日の高値を超えたら買い", per, pd.concat(Ts, ignore_index=True), tf="15分足")

    print("\n==== G11 出来高で「静かな日」を判定：10日の平均の売買代金が、過去1年の中央値より少ない日だけ（A01・S1の形）")
    mv = {}
    for a in COINS:
        q = pd.Series(dq[a]).rolling(10).mean().shift(1)
        med = q.rolling(365, min_periods=200).median()
        mv[a] = np.where(med.isna(), 1.0, (q < med).astype(float))
    out["G11 A01"] = R6.scaled(ctx, "G11", "値幅ブレイク：売買代金の少ない日だけ（A01の形）", per0, T0, mv)
    parts = ctx["parts"]
    per, T_ = build(parts, lambda a, p: pd.Series(mv[a], index=D[a]["idx"]).reindex(p["idx"]).fillna(1.0).to_numpy())
    out["G11 S1"] = show_any("G11", "S1の静かな日を、値動きの代わりに売買代金で判定", per, T_)
    both = {a: mv[a] * ctx["Mq"][a] for a in COINS}
    out["G11 両方"] = R6.scaled(ctx, "G11", "値幅ブレイク：値動きも売買代金も少ない日だけ", per0, T0, both)
    keep("vol", rows={k: (brief(v) if isinstance(v, dict) and "年平均%" in v else {kk: brief(vv) for kk, vv in v.items()}) for k, v in out.items()})


# ---------------------------------------------------------------- 本体
SECTIONS = {"size": sec_size, "alt": sec_alt, "meta": sec_meta, "combo": sec_combo, "vol": sec_vol}


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
    parts = s1_parts(D, COINS, D["BTC"])
    ctx = dict(cache=cache, D=D, days=days, per0=per0, T0=T0, v0=v0, base=base, Mq=Mq, parts=parts)
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
