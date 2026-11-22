#!/usr/bin/env python3
"""相場に合わせてロジックを切り替えるBOTの検証（BTC・ETH、2018年〜）。

部品（どれも Vantage CFD、1R＝ATR1本分の値動き＝資金の0.5%、含み損込みの毎日の損益）:
  値幅ブレイク : ①買い＋②売り×0.5（第一候補、当日決済）
  トレンド     : 9日EMAを実体で抜けたら（50日平均の向き）、終値がEMAを戻ったら翌朝決済。損切りは抜けた足の安値（ema9_lab）
  逆張りRSI2   : RSI(2)が10未満で翌朝買い・90超で翌朝売り。終値が5日平均を超えたら（売りは下回ったら）翌朝決済、最長5日、損切りATR×2
  逆張りBB     : 終値がボリンジャーバンド（20日・2σ）の外で翌朝に逆張り。真ん中（20日平均）に戻ったら翌朝決済、最長10日、損切りATR×2
  買い持ち     : ずっと買い（スワップ込み）
相場の見分け方（その日の朝にわかる値だけを使う。銘柄ごと）:
  向き   : 前日の終値が200日平均の上なら「上昇」、下なら「下落」
  強さ   : ADX(14)が25以上なら「トレンド」、未満なら「レンジ」。参考に効率レシオ（20日）でも分ける（境目は前半の中央値）
  値動き : ATR（価格比）が過去1年の中央値より大きければ「荒い」、小さければ「静か」
切り替えのしかた:
  理屈どおり   : トレンドの日はトレンド部品、レンジの日は逆張り部品 など、先に決めた対応
  データで選ぶ : 前半（2018〜2022年）で、相場ごとに一番成績の良かった部品を選び、後半（2023年〜）で確かめる
  成績を追う   : 毎月初め、直近3か月（または6か月）で一番稼いだ部品に切り替える
比べる相手: 値幅ブレイクだけ／全部の部品を同じ量で持ちっぱなし（量の合計は1）。
使い方: python3 project/tools/regime_switch_lab.py <キャッシュ（BTC_15m.csv, ETH_15m.csv）> [vol]
        vol: 部品を切り替える代わりに、値動きの荒さで量を変える（静かな日を多め）
"""
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
import ema9_lab as E  # noqa: E402
import exit_combo_lab as EX  # noqa: E402
import more_methods_lab as MM  # noqa: E402
import portfolio_lab as PF  # noqa: E402
from registry import drop, metrics_row, record  # noqa: E402

COINS = ("BTC", "ETH")
SPLIT = PF.SPLIT
GROUP = "相場で切り替え"
GROUP_V = "値動きで量を変える"
PARTS = ["値幅ブレイク", "トレンド", "逆張りRSI2", "逆張りBB", "買い持ち"]
SWITCH_COST = 0.005          # 部品の量を1だけ増減するたびの費用（資金の%）。建玉の乗り換えのおおよその手数料


@njit
def gen_engine(O, H, L, C, P, atr, sig, xsig, side, stop_k, maxd, cost, swd, roll):
    """前日の終値で sig なら翌朝の始値で入る。損切りはATR×stop_k、xsig（前日の終値で判定）か maxd 日で翌朝決済。
    1R＝ATR1本分＝資金の0.5%。毎日の損益（資金の%）を返す。"""
    nd = sig.shape[0]
    N = nd * P
    pnl = np.zeros(nd)
    d = 1
    while d < nd:
        A = atr[d]
        if not sig[d - 1] or not (A > 0):
            d += 1
            continue
        k = d * P
        e = O[k]
        stop = e - side * stop_k * A
        cR = cost * e / A
        sR = swd * e / A
        pnl[d] -= cR * 0.5
        mark = 0.0
        held = 0
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
            if j > k and pos == roll:
                pnl[dd] -= sR * 0.5
            adv = L[j] if side == 1 else H[j]
            if side * (adv - stop) <= 0:
                if j == k:
                    x = stop
                elif side == 1:
                    x = min(O[j], stop)
                else:
                    x = max(O[j], stop)
                out = dd
                break
            if pos == P - 1:
                m = side * (C[j] - e) / A
                pnl[dd] += (m - mark) * 0.5
                mark = m
            j += 1
        if out < 0:
            break
        pnl[out] += (side * (x - e) / A - mark) * 0.5
        d = out + 1
    return pnl


def rsi(c, n):
    d = np.diff(c, prepend=c[0])
    up = pd.Series(np.clip(d, 0, None)).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    dn = pd.Series(np.clip(-d, 0, None)).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100 - 100 / (1 + up / dn)


def wilder(x, n):
    return pd.Series(x).ewm(alpha=1 / n, adjust=False).mean().to_numpy()


def adx(h, l, c, n=14):
    ph, pl, pc = E.prev(h), E.prev(l), E.prev(c)
    up, dn = h - ph, pl - l
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    at = wilder(np.nan_to_num(tr), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi, mdi = 100 * wilder(pdm, n) / at, 100 * wilder(mdm, n) / at
        dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    return wilder(np.nan_to_num(dx), n)


def parts_for(D, a):
    """部品ごとの毎日の損益（資金の%、その銘柄の分）。"""
    Da = D[a]
    B = E.daily(Da)
    O, H, L, C = B["flat"]
    P, dC, atr = Da["per_day"], B["dC"], Da["atr"]
    cost, swl, sws = PF.SPEC[a][0], PF.SPEC[a][1] / 36500, PF.SPEC[a][2] / 36500
    roll = EX.ROLL_UTC * P // 24
    idx = pd.DatetimeIndex(Da["days"])
    out = {}
    v = EX.vbo_exit_n(Da, a, 1, "そのまま")
    out["値幅ブレイク"] = (v.fillna(0) + 0.5 * EX.vbo_exit_n(Da, a, -1, "そのまま").fillna(0)).to_numpy() * 0.5
    tr = np.zeros(len(dC))
    for side in (1, -1):
        p, *_ = E.engine(O, H, L, C, P, B["dH"], B["dL"], dC, B["ema"], E.signal(B, side, "50日平均の上"), side, 0, 0, 0.0,
                         cost, swl if side == 1 else sws, roll, 0.005)
        tr += p
    out["トレンド"] = tr
    s = pd.Series(dC)
    r2, sma5 = rsi(dC, 2), s.rolling(5).mean().to_numpy()
    m20, sd20 = s.rolling(20).mean().to_numpy(), s.rolling(20).std().to_numpy()
    with np.errstate(invalid="ignore"):
        rules = {"逆張りRSI2": [(1, r2 < 10, dC > sma5, 5), (-1, r2 > 90, dC < sma5, 5)],
                 "逆張りBB": [(1, dC < m20 - 2 * sd20, dC > m20, 10), (-1, dC > m20 + 2 * sd20, dC < m20, 10)]}
    for name, legs in rules.items():
        tot = np.zeros(len(dC))
        for side, sig, xs, maxd in legs:
            tot += gen_engine(O, H, L, C, P, atr, sig.astype(np.bool_), xs.astype(np.bool_), side, 2.0, maxd, cost,
                              swl if side == 1 else sws, roll)
        out[name] = tot
    with np.errstate(invalid="ignore", divide="ignore"):
        hold = 0.5 * (dC - E.prev(dC)) / atr - 0.5 * swl * E.prev(dC) / atr
    out["買い持ち"] = np.nan_to_num(hold)
    # 相場の見分け（その日の朝にわかる値）
    sma200 = s.rolling(200).mean().to_numpy()
    er = (s - s.shift(20)).abs() / s.diff().abs().rolling(20).sum()
    atrp = pd.Series(atr / B["dO"])
    reg = pd.DataFrame({"上昇": E.prev(dC) > E.prev(sma200), "ADX": E.prev(adx(B["dH"], B["dL"], dC)),
                        "ER": E.prev(er.to_numpy()), "荒い": (atrp > atrp.shift(1).rolling(365, min_periods=200).median()).to_numpy(),
                        "荒い180": (atrp > atrp.shift(1).rolling(180, min_periods=120).median()).to_numpy(),
                        "荒い730": (atrp > atrp.shift(1).rolling(730, min_periods=200).median()).to_numpy(),
                        "有効": ~np.isnan(E.prev(sma200))}, index=idx)
    return pd.DataFrame(out, index=idx), reg


def clip(x):
    return x[(x.index >= MM.START) & (x.index <= MM.END)]


def ann(x):
    return x.mean() * 365 if len(x) else np.nan


def sharpe(x):
    return x.mean() / x.std() * np.sqrt(365) if len(x) > 30 and x.std() > 0 else np.nan


def apply(Pm, W):
    """部品の損益（銘柄ごと）に、毎日の量 W（銘柄ごと）をかけて合計。量が変わった分だけ切り替えの費用を引く。"""
    tot = None
    for a in COINS:
        w = W[a].reindex(Pm[a].index).fillna(0.0)
        x = (Pm[a][PARTS] * w[PARTS]).sum(axis=1)
        chg = w[[p for p in PARTS if p != "値幅ブレイク"]].diff().abs().sum(axis=1).fillna(0)
        x = x - chg * SWITCH_COST
        tot = x if tot is None else tot.add(x, fill_value=0)
    return clip(tot)


def onehot(index, choice):
    """choice（部品名か None の Series）から量の表。"""
    W = pd.DataFrame(0.0, index=index, columns=PARTS)
    for p in PARTS:
        W.loc[choice == p, p] = 1.0
    return W


def show(name, x, memo="", rec=True, group=GROUP):
    r = metrics_row(x)
    h2 = metrics_row(x[x.index >= SPLIT])
    print(f"  {name:34s} 年{r['年平均%']:+6.2f}% 落込{r['最大の落ち込み%']:5.1f}% 安定度{r['安定度']:5.2f} 負け年{r['マイナスの年']:16s}"
          f" ｜前半{r['前半の年平均%']:+5.1f}%→後半{r['後半の年平均%']:+5.1f}%（後半の安定度{h2['安定度']:5.2f}・落込{h2['最大の落ち込み%']:4.1f}%）")
    if rec:
        record(グループ=group, 名前=name, 銘柄="BTC,ETH", 時間足="日足（約定は15分足）", メモ=memo, **r)
    return r, h2


def main(cache):
    print(f"（古い「{GROUP}」の行を {drop(GROUP)} 行消して、計算し直す）")
    D = {a: EX.day_matrix_n(os.path.join(cache, f"{a}_15m.csv"), 96) for a in COINS}
    Pm, R = {}, {}
    for a in COINS:
        Pm[a], R[a] = parts_for(D, a)
    er_cut = np.nanmedian(pd.concat([R[a].ER[(R[a].index >= MM.START) & (R[a].index < SPLIT)] for a in COINS]))
    for a in COINS:
        R[a]["トレンド相場"] = R[a].ADX >= 25
        R[a]["効率が高い"] = R[a].ER >= er_cut
        R[a]["相場"] = np.where(R[a]["上昇"], "上昇", "下落") + "・" + np.where(R[a]["トレンド相場"], "トレンド", "レンジ")
        R[a]["値動き"] = np.where(R[a]["荒い"], "荒い", "静か")

    print("==== 部品それぞれ（BTC+ETH、切り替えなし）")
    for p in PARTS:
        x = clip(sum(Pm[a][p] for a in COINS))
        show(f"部品：{p}", x, rec=True)

    allr = pd.concat([R[a][(R[a].index >= MM.START) & (R[a].index <= MM.END)] for a in COINS])
    print(f"\n==== 相場の日数の割合（BTC・ETH合計、2018年〜）  効率レシオの境目（前半の中央値）{er_cut:.2f}")
    print("  " + " ／ ".join(f"{k} {v * 100:.0f}%" for k, v in allr["相場"].value_counts(normalize=True).sort_index().items())
          + f" ／ 荒い日 {allr['荒い'].mean() * 100:.0f}%")

    print("\n==== 相場ごとの部品の成績（その相場の日だけ持った場合の年率％。前半 → 後半）")
    def table(col, labels):
        rows = {}
        for p in PARTS:
            cells = {}
            for lab, cond in labels:
                v = []
                for half, (lo, hi) in (("前半", (MM.START, SPLIT)), ("後半", (SPLIT, MM.END + pd.Timedelta(days=1)))):
                    xs = pd.concat([Pm[a][p][(R[a].index >= lo) & (R[a].index < hi) & cond(R[a]) & R[a]["有効"]] for a in COINS])
                    v.append(ann(xs))
                cells[lab] = f"{v[0]:+6.1f} → {v[1]:+6.1f}"
            rows[p] = cells
        print(pd.DataFrame(rows).T.to_string())
    table("相場", [(lab, (lambda r, lab=lab: r["相場"] == lab)) for lab in ("上昇・トレンド", "上昇・レンジ", "下落・トレンド", "下落・レンジ")])
    print()
    table("値動き", [("荒い", lambda r: r["荒い"]), ("静か", lambda r: ~r["荒い"]),
                    ("効率が高い", lambda r: r["効率が高い"]), ("効率が低い", lambda r: ~r["効率が高い"])])

    print("\n==== 比べる相手")
    base = {}
    W = {a: onehot(Pm[a].index, pd.Series("値幅ブレイク", index=Pm[a].index)) for a in COINS}
    base["vbo"] = show("値幅ブレイクだけ", apply(Pm, W), "切り替えなし")
    W = {a: pd.DataFrame(1 / len(PARTS), index=Pm[a].index, columns=PARTS) for a in COINS}
    base["eq"] = show("全部の部品を同じ量で持ちっぱなし（量の合計1）", apply(Pm, W), "5つの部品を1/5ずつ")

    print("\n==== 理屈どおりに切り替える（量の合計は1）")
    plans = {
        "トレンドの日はトレンド、レンジの日は逆張りRSI2": lambda r: np.where(r["トレンド相場"], "トレンド", "逆張りRSI2"),
        "トレンドの日はトレンド、レンジの日は逆張りBB": lambda r: np.where(r["トレンド相場"], "トレンド", "逆張りBB"),
        "効率が高い日はトレンド、低い日は逆張りRSI2": lambda r: np.where(r["効率が高い"], "トレンド", "逆張りRSI2"),
        "トレンドの日はトレンド、レンジの日は値幅ブレイク": lambda r: np.where(r["トレンド相場"], "トレンド", "値幅ブレイク"),
        "上昇は買い持ち、下落は休む": lambda r: np.where(r["上昇"], "買い持ち", "なし"),
        "上昇は買い持ち、下落は値幅ブレイク": lambda r: np.where(r["上昇"], "買い持ち", "値幅ブレイク"),
        "荒い日は値幅ブレイク、静かな日は逆張りRSI2": lambda r: np.where(r["荒い"], "値幅ブレイク", "逆張りRSI2"),
        "荒い日は値幅ブレイク、静かな日は休む": lambda r: np.where(r["荒い"], "値幅ブレイク", "なし"),
    }
    for name, fn in plans.items():
        W = {a: onehot(Pm[a].index, pd.Series(fn(R[a]), index=Pm[a].index).where(R[a]["有効"], "値幅ブレイク")) for a in COINS}
        show(f"理屈：{name}", apply(Pm, W), "相場の見分けは前日までの値")

    print("\n==== 値幅ブレイクはずっと持ち、もう1つを相場で切り替える（量の合計は1.5）")
    plans2 = {
        "トレンドの日にトレンド×0.5": lambda r: np.where(r["トレンド相場"], "トレンド", "なし"),
        "トレンドの日にトレンド×0.5、レンジの日に逆張りRSI2×0.5": lambda r: np.where(r["トレンド相場"], "トレンド", "逆張りRSI2"),
        "上昇の日に買い持ち×0.5": lambda r: np.where(r["上昇"], "買い持ち", "なし"),
    }
    for name, fn in plans2.items():
        W = {}
        for a in COINS:
            w = 0.5 * onehot(Pm[a].index, pd.Series(fn(R[a]), index=Pm[a].index).where(R[a]["有効"], "なし"))
            w["値幅ブレイク"] += 1.0
            W[a] = w
        show(f"値幅ブレイク＋{name}", apply(Pm, W))
    W = {}
    for a in COINS:
        w = pd.DataFrame(0.0, index=Pm[a].index, columns=PARTS)
        w["値幅ブレイク"], w["トレンド"] = 1.0, 0.5
        W[a] = w
    show("値幅ブレイク＋トレンド×0.5（切り替えなし）", apply(Pm, W))

    print("\n==== データで選ぶ：前半（2018〜2022年）で相場ごとに一番良かった部品 → 後半で確かめる")
    for keyname, key in (("向き×強さ（4つ）", "相場"), ("値動き（2つ）", "値動き")):
        pick = {}
        for lab in sorted(set(pd.concat([R[a][key] for a in COINS]).dropna())):
            best, bs = None, 0.0
            for p in PARTS:
                xs = pd.concat([Pm[a][p][(R[a].index >= MM.START) & (R[a].index < SPLIT) & (R[a][key] == lab) & R[a]["有効"]] for a in COINS])
                s = sharpe(xs)
                if s == s and s > bs:
                    best, bs = p, s
            pick[lab] = best
        print(f"  {keyname}: " + " ／ ".join(f"{k}→{v or '休む'}" for k, v in pick.items()))
        W = {a: onehot(Pm[a].index, R[a][key].map(pick).where(R[a]["有効"], "値幅ブレイク")) for a in COINS}
        show(f"データで選ぶ：{keyname}", apply(Pm, W), "前半で選んだ対応。後半の成績が本当の実力")

    print("\n==== 成績を追う：毎月初め、直近の成績が一番良い部品に切り替える")
    for look in (63, 126):
        for how in ("一番稼いだ1つ", "稼いだ部品を成績に応じた量で"):
            W = {}
            for a in COINS:
                trail = Pm[a][PARTS].rolling(look).sum().shift(1)
                month_start = Pm[a].index.to_period("M") != pd.Index(Pm[a].index.to_period("M")).to_series().shift(1).to_numpy()
                w = pd.DataFrame(np.nan, index=Pm[a].index, columns=PARTS)
                if how == "一番稼いだ1つ":
                    best = trail.fillna(-np.inf).idxmax(axis=1).where(trail.max(axis=1) > 0)
                    w.loc[month_start] = onehot(Pm[a].index, best).loc[month_start]
                else:
                    pos = trail.clip(lower=0)
                    w.loc[month_start] = pos.div(pos.sum(axis=1).replace(0, np.nan), axis=0).fillna(0).loc[month_start]
                W[a] = w.ffill().fillna(0.0)
            show(f"成績を追う：直近{look // 21}か月で{how}", apply(Pm, W), "毎月初めに、前日までの成績だけで決める")


def vol_size(cache):
    """値動きの荒さで、部品を切り替えるのではなく「量」を変える（値幅ブレイク・トレンド）。"""
    D = {a: EX.day_matrix_n(os.path.join(cache, f"{a}_15m.csv"), 96) for a in COINS}
    Pm, R = {}, {}
    for a in COINS:
        Pm[a], R[a] = parts_for(D, a)
    print(f"（古い「{GROUP_V}」の行を {drop(GROUP_V)} 行消して、計算し直す）")
    print("==== 値動きの荒さで量を変える（静か＝ATRが過去の中央値より小さい日、その日の朝にわかる）")
    print(f"  静かな日の割合 {np.mean(np.concatenate([~R[a]['荒い'].to_numpy()[(R[a].index >= MM.START)] for a in COINS])) * 100:.0f}%")
    for mix, label in ((("値幅ブレイク", 1.0),), "値幅ブレイク"), ((("トレンド", 1.0),), "トレンド"), \
                      ((("値幅ブレイク", 1.0), ("トレンド", 0.5)), "値幅ブレイク＋トレンド×0.5"):
        print(f"  -- {label}")
        for col, lk in (("荒い", "1年"), ("荒い180", "半年"), ("荒い730", "2年")):
            for qs, vs, how in ((1.0, 1.0, "いつも同じ量"), (1.0, 0.0, "静かな日だけ"), (0.0, 1.0, "荒い日だけ"), (1.5, 0.5, "静か×1.5・荒い×0.5")):
                if col != "荒い" and how in ("いつも同じ量", "荒い日だけ"):
                    continue
                W = {}
                for a in COINS:
                    quiet = (~R[a][col].astype(bool)) & R[a]["有効"]
                    scale = np.where(quiet, qs, vs)
                    w = pd.DataFrame(0.0, index=Pm[a].index, columns=PARTS)
                    for p, k in mix:
                        w[p] = k * scale
                    W[a] = w
                name = f"{label}：{how}（中央値は過去{lk}）" if how != "いつも同じ量" else f"{label}：いつも同じ量"
                show(name, apply(Pm, W), "値動きの荒さ＝ATR（価格比）と、その過去の中央値の比較", group=GROUP_V)

    print("\n==== 確かめ：値幅ブレイク（中央値は過去1年）の銘柄別・年ごと")
    rows, ny = {}, {}
    for how, qs, vs in (("いつも同じ量", 1.0, 1.0), ("静かな日だけ", 1.0, 0.0), ("荒い日だけ", 0.0, 1.0), ("静か×1.5・荒い×0.5", 1.5, 0.5)):
        per = {}
        for a in COINS:
            quiet = (~R[a]["荒い"].astype(bool)) & R[a]["有効"]
            per[a] = clip(Pm[a]["値幅ブレイク"] * np.where(quiet, qs, vs))
            v = EX.vbo_exit_n(D[a], a, 1, "そのまま").notna() | EX.vbo_exit_n(D[a], a, -1, "そのまま").notna()
            v.index = pd.DatetimeIndex(v.index)
            ny[(how, a)] = clip(v & pd.Series(np.where(quiet, qs, vs) > 0, index=v.index)).sum() / 8.9
        x = per["BTC"].add(per["ETH"], fill_value=0)
        rows[how] = x.groupby(x.index.year).sum()
        rb, re_ = metrics_row(per["BTC"]), metrics_row(per["ETH"])
        print(f"  {how:12s} BTCだけ 年{rb['年平均%']:+5.2f}% 落込{rb['最大の落ち込み%']:4.1f}% 負け年{rb['マイナスの年']:12s}"
              f" ETHだけ 年{re_['年平均%']:+5.2f}% 落込{re_['最大の落ち込み%']:4.1f}% 負け年{re_['マイナスの年']:12s}"
              f" ｜取引のある日 年{ny[(how, 'BTC')] + ny[(how, 'ETH')]:.0f}日")
    print(pd.DataFrame(rows).T.round(1).to_string())


if __name__ == "__main__" and len(sys.argv) == 2:
    main(sys.argv[1])

if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "vol":
    vol_size(sys.argv[1])
