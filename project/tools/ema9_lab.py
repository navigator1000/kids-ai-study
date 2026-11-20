#!/usr/bin/env python3
"""ラリー・ウィリアムズの手法として紹介された「9日EMA」ルールの検証（BTC・ETH、日足、2018年〜）。

紹介されたルール: 日足・9日EMA。トレンドを確認し、ローソク足の実体がEMAを上に抜けたら買い（売りは逆）。
                  損切りは抜けた足の安値（売りは高値）。
はっきりしない部分は、いくつかの決め方を試す。
  抜けた足（信号）: 始値がEMAの下、終値がEMAの上（売りは逆）
  トレンドの確認  : なし／20日平均が上向き／50日平均の上／200日平均の上（売りは逆）
                    ※EMA9の向きは条件にならない。終値がEMAの上なら、EMAは必ず上を向いているため。
  入り方          : 翌朝9時の始値／翌日、信号の足の高値を超えたら（逆指値、翌日だけ有効）
  損切り          : 信号の足の安値（売りは高値）
  決済            : 終値がEMAを下回ったら翌朝／損切りを前日の安値へ上げていく／利確1R・2R・3R／
                    最初に利益が出た朝（ラリー・ウィリアムズがよく使う決済）／5日後の朝
  参考            : 「9.1」と呼ばれる型（EMAが上向きに変わった足の高値を超えたら買い、EMAが下向きに変わったら翌朝決済）
計算は15分足で行う（日足で信号、15分足で約定）。同じ15分足で損切りと利確の両方に届いたら、損切りを先とする。
1回の量: 損切りにかかると資金の0.5%を失う大きさ（建玉は資金の1倍まで）。含み損込みで毎日の損益を計算。
コスト : Vantage CFD（往復0.05%、買いのスワップ年25〜35%をUTC21時をまたぐごと、売り0%）。参考に国内現物（買いだけ、往復0.15%）。
使い方 : python3 project/tools/ema9_lab.py <キャッシュ（BTC_15m.csv, ETH_15m.csv）> [check]
          check: 一番良かった形の設定ずらし（EMAの日数×確認の平均の日数）、中身（大勝ち頼みの度合い・連敗）、年ごとの損益
"""
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
import exit_combo_lab as EX  # noqa: E402
import more_methods_lab as MM  # noqa: E402
import portfolio_lab as PF  # noqa: E402
from registry import drop, metrics_row, record  # noqa: E402

GROUP = "9日EMA（ラリー・ウィリアムズ）"
COINS = ("BTC", "ETH")
SPLIT = PF.SPLIT
SPOT_COST = 0.0015
SIZE = "損切りにかかると資金の0.5%（建玉は資金の1倍まで）"
FILTERS = ["なし", "20日平均が上向き", "50日平均の上", "200日平均の上"]
ENTRIES = [(0, "翌朝の始値"), (1, "高値を超えたら")]
EXITS = [(0, 0, "終値がEMAを下回ったら翌朝"), (1, 0, "損切りを前日の安値へ"), (2, 1, "利確1R"), (2, 2, "利確2R"), (2, 3, "利確3R"),
         (3, 0, "最初に利益が出た朝"), (4, 5, "5日後の朝")]


@njit
def engine(O, H, L, C, P, dH, dL, dC, ema, sig, side, entry_mode, exit_mode, xp, cost, swd, roll, cap):
    """日足の信号で入り、15分足で決済まで追う。返り値: 毎日の損益（資金の%）、取引ごとの 入った日・出た日・R・損切りまでの幅（価格比）。"""
    nd = dC.shape[0]
    N = nd * P
    pnl = np.zeros(nd)
    te = np.empty(nd, np.int64)
    tx = np.empty(nd, np.int64)
    tr = np.empty(nd)
    tq = np.empty(nd)
    nt = 0
    d = 1
    while d < nd:
        if not sig[d - 1]:
            d += 1
            continue
        stop = dL[d - 1] if side == 1 else dH[d - 1]
        b0 = d * P
        k = -1
        e = 0.0
        if entry_mode == 0:
            k = b0
            e = O[b0]
        else:
            lvl = dH[d - 1] if side == 1 else dL[d - 1]
            for j in range(b0, b0 + P):
                if side == 1:
                    if H[j] >= lvl:
                        k = j
                        e = max(O[j], lvl)
                        break
                else:
                    if L[j] <= lvl:
                        k = j
                        e = min(O[j], lvl)
                        break
        if k < 0:
            d += 1
            continue
        risk = side * (e - stop)
        if risk <= 0.0:
            d += 1
            continue
        w = min(1.0, risk / e / cap)                       # 建玉は資金の1倍まで
        scale = 0.5 * w
        cR = cost * e / risk
        sR = swd * e / risk
        tp = e + side * xp * risk
        pnl[d] -= cR * scale
        mark = 0.0
        nroll = 0
        held = 0
        x = 0.0
        out = -1
        j = k
        while j < N:
            dd = j // P
            pos = j - dd * P
            if j > k and pos == 0:                         # 新しい日の始まり
                held += 1
                pv = dd - 1
                hit = False
                if exit_mode == 0 and side * (dC[pv] - ema[pv]) < 0:
                    hit = True
                elif exit_mode == 3 and side * (O[j] - e) > 0:
                    hit = True
                elif exit_mode == 4 and held >= xp:
                    hit = True
                elif exit_mode == 5 and side * (ema[pv] - ema[pv - 1]) < 0:
                    hit = True
                if hit:
                    x = O[j]
                    out = dd
                    break
                if exit_mode == 1:
                    nl = dL[pv] if side == 1 else dH[pv]
                    if side * (nl - stop) > 0:
                        stop = nl
            if j > k and pos == roll:                      # スワップ
                nroll += 1
                pnl[dd] -= sR * scale
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
            if exit_mode == 2:
                fav = H[j] if side == 1 else L[j]
                if side * (fav - tp) >= 0:
                    if j == k:
                        x = tp
                    elif side == 1:
                        x = max(O[j], tp)
                    else:
                        x = min(O[j], tp)
                    out = dd
                    break
            if pos == P - 1:                               # 日の終わりに含み損益を計上
                m = side * (C[j] - e) / risk
                pnl[dd] += (m - mark) * scale
                mark = m
            j += 1
        if out < 0:
            break
        rx = side * (x - e) / risk
        pnl[out] += (rx - mark) * scale
        te[nt] = d
        tx[nt] = out
        tr[nt] = rx - cR - nroll * sR
        tq[nt] = risk / e
        nt += 1
        d = out + 1
    return pnl, te[:nt], tx[:nt], tr[:nt], tq[:nt]


def daily(D):
    M = D["M"]
    dC = M["close"][:, -1]
    s = pd.Series(dC)
    return dict(dO=M["open"][:, 0], dH=M["high"].max(1), dL=M["low"].min(1), dC=dC,
                ema=s.ewm(span=9, adjust=False).mean().to_numpy(), sma20=s.rolling(20).mean().to_numpy(),
                sma50=s.rolling(50).mean().to_numpy(), sma200=s.rolling(200).mean().to_numpy(),
                flat=tuple(np.ascontiguousarray(M[k].ravel()) for k in ("open", "high", "low", "close")))


def prev(a):
    return np.r_[np.nan, a[:-1]]


def signal(B, side, filt, kind="実体"):
    dO, dC, ema = B["dO"], B["dC"], B["ema"]
    with np.errstate(invalid="ignore"):
        if kind == "実体":
            s = (side * (ema - dO) > 0) & (side * (dC - ema) > 0)
        else:                                              # 9.1：EMAの向きが変わった足
            de = ema - prev(ema)
            s = (side * de > 0) & (side * prev(de) <= 0)
        if filt == "20日平均が上向き":
            s &= side * (B["sma20"] - prev(B["sma20"])) > 0
        elif filt == "50日平均の上":
            s &= side * (dC - B["sma50"]) > 0
        elif filt == "200日平均の上":
            s &= side * (dC - B["sma200"]) > 0
    return s.astype(np.bool_)


def run(D, B, filt, entry_mode, exit_mode, xp, kind="実体", spot=False, sides=(1, -1)):
    """側（1=買い、-1=売り）ごとに、BTC+ETHの毎日の損益（資金の%）と取引一覧。"""
    out = {}
    for side in sides:
        tot, trades = None, []
        for a in COINS:
            Ba, P = B[a], D[a]["per_day"]
            O, H, L, C = Ba["flat"]
            cost = SPOT_COST if spot else PF.SPEC[a][0]
            swd = 0.0 if spot else (PF.SPEC[a][1] if side == 1 else PF.SPEC[a][2]) / 36500
            pnl, te, tx, tr, tq = engine(O, H, L, C, P, Ba["dH"], Ba["dL"], Ba["dC"], Ba["ema"], signal(Ba, side, filt, kind),
                                         side, entry_mode, exit_mode, float(xp), cost, swd, EX.ROLL_UTC * P // 24, 0.005)
            s = pd.Series(pnl, index=D[a]["days"])
            tot = s if tot is None else tot.add(s, fill_value=0)
            days = D[a]["days"]
            trades.append(pd.DataFrame({"coin": a, "entry": days[te], "exit": days[tx], "R": tr, "q": tq}))
        T = pd.concat(trades, ignore_index=True)
        out[side] = (MM.clip(tot), T[(T.entry >= MM.START) & (T.entry <= MM.END)])
    return out


def tstats(T, x):
    yrs = max((x.index[-1] - x.index[0]).days / 365.25, 0.5)
    win = T.R[T.R > 0]
    return dict(n=len(T) / yrs, wr=len(win) / max(len(T), 1) * 100, aw=win.mean(), al=T.R[T.R <= 0].mean(),
                hold=(T.exit - T.entry).dt.days.mean(), q=T.q.median() * 100)


def fmt(r):
    return f"年{r['年平均%']:+6.2f}% 落込{r['最大の落ち込み%']:5.1f}% 安定度{r['安定度']:5.2f} 負け年{r['マイナスの年']:15s} 前半{r['前半の年平均%']:+5.1f}→後半{r['後半の年平均%']:+5.1f}"


def memo_of(st, extra=""):
    return (f"年{st['n']:.0f}回・勝率{st['wr']:.0f}%・平均の勝ち{st['aw']:+.2f}R・負け{st['al']:+.2f}R・平均保有{st['hold']:.1f}日・"
            f"損切りまでの幅（中央値）価格の{st['q']:.1f}%" + (f"。{extra}" if extra else ""))


def main(cache):
    print(f"（古い「{GROUP}」の行を {drop(GROUP)} 行消して、計算し直す）")
    D = {a: EX.day_matrix_n(os.path.join(cache, f"{a}_15m.csv"), 96) for a in COINS}
    B = {a: daily(D[a]) for a in COINS}

    print("==== 信号の数（2018年〜、BTC+ETH、年あたり）")
    for filt in FILTERS:
        n = {side: sum(int(signal(B[a], side, filt)[pd.DatetimeIndex(D[a]["days"]) >= MM.START].sum()) for a in COINS) for side in (1, -1)}
        print(f"  {filt:10s} 買い{n[1] / 8.9:5.1f}回 売り{n[-1] / 8.9:5.1f}回")

    print("\n==== 9日EMAを実体で抜けたら（買い＋売り、BTC+ETH、Vantage CFD）")
    res = {}
    for em, ename in ENTRIES:
        for filt in FILTERS:
            print(f"  -- 入り方：{ename}／トレンドの確認：{filt}")
            for xm, xp, xname in EXITS:
                S = run(D, B, filt, em, xm, xp)
                x = S[1][0].add(S[-1][0], fill_value=0)
                T = pd.concat([S[1][1], S[-1][1]])
                st = tstats(T, x)
                r = metrics_row(x, st["n"])
                r1, r2 = metrics_row(S[1][0]), metrics_row(S[-1][0])
                h1 = metrics_row(x[x.index < SPLIT])
                res[(ename, filt, xname)] = dict(r=r, st=st, r1=r1, r2=r2, h1=h1, x=x, S=S)
                print(f"    {xname:16s} {fmt(r)} ｜年{st['n']:3.0f}回 勝率{st['wr']:3.0f}% 保有{st['hold']:4.1f}日"
                      f" ｜買いだけ{r1['年平均%']:+5.1f}%（負け年{r1['マイナスの年']}） 売りだけ{r2['年平均%']:+5.1f}%")
                record(グループ=GROUP, 名前=f"{ename}・確認{filt}・{xname}（買い＋売り）", 銘柄="BTC,ETH", 時間足="日足（約定は15分足）",
                       **{"1回の量": SIZE}, メモ=memo_of(st, f"買いだけ年{r1['年平均%']:+.1f}%、売りだけ年{r2['年平均%']:+.1f}%"), **r)

    print("\n==== まとめ（56通り）")
    vals = list(res.values())
    both = sum(1 for v in vals if v["r"]["前半の年平均%"] > 0 and v["r"]["後半の年平均%"] > 0)
    nolose = sum(1 for v in vals if v["r"]["マイナスの年"] == "なし")
    print(f"  前半・後半ともプラス {both}通り ／ マイナスの年なし {nolose}通り ／ 年平均の中央値 {np.median([v['r']['年平均%'] for v in vals]):+.2f}%"
          f" ／ 買いだけの年平均の中央値 {np.median([v['r1']['年平均%'] for v in vals]):+.2f}% ／ 売りだけ {np.median([v['r2']['年平均%'] for v in vals]):+.2f}%")
    for key in ("入り方", "確認", "決済"):
        i = {"入り方": 0, "確認": 1, "決済": 2}[key]
        groups = {}
        for k, v in res.items():
            groups.setdefault(k[i], []).append(v["r"]["年平均%"])
        print(f"  {key}ごとの年平均の中央値: " + " ／ ".join(f"{g} {np.median(v):+.2f}%" for g, v in groups.items()))
    top = sorted(res.items(), key=lambda kv: -kv[1]["h1"]["安定度"])[:5]
    print("  前半（2018〜2022年）の安定度が高かった5つ → 後半（2023年〜）")
    for k, v in top:
        print(f"    {'・'.join(k):40s} 前半の安定度{v['h1']['安定度']:5.2f} 前半{v['r']['前半の年平均%']:+5.1f}% → 後半{v['r']['後半の年平均%']:+5.1f}%"
              f"（後半だけの安定度{metrics_row(v['x'][v['x'].index >= SPLIT])['安定度']:5.2f}）")

    print("\n==== 参考：「9.1」型（EMAが向きを変えた足の高値を超えたら入り、EMAが逆を向いたら翌朝決済）")
    for filt in FILTERS:
        S = run(D, B, filt, 1, 5, 0, kind="9.1")
        x = S[1][0].add(S[-1][0], fill_value=0)
        T = pd.concat([S[1][1], S[-1][1]])
        st = tstats(T, x)
        r = metrics_row(x, st["n"])
        r1, r2 = metrics_row(S[1][0]), metrics_row(S[-1][0])
        print(f"  確認{filt:10s} {fmt(r)} ｜年{st['n']:3.0f}回 勝率{st['wr']:3.0f}% 保有{st['hold']:4.1f}日 ｜買いだけ{r1['年平均%']:+5.1f}% 売りだけ{r2['年平均%']:+5.1f}%")
        record(グループ=GROUP, 名前=f"9.1型・確認{filt}（買い＋売り）", 銘柄="BTC,ETH", 時間足="日足（約定は15分足）", **{"1回の量": SIZE},
               メモ=memo_of(st, f"買いだけ年{r1['年平均%']:+.1f}%、売りだけ年{r2['年平均%']:+.1f}%"), **r)

    print("\n==== 参考：国内現物（買いだけ、往復0.15%、スワップなし）")
    for ename, filt, xname in [("翌朝の始値", "200日平均の上", "終値がEMAを下回ったら翌朝"), ("翌朝の始値", "200日平均の上", "損切りを前日の安値へ"),
                               ("翌朝の始値", "なし", "終値がEMAを下回ったら翌朝"), ("高値を超えたら", "200日平均の上", "利確2R")]:
        em = dict((n, m) for m, n in ENTRIES)[ename]
        xm, xp = [(m, p) for m, p, n in EXITS if n == xname][0]
        S = run(D, B, filt, em, xm, xp, spot=True, sides=(1,))
        x, T = S[1]
        st = tstats(T, x)
        r = metrics_row(x, st["n"])
        print(f"  {ename}・確認{filt}・{xname:16s} {fmt(r)} ｜年{st['n']:3.0f}回 勝率{st['wr']:3.0f}%")
        record(グループ=GROUP,名前=f"国内現物・{ename}・確認{filt}・{xname}（買いだけ）", 銘柄="BTC,ETH", 時間足="日足（約定は15分足）",
               コストの前提="国内現物（往復0.15%・スワップなし）", **{"1回の量": SIZE}, メモ=memo_of(st), **r)

    print("\n==== 値幅ブレイク（第一候補）との関係")
    v, _ = EX.run_plan(D, "そのまま")
    rv = metrics_row(v)
    print(f"  値幅ブレイク（①＋②×0.5）        {fmt(rv)}")
    picks = [top[0][0], ("翌朝の始値", "200日平均の上", "終値がEMAを下回ったら翌朝"), ("翌朝の始値", "なし", "終値がEMAを下回ったら翌朝")]
    seen = set()
    for k in picks:
        if k in seen:
            continue
        seen.add(k)
        e9 = res[k]["x"]
        corr = v.resample("ME").sum().corr(e9.resample("ME").sum())
        print(f"  9日EMA：{'・'.join(k)} ｜月ごとの損益の相関 {corr:+.2f}")
        for wgt in (0.5, 1.0):
            x = v.add(wgt * e9, fill_value=0)
            r = metrics_row(x)
            print(f"    値幅ブレイク＋9日EMA×{wgt:g}        {fmt(r)}")
            record(グループ=GROUP, 名前=f"値幅ブレイク＋9日EMA×{wgt:g}（{'・'.join(k)}）", 銘柄="BTC,ETH", 時間足="日足（約定は15分足）",
                   メモ=f"値幅ブレイク（①＋②×0.5、1回0.5%）に9日EMA（買い＋売り）を足す。月ごとの損益の相関{corr:+.2f}", **r)


if __name__ == "__main__" and len(sys.argv) == 2:
    main(sys.argv[1])


def custom(B, span, ma):
    """EMAの日数と、トレンド確認の平均の日数を変えた B（「50日平均の上」の枠に入れる）。"""
    out = {}
    for a, b in B.items():
        s = pd.Series(b["dC"])
        out[a] = dict(b, ema=s.ewm(span=span, adjust=False).mean().to_numpy(), sma50=s.rolling(ma).mean().to_numpy())
    return out


def streak(T):
    """連続で負けた最長回数。T は取引一覧（DataFrame）か、取引ごとのR（Series）。"""
    R = T.sort_values("exit").R if isinstance(T, pd.DataFrame) else T.sort_index()
    run = best = 0
    for r in R:
        run = run + 1 if r <= 0 else 0
        best = max(best, run)
    return best


def check(cache):
    D = {a: EX.day_matrix_n(os.path.join(cache, f"{a}_15m.csv"), 96) for a in COINS}
    B = {a: daily(D[a]) for a in COINS}
    v, Tv = EX.run_plan(D, "そのまま")

    print("==== 設定ずらし（翌朝の始値で入る・終値がEMAを下回ったら翌朝決済・買い＋売り）")
    print("     セル＝年平均％／安定度／負け年の数")
    spans, mas = (5, 7, 9, 12, 15, 20), (30, 50, 75, 100, 150, 200)
    grid, cells = {}, []
    for span in spans:
        for ma in mas:
            Bc = custom(B, span, ma)
            S = run(D, Bc, "50日平均の上", 0, 0, 0)
            x = S[1][0].add(S[-1][0], fill_value=0)
            r = metrics_row(x)
            nneg = 0 if r["マイナスの年"] == "なし" else len(r["マイナスの年"].split(","))
            grid.setdefault(f"EMA{span}", {})[f"{ma}日平均"] = f"{r['年平均%']:+5.1f}/{r['安定度']:.2f}/{nneg}"
            cells.append((span, ma, r, nneg))
    print(pd.DataFrame(grid).T.to_string())
    print(f"  → 36通り中 負け年なし {sum(1 for c in cells if c[3] == 0)}通り、負け年1つ {sum(1 for c in cells if c[3] == 1)}通り、"
          f"安定度の中央値 {np.median([c[2]['安定度'] for c in cells]):.2f}、後半もプラス {sum(1 for c in cells if c[2]['後半の年平均%'] > 0)}通り")

    print("\n==== 一番良かった形（EMA9・50日平均・翌朝の始値・終値がEMAを下回ったら翌朝）の中身")
    S = run(D, B, "50日平均の上", 0, 0, 0)
    x = S[1][0].add(S[-1][0], fill_value=0)
    T = pd.concat([S[1][1], S[-1][1]]).sort_values("entry")
    st = tstats(T, x)
    w = T.R[T.R > 0].sort_values(ascending=False)
    gross = T.R.sum()
    print(f"  取引{len(T)}回（年{st['n']:.0f}回）、勝率{st['wr']:.0f}%、平均の勝ち{st['aw']:+.2f}R、平均の負け{st['al']:+.2f}R、一番大きな勝ち{w.iloc[0]:+.1f}R、"
          f"平均保有{st['hold']:.1f}日、損切りまでの幅（中央値）価格の{st['q']:.1f}%")
    print(f"  合計{gross:+.0f}Rのうち、上位5回の勝ちで{w.iloc[:5].sum():+.0f}R（{w.iloc[:5].sum() / gross * 100:.0f}%）、上位20回で{w.iloc[:20].sum() / gross * 100:.0f}%")
    print(f"  連続で負けた最長回数 {streak(T)}回（値幅ブレイクは {streak(Tv)}回）")
    for a in COINS:
        Ta = T[T.coin == a]
        print(f"  {a}: {len(Ta)}回 勝率{(Ta.R > 0).mean() * 100:.0f}% 合計{Ta.R.sum():+.0f}R")
    big = T.sort_values("R", ascending=False).head(5)
    print("  大きく勝った取引: " + " ／ ".join(f"{r.coin} {r.entry:%Y-%m-%d}〜{r.exit:%m-%d} {r.R:+.1f}R" for r in big.itertuples()))

    print("\n==== 年ごとの損益（%）")
    nb = run(D, B, "なし", 0, 0, 0)
    xn = nb[1][0].add(nb[-1][0], fill_value=0)
    rows = {"値幅ブレイク": v, "9日EMA（50日平均）": x, "9日EMA（買いだけ）": S[1][0], "9日EMA（売りだけ）": S[-1][0],
            "9日EMA（確認なし）": xn, "値幅ブレイク＋9日EMA×0.5": v.add(0.5 * x, fill_value=0), "値幅ブレイク×2": 2 * v}
    print(pd.DataFrame({k: s.groupby(s.index.year).sum() for k, s in rows.items()}).round(1).T.to_string())
    for k in ("値幅ブレイク", "9日EMA（50日平均）", "値幅ブレイク＋9日EMA×0.5", "値幅ブレイク×2"):
        s = rows[k]
        r, h2 = metrics_row(s), metrics_row(s[s.index >= SPLIT])
        print(f"  {k:18s} {fmt(r)} ｜後半だけの安定度{h2['安定度']:5.2f}・後半の落ち込み{h2['最大の落ち込み%']:.1f}%")


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "check":
    check(sys.argv[1])
