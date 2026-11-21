#!/usr/bin/env python3
"""Zennの記事「Claude Codeで仮想通貨自動売買Botを作った話」にある7つの売買戦略を、検証できる範囲で確かめる（BTC・ETH、2018年〜）。

記事に書かれている戦略（パラメータや成績の記載はない）:
  1 ルールベース       : 移動平均クロス・RSI（過熱判定）・MACD の多数決。3つ中2つが同じ向きなら売買、割れたら何もしない
  2 グリッド           : 価格帯を分割して自動売買
  3 DCA                : 一定間隔で買い続ける
  4 AI/LLM             : LLM（Groq）に市場データを渡して判定        → 過去での検証はできない
  5 裁定取引           : 取引所間の価格差が閾値を超えたら実行      → 複数の取引所の同時刻の価格・板がないので検証できない
  6 スキャルピング     : 板の厚みを見て即時売買                    → 板の履歴がないので検証できない
  7 マーケットメイキング: 買い・売り両方に指値                      → 板と約定の順番のデータがないので検証できない
記事にない部分は、よく使われる値で決める。
  1: 移動平均 5/25・10/30・20/50本、RSI14（30以下で買い・70以上で売り）、MACD 12/26/9（MACDがシグナルより上で買い）。
     数え方は「状態」（短い平均が長い平均の上なら買い…）と「クロスした瞬間」の2通り。足は15分・1時間・4時間・日足。
     記事の安全装置「同じ向きのシグナルが2回続いたら発注」あり／なし。
     国内現物（買いだけ、往復0.15%）と Vantage CFD（買い＋売り、往復0.05%＋買いのスワップ）。
     量は ATR（日足14日）1本分の値動き＝資金の0.5%（単利）。
  2: 現物のグリッド。張った時の価格±10%・±20%を10・20に分け、下の線で買い・1つ上の線で売る。
     始めに上半分の区間ぶんを成行で買う。週ごと・月ごとに全部売って張り直す。資金はすべてグリッドに使う（単利）。
     手数料は、海外取引所（指値・成行とも片道0.1%）と、SBI VCトレード（指値−0.01%、張り直しの成行0.05%）。
     口座の増減（複利）で評価し、「資金の半分でコインを買って持っていただけ」と比べる。
  3: 毎週月曜（UTC0時）に同じ金額で買う（手数料0.1%）。始めた年 2018・2020・2022・2024年で比べる。
使い方: python3 project/tools/article7_lab.py <キャッシュ（BTC_15m.csv, ETH_15m.csv）>
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

COINS = ("BTC", "ETH")
G1, G2, G3 = "記事の戦略：多数決（移動平均・RSI・MACD）", "記事の戦略：グリッド（現物）", "記事の戦略：DCA（積立）"
TF = [("15分足", 1), ("1時間足", 4), ("4時間足", 16), ("日足", 96)]
MAS = [(5, 25), (10, 30), (20, 50)]
SPOT_RT = 0.0015
SIZE = "ATR（日足）1本分の値動き＝資金の0.5%（単利）"


# ---------------------------------------------------------------- 1 ルールベース（多数決）
def bars(D, k):
    M = D["M"]
    O = M["open"].ravel().reshape(-1, k)[:, 0]
    H = M["high"].ravel().reshape(-1, k).max(1)
    L = M["low"].ravel().reshape(-1, k).min(1)
    C = M["close"].ravel().reshape(-1, k)[:, -1]
    n = len(O)
    t = pd.DatetimeIndex(D["days"][0] + pd.to_timedelta(np.arange(n) * k * 15, unit="min"))
    return O, H, L, C, t, np.arange(n) * k // 96


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0])
    up = pd.Series(np.clip(d, 0, None)).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    dn = pd.Series(np.clip(-d, 0, None)).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100 - 100 / (1 + up / dn)


def majority(C, ms, ml, kind="状態"):
    """1=買い、-1=売り、0=何もしない（意見が割れた）。"""
    s = pd.Series(C)
    ma = np.sign((s.rolling(ms).mean() - s.rolling(ml).mean()).to_numpy())
    m = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
    macd = np.sign((m - m.ewm(span=9, adjust=False).mean()).to_numpy())
    with np.errstate(invalid="ignore"):
        if kind == "クロスの瞬間":
            ma = np.where(np.r_[np.nan, ma[:-1]] * ma < 0, ma, 0)
            macd = np.where(np.r_[np.nan, macd[:-1]] * macd < 0, macd, 0)
        r = rsi(C)
        rs = np.where(r < 30, 1, np.where(r > 70, -1, 0))
    buy = (ma == 1).astype(int) + (rs == 1) + (macd == 1)
    sell = (ma == -1).astype(int) + (rs == -1) + (macd == -1)
    return np.where(buy >= 2, 1, np.where(sell >= 2, -1, 0))


def positions(dec, spot, confirm):
    if confirm:                                            # 同じ向きのシグナルが2回続いたら発注
        dec = np.where((dec != 0) & (dec == np.r_[0, dec[:-1]]), dec, 0)
    if spot:
        x = np.where(dec == 1, 1.0, np.where(dec == -1, 0.0, np.nan))
    else:
        x = np.where(dec != 0, dec.astype(float), np.nan)
    return pd.Series(x).ffill().fillna(0.0).to_numpy()


def rule_run(D, a, k, ms, ml, kind, spot, confirm):
    O, H, L, C, t, day = bars(D, k)
    pos = positions(majority(C, ms, ml, kind), spot, confirm)
    with np.errstate(divide="ignore", invalid="ignore"):
        f_day = np.nan_to_num(0.005 / (D["atr"] / D["dO"]))
    f = f_day[day]
    cost_rt = SPOT_RT if spot else PF.SPEC[a][0]
    sw = 0.0 if spot else PF.SPEC[a][1] / 36500
    r = np.r_[0.0, C[1:] / C[:-1] - 1]
    held = np.r_[0.0, pos[:-1]]                            # 前の足の終わりに決めた建玉
    change = np.abs(np.diff(np.r_[0.0, held]))
    gross = held * r * f * 100
    mins = np.asarray(t.hour * 60 + t.minute)
    roll = (mins <= EX.ROLL_UTC * 60) & (EX.ROLL_UTC * 60 < mins + k * 15)
    swap = np.where(roll & (held > 0), sw * f * 100, 0.0)
    bar = gross - change * cost_rt / 2 * f * 100 - swap
    daily = pd.Series(bar, index=t).groupby(t.normalize()).sum()
    seg = np.cumsum(change > 0)
    on = held != 0
    tr = pd.DataFrame({"seg": seg[on], "g": (gross - swap)[on], "f": f[on], "t": t[on]}).groupby("seg").agg(
        g=("g", "sum"), f=("f", "first"), t=("t", "first"))
    tr["net"] = tr.g - cost_rt * tr.f * 100
    tr = tr[(tr.t >= MM.START) & (tr.t <= MM.END)]
    return MM.clip(daily), tr


def rules(D):
    print("==== 1 ルールベース：移動平均・RSI・MACDの多数決（BTC+ETH）")
    out = {}
    for kind in ("状態", "クロスの瞬間"):
        for spot in (True, False):
            mode = "国内現物（買いだけ）" if spot else "CFD（買い＋売り）"
            print(f"  -- 数え方：{kind}／{mode}")
            for tf, k in TF:
                for ms, ml in MAS:
                    for confirm in (False, True):
                        x, trs = None, []
                        for a in COINS:
                            s, tr = rule_run(D[a], a, k, ms, ml, kind, spot, confirm)
                            x = s if x is None else x.add(s, fill_value=0)
                            trs.append(tr)
                        T = pd.concat(trs).sort_values("t")
                        yrs = (x.index[-1] - x.index[0]).days / 365.25
                        run, hits = 0, 0
                        for v in T.net:
                            run = run + 1 if v <= 0 else 0
                            hits += run == 5
                        stops = hits / yrs
                        n, wr = len(T) / yrs, (T.net > 0).mean() * 100 if len(T) else 0
                        r = metrics_row(x, n)
                        name = f"{tf}・移動平均{ms}/{ml}・{'2回続いたら発注' if confirm else 'すぐ発注'}・{mode}"
                        out[(kind, spot, tf, ms, confirm)] = r
                        print(f"    {tf:5s} 平均{ms:2d}/{ml:2d} {'2回確認' if confirm else '確認なし'} 年{r['年平均%']:+7.2f}% 落込{r['最大の落ち込み%']:6.1f}%"
                              f" 安定度{r['安定度']:5.2f} 負け年{r['マイナスの年']:24s} ｜年{n:5.0f}回 勝率{wr:3.0f}% 5連敗{stops:4.0f}回/年")
                        if kind == "状態" or (ms, ml) == (5, 25):
                            record(グループ=G1, 名前=f"{name}（{kind}で数える）", 銘柄="BTC,ETH", 時間足=tf,
                                   コストの前提="国内現物（往復0.15%・スワップなし）" if spot else "Vantage CFD（往復0.05%・買いスワップ年25〜35%・売り0%）",
                                   **{"1回の量": SIZE}, メモ=f"年{n:.0f}回・勝率{wr:.0f}%・5連敗が年{stops:.0f}回（記事の安全装置なら、そのたびに停止）", **r)
    print("\n  まとめ（年平均の中央値、移動平均3通り×確認あり／なし）")
    for kind in ("状態", "クロスの瞬間"):
        for spot in (True, False):
            cells = []
            for tf, _ in TF:
                v = [out[(kind, spot, tf, ms, c)]["年平均%"] for ms, _ in MAS for c in (False, True)]
                nl = sum(1 for ms, _ in MAS for c in (False, True) if out[(kind, spot, tf, ms, c)]["マイナスの年"] == "なし")
                cells.append(f"{tf} {np.median(v):+.2f}%（負け年なし{nl}/6）")
            print(f"    {kind}・{'国内現物' if spot else 'CFD'}: " + " ／ ".join(cells))
    return out


# ---------------------------------------------------------------- 2 グリッド（現物）
@njit
def grid_engine(O, H, L, C, reset, r, N, fm, ft, compound):
    """資金1でグリッドを張り、reset の足の始まりで全部売って（成行）張り直す。各足の終値での評価損益を返す。
    compound=False: 張り直すたびに資金1で始める（単利の累計）。True: その時の資金をすべて使う（複利、口座の増減そのもの）。"""
    n = O.shape[0]
    out = np.empty(n)
    cash = np.zeros(N)
    coin = np.zeros(N)
    lv = np.zeros(N + 1)
    done = 0.0                                             # 張り直しまでに確定した損益の合計（単利）
    eq = 1.0                                               # 張り直す時の資金（複利）
    for t in range(n):
        if t == 0 or reset[t]:
            if t > 0:
                now = cash.sum() + coin.sum() * O[t] * (1 - ft)
                if compound:
                    eq = now
                else:
                    done += now - 1.0
            p0 = O[t]
            for i in range(N + 1):
                lv[i] = p0 * (1 - r) + 2 * r * p0 * i / N
            for i in range(N):
                if lv[i] >= p0:                            # 上側の区間：始めに成行で買い、1つ上の線で売る
                    coin[i] = (eq / N) / p0 * (1 - ft)
                    cash[i] = 0.0
                else:                                      # 下側の区間：下の線で買う
                    coin[i] = 0.0
                    cash[i] = eq / N
        for i in range(N):
            if coin[i] > 0:
                if H[t] >= lv[i + 1]:
                    px = max(O[t], lv[i + 1])
                    cash[i] = coin[i] * px * (1 - fm)
                    coin[i] = 0.0
            elif L[t] <= lv[i]:
                px = min(O[t], lv[i])
                coin[i] = cash[i] / px * (1 - fm)
                cash[i] = 0.0
        out[t] = done + cash.sum() + coin.sum() * C[t] - 1.0
    return out


def compound_stats(v):
    """口座の評価額（1から始まる日々の値）から、年率・最大の落ち込み・マイナスの年。"""
    yrs = (v.index[-1] - v.index[0]).days / 365.25
    ye = v.groupby(v.index.year).last()
    yr = ye / np.r_[1.0, ye.to_numpy()[:-1]] - 1
    neg = [str(y) for y, x in yr.items() if x < 0]
    return dict(cagr=(v.iloc[-1] ** (1 / yrs) - 1) * 100, mdd=((v.cummax() - v) / v.cummax()).max() * 100,
                neg=",".join(neg) or "なし", worst=yr.min() * 100)


def grid(D):
    print("\n==== 2 グリッド（現物、資金をすべてグリッドに使う。BTCとETHに半分ずつ）")
    print("     口座の増減（複利）で表示。比べる相手：資金の半分でコインを買って持っていただけの場合")
    fees = [("海外取引所（片道0.1%）", 0.001, 0.001), ("SBI VCトレード（指値−0.01%・成行0.05%）", -0.0001, 0.0005)]
    px = {}
    for a in COINS:
        O, H, L, C, t, day = bars(D[a], 1)
        m = np.asarray((t >= MM.START) & (t < MM.END + pd.Timedelta(days=1)))
        px[a] = (O[m], H[m], L[m], C[m], t[m])
    hold = 0.5 * sum(0.5 + 0.5 * pd.Series(px[a][3], index=px[a][4]).groupby(px[a][4].normalize()).last() / px[a][0][0] for a in COINS)
    hs = compound_stats(hold)
    print(f"  資金の半分を持っていただけ: 年率{hs['cagr']:+6.1f}% 口座の最大の落ち込み{hs['mdd']:4.0f}% マイナスの年{hs['neg']}")
    rows = []
    for fname, fm, ft in fees:
        print(f"  -- 手数料：{fname}")
        for every in ("週", "月"):
            for r in (0.10, 0.20):
                for N in (10, 20):
                    si, cp = {}, {}
                    for a in COINS:
                        O, H, L, C, t = px[a]
                        key = t.to_period("W") if every == "週" else t.to_period("M")
                        reset = np.r_[True, np.asarray(key[1:] != key[:-1])]
                        v = pd.Series(grid_engine(O, H, L, C, reset, r, N, fm, ft, False) * 100, index=t).groupby(t.normalize()).last()
                        si[a] = v.diff().fillna(v.iloc[0])
                        cp[a] = 1 + pd.Series(grid_engine(O, H, L, C, reset, r, N, fm, ft, True), index=t).groupby(t.normalize()).last()
                    both = 0.5 * si["BTC"].add(si["ETH"], fill_value=0)
                    acc = 0.5 * cp["BTC"] + 0.5 * cp["ETH"]
                    cs = compound_stats(acc)
                    m = metrics_row(both)
                    memo = (f"口座の増減（複利）：年率{cs['cagr']:+.1f}%・最大の落ち込み{cs['mdd']:.0f}%・マイナスの年{cs['neg']}。"
                            f"資金の半分を持っていただけなら年率{hs['cagr']:+.1f}%・落ち込み{hs['mdd']:.0f}%。標準の列は単利（張り直しごとに資金を元に戻す）")
                    print(f"    {every}ごと ±{r:.0%} {N:2d}分割（間隔{2 * r / N:.0%}） 年率{cs['cagr']:+6.1f}% 口座の最大の落ち込み{cs['mdd']:4.0f}%"
                          f" 最悪の年{cs['worst']:+5.0f}% マイナスの年{cs['neg']:20s} ｜BTCだけ{compound_stats(cp['BTC'])['cagr']:+6.1f}% ETHだけ{compound_stats(cp['ETH'])['cagr']:+6.1f}%")
                    record(グループ=G2, 名前=f"{every}ごとに張り直し・±{r:.0%}・{N}分割・{fname}", 銘柄="BTC,ETH", 時間足="15分足",
                           コストの前提=fname, **{"1回の量": "資金をすべてグリッドに使う（BTC・ETHに半分ずつ）"}, メモ=memo, **m)
                    rows.append(cs)
    return rows


# ---------------------------------------------------------------- 3 DCA（積立）
def irr(flows, when, value, end):
    yrs = np.array([(end - w).days / 365.25 for w in when])
    lo, hi = -0.99, 5.0
    for _ in range(200):
        mid = (lo + hi) / 2
        fv = sum(f * (1 + mid) ** y for f, y in zip(flows, yrs))
        lo, hi = (mid, hi) if fv < value else (lo, mid)
    return mid


def dca(D):
    print("\n==== 3 DCA（毎週月曜に同じ金額で買う、手数料0.1%、2026年11月16日の終値で評価）")
    end = MM.END
    for start in (2018, 2020, 2022, 2024):
        for label, w in (("BTC", {"BTC": 1.0}), ("ETH", {"ETH": 1.0}), ("BTC・ETH半分ずつ", {"BTC": 0.5, "ETH": 0.5})):
            val, when = None, []
            for a, wa in w.items():                           # 毎週1ずつ、銘柄の割合で分けて買う
                days = pd.DatetimeIndex(D[a]["days"])
                o, c = D[a]["dO"], D[a]["M"]["close"][:, -1]
                buy = (days.weekday == 0) & (days >= pd.Timestamp(f"{start}-01-01")) & (days <= end)
                qty = np.cumsum(np.where(buy, wa / o * (1 - 0.001), 0.0))
                v = pd.Series(qty * c, index=days)
                val = v if val is None else val.add(v, fill_value=0)
                when = list(days[buy])
            flows = [1.0] * len(when)
            inv = pd.Series(np.cumsum([1.0] * len(when)), index=when).reindex(val.index).ffill().fillna(0)
            val, inv = val[(val.index >= pd.Timestamp(f"{start}-01-01")) & (val.index <= end)], inv[(inv.index >= pd.Timestamp(f"{start}-01-01")) & (inv.index <= end)]
            gain = val - inv
            peak = val.cummax()
            dd = ((peak - val) / peak).max() * 100
            under = (gain < 0).mean() * 100
            worst = (gain / inv).min() * 100
            ann = irr(flows, when, val.iloc[-1], end) * 100
            memo = (f"{start}年1月から毎週同じ金額（{len(when)}回）。投資額に対し{(val.iloc[-1] / inv.iloc[-1] - 1) * 100:+.0f}%、"
                    f"年率{ann:+.1f}%（お金を入れた時期を考えた利回り）、評価額の高値からの最大の落ち込み{dd:.0f}%、"
                    f"含み損だった期間{under:.0f}%、一番悪い時の損益{worst:+.0f}%")
            print(f"  {start}年〜 {label:10s} 投資額に対し{(val.iloc[-1] / inv.iloc[-1] - 1) * 100:+5.0f}% 年率{ann:+6.1f}%"
                  f" 評価額の最大の落ち込み{dd:3.0f}% 含み損の期間{under:3.0f}% 一番悪い時{worst:+4.0f}%")
            record(グループ=G3, 名前=f"{start}年から毎週・{label}", 銘柄=label.replace("・半分ずつ", ""), 時間足="日足",
                   期間=f"{start}-01〜2026-11", コストの前提="手数料0.1%", **{"1回の量": "毎週同じ金額"}, メモ=memo)


def main(cache):
    for g in (G1, G2, G3):
        drop(g)
    D = {a: EX.day_matrix_n(os.path.join(cache, f"{a}_15m.csv"), 96) for a in COINS}
    rules(D)
    grid(D)
    dca(D)
    v, _ = EX.run_plan(D, "そのまま")
    print("\n参考：値幅ブレイク（第一候補） " + " ".join(f"{k}{metrics_row(v)[k]}" for k in ("年平均%", "最大の落ち込み%", "安定度", "マイナスの年")))


if __name__ == "__main__":
    main(sys.argv[1])
