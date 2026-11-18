#!/usr/bin/env python3
"""「見栄えの良いバックテスト」はどう作られるかの実演（BTC 1時間足、2018年〜）。

1. 432通りのルール（順張り/逆張り × 買い/売り × 期間・損切り・利確・保有時間）を全期間で試し、
   一番成績が良いものを選ぶ（＝全期間で最適化。EAの宣伝でよくある作り方）。
2. 1回の取引で資金の何%をリスクにさらすかを決め、複利で100万円から運用した結果を出す。
3. 同じルールに、手数料（スプレッド往復0.1%）とスワップ（買い年25%＝Vantageの表示）を入れる。
4. 前半（2018〜2022年）だけで選び、後半（2023年〜）で確かめる。
5. 条件（取引する時間帯）を1つ足し、全期間の結果を見て選ぶと、見た目がどこまで良くなるか。
使い方: python3 project/tools/backtest_demo.py <キャッシュ用ディレクトリ（BTC_1h.csv があるところ）>
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
import shortterm_lab as T  # noqa: E402

START, SPLIT = "2018-01-01", "2023-01-01"
SWAP_LONG = 25.0     # 年率%（Vantage BTCUSD 買い）
SPREAD = 0.001       # 往復0.1%


def load(cache):
    df = pd.read_csv(os.path.join(cache, "BTC_1h.csv"), index_col=0, parse_dates=True)
    return T.prep(df[df.index >= START])


def trades(P, fam, side, N, M, s, tR, H, cost=0.0, swap=0.0):
    tr = T.run(P, T.make_signal(P, fam, side, N, M), side, s, tR, H, cost)
    if len(tr) and swap and side == 1:
        tr["R"] = tr.R - swap / 36500 * tr.nights * tr.lev
    return tr


def equity(tr, f, cap=1_000_000):
    eq, peak, dd, gain, loss, streak, maxl = cap, cap, 0.0, 0.0, 0.0, 0, 0
    for r in tr.R.to_numpy():
        pnl = eq * f * r
        eq += pnl
        if pnl > 0:
            gain += pnl
            streak = 0
        else:
            loss -= pnl
            streak += 1
            maxl = max(maxl, streak)
        peak = max(peak, eq)
        dd = max(dd, 1 - eq / peak)
    yrs = (tr.t_out.iloc[-1] - tr.t_in.iloc[0]).days / 365.25
    return dict(純利益=eq - cap, PF=gain / loss if loss else np.inf, 最大DD=dd, 取引数=len(tr), 勝率=(tr.R > 0).mean(),
                最大連敗=maxl, 年率=(eq / cap) ** (1 / yrs) - 1 if eq > 0 else -1)


def pick_risk(tr, dd_cap=0.18):
    best = 0.0025
    for f in np.arange(0.0025, 0.0501, 0.0025):
        if equity(tr, f)["最大DD"] <= dd_cap:
            best = f
    return best


def show(lab, tr, f):
    e = equity(tr, f)
    print(f"{lab}: 1回のリスク{f*100:.2f}% ｜ 純利益 {e['純利益']:+,.0f}円 ｜ PF {e['PF']:.2f} ｜ 最大DD {e['最大DD']*100:.1f}% ｜ "
          f"取引{e['取引数']}回 ｜ 勝率{e['勝率']*100:.1f}% ｜ 最大{e['最大連敗']}連敗 ｜ 年率{e['年率']*100:+.1f}%")


def pf(tr):
    g, l = tr.R[tr.R > 0].sum(), -tr.R[tr.R < 0].sum()
    return g / l if l else np.inf


def main(cache):
    P = load(cache)
    grid = list(itertools.product(("bo", "mr"), (1, -1), (24, 72), (0, 480, 4800), (1, 2, 3), (1, 2, 3), (24, 48)))
    rows = []
    for g in grid:
        tr = trades(P, *g)
        if len(tr) < 500:
            continue
        sp = tr.t_in < pd.Timestamp(SPLIT)
        rows.append(dict(g=g, n=len(tr), pf_all=pf(tr), pf_in=pf(tr[sp]), pf_out=pf(tr[~sp])))
    R = pd.DataFrame(rows)
    print(f"試したルール {len(grid)}通り（取引500回以上: {len(R)}通り）")
    print(f"全期間・手数料なしのPF: 中央値 {R.pf_all.median():.2f} ／ 1.5超 {int((R.pf_all > 1.5).sum())}通り ／ 最大 {R.pf_all.max():.2f}")

    best = R.sort_values("pf_all", ascending=False).iloc[0]
    g = best.g
    lab = f"{'順張り' if g[0] == 'bo' else '逆張り'}{'買い' if g[1] == 1 else '売り'}・{g[2]}時間・トレンド{g[3]}本・損切りATR×{g[4]}・利確{g[5]}R・最大{g[6]}時間"
    print(f"\n【1】全期間で一番良かったルール: {lab}")
    tr0 = trades(P, *g)
    f = pick_risk(tr0)
    show(" 手数料・スワップなし", tr0, f)
    tr1 = trades(P, *g, cost=SPREAD, swap=SWAP_LONG)
    show(" 手数料・スワップあり", tr1, f)

    b_in = R.sort_values("pf_in", ascending=False).iloc[0]
    g2 = b_in.g
    print(f"\n【2】前半（2018〜2022年）だけで一番良かったルール: PF 前半{b_in.pf_in:.2f} → 後半{b_in.pf_out:.2f}（手数料なし）")
    tr2 = trades(P, *g2, cost=SPREAD, swap=SWAP_LONG)
    tr2 = tr2[tr2.t_in >= pd.Timestamp(SPLIT)]
    show(" 後半（2023年〜）・手数料・スワップあり", tr2, pick_risk(trades(P, *g2)))
    top = R.sort_values("pf_in", ascending=False).head(20)
    print(f" 前半の上位20通りの後半PF: 中央値 {top.pf_out.median():.2f}（1未満 {int((top.pf_out < 1).sum())}通り）")
    hour_filter_demo(P, g)


def hour_filter_demo(P, g):
    """条件を1つ足すだけ（取引する時間帯を選ぶ）で、見た目の成績がどう変わるか。"""
    tr = trades(P, *g)
    tr["hour"] = pd.DatetimeIndex(tr.t_in).hour
    sp = tr.t_in < pd.Timestamp(SPLIT)
    good = [h for h, v in tr[sp].groupby("hour").R.mean().items() if v > 0]       # 前半で成績が良かった時間帯
    full_good = [h for h, v in tr.groupby("hour").R.mean().items() if v > 0]      # 全期間で成績が良かった時間帯
    f = tr[tr.hour.isin(full_good)]
    print(f"\n【3】時間帯の条件を足す（全期間で選んだ{len(full_good)}時間に限定）")
    show(" 手数料・スワップなし", f, pick_risk(f))
    print(f" 前半で選んだ{len(good)}時間に限定した場合のPF: 前半 {pf(tr[sp & tr.hour.isin(good)]):.2f} → 後半 {pf(tr[~sp & tr.hour.isin(good)]):.2f}"
          f"（条件なしの後半 {pf(tr[~sp]):.2f}）")
    for risk in (0.02, 0.03):
        show(f" 同じ結果を1回のリスク{risk*100:.0f}%で複利運用", f, risk)


if __name__ == "__main__":
    main(sys.argv[1])
