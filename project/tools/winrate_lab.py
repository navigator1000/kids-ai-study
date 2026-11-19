#!/usr/bin/env python3
"""「1日1〜2回・勝ちを重ねる・マイナスの年なし」の条件で短期ルールを探す（BTC/ETH 30分足、EAでの運用を想定）。

パターン
  RSI逆張り     : 30分足 RSI(2) が L 未満で買い／100−L 超で売り
  バンド逆張り   : ボリンジャーバンド(20本, 2σ) の外で逆向きに入る
  短期ブレイク   : 直近 N 本の高値（安値）を終値で抜けたら同じ向きに入る
  絞り込み      : なし／日足5日平均・20日平均の向きだけ（買いは平均より上、売りは下）
出口: 損切り ATR(30分足)×s、利確は損切り幅の tR 倍、最大 H 本（30分足）。逆張りは「中央線に戻ったら決済」も。
コスト: スプレッド往復（基本0.05%）、買いのスワップ年25%（保有時間で按分）、売りのスワップ0。
評価: 前半（2018〜2022年）で選び、後半（2023年〜）とETHで確認。年別の損益でマイナスの年を数える。
使い方: python3 project/tools/winrate_lab.py <キャッシュ（BTC_30m.csv, ETH_30m.csv）>
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
import shortterm_lab as T  # noqa: E402
from rsi_lab import rsi, run  # noqa: E402

START, TRAIN_END = "2018-01-01", "2023-01-01"
SWAP_LONG = 25.0


def prep(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df[~df.index.duplicated()]
    P = T.prep(df)
    c = P["c"]
    P["rsi2"] = rsi(c, 2)
    s = pd.Series(c)
    mid, sd = s.rolling(20).mean(), s.rolling(20).std()
    P["mid"], P["up"], P["lo"] = mid.to_numpy(), (mid + 2 * sd).to_numpy(), (mid - 2 * sd).to_numpy()
    P["sma"] = {n: s.rolling(48 * n).mean().to_numpy() for n in (5, 20)}
    P["hhN"] = {n: pd.Series(P["h"]).rolling(n).max().shift(1).to_numpy() for n in (24, 48)}
    P["llN"] = {n: pd.Series(P["l"]).rolling(n).min().shift(1).to_numpy() for n in (24, 48)}
    return P


def signal(P, fam, side, x, filt):
    c = P["c"]
    if fam == "RSI逆張り":
        sig = P["rsi2"] < x if side == 1 else P["rsi2"] > 100 - x
    elif fam == "バンド逆張り":
        sig = c < P["lo"] if side == 1 else c > P["up"]
    else:
        sig = c > P["hhN"][x] if side == 1 else c < P["llN"][x]
    if filt:
        sig = sig & ((c - P["sma"][filt]) * side > 0)
    return sig


def evaluate(tr):
    tr = tr[tr.t >= pd.Timestamp(START)]
    if len(tr) < 100:
        return None
    out = {}
    for name, x in (("前半", tr[tr.t < pd.Timestamp(TRAIN_END)]), ("後半", tr[tr.t >= pd.Timestamp(TRAIN_END)]), ("全期間", tr)):
        R = x.R.to_numpy()
        if len(R) < 30:
            return None
        yrs = max((x.t.iloc[-1] - x.t.iloc[0]).days / 365.25, 0.5)
        yearly = x.groupby(x.t.dt.year).R.sum()
        eq = np.cumsum(R)
        out[name] = dict(n_yr=len(R) / yrs, win=(R > 0).mean(), pf=R[R > 0].sum() / max(-R[R < 0].sum(), 1e-9),
                         exp=R.mean(), neg_years=int((yearly < 0).sum()), worst_year=yearly.min(),
                         dd=(np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max(), worst=R.min(),
                         avg_win=R[R > 0].mean() if (R > 0).any() else 0, avg_loss=-R[R < 0].mean() if (R < 0).any() else 0)
    return out


def grid():
    exits = [(s, tR, H, False) for s, tR, H in itertools.product((1.5, 3.0), (0.33, 0.5, 1.0), (12, 48))]
    mean_exits = [(s, None, H, True) for s, H in itertools.product((1.5, 3.0), (12, 48))]
    for fam, xs in (("RSI逆張り", (5, 10, 20)), ("バンド逆張り", (0,)), ("短期ブレイク", (24, 48))):
        for x, filt, side in itertools.product(xs, (0, 5, 20), (1, -1)):
            for ex in exits + (mean_exits if fam != "短期ブレイク" else []):
                yield fam, x, filt, side, ex


def simulate(P, g, cost):
    fam, x, filt, side, (s, tR, H, mean_exit) = g
    sig = signal(P, fam, side, x, filt)
    ex_sig = None
    if mean_exit:
        ex_sig = P["c"] > P["mid"] if side == 1 else P["c"] < P["mid"]
    return run(P, sig, side, s, tR, H, cost, SWAP_LONG if side == 1 else 0.0, ex_sig)


def label(g):
    fam, x, filt, side, (s, tR, H, mean_exit) = g
    head = {"RSI逆張り": f"RSI逆張り(RSI2<{x})", "バンド逆張り": "バンド逆張り", "短期ブレイク": f"短期ブレイク({x // 2}時間高安)"}[fam]
    ex = "中央線で決済" if mean_exit else f"利確=損切りの{tR:g}倍"
    return f"{head}・{'買い' if side == 1 else '売り'}・絞り込み{('日足' + str(filt) + '日平均') if filt else 'なし'}・損切りATR×{s:g}・{ex}・最大{H // 2}時間"


def line(tag, m):
    return (f"{tag}: 年{m['n_yr']:.0f}回 勝率{m['win']*100:.0f}% PF{m['pf']:.2f} 平均{m['exp']:+.3f}R"
            f"（勝ち平均{m['avg_win']:.2f}R・負け平均{m['avg_loss']:.2f}R） マイナスの年{m['neg_years']} 最悪の年{m['worst_year']:+.0f}R")


def main(cache, cost=0.0005):
    B, E = prep(os.path.join(cache, "BTC_30m.csv")), prep(os.path.join(cache, "ETH_30m.csv"))
    rows = []
    for g in grid():
        mb, me = evaluate(simulate(B, g, cost)), evaluate(simulate(E, g, cost))
        if mb and me:
            rows.append((g, mb, me))
    print(f"######## スプレッド往復{cost*100:.2f}% ｜ 試した数 {len(rows)}通り")
    ok_train = [r for r in rows if r[1]["前半"]["n_yr"] >= 300 and r[1]["前半"]["win"] >= 0.55 and r[1]["前半"]["neg_years"] == 0]
    ok_test = [r for r in ok_train if r[1]["後半"]["neg_years"] == 0 and r[1]["後半"]["pf"] > 1]
    ok_eth = [r for r in ok_test if r[2]["全期間"]["pf"] > 1]
    print(f"前半で3条件（年300回以上・勝率55%以上・マイナスの年なし）を満たす: {len(ok_train)}通り"
          f" → 後半もマイナスの年なし＆PF>1: {len(ok_test)}通り → ETHもPF>1: {len(ok_eth)}通り")
    for fam in ("RSI逆張り", "バンド逆張り", "短期ブレイク"):
        xs = [r for r in rows if r[0][0] == fam]
        hi = [r for r in xs if r[1]["前半"]["win"] >= 0.55]
        print(f"\n== {fam}: {len(xs)}通り ｜ 前半の勝率55%以上 {len(hi)}通り ｜ 前半PF>1 {sum(r[1]['前半']['pf'] > 1 for r in xs)}通り"
              f" ｜ 後半PF>1 {sum(r[1]['後半']['pf'] > 1 for r in xs)}通り")
        cand = sorted([r for r in xs if r[1]["前半"]["n_yr"] >= 150], key=lambda r: r[1]["前半"]["pf"], reverse=True)[:2]
        for g, mb, me in cand:
            print(f"* {label(g)}")
            print("   " + line("前半", mb["前半"]))
            print("   " + line("後半", mb["後半"]))
            print("   " + line("ETH全期間", me["全期間"]))
    return rows


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.0005)
