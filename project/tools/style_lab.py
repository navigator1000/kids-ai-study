#!/usr/bin/env python3
"""取引スタイル別の比較（暗号資産8銘柄・日足・2018年〜、1日0.5〜1回前後を目安）。

設定はこれまでの検証で決まったもの／一般的な決め方をそのまま使い、ここでは探し直さない。
  ① 小さく積み重ねる : 値幅ブレイク(k=0.8・5日平均の上) で買い、ATR×0.5で利確、届かなければ当日の終値で決済
  ② 押し目買い      : 200日平均の上で RSI(2)<10 → 翌日の始値で買い、RSI(2)>70 の終値で売り（損切りATR×3・最大10日）
  ③ バランス        : 値幅ブレイク(k=0.8・5日平均の上)、当日の終値で決済
  ④ 少し伸ばす      : 値幅ブレイクで買い、終値が5日平均を割ったら決済（損切りATR×2・最大10日）
  ⑤ 大きく狙う      : 終値が10日平均の上なら翌日の始値で買い、下回った終値で売り（損切りATR×3）
  ⑥ 参考・大きく狙う : 200日平均の上で20日高値更新 → 損切りATR×2・利確4R・最大40日（最初の毎朝ルール）
コスト: 手数料 往復 BTC/ETH 0.05%・アルト 0.15%。スワップ（買い）年 BTC25%・ETH35%・アルト30%（Vantage並み）。
損益: ATR1本分の値動きで資金の0.5%が増減する大きさで売買（単利）として％で表示。
使い方: python3 project/tools/style_lab.py <キャッシュ（BTC.csv など日足）>
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
import shortterm_lab as T  # noqa: E402
import strategy_lab as S  # noqa: E402
from rsi_lab import rsi, run  # noqa: E402
from winrate_lab import vbo_daily_tp  # noqa: E402

START, SPLIT = pd.Timestamp("2018-01-01"), pd.Timestamp("2023-01-01")
COINS = {"BTC": (0.0005, 25), "ETH": (0.0005, 35), "SOL": (0.0015, 30), "XRP": (0.0015, 30),
         "BNB": (0.0015, 30), "ADA": (0.0015, 30), "LTC": (0.0015, 30), "DOGE": (0.0015, 30)}
RISK = 0.5   # ATR1本分の値動き＝資金の0.5%


def load(cache, a):
    df = pd.read_csv(os.path.join(cache, f"{a}.csv"), index_col=0, parse_dates=True)
    return df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]


def vbo_multiday(df, cost, swap, k=0.8, n=5, s=2.0, maxd=10):
    o, h, l, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - l, abs(h - pc), abs(l - pc)]), axis=0)
    atr = pd.Series(tr).rolling(14).mean().shift(1).to_numpy()
    rng = np.r_[np.nan, (h - l)[:-1]]
    filt = o > pd.Series(c).rolling(n).mean().shift(1).to_numpy()
    ma5 = pd.Series(c).rolling(5).mean().to_numpy()
    trig = o + k * rng
    rows, d, N = [], 0, len(c)
    while d < N:
        if not (filt[d] and atr[d] > 0 and rng[d] > 0 and h[d] >= trig[d]):
            d += 1
            continue
        entry, unit = trig[d], atr[d]
        stop, ex, j = entry - s * unit, None, d
        while True:
            if j == d:
                ex = stop if l[d] <= stop else None
            elif o[j] <= stop:
                ex = o[j]
            elif l[j] <= stop:
                ex = stop
            if ex is None and (c[j] < ma5[j] or j - d + 1 >= maxd):
                ex = c[j]
            if ex is not None or j == N - 1:
                break
            j += 1
        if ex is None:
            break
        nights = j - d + 1
        rows.append((df.index[d], df.index[j], (ex - entry) / unit - cost * entry / unit - swap * nights * entry / unit, nights))
        d = j + 1
    return pd.DataFrame(rows, columns=["t", "exit_t", "R", "nights"])


def from_run(df, sig, exit_sig, s, H, cost, swap_pct):
    """rsi_lab.run を使い、R を「ATR1本＝1R」に直す。決済日も付ける。"""
    P = T.prep(df)
    tr = run(P, sig, 1, s, None, H, cost, swap_pct, exit_sig)
    rows = []
    for t, R in tr.itertuples(index=False):
        rows.append((t, R * s))
    out = pd.DataFrame(rows, columns=["t", "R"])
    out["exit_t"], out["nights"] = out.t, np.nan          # 決済日は不明なので入る日で代用（年別の集計用）
    return out


def style_trades(style, df, cost, swap_pct):
    swap = swap_pct / 36500
    c = df.close.to_numpy(float)
    if style == 1:
        x = vbo_daily_tp(df, 0.8, 5, 0.5, cost, swap)
    elif style == 3:
        x = vbo_daily_tp(df, 0.8, 5, None, cost, swap)
    elif style == 2:
        up = c > pd.Series(c).rolling(200).mean().to_numpy()
        r2 = rsi(c, 2)
        return from_run(df, up & (r2 < 10), r2 > 70, 3.0, 10, cost, swap_pct)
    elif style == 4:
        return vbo_multiday(df, cost, swap)
    elif style == 5:
        sma = pd.Series(c).rolling(10).mean().to_numpy()
        return from_run(df, c > sma, c < sma, 3.0, 400, cost, swap_pct)
    else:
        S.HOLD = 40
        P = S.prep(df)
        rows = []
        for t in S.run(P, 1, 200, "bo", 2.0, 4.0, cost):
            if t["open"]:
                continue
            nights = (t["exit_t"] - t["t"]).days + 1
            atr = t["risk"] / 2.0
            rows.append((t["t"], t["exit_t"], t["R"] * 2.0 - swap * nights * t["fill"] / atr, nights))
        return pd.DataFrame(rows, columns=["t", "exit_t", "R", "nights"])
    x["exit_t"], x["nights"] = x.t, 1
    return x


def summarize(x):
    x = x[x.t >= START].sort_values("exit_t")
    R = x.R.to_numpy()
    days = (x.exit_t.max() - START).days
    pct = R * RISK
    yearly = pd.Series(pct).groupby(x.exit_t.dt.year.to_numpy()).sum()
    eq = np.cumsum(pct)
    mx = cur = 0
    for r in R:
        cur = cur + 1 if r <= 0 else 0
        mx = max(mx, cur)
    te = R[(x.t >= SPLIT).to_numpy()]
    return dict(per_day=len(R) / days, win=(R > 0).mean(), aw=R[R > 0].mean(), al=-R[R < 0].mean(),
                pf=R[R > 0].sum() / -R[R < 0].sum(), pf_te=te[te > 0].sum() / -te[te < 0].sum(),
                yearly=yearly, neg=[int(y) for y, v in yearly.items() if v < 0], dd=(np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max(),
                worst_day=pd.Series(pct).groupby(x.exit_t.to_numpy()).sum().min(), maxl=mx,
                hold=np.nanmean(x.nights.to_numpy(float)) if x.nights.notna().any() else np.nan, annual=pct.sum() / (days / 365.25))


NAMES = {1: "①小さく積み重ねる（値幅ブレイク＋早め利確）", 2: "②押し目買い（RSI2）", 3: "③バランス（値幅ブレイク・当日決済）",
         4: "④少し伸ばす（値幅ブレイク＋数日保有）", 5: "⑤大きく狙う（10日平均の上で保有）", 6: "⑥参考：大きく狙う（20日高値・最大40日）"}


def main(cache):
    data = {a: load(cache, a) for a in COINS}
    for style in (1, 2, 3, 4, 5, 6):
        for swap_on in ((True, False) if style in (4, 5, 6) else (True,)):
            x = pd.concat([style_trades(style, df, COINS[a][0], COINS[a][1] if swap_on else 0.0).assign(asset=a) for a, df in data.items()])
            m = summarize(x)
            tag = "" if style not in (4, 5, 6) else ("（スワップあり）" if swap_on else "（現物・スワップなし）")
            print(f"\n{NAMES[style]}{tag}")
            print(f"  1日{m['per_day']:.2f}回 勝率{m['win']*100:.0f}% 勝ち平均{m['aw']:.2f}R 負け平均{m['al']:.2f}R（損益比{m['aw']/m['al']:.2f}）"
                  f" PF 全期間{m['pf']:.2f}・後半{m['pf_te']:.2f} 最大{m['maxl']}連敗"
                  + (f" 平均保有{m['hold']:.1f}日" if not np.isnan(m['hold']) else ""))
            print(f"  年平均{m['annual']:+.0f}% 最大の落ち込み{m['dd']:.0f}% 最悪の1日{m['worst_day']:.1f}% マイナスの年{m['neg'] or 'なし'}")
            print("  年別: " + " ".join(f"{y}:{v:+.0f}%" for y, v in m["yearly"].items()))


if __name__ == "__main__":
    main(sys.argv[1])
