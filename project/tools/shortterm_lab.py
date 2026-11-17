#!/usr/bin/env python3
"""保有期間とスワップの検証ラボ。

(1) 日足ルール（最大40日保有）にスワップを入れると期待値がどう変わるか。
    スワップは「1晩あたり建玉金額の何%か」で表し、利益が残る上限（損益分岐の率）を求める。
(2) 1時間足で、保有を最大24〜48時間に限ったルールに統計的な優位性があるか。
    学習期間で選び、検証期間で確かめる（検証期間: BTC/ETH 2023年〜、金 2024年〜）。
    金はPAXG（金1オンス連動トークン、Binance）で代用し、金市場の休場時間は除外する。
    コスト（往復）: BTC/ETH 0.1%、金 0.03%（CFDのスプレッド相当）。
スワップの付与は米東部17時（Vantageなど多くのCFD業者の日替わり時刻）で数える。
使い方: python3 shortterm_lab.py <キャッシュ> [holds｜checks]
  引数なし: (1)(2)  holds: (5)保有上限とスワップの早見表  checks: (3)朝晩チェック運用
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
import strategy_lab as S  # noqa: E402
from marketdata import binance_klines, load_all  # noqa: E402

COST = {"BTC": 0.001, "ETH": 0.001, "Gold": 0.0003}
SPLIT = {"BTC": "2023-01-01", "ETH": "2023-01-01", "Gold": "2024-01-01"}


def ny_day(t):
    t = pd.DatetimeIndex(t)
    return (t.tz_localize("UTC").tz_convert("America/New_York") + pd.Timedelta(hours=7)).normalize()


def nights(t_in, t_out):
    """米東部17時（スワップ付与時刻）をまたいだ回数。"""
    return np.asarray((ny_day(t_out) - ny_day(t_in)).days, float)


# ---------------------------------------------------------------- (1) 日足ルール
def daily_swap_report(cache):
    S.HOLD = 40
    data = load_all(cache)
    print("===== (1) 日足ルール（200日線＋20日高値更新、損切りATR×2、利確4R、最大40日）とスワップ =====")
    for a, df in data.items():
        P = S.prep(df)
        tr = [x for x in S.run(P, 1, 200, "bo", 2.0, 4.0, S.COST[a]) if not x["open"]]
        R = np.array([x["R"] for x in tr])
        lev = np.array([x["fill"] / x["risk"] for x in tr])           # 建玉金額 ÷ 1R
        nt = np.array([(x["exit_t"] - x["t"]).days + (1 if x["why"] == "時間切れ" else 0.5) for x in tr])
        per = (lev * nt).mean()                                       # スワップ1%/晩あたりの損失（R）×100
        sp = pd.Timestamp(S.SPLIT[a])
        out = np.array([x["t"] >= sp for x in tr])
        be_all = R.mean() / per
        be_out = R[out].mean() / (lev[out] * nt[out]).mean()
        print(f"{a:5s} 取引{len(R)}回 平均保有{nt.mean():.0f}晩 建玉=1Rの{lev.mean():.0f}倍 ｜ "
              f"スワップ0.01%/晩で −{per*0.0001:.2f}R/回 ｜ 期待値 全期間{R.mean():+.2f}R・検証期間{R[out].mean():+.2f}R ｜ "
              f"損益分岐のスワップ 全期間{be_all*100:.3f}%/晩(年{be_all*36500:.0f}%)・検証期間{be_out*100:.3f}%/晩(年{be_out*36500:.0f}%)")


# ---------------------------------------------------------------- (2) 1時間足
def load_hourly(cache):
    out = {}
    for name, sym in (("BTC", "BTCUSDT"), ("ETH", "ETHUSDT"), ("Gold", "PAXGUSDT")):
        path = f"{cache}/{'PAXG' if name == 'Gold' else name}_1h.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            df = binance_klines(sym, "1h")
            df.to_csv(path)
        if name == "Gold":   # 金市場の休場（米東部 金17時〜日18時）を除く
            ny = df.index.tz_localize("UTC").tz_convert("America/New_York")
            wd, hr = ny.weekday, ny.hour
            df = df[~(((wd == 4) & (hr >= 17)) | (wd == 5) | ((wd == 6) & (hr < 18)))]
        out[name] = df
    return out


def prep(df):
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
    return dict(o=o, h=h, l=l, c=c, atr=atr, t=df.index.values, hour=df.index.hour.to_numpy(),
                sma={}, hh={}, ll={})


def make_signal(P, fam, side, N, M):
    c = P["c"]
    if M:
        if M not in P["sma"]:
            P["sma"][M] = pd.Series(c).rolling(M).mean().to_numpy()
        sig = (c - P["sma"][M]) * side > 0
    else:
        sig = np.ones(len(c), bool)
    if fam == "rnd":
        return sig
    if N not in P["hh"]:
        P["hh"][N] = pd.Series(P["h"]).rolling(N).max().shift(1).to_numpy()
        P["ll"][N] = pd.Series(P["l"]).rolling(N).min().shift(1).to_numpy()
    hh, ll = P["hh"][N], P["ll"][N]
    if fam == "bo":      # 順張り: 直近N時間の高値更新で買い／安値更新で売り
        cond = (c > hh) if side == 1 else (c < ll)
    else:                # 逆張り: 直近N時間の安値更新で買い（押し目）／高値更新で売り（戻り）
        cond = (c < ll) if side == 1 else (c > hh)
    return sig & cond


def run(P, sig, side, s, tR, H, cost):
    o, h, l, c, atr, t = P["o"], P["h"], P["l"], P["c"], P["atr"], P["t"]
    N, free = len(c), 0
    rows = []
    for i in np.flatnonzero(sig[:-1]):
        if i < free:
            continue
        j = i + 1
        fill, risk = o[j], s * atr[i]
        if not risk > 0:
            continue
        stop, tgt = fill - side * risk, fill + side * tR * risk
        end = min(j + H, N)
        adv = l[j:end] if side == 1 else h[j:end]
        fav = h[j:end] if side == 1 else l[j:end]
        hs = (adv - stop) * side <= 0
        ht = (fav - tgt) * side >= 0
        a = hs.argmax() if hs.any() else 10**9
        b = ht.argmax() if ht.any() else 10**9
        if a == b == 10**9:
            if end - j < H:
                break                                   # データ末尾で未決済
            k, ex, dt = end - 1, c[end - 1], np.timedelta64(60, "m")
        elif a <= b:                                    # 同じ足で両方なら損切り扱い
            k = j + a
            ex = o[k] if k > j and (o[k] - stop) * side <= 0 else stop
            dt = np.timedelta64(30, "m")
        else:
            k = j + b
            ex = o[k] if k > j and (o[k] - tgt) * side >= 0 else tgt
            dt = np.timedelta64(30, "m")
        rows.append((t[j], t[k] + dt, side * (ex - fill) / risk - cost * fill / risk, fill / risk))
        free = k
    if not rows:
        return pd.DataFrame(columns=["t_in", "t_out", "R", "lev", "nights"])
    df = pd.DataFrame(rows, columns=["t_in", "t_out", "R", "lev"])
    df["nights"] = nights(df.t_in, df.t_out)
    return df


def stats(df):
    if len(df) < 2:
        return dict(n=len(df), win=np.nan, exp=np.nan, t=np.nan, be=np.nan, nights=np.nan)
    R = df.R.to_numpy()
    sw = (df.lev * df.nights).mean()
    return dict(n=len(R), win=(R > 0).mean(), exp=R.mean(), t=R.mean() / R.std(ddof=1) * np.sqrt(len(R)),
                be=(R.mean() / sw * 100) if sw > 0 else np.inf, nights=df.nights.mean())


def hourly_grid(cache):
    data = load_hourly(cache)
    Ps = {k: prep(v) for k, v in data.items()}
    rows = []
    grid = [(f, n) for f in ("bo", "mr") for n in (24, 72)] + [("rnd", 0)]
    for (fam, N), side, M, s, tR, H in itertools.product(grid, (1, -1), (0, 480, 4800), (1, 2, 3), (1, 2, 3), (24, 48)):
        for a, P in Ps.items():
            tr = run(P, make_signal(P, fam, side, N, M), side, s, tR, H, COST[a])
            sp = pd.Timestamp(SPLIT[a])
            si, so = stats(tr[tr.t_in < sp]), stats(tr[tr.t_in >= sp])
            rows.append(dict(fam=fam, side=side, N=N, M=M, s=s, tR=tR, H=H, asset=a,
                             **{f"in_{k}": v for k, v in si.items()}, **{f"out_{k}": v for k, v in so.items()}))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- (3) 朝晩2回チェックの運用
CHECK_BARS = (23, 11)   # 23時台・11時台(UTC)の足の確定時 ＝ 日本時間 9:00 / 21:00 にチェック


def run_checks(P, N, M, s, tR, mode, cost):
    """チェック時に、上昇中なら「直近N時間の高値」に逆指値の買い注文（次のチェックまで有効）。
    損切り = 注文価格 − ATR×s、利確 = 注文価格 + 損切り幅×tR。
    mode V1: 朝晩とも注文、24時間以上たったら次のチェックで決済（最大36時間）
         V2: 朝晩とも注文、約定後の次のチェックで決済（最大12時間）
         V3: 朝だけ注文、夜のチェックで決済（日をまたがない＝スワップなし）"""
    o, h, l, c, atr, t, hr = P["o"], P["h"], P["l"], P["c"], P["atr"], P["t"], P["hour"]
    if M and M not in P["sma"]:
        P["sma"][M] = pd.Series(c).rolling(M).mean().to_numpy()
    key = ("incl", N)
    if key not in P["hh"]:
        P["hh"][key] = pd.Series(h).rolling(N).max().to_numpy()
    hh = P["hh"][key]
    checks = np.flatnonzero(np.isin(hr, CHECK_BARS))
    place = checks if mode in ("V1", "V2") else np.flatnonzero(hr == CHECK_BARS[0])
    rows, free = [], -1
    for i in place:
        if i <= free or i + 1 >= len(c):
            continue
        if M and not c[i] > P["sma"][M][i]:
            continue
        L, A = hh[i], atr[i]
        if np.isnan(L) or not A > 0:
            continue
        risk = s * A
        stop, tgt = L - risk, L + tR * risk
        q = np.searchsorted(checks, i, side="right")
        if q >= len(checks):
            break
        nxt = checks[q]
        hit = np.flatnonzero(h[i + 1:nxt + 1] >= L)
        if not len(hit):
            continue                                     # 約定せず失効
        j = i + 1 + hit[0]
        fill = max(o[j], L)
        if mode == "V1":
            later = checks[checks >= j]
            later = later[(t[later] + np.timedelta64(1, "h") - t[j]) >= np.timedelta64(24, "h")]
        else:
            later = checks[checks >= j]
        if not len(later):
            break
        m = later[0]
        hs = l[j:m + 1] <= stop
        ht = h[j:m + 1] >= tgt
        a = hs.argmax() if hs.any() else 10**9
        b = ht.argmax() if ht.any() else 10**9
        if a == b == 10**9:
            k, ex, tout = m, c[m], t[m] + np.timedelta64(60, "m")
        elif a <= b:
            k = j + a
            ex = o[k] if k > j and o[k] <= stop else stop
            tout = t[k] + np.timedelta64(30, "m")
        else:
            k = j + b
            ex = o[k] if k > j and o[k] >= tgt else tgt
            tout = t[k] + np.timedelta64(30, "m")
        rows.append((t[j] + np.timedelta64(30, "m"), tout, (ex - fill) / risk - cost * fill / risk, fill / risk))
        free = k - 1
    df = pd.DataFrame(rows, columns=["t_in", "t_out", "R", "lev"])
    df["nights"] = nights(df.t_in, df.t_out) if len(df) else []
    return df


def checks_grid(cache):
    Ps = {k: prep(v) for k, v in load_hourly(cache).items()}
    rows = []
    for N, M, s, tR, mode in itertools.product((24, 72), (0, 480), (2, 3), (2, 3), ("V1", "V2", "V3")):
        for a, P in Ps.items():
            tr = run_checks(P, N, M, s, tR, mode, COST[a])
            sp = pd.Timestamp(SPLIT[a])
            si, so = stats(tr[tr.t_in < sp]), stats(tr[tr.t_in >= sp])
            yrs = (P["t"][-1] - sp.to_datetime64()) / np.timedelta64(365, "D")
            rows.append(dict(N=N, M=M, s=s, tR=tR, mode=mode, asset=a,
                             **{f"in_{k}": v for k, v in si.items()}, **{f"out_{k}": v for k, v in so.items()},
                             out_per_yr=so["n"] / yrs))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- (4) 朝晩チェック時の成行買い
def run_at_check(P, N, M, s, tR, mode, cost):
    """チェック時の終値が「前回チェック時点（12時間前）の直近N時間高値」を上回っていれば成行で買う。
    逆指値の買い（run_checks）と違い、高値更新が終値で確認されてから入る。mode は run_checks と同じ。"""
    o, h, l, c, atr, t, hr = P["o"], P["h"], P["l"], P["c"], P["atr"], P["t"], P["hour"]
    if M and M not in P["sma"]:
        P["sma"][M] = pd.Series(c).rolling(M).mean().to_numpy()
    hh = pd.Series(h).rolling(N).max().to_numpy()
    checks = np.flatnonzero(np.isin(hr, CHECK_BARS))
    place = checks if mode in ("V1", "V2") else np.flatnonzero(hr == CHECK_BARS[0])
    rows, free = [], -1
    for i in place:
        if i <= free or i + 1 >= len(c) or i < 12:
            continue
        if M and not c[i] > P["sma"][M][i]:
            continue
        L = hh[i - 12]
        if np.isnan(L) or not c[i] > L:
            continue
        j = i + 1
        fill, risk = o[j], s * atr[i]
        stop, tgt = fill - risk, fill + tR * risk
        later = checks[checks >= j]
        if mode == "V1":
            later = later[(t[later] + np.timedelta64(1, "h") - t[j]) >= np.timedelta64(24, "h")]
        if not len(later):
            break
        m = later[0]
        hs, ht = l[j:m + 1] <= stop, h[j:m + 1] >= tgt
        a = hs.argmax() if hs.any() else 10**9
        b = ht.argmax() if ht.any() else 10**9
        if a == b == 10**9:
            k, ex, tout = m, c[m], t[m] + np.timedelta64(60, "m")
        elif a <= b:
            k = j + a
            ex = o[k] if k > j and o[k] <= stop else stop
            tout = t[k] + np.timedelta64(30, "m")
        else:
            k = j + b
            ex = o[k] if k > j and o[k] >= tgt else tgt
            tout = t[k] + np.timedelta64(30, "m")
        rows.append((t[j], tout, (ex - fill) / risk - cost * fill / risk, fill / risk))
        free = k - 1
    df = pd.DataFrame(rows, columns=["t_in", "t_out", "R", "lev"])
    df["nights"] = nights(df.t_in, df.t_out) if len(df) else []
    return df


# ---------------------------------------------------------------- (5) 日足ルールの保有上限とスワップの早見表
HOLD_OPTIONS = {"最大40日": (40, 2.0), "最大10日": (10, 1.5), "最大5日": (5, 1.0)}   # (保有上限, 損切りATR倍)


def hold_options_report(cache, swaps=(5, 10, 20, 30)):
    """200日線＋20日高値更新・利確4R で、保有上限を変えた場合の検証期間の成績と、
    スワップ年率ごとの1回あたり期待値（R）を表示する。"""
    data = load_all(cache)
    Ps = {k: S.prep(v) for k, v in data.items()}
    for lab, (H, s) in HOLD_OPTIONS.items():
        S.HOLD = H
        print(f"\n=== {lab}: 損切りATR×{s:g}、利確4R ===")
        for a, P in Ps.items():
            tr = [x for x in S.run(P, 1, 200, "bo", s, 4.0, S.COST[a]) if not x["open"]]
            sp = pd.Timestamp(S.SPLIT[a])
            te = [x for x in tr if x["t"] >= sp]
            R = np.array([x["R"] for x in te])
            lev = np.array([x["fill"] / x["risk"] for x in te])
            nt = np.array([(x["exit_t"] - x["t"]).days + (1 if x["why"] == "時間切れ" else 0.5) for x in te])
            yrs = (P["t"][-1] - sp).days / 365.25
            per_pct = (lev * nt).mean() / 36500
            sm = S.summarize(R)
            row = " ".join(f"年{r}%→{R.mean() - r * per_pct:+.2f}R" for r in swaps)
            print(f"{a:4s} 年{len(R)/yrs:.1f}回 勝率{sm['win']*100:.0f}% 期待値{R.mean():+.2f}R 最大{sm['maxl']}連敗 "
                  f"平均{nt.mean():.0f}晩 ｜ スワップ年率別: {row}")


if __name__ == "__main__":
    cache = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "holds":
        hold_options_report(cache)
        sys.exit()
    if len(sys.argv) > 2 and sys.argv[2] == "checks":
        df = checks_grid(cache)
        df.to_csv(f"{cache}/checks_results.csv", index=False)
        g = df.groupby(["mode", "N", "M", "s", "tR"]).agg(
            in_exp=("in_exp", "mean"), in_min=("in_exp", "min"), out_exp=("out_exp", "mean"),
            out_min=("out_exp", "min"), out_n=("out_n", "sum"), per_yr=("out_per_yr", "sum"),
            be=("out_be", "min"), nights=("out_nights", "mean")).reset_index()
        pd.set_option("display.width", 220)
        for mode in ("V1", "V2", "V3"):
            x = g[g["mode"] == mode].sort_values("in_exp", ascending=False)
            print(f"\n--- {mode}: 全{len(x)}通り 学習プラス{(x.in_exp > 0).mean()*100:.0f}% 検証プラス{(x.out_exp > 0).mean()*100:.0f}%")
            print(x.round(3).to_string(index=False))
        sys.exit()
    daily_swap_report(cache)
    df = hourly_grid(cache)
    df.to_csv(f"{cache}/shortterm_results.csv", index=False)
    key = ["fam", "side", "N", "M", "s", "tR", "H"]
    g = df.groupby(key).agg(in_exp=("in_exp", "mean"), in_min=("in_exp", "min"), in_n=("in_n", "min"),
                            out_exp=("out_exp", "mean"), out_min=("out_exp", "min"), out_n=("out_n", "sum"),
                            out_be=("out_be", "min"), nights=("out_nights", "mean")).reset_index()
    pd.set_option("display.width", 220)
    print("\n===== (2) 1時間足・最大24/48時間保有（コスト込み、スワップ抜き） =====")
    for fam in ("bo", "mr", "rnd"):
        for side in (1, -1):
            x = g[(g.fam == fam) & (g.side == side) & (g.in_n >= 100)].sort_values("in_exp", ascending=False)
            lab = {"bo": "順張り(高値/安値更新)", "mr": "逆張り(押し目/戻り)", "rnd": "ランダム(比較用)"}[fam]
            print(f"\n--- {lab} {'買い' if side == 1 else '売り'}: 全{len(x)}通り 学習プラス{(x.in_exp > 0).mean()*100:.0f}% "
                  f"検証プラス{(x.out_exp > 0).mean()*100:.0f}% ／ 学習上位5")
            print(x.head(5).round(3).to_string(index=False))
