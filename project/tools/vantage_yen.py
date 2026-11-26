#!/usr/bin/env python3
"""ランキング上位3つ（R03・M08・R02）を、Vantage の実際の条件で円建てで回した場合（2018-01〜2026-11）。

Vantage の条件（アプリの「情報」、2026-11-27 に確認）:
  BTCUSD・ETHUSD とも 契約サイズ1.0（1ロット＝1枚）・最小0.01ロット・0.01刻み・証拠金率0.20%（500倍）
  スワップ：買い BTC −25.0・ETH −35.0（年率%、現在の値段、360日割り＝365日で25.35%・35.49%）、売り0、金曜は3日分
  取引時間は GMT+2（日本時間7時に日付が変わる。夏時間は6時）
計算のしかた:
  ・取引は検証と同じ（S1の土台は round2_lab の値幅ブレイク、M08 は上下の逆指値で先に約定した方だけ）。
  ・ロット＝倍率 × 資金 × 0.5% ÷（ATR × ドル円）を0.01ロットに丸める（最小0.01）。ドル円は日ごと。資金は固定（単利）。
  ・損益（円）＝ロット ×（R × ATR）× ドル円。R には往復0.05%とスワップ（毎晩1日分）が入っている。
  ・スワップの付き方（金曜3日分）とスプレッドの広さで、成績がどれだけ変わるかも見る。
使い方: python3 project/tools/vantage_yen.py <キャッシュ（BTC_15m.csv・ETH_15m.csv・USDJPY.csv）>
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import more_methods_lab as MM  # noqa: E402
import portfolio_lab as PF  # noqa: E402
import round1_lab as R1  # noqa: E402
import round4_lab as R4  # noqa: E402
import round7_lab as R7  # noqa: E402

COINS = R7.COINS
COST = 0.0005
MARGIN = 0.002
SWD = {a: PF.SPEC[a][1] / 36500 for a in COINS}
CAPS = (100_000, 300_000, 1_000_000, 3_000_000)


def load(cache):
    D = R1.prep(cache)
    days = D["BTC"]["idx"]
    assert all((D[a]["idx"] == days).all() for a in COINS)
    fx = pd.read_csv(os.path.join(cache, "USDJPY.csv"), index_col=0, parse_dates=True)["close"]
    fx = fx[~fx.index.duplicated()].sort_index()
    fxd = fx.reindex(days.union(fx.index)).ffill().reindex(days).bfill().to_numpy()
    return D, days, fxd


def s1_trades(D):
    """S1 の向きで出た値幅ブレイクの取引（静かな日の判定の前）に、約定の値段と、形ごとの量の倍率をつける。"""
    parts = R7.s1_parts(D, COINS, D["BTC"])
    rules = R7.size_rules(parts, COINS)

    def r02_05(a, p):
        n, w, _ = R7.day_stats(R7.taken(p), a, p["idx"])
        return p["mq"] * np.where(R7.lost_last(n, w, 2), 2.0, np.where(R7.won_last(n, w, 1), 0.5, 1.0))
    mult = {"S1": rules["S1"], "S4案": rules["S4案"], "R03": rules["R03"], "R02": r02_05}
    rows = []
    for a in COINS:
        x, p = D[a], parts[a]
        T = p["T"].reset_index(drop=True)
        di = x["idx"].get_indexer(pd.DatetimeIndex(T.day))
        sd, pos = T.side.to_numpy(), T.pos.to_numpy().astype(int)
        trig = x["dO"][di] + sd * 0.8 * x["rng"][di]
        o = x["M"]["open"][di, pos]
        e = np.where(sd == 1, np.maximum(o, trig), np.minimum(o, trig))
        A = x["atr"][di]
        chk = sd * (x["M"]["close"][di, -1] - e) / A - COST * e / A - np.where((sd == 1) & (pos < 84), SWD[a] * e / A, 0)
        assert np.allclose(chk, T.R.to_numpy(), atol=1e-9)
        df = pd.DataFrame(dict(coin=a, d=di, side=sd, w=T.w.to_numpy(), cw=1.0, R=T.R.to_numpy(), e=e, pos=pos, A=A))
        for k, fn in mult.items():
            df["m_" + k] = np.asarray(fn(a, p), float)[di]
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def oco(x, a, Lu, Ld, ok, last, wl, ws, cw):
    """上下に逆指値、先に約定した方だけ持ち、日の終わり（翌朝9時）に決済。同じ15分足で両方に届いた日は見送る（R1.oco_engine と同じ）。"""
    H, L, O, C = x["M"]["high"], x["M"]["low"], x["M"]["open"], x["M"]["close"]
    with np.errstate(invalid="ignore"):
        up = H[:, :last] >= Lu[:, None]
        dn = L[:, :last] <= Ld[:, None]
    big = 10 ** 6
    ju = np.where(up.any(1), up.argmax(1), big)
    jd = np.where(dn.any(1), dn.argmax(1), big)
    ok = ok & ~np.isnan(Lu) & (x["atr"] > 0)
    d = np.flatnonzero(ok & (np.minimum(ju, jd) < big) & (ju != jd))
    sd = np.where(ju[d] < jd[d], 1, -1)
    j = np.minimum(ju[d], jd[d])
    o = O[d, j]
    e = np.where(sd == 1, np.maximum(o, Lu[d]), np.minimum(o, Ld[d]))
    A = x["atr"][d]
    R = sd * (C[d, -1] - e) / A - COST * e / A - np.where((sd == 1) & (j < 84), SWD[a] * e / A, 0)
    return pd.DataFrame(dict(coin=a, d=d, side=sd, w=np.where(sd == 1, wl, ws), cw=cw, R=R, e=e, pos=j, A=A))


def m08_trades(D, win):
    """M08：向きなし・静かな日だけ・5時まで（80本目まで）の値幅ブレイク ＋ はらみ足（F02）×0.5 ＋ 持ち合い（F08）×0.5。"""
    rows = []
    for a in COINS:
        x = D[a]
        q = R4.mult(x, "静かな日だけ") > 0
        rows.append(oco(x, a, x["dO"] + 0.8 * x["rng"], x["dO"] - 0.8 * x["rng"], q & ~np.isnan(x["rng"]), 80, 1.0, 0.5, 1.0))
        with np.errstate(invalid="ignore"):
            m = (x["dH"] < np.r_[np.nan, x["dH"][:-1]]) & (x["dL"] > np.r_[np.nan, x["dL"][:-1]])
        rows.append(oco(x, a, np.r_[np.nan, x["dH"][:-1]], np.r_[np.nan, x["dL"][:-1]], np.r_[False, m[:-1]], 92, 1.0, 0.5, 0.5))
        hh, ll = pd.Series(x["dH"]).rolling(5).max(), pd.Series(x["dL"]).rolling(5).min()
        with np.errstate(invalid="ignore"):
            m = ((hh < hh.shift(5)) & (ll > ll.shift(5))).to_numpy()
        rows.append(oco(x, a, np.r_[np.nan, hh.to_numpy()[:-1]], np.r_[np.nan, ll.to_numpy()[:-1]], np.r_[False, m[:-1]], 92, 1.0, 0.5, 0.5))
    T = pd.concat(rows, ignore_index=True)
    T = T[win[T.d]].reset_index(drop=True)
    T["m_M08"] = 1.0
    return T


def run(T, key, cap, days, fxd, lots_round=True, cost_add=0.0, sched=None):
    """円の毎日の損益・証拠金・ロット。sched：曜日（月〜日）ごとのスワップの日数（None なら毎晩1日分＝検証と同じ）。"""
    f = fxd[T.d.to_numpy()]
    tgt = T["m_" + key].to_numpy() * T.cw.to_numpy() * T.w.to_numpy() * cap * 0.005 / (T.A.to_numpy() * f)
    lots = np.where(tgt > 0, np.maximum(np.round(tgt / 0.01) * 0.01, 0.01), 0.0) if lots_round else tgt
    usd = T.R.to_numpy() * T.A.to_numpy() - cost_add * T.e.to_numpy()
    if sched is not None:
        wd = days[T.d.to_numpy()].dayofweek.to_numpy()
        paid = (T.side.to_numpy() == 1) & (T.pos.to_numpy() < 84)
        swd = np.where(T.coin.to_numpy() == "BTC", SWD["BTC"], SWD["ETH"])
        usd = usd - np.where(paid, (np.asarray(sched)[wd] - 1) * swd * T.e.to_numpy(), 0.0)
    yen = lots * usd * f
    n = len(days)
    daily = pd.Series(np.bincount(T.d.to_numpy(), weights=yen, minlength=n), index=days)
    margin = pd.Series(np.bincount(T.d.to_numpy(), weights=lots * T.e.to_numpy() * f * MARGIN, minlength=n), index=days)
    return daily, margin, lots, tgt, yen


def stats(daily, cap, win):
    daily = daily[win]
    yrs = (MM.END - MM.START).days / 365.25
    eq = daily.cumsum()
    dd = float((eq.cummax().clip(lower=0) - eq).max())
    yearly = daily.groupby(daily.index.year).sum()
    mon = daily.resample("ME").sum()
    return dict(a=daily.sum() / yrs, ap=daily.sum() / yrs / cap * 100, dd=dd, ddp=dd / cap * 100, st=mon.mean() / mon.std(ddof=1) * np.sqrt(12),
                neg=",".join(str(y) for y, v in yearly.items() if v < 0) or "なし", worst_day=float(daily.min()), yearly=yearly)


def pick(T, key):
    return T if key == "M08" else T[T["m_" + key] > 0]


def main(cache):
    D, days, fxd = load(cache)
    win = ((days >= MM.START) & (days <= MM.END))
    TS1, TM8 = s1_trades(D), m08_trades(D, win)
    top = (("R03", TS1, "1位 R03"), ("M08", TM8, "2位 M08"), ("R02", TS1, "3位 R02"))
    print("==== 丸めなし（検証と同じ量。資金100万円で円に直した確かめ）")
    for key, T, lab in top + (("S4案", TS1, "参考 S4案"), ("S1", TS1, "参考 S1")):
        s = stats(run(pick(T, key), key, 1_000_000, days, fxd, lots_round=False)[0], 1_000_000, win)
        print(f"  {lab}: 年{s['ap']:+.2f}%（{s['a']:,.0f}円） 落込{s['ddp']:.2f}% 安定度{s['st']:.2f} 負けた年{s['neg']}")

    print("\n==== Vantage の実際のロット（0.01単位・最小0.01）")
    for cap in CAPS:
        print(f"-- 資金{cap:,}円")
        for key, T, lab in top:
            TT = pick(T, key)
            daily, mg, lots, tgt, yen = run(TT, key, cap, days, fxd)
            s = stats(daily, cap, win)
            btc = (TT.coin == "BTC").to_numpy()
            over = lots[btc] / tgt[btc]
            print(f"  {lab}: 年{s['a']:+,.0f}円（{s['ap']:+.1f}%） 最大の落ち込み{s['dd']:,.0f}円（{s['ddp']:.1f}%） 安定度{s['st']:.2f} 負けた年{s['neg']} "
                  f"最悪の日{s['worst_day']:+,.0f}円 証拠金の最大{mg[win].max():,.0f}円｜BTCのロットがほしい量の1.5倍以上{np.mean(over >= 1.5) * 100:.0f}%（中央値{np.median(over):.1f}倍）")

    print("\n==== 資金100万円の内訳（実際のロット）")
    yt = {}
    for key, T, lab in top:
        TT = pick(T, key)
        daily, mg, lots, tgt, yen = run(TT, key, 1_000_000, days, fxd)
        expo = (mg[win] / MARGIN / 1_000_000)
        on = expo[expo > 0]
        print(f"  {lab}: 1回の損益 勝ちの平均{yen[yen > 0].mean():+,.0f}円・負けの平均{yen[yen <= 0].mean():+,.0f}円・最大の負け{yen.min():+,.0f}円・勝率{(yen > 0).mean() * 100:.0f}%。"
              f"建玉の合計は資金の{on.median():.2f}倍（中央値）・最大{on.max():.2f}倍")
        yt[lab] = stats(daily, 1_000_000, win)["yearly"].round(-2)
    print("  年ごとの損益（円、2026年は11月16日まで）\n" + "\n".join("    " + ln for ln in pd.DataFrame(yt).to_string().splitlines()))

    print("\n==== スワップの付き方（資金100万円・丸めなし）：毎晩1日分（検証）／月〜木1・金3・土日0／月〜木1・金3・土日1")
    for key, T, lab in top:
        out = []
        for sched in (None, [1, 1, 1, 1, 3, 0, 0], [1, 1, 1, 1, 3, 1, 1]):
            s = stats(run(pick(T, key), key, 1_000_000, days, fxd, lots_round=False, sched=sched)[0], 1_000_000, win)
            out.append(f"年{s['ap']:+.2f}%・安定度{s['st']:.2f}")
        print(f"  {lab}: " + " ／ ".join(out))

    print("\n==== スプレッドが広い場合（往復0.05%・0.10%・0.15%。資金100万円・丸めなし）")
    for key, T, lab in top:
        out = []
        for add in (0.0, 0.0005, 0.0010):
            s = stats(run(pick(T, key), key, 1_000_000, days, fxd, lots_round=False, cost_add=add)[0], 1_000_000, win)
            out.append(f"往復{(COST + add) * 100:.2f}%：年{s['ap']:+.2f}%・安定度{s['st']:.2f}・負けた年{s['neg']}")
        print(f"  {lab}: " + " ／ ".join(out))

    print("\n==== 資金が少ない時：ETHだけ（実際のロット。1回の量は資金の0.5%と、BTCの分も回した1.0%）")
    for cap in CAPS[:2]:
        for risk in (0.005, 0.01):
            out = []
            for key, T, lab in (("S4案", TS1, "S4案"), ("R02", TS1, "R02"), ("M08", TM8, "M08")):
                TT = pick(T, key)
                TT = TT[TT.coin == "ETH"]
                s = stats(run(TT, key, cap * risk / 0.005, days, fxd)[0], cap, win)
                out.append(f"{lab} 年{s['a']:+,.0f}円（{s['ap']:+.1f}%）・落込{s['dd']:,.0f}円（{s['ddp']:.1f}%）・安定度{s['st']:.2f}・負けた年{s['neg']}")
            print(f"  資金{cap:,}円・1回{risk * 100:.1f}%: " + " ／ ".join(out))

    print("\n==== 直近1年の、取引した日のATR（中央値）と、資金ごとの1倍のロット（ドル円160円）")
    for a in COINS:
        T = TS1[(TS1.coin == a) & (TS1.m_S1 > 0) & (days[TS1.d.to_numpy()] >= MM.END - pd.Timedelta(days=365))]
        A, px = float(np.median(T.A)), float(np.median(T.e))
        print(f"  {a}: ATR {A:,.1f}ドル（値段の{A / px * 100:.1f}%）。" + "・".join(f"{c // 10000}万円→{c * 0.005 / (A * 160):.3f}枚" for c in CAPS))


if __name__ == "__main__":
    main(sys.argv[1])
