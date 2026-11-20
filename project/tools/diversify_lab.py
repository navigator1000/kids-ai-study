#!/usr/bin/env python3
"""分散のパターン検証（BTC・ETH）と、結果の記録（project/research/results.csv）。

土台: ①値幅ブレイク買い＋②値幅ブレイク売り×0.5（5日平均の向きに、前日値幅×0.8の逆指値、当日決済）
  A 時間をずらす : 区切りを日本時間 3・9・15・21時の4つにして、1/4ずつ運用（1時間足）
  B 設定をずらす : k＝0.6・0.8・1.0 を1/3ずつ（日足・9時区切り）
  C 曜日         : 曜日ごとの成績（前半/後半）と、週末（土日）に入る取引を除いた場合
  D 全体の量を調整: 過去30日の損益の振れが大きい時は量を減らす（0.25〜2倍、目安は前半の振れの中央値）
  E トレンド部品 : 一目均衡表(BTC200日上)の代わりに、ドンチャン（10・20・55日高値で買い、半分の日数の安値で売り）を1/3ずつ
また、portfolio_lab の部品①〜⑧と主な組み合わせも同じ物差しで記録する。
使い方: python3 project/tools/diversify_lab.py <キャッシュ（BTC.csv, ETH.csv, BTC_1h.csv, ETH_1h.csv）> [older]
  older: ルールを決めた期間より前（2014〜2017年）のBTCで土台を確認
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
import intraday_lab as IL  # noqa: E402
import portfolio_lab as PF  # noqa: E402
from btceth_lab import ichimoku_pos, ind, load, state  # noqa: E402
from registry import metrics_row, record  # noqa: E402

RISK, START, SPLIT = PF.RISK, PF.START, PF.SPLIT
SW = {"BTC": 25.35, "ETH": 35.49}


def per_year(n, x):
    return n / max((x.index[-1] - x.index[0]).days / 365.25, 0.5)


def daily_vbo_ls(I, k=0.8, short_w=0.5):
    """日足の値幅ブレイク（買い＋売り×short_w）。毎日の損益（R）と取引数。"""
    L = sum(PF.vbo_side(I[a], 1, PF.SPEC[a][0], PF.SPEC[a][1], k=k) for a in ("BTC", "ETH"))
    S = sum(PF.vbo_side(I[a], -1, PF.SPEC[a][0], PF.SPEC[a][2], k=k) for a in ("BTC", "ETH"))
    n = int((L != 0).sum() + (S != 0).sum())
    return (L + short_w * S), n


def boundary_vbo(cache, utc_hour, k=0.8, short_w=0.5):
    """区切りを UTC utc_hour 時にした値幅ブレイク（買い＋売り×short_w）。決済した日（UTC）ごとの損益（R）。"""
    out, n = [], 0
    for a in ("BTC", "ETH"):
        D = IL.day_matrix_shift(os.path.join(cache, f"{a}_1h.csv"), utc_hour)
        IL.SWAP_LONG = SW[a] / 365 / 100
        for side, w in ((1, 1.0), (-1, short_w)):
            tr = IL.vbo(D, side, k, 5, 99, 24, 0.0005)
            n += len(tr)
            when = pd.DatetimeIndex(pd.to_datetime(tr.t)) + pd.Timedelta(hours=utc_hour + 24)
            out.append(pd.Series(tr.R.to_numpy() * w, index=when.normalize()))
    s = pd.concat(out).groupby(level=0).sum()
    return s, n


def donchian(I, n):
    H, L, c = pd.Series(I["h"]), pd.Series(I["l"]), I["c"]
    return state(c > H.rolling(n).max().shift(1).to_numpy(), c < L.rolling(max(n // 2, 2)).min().shift(1).to_numpy())


def trend_long(I, pos_fn):
    out = None
    for a in ("BTC", "ETH"):
        pos = pos_fn(I[a]) * I[a]["btc200"]
        s = PF.pos_side(I[a], pos, 1, PF.SPEC[a][0], PF.SPEC[a][1])
        out = s if out is None else out.add(s, fill_value=0)
    return out


def clip(x):
    x = x[x.index >= START]
    return x[x.index <= pd.Timestamp("2026-11-16")]


def line(name, x, n=None):
    r = metrics_row(x, per_year(n, x) if n else np.nan)
    print(f"  {name:34s} 年{r['年平均%']:+6.2f}% 落込{r['最大の落ち込み%']:5.2f}% 効率{r['効率']:4.2f} マイナスの年 {r['マイナスの年']:10s}"
          f" 最悪の年{r['最悪の年%']:+5.1f}% プラスの月{r['プラスの月%']:4.0f}% 連続マイナス月{r['最長連続マイナス月']} 回復最長{r['高値回復の最長日数']}日"
          + (f" ｜ 前半{r['前半の年平均%']:+.1f}%→後半{r['後半の年平均%']:+.1f}%" if r['後半の年平均%'] != "" else ""))
    return r


def main(cache):
    btc = load(cache, "BTC")
    I = {a: ind(load(cache, a), btc) for a in ("BTC", "ETH")}

    # ---- これまでの部品と組み合わせを同じ物差しで記録
    X = PF.build(cache)
    for c in X.columns:
        record(グループ="部品", 名前=c, 銘柄="BTC,ETH" if "金" not in c else "金", 時間足="日足", **metrics_row(clip(X[c])))
    base, n_base = daily_vbo_ls(I)
    base = clip(base * RISK)
    record(グループ="組み合わせ", 名前="①＋②×0.5（9時区切り）", 銘柄="BTC,ETH", 時間足="日足",
           メモ="5日平均の向きに値幅ブレイク（k=0.8）、当日決済", **metrics_row(base, per_year(n_base, base)))

    print("==== 土台：①＋②×0.5（9時区切り、日足）")
    line("土台", base, n_base)

    print("\n==== A 時間をずらす（1時間足、区切り＝日本時間）")
    parts = {}
    for jst in (3, 9, 15, 21):
        s, n = boundary_vbo(cache, (jst - 9) % 24)
        parts[jst] = (clip(s * RISK), n)
        line(f"{jst}時区切りだけ（量1）", parts[jst][0], n)
    combo = pd.concat([p[0] for p in parts.values()], axis=1).fillna(0).sum(axis=1) / 4
    n_all = sum(p[1] for p in parts.values())
    r = line("4つの時間帯を1/4ずつ", combo, n_all)
    record(グループ="分散", 名前="A 4つの時間帯（3・9・15・21時）を1/4ずつ", 銘柄="BTC,ETH", 時間足="1時間足",
           メモ="①＋②×0.5を区切りの時刻だけ変えて同時運用", **metrics_row(combo, per_year(n_all, combo)))

    print("\n==== B 設定をずらす（k＝0.6・0.8・1.0 を1/3ずつ、9時区切り）")
    ks = {}
    for k in (0.6, 0.8, 1.0):
        s, n = daily_vbo_ls(I, k=k)
        ks[k] = (clip(s * RISK), n)
        line(f"k={k}だけ（量1）", ks[k][0], n)
    kc = sum(v[0] for v in ks.values()) / 3
    nk = sum(v[1] for v in ks.values())
    line("3つを1/3ずつ", kc, nk)
    record(グループ="分散", 名前="B k=0.6・0.8・1.0を1/3ずつ", 銘柄="BTC,ETH", 時間足="日足", **metrics_row(kc, per_year(nk, kc)))

    print("\n==== C 曜日（土台の取引を、入った曜日で分ける）1回あたりの平均（R） 前半／後半")
    days = ["月", "火", "水", "木", "金", "土", "日"]
    raw = base / RISK
    for d in range(7):
        a = raw[(raw.index.weekday == d) & (raw.index < SPLIT) & (raw != 0)]
        b = raw[(raw.index.weekday == d) & (raw.index >= SPLIT) & (raw != 0)]
        print(f"  {days[d]}曜: {a.mean():+.3f}R（{len(a)}回）／ {b.mean():+.3f}R（{len(b)}回）")
    wk = base[base.index.weekday < 5]
    wk = wk.reindex(base.index).fillna(0)
    line("土日に入る取引を除く", wk)
    record(グループ="分散", 名前="C 土日に入る取引を除く", 銘柄="BTC,ETH", 時間足="日足", **metrics_row(wk))

    print("\n==== D 全体の量を、最近の損益の振れに合わせて調整（0.25〜2倍）")
    for name, x in (("土台", base), ("A 4つの時間帯", combo)):
        vol = x.rolling(30).std().shift(1)
        target = vol[vol.index < SPLIT].median()
        scaled = (x * (target / vol).clip(0.25, 2.0)).fillna(0)
        line(f"{name}・量を調整", scaled)
        record(グループ="分散", 名前=f"D {name}・振れに合わせて量を調整", 銘柄="BTC,ETH", 時間足="日足" if name == "土台" else "1時間足",
               メモ="過去30日の損益の振れで量を0.25〜2倍に調整（目安は前半の中央値）", **metrics_row(scaled))

    print("\n==== E トレンド部品（BTCが200日平均の上の時だけ買い）")
    ichi = clip(trend_long(I, ichimoku_pos) * RISK)
    line("⑧一目均衡表", ichi)
    dons = {n: clip(trend_long(I, lambda J, n=n: donchian(J, n)) * RISK) for n in (10, 20, 55)}
    for n, x in dons.items():
        line(f"ドンチャン{n}日", x)
        record(グループ="部品", 名前=f"ドンチャン{n}日（BTC200日上・買い）", 銘柄="BTC,ETH", 時間足="日足", **metrics_row(x))
    dc = sum(dons.values()) / 3
    line("ドンチャン10・20・55日を1/3ずつ", dc)
    record(グループ="部品", 名前="ドンチャン10・20・55日を1/3ずつ（BTC200日上・買い）", 銘柄="BTC,ETH", 時間足="日足", **metrics_row(dc))

    print("\n==== まとめの組み合わせ")
    finals = {
        "土台＋一目×0.5": base.add(0.5 * ichi, fill_value=0),
        "土台＋ドンチャン組×0.5": base.add(0.5 * dc, fill_value=0),
        "A 4つの時間帯＋ドンチャン組×0.5": combo.add(0.5 * dc, fill_value=0),
        "A 4つの時間帯＋一目×0.5": combo.add(0.5 * ichi, fill_value=0),
    }
    for name, x in finals.items():
        x = clip(x)
        line(name, x)
        record(グループ="組み合わせ", 名前=name, 銘柄="BTC,ETH", 時間足="日足/1時間足", **metrics_row(x))


if __name__ == "__main__" and len(sys.argv) == 2:
    main(sys.argv[1])


def older_period(cache):
    """ルールを決めた期間より前（2014年9月〜2017年12月）のBTCで、土台（①＋②×0.5）を確かめる。データ: BTC-USD（Yahoo日足）。"""
    df = pd.read_csv(os.path.join(cache, "BTC-USD_yahoo.csv"), index_col=0, parse_dates=True)
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    df = df[~df.index.duplicated()].sort_index()
    df = df[df.index < pd.Timestamp("2018-01-01")]
    I = ind(df, df)
    L = PF.vbo_side(I, 1, 0.0005, SW["BTC"])
    S = PF.vbo_side(I, -1, 0.0005, 0.0)
    x = ((L + 0.5 * S) * RISK)
    x = x[x.index >= pd.Timestamp("2014-10-15")]
    n = int((L != 0).sum() + (S != 0).sum())
    print("==== ルールを決めた期間より前（BTC 2014年10月〜2017年12月）での土台")
    r = line("土台（①＋②×0.5）BTCのみ", x, n)
    print("  年別: " + " ".join(f"{y}:{v:+.1f}%" for y, v in x.groupby(x.index.year).sum().items()))
    record(グループ="過去の別期間", 名前="土台（①＋②×0.5）BTC 2014-10〜2017-12", 銘柄="BTC", 時間足="日足",
           メモ="ルールを決めた2018年以降より前の期間で確認（Yahoo BTC-USD）", **metrics_row(x, per_year(n, x)))


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "older":
    older_period(sys.argv[1])
