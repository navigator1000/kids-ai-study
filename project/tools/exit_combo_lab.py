#!/usr/bin/env python3
"""(1) 値幅ブレイクの決済の工夫を15分足で確かめる  (2) デュアル・スラストと値幅ブレイクの組み合わせ（BTC・ETH、2018年〜）

(1) 15分足（1日96本・UTC0時区切り＝日本時間9時）で、買い＋売り×0.5の値幅ブレイク（k=0.8・5日平均の向き）の決済を比べる。
    そのまま（翌朝決済）／保険の損切りATR×2・×3／含み益がATR×0.3〜1で損切りを建値へ／高値からATR×1〜2戻ったら決済／
    入ってから6・12・18時間で決済／利確ATR×0.5〜3（届かなければ翌朝）／利確と損切りの組み合わせ
    足の中で高値と安値のどちらが先に付いたかはわからない。そこで次のように扱う。
      ・損切りを動かした足の終値が、新しい損切りより不利 → その足の中で確実にかかっている（甘め・厳しめとも損切り）
      ・「甘め」：それ以外は、動かした足の中では損切りにかからない
      ・「厳しめ」：動かした足の安値（売りは高値）が新しい損切りを越えていたら損切り
        （約定した足は、安値が約定より前に付いたかもしれないので、終値だけで判断）
    本当の成績は甘めと厳しめの間にある。1時間足・30分足でも同じ計算をして、足を細かくすると差が縮まるかを見る。
    利確だけ（損切りなし）の場合は順番の問題がない（買いなら、約定してから高値を付けるので）。
    確認: 銘柄別（BTCだけ・ETHだけ）、月ごとの損益の安定度（平均÷ばらつき、年換算）、1回の取引で一番大きな負け。
    スワップは、UTC21時（Vantageのサーバー時間の0時。冬は22時）をまたいで持った時だけ払う。
(2) デュアル・スラスト（4日間の値幅、k=0.6）の設定ずらしと、値幅ブレイクとの組み合わせ（量の配分・相関・同じ日に重なる割合）。
共通: 1R＝ATR1本分＝資金の0.5%（単利）、Vantage CFD想定。結果は results.csv に記録。
使い方: python3 project/tools/exit_combo_lab.py <キャッシュ> [combo|exits]
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
import more_methods_lab as MM  # noqa: E402
import portfolio_lab as PF  # noqa: E402
from btceth_lab import ind, load  # noqa: E402
from registry import drop, metrics_row, record  # noqa: E402

RISK, SPEC = 0.5, PF.SPEC
ROLL_UTC = 21          # スワップがかかる時刻（UTC）
GROUP = "値幅ブレイクの決済"


def trade_stats(t, x):
    yrs = max((x.index[-1] - x.index[0]).days / 365.25, 0.5)
    win, loss = t[t > 0], t[t <= 0]
    return dict(n_year=len(t) / yrs, wr=len(win) / len(t) * 100, aw=win.mean(), al=loss.mean())


def line(name, x, group, tf, memo="", t=None, quiet=False):
    st = trade_stats(t, x) if t is not None else None
    r = metrics_row(x, st["n_year"] if st else np.nan)
    if not quiet:
        extra = f" 勝率{st['wr']:3.0f}% 平均の勝ち{st['aw']:+.2f}R 負け{st['al']:+.2f}R" if st else ""
        print(f"  {name:36s} 年{r['年平均%']:+6.2f}% 落込{r['最大の落ち込み%']:5.2f}% 効率{r['効率']:5.2f} 安定度{r['安定度']:5.2f} マイナスの年 {r['マイナスの年']:4s}"
              f" 最悪の年{r['最悪の年%']:+5.1f}% プラスの月{r['プラスの月%']:4.0f}%{extra} ｜ 前半{r['前半の年平均%']:+.1f}%→後半{r['後半の年平均%']:+.1f}%")
    if st:
        memo = (memo + "。" if memo else "") + f"勝率{st['wr']:.0f}%・平均の勝ち{st['aw']:+.2f}R・平均の負け{st['al']:+.2f}R"
    record(グループ=group, 名前=name, 銘柄="BTC,ETH", 時間足=tf, メモ=memo, **r)
    return r


# ---------------------------------------------------------------- (2) 組み合わせ
def combo(cache):
    btc = load(cache, "BTC")
    I = {a: ind(load(cache, a), btc) for a in ("BTC", "ETH")}

    def vbo(a):
        return PF.vbo_side(I[a], 1, SPEC[a][0], SPEC[a][1]) + 0.5 * PF.vbo_side(I[a], -1, SPEC[a][0], SPEC[a][2])

    def dt(a, k=0.6, N=4):
        L_, _ = MM.dual_thrust(I[a], 1, a, k, N)
        S_, _ = MM.dual_thrust(I[a], -1, a, k, N)
        return L_ + 0.5 * S_

    print("==== デュアル・スラストの設定ずらし（買い＋売り×0.5、BTC+ETH）")
    cells = []
    for N in (2, 3, 4, 5, 7):
        row = []
        for k in (0.5, 0.6, 0.7):
            x = MM.clip((dt("BTC", k, N) + dt("ETH", k, N)) * RISK)
            r = metrics_row(x)
            row.append(f"k{k}:効率{r['効率']:.2f}・負け年{r['マイナスの年']}")
            cells.append(r)
        print(f"  {N}日: " + " ／ ".join(row))
    print(f"  → 15通り中 マイナスの年なし {sum(1 for r in cells if r['マイナスの年'] == 'なし')}通り、効率の中央値 {np.median([r['効率'] for r in cells]):.2f}")

    v = MM.clip((vbo("BTC") + vbo("ETH")) * RISK)
    d = MM.clip((dt("BTC") + dt("ETH")) * RISK)
    raw_v = {a: vbo(a) for a in ("BTC", "ETH")}
    raw_d = {a: dt(a) for a in ("BTC", "ETH")}
    both_days = sum(int(((raw_v[a] != 0) & (raw_d[a] != 0)).sum()) for a in raw_v)
    any_days = sum(int(((raw_v[a] != 0) | (raw_d[a] != 0)).sum()) for a in raw_v)
    corr_m = v.resample("ME").sum().corr(d.resample("ME").sum())
    print(f"\n==== 組み合わせ ｜ 同じ日に両方が入った割合 {both_days / any_days * 100:.0f}% ｜ 月ごとの損益の相関 {corr_m:.2f}")
    prio = MM.clip(sum((raw_v[a].where(raw_v[a] != 0, raw_d[a])) for a in raw_v) * RISK)
    plans = {
        "値幅ブレイクだけ": v,
        "デュアル・スラストだけ": d,
        "半分ずつ（合計の量は同じ）": 0.5 * v + 0.5 * d,
        "両方とも1倍（合計の量は2倍）": v + d,
        "値幅ブレイク優先、出ない日はデュアル・スラスト": prio,
    }
    for name, x in plans.items():
        line(f"組み合わせ：{name}", x, "値幅ブレイクの仲間", "日足")


# ---------------------------------------------------------------- (1) 決済の工夫（15分足など）
def day_matrix_n(path, per_day):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df[~df.index.duplicated()].sort_index()
    step = pd.Timedelta(hours=24) / per_day
    full = pd.date_range(df.index[0].normalize() + pd.Timedelta(days=1), df.index[-1].normalize() - step, freq=step)
    df = df.reindex(full).ffill()
    days = df.index.normalize().unique()
    nd = len(df) // per_day
    M = {k: df[k].to_numpy()[: nd * per_day].reshape(nd, per_day) for k in ("open", "high", "low", "close")}
    dO, dH, dL, dC = M["open"][:, 0], M["high"].max(1), M["low"].min(1), M["close"][:, -1]
    pc = np.r_[np.nan, dC[:-1]]
    tr = np.nanmax(np.vstack([dH - dL, abs(dH - pc), abs(dL - pc)]), axis=0)
    return dict(days=pd.DatetimeIndex(days[:nd]), M=M, dO=dO, atr=pd.Series(tr).rolling(14).mean().shift(1).to_numpy(),
                rng=np.r_[np.nan, (dH - dL)[:-1]], sma5=pd.Series(dC).rolling(5).mean().shift(1).to_numpy(), per_day=per_day)


def vbo_exit_n(D, a, side, mode, param=0.0, strict=False, k=0.8, init_stop=None, tp=None):
    """1日ごとの損益（R）。取引のない日は NaN。
    mode: そのまま／損切り（param=ATR倍）／建値に移す（param=含み益のATR倍）／高値から戻ったら決済（param=ATR倍）／
          時間で決済（param=時間）／利確（param=ATR倍）
    init_stop: 最初から置く損切り（ATR倍）。None なら置かない（「損切り」では param を使う）。
    tp: どのモードでも置ける利確（ATR倍）。同じ足で損切りと利確の両方に届いたら、損切りを先とみなす。"""
    O, H, L, C = D["M"]["open"], D["M"]["high"], D["M"]["low"], D["M"]["close"]
    dO, atr, rng, P = D["dO"], D["atr"], D["rng"], D["per_day"]
    ok = ~np.isnan(atr) & ~np.isnan(rng) & ((dO - D["sma5"]) * side > 0)
    trig = dO + side * k * rng
    cost = SPEC[a][0]
    sw = (SPEC[a][1] if side == 1 else SPEC[a][2]) / 36500
    roll = ROLL_UTC * P // 24
    R = np.full(len(dO), np.nan)
    last_entry = P - P // 24                                  # 最後の1時間には入らない
    for d in np.flatnonzero(ok):
        hit = np.flatnonzero((H[d] >= trig[d]) if side == 1 else (L[d] <= trig[d]))
        hit = hit[hit < last_entry]
        if not len(hit):
            continue
        h0 = hit[0]
        e = max(O[d, h0], trig[d]) if side == 1 else min(O[d, h0], trig[d])
        A = atr[d]
        first = param if mode == "損切り" else init_stop
        stop = None if first is None else e - side * first * A
        tpm = param if mode == "利確" else tp
        tp_px = None if tpm is None else e + side * tpm * A
        best, ex, out = e, None, None
        last = P if mode != "時間で決済" else min(h0 + int(round(param * P / 24)), P)
        for t in range(h0, last):
            adv = L[d, t] if side == 1 else H[d, t]
            fav = H[d, t] if side == 1 else L[d, t]
            if stop is not None and (adv - stop) * side <= 0:
                ex = stop if t == h0 else (min(O[d, t], stop) if side == 1 else max(O[d, t], stop))
                out = t
                break
            if tp_px is not None and (fav - tp_px) * side >= 0:
                ex = tp_px if t == h0 else (max(O[d, t], tp_px) if side == 1 else min(O[d, t], tp_px))
                out = t
                break
            if (fav - best) * side > 0:
                best = fav
            new = None
            if mode == "建値に移す" and (best - e) * side >= param * A:
                new = e
            elif mode == "高値から戻ったら決済":
                new = best - side * param * A
            if new is not None and (stop is None or (new - stop) * side > 0):
                stop = new
                worst = adv if (strict and t > h0) else C[d, t]
                if (worst - stop) * side <= 0:
                    ex, out = stop, t
                    break
        if ex is None:
            ex = C[d, last - 1]
            crosses = h0 < roll < last                       # 足の終わり（次の足の始まり）で決済
        else:
            crosses = h0 < roll <= out
        R[d] = side * (ex - e) / A - cost * e / A - (sw * e / A if crosses else 0)
    return pd.Series(R, index=D["days"])


def run_plan(D, mode, param=0.0, strict=False, init_stop=None, tp=None, coins=("BTC", "ETH")):
    """買い＋売り×0.5。毎日の損益（資金の%）と、取引ごとの損益（R）。"""
    x, trades = None, []
    for a in coins:
        for side, w in ((1, 1.0), (-1, 0.5)):
            r = vbo_exit_n(D[a], a, side, mode, param, strict, init_stop=init_stop, tp=tp)
            trades.append(r.dropna())
            s = (w * r).fillna(0)
            x = s if x is None else x.add(s, fill_value=0)
    return MM.clip(x * RISK), MM.clip(pd.concat(trades))


def lab(s):
    return "厳しめ" if s else "甘め"


PLANS = [  # (mode, param, strict, 最初の損切り, 利確, 名前)
    ("そのまま", 0, False, None, None, "そのまま（翌朝9時に決済）"),
    ("損切り", 2, False, None, None, "保険の損切りATR×2"),
    ("損切り", 3, False, None, None, "保険の損切りATR×3"),
    *[("建値に移す", p, s, None, None, f"含み益ATR×{p:g}で損切りを建値へ（{lab(s)}）") for p in (0.3, 0.5, 0.75, 1.0) for s in (False, True)],
    *[("建値に移す", 0.5, s, 3.0, None, f"保険の損切りATR×3＋含み益ATR×0.5で建値へ（{lab(s)}）") for s in (False, True)],
    *[("高値から戻ったら決済", p, s, None, None, f"高値からATR×{p:g}戻ったら決済（{lab(s)}）") for p in (1.0, 1.5, 2.0) for s in (False, True)],
    *[("時間で決済", h, False, None, None, f"入ってから{h}時間で決済") for h in (6, 12, 18)],
    *[("利確", p, False, None, None, f"利確ATR×{p:g}（届かなければ翌朝）") for p in (0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)],
    ("損切り", 3, False, None, 2.0, "利確ATR×2＋保険の損切りATR×3"),
    *[("建値に移す", 0.5, s, None, 2.0, f"利確ATR×2＋含み益ATR×0.5で建値へ（{lab(s)}）") for s in (False, True)],
]
CONV = [p for p in PLANS if p[4] is None and (p[0] == "そのまま" or (p[0] == "建値に移す" and p[1] == 0.5 and p[3] is None)
                                            or (p[0] == "高値から戻ったら決済" and p[1] == 1.0))]
CHECK = ["そのまま（翌朝9時に決済）", "保険の損切りATR×3", "含み益ATR×0.5で損切りを建値へ（甘め）", "含み益ATR×0.5で損切りを建値へ（厳しめ）",
         "利確ATR×1.5（届かなければ翌朝）", "利確ATR×2（届かなければ翌朝）", "利確ATR×2＋保険の損切りATR×3",
         "利確ATR×2＋含み益ATR×0.5で建値へ（甘め）", "利確ATR×2＋含み益ATR×0.5で建値へ（厳しめ）"]


def exits(cache):
    print(f"（古い「{GROUP}」の行を {drop(GROUP)} 行消して、計算し直す）")
    D = {a: day_matrix_n(os.path.join(cache, f"{a}_15m.csv"), 96) for a in ("BTC", "ETH")}
    print("==== 15分足で、値幅ブレイク（買い＋売り×0.5、9時区切り）の決済を比べる")
    keep = {}
    for mode, param, strict, init, tp, name in PLANS:
        x, t = run_plan(D, mode, param, strict, init, tp)
        line(f"値幅ブレイク（15分足）：{name}", x, GROUP, "15分足", t=t)
        keep[name] = x

    print("\n==== 確かめ：銘柄別（1回0.5%のまま）・月の損益の安定度・1回で一番大きな負け")
    rows = {}
    for mode, param, strict, init, tp, name in (p for p in PLANS if p[5] in CHECK):
        x, t = run_plan(D, mode, param, strict, init, tp)
        row = {"安定度": metrics_row(x)["安定度"], "一番大きな負け": f"{t.min():+.2f}R（{t.idxmin():%Y-%m-%d}）"}
        for a in ("BTC", "ETH"):
            xa, _ = run_plan(D, mode, param, strict, init, tp, coins=(a,))
            ra = metrics_row(xa)
            row[f"{a}だけ"] = f"{ra['年平均%']:+.2f}%／落込{ra['最大の落ち込み%']:.2f}%／負け年{ra['マイナスの年']}"
        rows[name] = row
    print(pd.DataFrame(rows).T.to_string())

    print("\n==== 年ごとの損益（%）")
    show = ["そのまま（翌朝9時に決済）", "含み益ATR×0.5で損切りを建値へ（甘め）", "含み益ATR×0.5で損切りを建値へ（厳しめ）",
            "保険の損切りATR×3", "利確ATR×2（届かなければ翌朝）", "利確ATR×2＋保険の損切りATR×3", "入ってから12時間で決済"]
    yearly = pd.DataFrame({n: keep[n].groupby(keep[n].index.year).sum() for n in show}).round(1)
    print(yearly.T.to_string())

    print("\n==== 足を細かくすると、甘め・厳しめの差は縮まるか（年平均％／効率）")
    table = {}
    for tf, per_day, f in (("1時間足", 24, "1h"), ("30分足", 48, "30m"), ("15分足", 96, "15m")):
        Dx = D if f == "15m" else {a: day_matrix_n(os.path.join(cache, f"{a}_{f}.csv"), per_day) for a in ("BTC", "ETH")}
        for mode, param, strict, init, tp, name in CONV:
            x, t = run_plan(Dx, mode, param, strict, init, tp)
            r = line(f"値幅ブレイク（{tf}）：{name}", x, GROUP, tf, t=t, quiet=True)
            table.setdefault(name, {})[tf] = f"{r['年平均%']:+.2f}%／{r['効率']:.2f}"
    print(pd.DataFrame(table).T.to_string())


if __name__ == "__main__":
    what = sys.argv[2] if len(sys.argv) > 2 else "combo"
    (combo if what == "combo" else exits)(sys.argv[1])
