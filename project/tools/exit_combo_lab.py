#!/usr/bin/env python3
"""(1) 値幅ブレイクの決済の工夫を15分足で確かめる  (2) デュアル・スラストと値幅ブレイクの組み合わせ（BTC・ETH、2018年〜）

(1) 15分足（1日96本・UTC0時区切り＝日本時間9時）で、買い＋売り×0.5の値幅ブレイク（k=0.8・5日平均の向き）の決済を比べる。
    そのまま（翌朝決済）／損切りATR×3／含み益がATR×0.5(1.0)で損切りを買値へ／高値からATR×1(1.5)戻ったら決済／12時間で決済
    「甘め」：損切りを動かした15分足の中では損切りにかからない。「厳しめ」：その15分足の中で戻っていたら損切り（約定した足は除く）。
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
from registry import metrics_row, record  # noqa: E402

RISK, SPEC = 0.5, PF.SPEC


def line(name, x, group, tf, memo=""):
    r = metrics_row(x)
    print(f"  {name:44s} 年{r['年平均%']:+6.2f}% 落込{r['最大の落ち込み%']:5.2f}% 効率{r['効率']:5.2f} マイナスの年 {r['マイナスの年']:10s}"
          f" 最悪の年{r['最悪の年%']:+5.1f}% プラスの月{r['プラスの月%']:4.0f}% ｜ 前半{r['前半の年平均%']:+.1f}%→後半{r['後半の年平均%']:+.1f}%")
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


# ---------------------------------------------------------------- (1) 15分足で決済の工夫
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
    return dict(days=days[:nd], M=M, dO=dO, atr=pd.Series(tr).rolling(14).mean().shift(1).to_numpy(),
                rng=np.r_[np.nan, (dH - dL)[:-1]], sma5=pd.Series(dC).rolling(5).mean().shift(1).to_numpy(), per_day=per_day)


def vbo_exit_n(D, a, side, mode, param=0.0, strict=False, k=0.8):
    O, H, L, C = D["M"]["open"], D["M"]["high"], D["M"]["low"], D["M"]["close"]
    dO, atr, rng, P = D["dO"], D["atr"], D["rng"], D["per_day"]
    ok = ~np.isnan(atr) & ~np.isnan(rng) & ((dO - D["sma5"]) * side > 0)
    trig = dO + side * k * rng
    cost = SPEC[a][0]
    sw = (SPEC[a][1] if side == 1 else SPEC[a][2]) / 36500
    R = np.zeros(len(dO))
    last_entry = P - P // 24                                  # 最後の1時間には入らない
    for d in np.flatnonzero(ok):
        hit = np.flatnonzero((H[d] >= trig[d]) if side == 1 else (L[d] <= trig[d]))
        hit = hit[hit < last_entry]
        if not len(hit):
            continue
        h0 = hit[0]
        e = max(O[d, h0], trig[d]) if side == 1 else min(O[d, h0], trig[d])
        A = atr[d]
        stop = None if mode == "そのまま" else e - side * 3 * A
        best, ex = e, None
        last = P if mode != "時間で決済" else min(h0 + int(param * P / 24), P)
        for t in range(h0, last):
            adv = L[d, t] if side == 1 else H[d, t]
            if stop is not None and (adv - stop) * side <= 0:
                ex = stop if t == h0 else (min(O[d, t], stop) if side == 1 else max(O[d, t], stop))
                break
            fav = H[d, t] if side == 1 else L[d, t]
            if (fav - best) * side > 0:
                best = fav
            moved = False
            if mode == "建値に移す" and (best - e) * side >= param * A and (e - stop) * side > 0:
                stop, moved = e, True
            if mode == "高値から戻ったら決済":
                trail = best - side * param * A
                if (trail - stop) * side > 0:
                    stop, moved = trail, True
            if strict and moved and t > h0 and (adv - stop) * side <= 0:
                ex = stop
                break
        overnight = True
        if ex is None:
            ex = C[d, last - 1]
            overnight = last == P
        R[d] = side * (ex - e) / A - cost * e / A - (sw * e / A if overnight else 0)
    return pd.Series(R, index=D["days"])


def exits(cache):
    D = {a: day_matrix_n(os.path.join(cache, f"{a}_15m.csv"), 96) for a in ("BTC", "ETH")}
    print("==== 15分足で、値幅ブレイク（買い＋売り×0.5、9時区切り）の決済を比べる")
    plans = [("そのまま", 0.0, False, "そのまま（翌朝決済）"), ("損切り", 0.0, False, "損切りATR×3"),
             ("建値に移す", 0.5, False, "含み益ATR×0.5で損切りを買値へ（甘め）"), ("建値に移す", 0.5, True, "含み益ATR×0.5で損切りを買値へ（厳しめ）"),
             ("建値に移す", 1.0, False, "含み益ATR×1で損切りを買値へ（甘め）"), ("建値に移す", 1.0, True, "含み益ATR×1で損切りを買値へ（厳しめ）"),
             ("高値から戻ったら決済", 1.0, False, "高値からATR×1戻ったら決済（甘め）"), ("高値から戻ったら決済", 1.0, True, "高値からATR×1戻ったら決済（厳しめ）"),
             ("高値から戻ったら決済", 1.5, False, "高値からATR×1.5戻ったら決済（甘め）"), ("高値から戻ったら決済", 1.5, True, "高値からATR×1.5戻ったら決済（厳しめ）"),
             ("時間で決済", 12, False, "12時間で決済")]
    for mode, param, strict, name in plans:
        x = None
        for a in ("BTC", "ETH"):
            s = vbo_exit_n(D[a], a, 1, mode, param, strict).add(0.5 * vbo_exit_n(D[a], a, -1, mode, param, strict), fill_value=0)
            x = s if x is None else x.add(s, fill_value=0)
        x.index = pd.DatetimeIndex(x.index)
        line(f"値幅ブレイク（15分足）：{name}", MM.clip(x * RISK), "値幅ブレイクの決済", "15分足")


if __name__ == "__main__":
    what = sys.argv[2] if len(sys.argv) > 2 else "combo"
    (combo if what == "combo" else exits)(sys.argv[1])
