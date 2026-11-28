#!/usr/bin/env python3
"""デイトレードの計画書（project/research/daytrade_plan.md）第1回：優先「高」の19件。

  TA01〜TA04 寄り付き・時間帯のレンジのブレイク（東京・ロンドン・ニューヨーク・アジア時間）
  TB01・TB02 時間帯の勢い（アジア時間の続き、1日の最初の数時間の勢い）
  TC01〜TC03 短い足のトレンド（15分足のドンチャン、移動平均の向き、数時間の勢い）
  TD01 急な動きの後の戻り（と、同じ向きに乗る形）
  TE01 第1回の形に「静かな日だけ」をかける
  TG01・TG02 成行の買いの比率、出来高の急増した足
  TH01 BTCが先に動き、ETHが後を追う
  TJ01・TJ02 CMEの窓（週末）、資金調達の時刻の前後
  TN03 短い時間の値動きの続きやすさ（土台の確認）
  TP01 偶然の目安（ランダムに入った場合）・TP02 コストへの強さ（すべての形を往復0%・0.05%・0.10%で）
共通: Binance の15分足（出来高つき）。取引は日本時間9時〜翌5時（UTCの0〜80本目）で、5時には決済（形によって書く）。
      1R＝日足のATR1本分の値動き＝資金の0.5%（単利）。往復0.05%、スワップは日本時間7時（夏は6時）をまたいだ買いに1日分・金曜3日分・土日なし。
      合図は足が終わってから見て次の足で入る。逆指値は足の中で約定、損切りと利確が同じ足なら損切りを先とみなす。
      前半（2018〜2022年）で選び、後半（2023年〜）で確かめる。結果は results.csv（グループ「計画・デイトレ第1回」）と daytrade_plan.csv。
使い方: python3 project/tools/dt1_lab.py <キャッシュ（BTC_15m_vol.csv・ETH_15m_vol.csv・BTC_15m.csv・ETH_15m.csv）> [項目（カンマ区切り）]
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import more_methods_lab as MM  # noqa: E402
import portfolio_lab as PF  # noqa: E402
import round1_lab as R1  # noqa: E402
import round4_lab as R4  # noqa: E402
import round7_lab as R7  # noqa: E402
import vantage_exit_time as VX  # noqa: E402
from registry import drop, metrics_row, record  # noqa: E402

warnings.filterwarnings("ignore")
GROUP = "計画・デイトレ第1回"
COINS = ("BTC", "ETH")
DAY = 96
COST = 0.0005
END = 80                      # 日本時間5時（UTC20時）。この足から先は持たない
SWMULT = np.array([1, 1, 1, 1, 3, 0, 0])
SUM, BEST, BUILD = {}, {}, {}


# ---------------------------------------------------------------- データ
def load(cache):
    raw = {}
    for a in COINS:
        df = pd.read_csv(os.path.join(cache, f"{a}_15m_vol.csv"), index_col=0, parse_dates=True)
        raw[a] = df[~df.index.duplicated()].sort_index()
    start = max(raw[a].index[0] for a in COINS).normalize() + pd.Timedelta(days=1)
    end = min(raw[a].index[-1] for a in COINS).normalize()
    full = pd.date_range(start, end, freq="15min", inclusive="left")
    nd = len(full) // DAY
    days = pd.DatetimeIndex(full[::DAY][:nd])
    X = {}
    for a in COINS:
        df = raw[a].reindex(full)
        for c in ("open", "high", "low", "close"):
            df[c] = df[c].ffill()
        for c in ("volume", "quote_volume", "trades", "taker_buy"):
            df[c] = df[c].fillna(0.0)
        x = {k: np.ascontiguousarray(df[c].to_numpy()[:nd * DAY]) for k, c in
             (("O", "open"), ("H", "high"), ("L", "low"), ("C", "close"), ("V", "volume"), ("TB", "taker_buy"))}
        M = {k: x[k].reshape(nd, DAY) for k in ("O", "H", "L", "C")}
        dO, dH, dL, dC = M["O"][:, 0], M["H"].max(1), M["L"].min(1), M["C"][:, -1]
        pc = np.r_[np.nan, dC[:-1]]
        tr = np.nanmax(np.vstack([dH - dL, np.abs(dH - pc), np.abs(dL - pc)]), axis=0)
        x.update(M=M, idx=days, nd=nd, dO=dO, dH=dH, dL=dL, dC=dC, prevC=pc, rng=np.r_[np.nan, (dH - dL)[:-1]],
                 atr=pd.Series(tr).rolling(14).mean().shift(1).to_numpy(), bd=np.repeat(np.arange(nd), DAY),
                 bod=np.tile(np.arange(DAY), nd))
        x["q"] = R4.mult(x, "静かな日だけ") > 0
        X[a] = x
    b = X["BTC"]
    sdir = np.nan_to_num(np.sign(b["dO"] - R1.sma_prev(b, 5)))
    roll = np.where(VX.summer(days), 84, 88)
    for a in COINS:
        X[a].update(sdir=sdir, roll=roll, wd=days.dayofweek.to_numpy())
    return X


def uk_summer(days):
    """英国の夏時間（3月の最終日曜〜10月の最終日曜）。"""
    out = np.zeros(len(days), bool)
    for y in range(days[0].year, days[-1].year + 1):
        mar, octo = pd.Timestamp(y, 3, 31), pd.Timestamp(y, 10, 31)
        s = mar - pd.Timedelta(days=(mar.dayofweek + 1) % 7)
        e = octo - pd.Timedelta(days=(octo.dayofweek + 1) % 7)
        out |= np.asarray((days >= s) & (days < e))
    return out


# ---------------------------------------------------------------- 損益の計算（2つの形）
def res_pos(x, a, pos):
    """建玉の並び（15分足ごと、前の足の終値で決めてその足を持つ）から、日ごとの損益（コスト前・コスト・スワップ、資金の%）と取引の数字。"""
    C = x["C"]
    N = len(C)
    bd = x["bd"]
    A = x["atr"][bd]
    pos = np.where(np.isfinite(A), np.nan_to_num(pos), 0.0)
    pc = np.r_[C[0], C[:-1]]
    g = 0.5 * pos * (C - pc) / np.where(np.isfinite(A), A, 1.0)
    dpos = np.abs(np.diff(np.r_[0.0, pos]))
    c = 0.5 * dpos * (COST / 2) * pc / np.where(np.isfinite(A), A, 1.0)
    sw = np.zeros(N)
    rb = np.arange(x["nd"]) * DAY + x["roll"]
    k = (rb > 0) & (rb < N)
    rb, dd = rb[k], np.arange(x["nd"])[k]
    held = (pos[rb - 1] > 0) & (pos[rb] > 0)
    sw[rb[held]] = 0.5 * pos[rb[held]] * PF.SPEC[a][1] / 36500 * C[rb[held]] / A[rb[held]] * SWMULT[x["wd"][dd[held]]]
    nd = x["nd"]
    sg = np.sign(pos)
    ent = (sg != 0) & (np.r_[0.0, sg[:-1]] != sg)
    seg = np.cumsum(ent)
    on = sg != 0
    ret = np.bincount(seg[on], weights=(sg * (C - pc) / pc)[on], minlength=seg.max() + 1)[1:]
    hold = np.bincount(seg[on], minlength=seg.max() + 1)[1:]
    eday = bd[ent]
    return dict(g=np.bincount(bd, weights=g, minlength=nd), c=np.bincount(bd, weights=c, minlength=nd),
                s=np.bincount(bd, weights=sw, minlength=nd), ret=ret, hold=hold * 0.25, eday=eday)


def res_trades(x, a, day, side, e, xp, w, ebar, xbar):
    """取引の一覧（入った日・向き・値段・量・入った足・出た足）から、日ごとの損益。
    買いがスワップの時刻（日本時間7時、夏は6時）をまたいだら、その日の曜日の日数分のスワップを引く。"""
    nd = x["nd"]
    A = x["atr"][day]
    g = 0.5 * w * side * (xp - e) / A
    c = 0.5 * w * COST * e / A
    units = np.zeros(len(day))
    for k in range(4):                      # 入った日から4日先までのスワップの時刻
        dk = np.minimum(day + k, nd - 1)
        rb = dk * DAY + x["roll"][dk]
        cross = (day + k < nd) & (ebar < rb) & (xbar >= rb)
        units += np.where(cross, SWMULT[x["wd"][dk]], 0)
    sw = np.where(side == 1, 0.5 * w * PF.SPEC[a][1] / 36500 * e / A * units, 0.0)
    return dict(g=np.bincount(day, weights=g, minlength=nd), c=np.bincount(day, weights=c, minlength=nd),
                s=np.bincount(day, weights=sw, minlength=nd), ret=side * (xp - e) / e, hold=(xbar - ebar + 1) * 0.25, eday=day)


def empty(x):
    nd = x["nd"]
    return dict(g=np.zeros(nd), c=np.zeros(nd), s=np.zeros(nd), ret=np.zeros(0), hold=np.zeros(0), eday=np.zeros(0, int))


@njit
def brk_engine(O, H, L, C, start, end, xbar, up, dn, slu, sld, tpu, tpd, ok):
    """日ごとに、start〜end 本目の間で up（買い）・dn（売り）の逆指値の先に届いた方に入る（同じ足で両方なら見送り）。
    決済は xbar 本目の始まり（その前の足の終値）か、損切り・利確。入った足で損切りにも届いていたら、損切りを先とみなす。"""
    nd = start.shape[0]
    N = O.shape[0]
    rd = np.empty(nd, np.int64)
    rs = np.empty(nd, np.int64)
    re = np.empty(nd)
    rx = np.empty(nd)
    rb = np.empty(nd, np.int64)
    rxb = np.empty(nd, np.int64)
    nt = 0
    for d in range(nd):
        if not ok[d]:
            continue
        b0 = d * 96
        last = b0 + xbar[d]
        if last > N:
            continue
        k = -1
        side = 0
        e = 0.0
        for j in range(b0 + start[d], min(b0 + end[d], last)):
            u = (not np.isnan(up[d])) and H[j] >= up[d]
            v = (not np.isnan(dn[d])) and L[j] <= dn[d]
            if u and v:
                break
            if u:
                k, side, e = j, 1, max(O[j], up[d])
                break
            if v:
                k, side, e = j, -1, min(O[j], dn[d])
                break
        if k < 0:
            continue
        sl = slu[d] if side == 1 else sld[d]
        tp = tpu[d] if side == 1 else tpd[d]
        x = C[last - 1]
        xb = last - 1
        for j in range(k, last):
            if side == 1:
                if (not np.isnan(sl)) and L[j] <= sl:
                    x = sl if j == k else min(O[j], sl)
                    xb = j
                    break
                if j > k and (not np.isnan(tp)) and H[j] >= tp:
                    x = max(O[j], tp)
                    xb = j
                    break
            else:
                if (not np.isnan(sl)) and H[j] >= sl:
                    x = sl if j == k else max(O[j], sl)
                    xb = j
                    break
                if j > k and (not np.isnan(tp)) and L[j] <= tp:
                    x = min(O[j], tp)
                    xb = j
                    break
        rd[nt], rs[nt], re[nt], rx[nt], rb[nt], rxb[nt] = d, side, e, x, k, xb
        nt += 1
    return rd[:nt], rs[:nt], re[:nt], rx[:nt], rb[:nt], rxb[:nt]


def brk(x, a, start, end, xbar, up, dn, slu=None, sld=None, tpu=None, tpd=None, ok=None, wl=1.0, ws=1.0, mask=None):
    nd = x["nd"]
    nan = np.full(nd, np.nan)
    f = lambda v: np.asarray(nan if v is None else v, float)
    as_i = lambda v: np.asarray(np.broadcast_to(v, (nd,)), np.int64)
    okv = np.isfinite(x["atr"]) & (np.ones(nd, bool) if ok is None else ok) & (np.ones(nd, bool) if mask is None else mask)
    d, s, e, xp, eb, xb = brk_engine(x["O"], x["H"], x["L"], x["C"], as_i(start), as_i(end), as_i(xbar), f(up), f(dn),
                                     f(slu), f(sld), f(tpu), f(tpd), okv)
    return res_trades(x, a, d, s, e, xp, np.where(s == 1, wl, ws), eb, xb)


@njit
def chan_pos(C, hiN, loN, hiM, loM, al, ash, win):
    n = C.shape[0]
    pos = np.zeros(n)
    p = 0.0
    for t in range(1, n):
        if not win[t]:
            p = 0.0
        else:
            c = C[t - 1]
            if p == 0.0:
                if al[t] and c > hiN[t - 1]:
                    p = 1.0
                elif ash[t] and c < loN[t - 1]:
                    p = -1.0
            elif p > 0.0 and c < loM[t - 1]:
                p = 0.0
            elif p < 0.0 and c > hiM[t - 1]:
                p = 0.0
        pos[t] = p
    return pos


@njit
def hold_after(sig, H, win):
    """足 t の終値で出た合図（+1・−1）で、t+1〜t+H 本目を持つ（新しい合図が出たら、そこから数え直す）。取引時間の外は持たない。"""
    n = sig.shape[0]
    pos = np.zeros(n)
    left = 0
    cur = 0.0
    for t in range(n):
        if not win[t]:
            left = 0
        elif left > 0:
            pos[t] = cur
            left -= 1
        if win[t] and sig[t] != 0.0:
            cur = sig[t]
            left = H
    return pos


def window(x, end=END):
    return (x["bod"] < end) & np.isfinite(x["atr"][x["bd"]])


def day_mask(x, mask):
    return np.ones(len(x["C"]), bool) if mask is None else np.asarray(mask)[x["bd"]]


# ---------------------------------------------------------------- 数字と記録
def total(res, costmult=1.0, coins=COINS, idx=None):
    s = None
    for a in coins:
        r = res[a]
        v = pd.Series(r["g"] - costmult * r["c"] - r["s"], index=idx)
        s = v if s is None else s.add(v, fill_value=0)
    return MM.clip(s)


def trades_in(res, idx, coins=COINS):
    n, ret, hold = 0, [], []
    for a in coins:
        r = res[a]
        dd = idx[r["eday"]] if len(r["eday"]) else pd.DatetimeIndex([])
        k = np.asarray((dd >= MM.START) & (dd <= MM.END)) if len(dd) else np.zeros(0, bool)
        n += int(k.sum())
        ret.append(np.asarray(r["ret"])[k])
        hold.append(np.asarray(r["hold"])[k])
    return n, np.concatenate(ret) if ret else np.zeros(0), np.concatenate(hold) if hold else np.zeros(0)


def show(ctx, pid, name, res, memo="", coins=COINS):
    idx = ctx["idx"]
    r = R1.stats(total(res, 1.0, coins, idx))
    r0 = R1.stats(total(res, 0.0, coins, idx))
    r2 = R1.stats(total(res, 2.0, coins, idx))
    n, ret, hold = trades_in(res, idx, coins)
    yrs = (MM.END - MM.START).days / 365.25
    r["_n"] = n / yrs
    r["年の取引回数"] = round(n / yrs, 1)
    gross = float(np.mean(ret) * 100) if len(ret) else np.nan
    wr = float(np.mean(ret - COST > 0) * 100) if len(ret) else np.nan
    hh = float(np.mean(hold)) if len(hold) else np.nan
    cb = {a: metrics_row(MM.clip(pd.Series(res[a]["g"] - res[a]["c"] - res[a]["s"], index=idx)))["年平均%"] for a in coins}
    r.update(_x=total(res, 1.0, coins, idx), _r0=r0, _r2=r2, _gross=gross, _wr=wr, _hold=hh, _coins=cb)
    ratio = gross / (COST * 100) if gross == gross else np.nan
    extra = (f" ｜コストなし{r0['年平均%']:+.1f}%（{r0['安定度']:.2f}）・倍{r2['年平均%']:+.1f}%（{r2['安定度']:.2f}）・1回{gross:+.3f}%"
             f"（コストの{ratio:.1f}倍）・勝率{wr:.0f}%・{hh:.1f}時間・" + " ".join(f"{a}{v:+.1f}%" for a, v in cb.items()))
    R1.line(name, r, extra)
    record(グループ=GROUP, 計画ID=pid, 名前=f"{pid} {name}", 銘柄=",".join(coins), 時間足="15分足",
           メモ=(memo + "。" if memo else "") + f"コストなし年{r0['年平均%']:+.2f}%・安定度{r0['安定度']:.2f}、往復0.10%で年{r2['年平均%']:+.2f}%・"
                                              f"安定度{r2['安定度']:.2f}。1回の平均{gross:+.3f}%（コスト前、値段に対して）・勝率{wr:.0f}%・平均{hh:.1f}時間。"
                                              + "・".join(f"{a}だけ年{v:+.1f}%" for a, v in cb.items()),
           **{k: v for k, v in r.items() if not k.startswith("_")})
    return r


def judge(ctx, pid, rs, coins=COINS, family=None):
    """前半の安定度で一番良い形を選び、1次・2次・3次を判定する。3次は S4案に、S4案の半分の振れ幅にそろえて足す。
    family：向きが逆の形を混ぜて調べた項目で、一番良い形と同じ考え方の設定だけで「7割以上がプラス」を見るための関数。"""
    ok = {k: v for k, v in rs.items() if v["_h1s"] == v["_h1s"]}
    bk = max(ok, key=lambda k: ok[k]["_h1s"])
    r = rs[bk]
    neg = 0 if r["マイナスの年"] == "なし" else len(str(r["マイナスの年"]).split(","))
    p1 = r["年平均%"] > 0 and float(r["前半の年平均%"] or -1) > 0 and float(r["後半の年平均%"] or -1) > 0 and r["_n"] >= 100
    fam = rs if family is None else {k: v for k, v in rs.items() if family(k) == family(bk)}
    frac = np.mean([v["年平均%"] > 0 for v in fam.values()])
    p2 = p1 and frac >= 0.7 and all(v > 0 for v in r["_coins"].values()) and r["安定度"] >= 0.8 and neg <= 2 and r["_r2"]["年平均%"] > 0
    s4 = ctx["s4"]
    x = r["_x"]
    sm, xm = s4.resample("ME").sum(), x.resample("ME").sum()
    corr = sm.corr(xm)
    sc = 0.5 * sm.std() / xm.std() if xm.std() > 0 else 0.0
    mix = R1.stats(s4.add(sc * x, fill_value=0))
    p3 = p2 and corr < 0.5 and mix["安定度"] > ctx["s4r"]["安定度"]
    v = "3次合格" if p3 else ("2次合格・3次不合格" if p2 else ("1次合格・2次不合格" if p1 else "1次不合格"))
    R1.line(f"→ S4案＋「{bk}」（S4案の半分の振れ幅）", mix, f" ｜月の相関{corr:+.2f}・量は{sc:.1f}倍")
    record(グループ=GROUP, 計画ID=pid, 名前=f"{pid} 値幅ブレイク（S4案）＋{bk}×0.5", 銘柄="BTC,ETH", 時間足="15分足・日足",
           メモ=f"S4案の半分の振れ幅にそろえて足す（量は{sc:.1f}倍）。月の相関{corr:+.2f}", **{k: v for k, v in mix.items() if not k.startswith("_")})
    print(f"    判定：{v}（{len(rs)}通り中{sum(v_['年平均%'] > 0 for v_ in rs.values())}通りがプラス"
          + (f"、同じ考え方の{len(fam)}通り中{sum(v_['年平均%'] > 0 for v_ in fam.values())}通り" if family else "") + f"、前半で一番良い形は「{bk}」）")
    SUM[pid] = dict(verdict=v, best=bk, r=r, n=len(fam), npos=sum(v_["年平均%"] > 0 for v_ in fam.values()), corr=corr, mix=mix, sc=sc)
    BEST[pid] = bk
    return SUM[pid]


def run_variants(ctx, pid, title, variants, coins=COINS, family=None):
    """variants: {表示名: 関数(mask) → {銘柄: 結果}}。すべて計算して記録し、判定する。"""
    print(f"\n==== {pid} {title}")
    rs = {}
    BUILD[pid] = variants
    for lab, fn in variants.items():
        rs[lab] = show(ctx, pid, lab, fn(None), coins=coins)
    return judge(ctx, pid, rs, coins, family)


# ---------------------------------------------------------------- 各項目
def sec_tn03(ctx):
    X = ctx["X"]
    print("\n==== TN03 短い時間の値動きの続きやすさ（隣どうしの騰落の相関。プラスなら続きやすく、マイナスなら戻りやすい）")
    h2 = ctx["idx"] >= pd.Timestamp("2023-01-01")
    out = {}
    for a in COINS:
        x = X[a]
        lr = np.diff(np.log(x["C"]))
        bd = x["bd"][1:]
        ok = (ctx["idx"][bd] >= MM.START) & (ctx["idx"][bd] <= MM.END)
        row = {}
        for q, lab in ((1, "15分"), (4, "1時間"), (16, "4時間"), (96, "1日")):
            n = (len(lr) // q) * q
            r = lr[:n].reshape(-1, q).sum(1)
            okq = ok[:n].reshape(-1, q).all(1)
            hq = np.asarray(h2[bd[:n].reshape(-1, q)[:, 0]])
            c = lambda m: float(pd.Series(r[m]).autocorr(1))
            row[lab] = (c(okq & ~hq), c(okq & hq))
        vr = {}
        for q in (4, 16, 96):
            r1 = lr[ok]
            n = (len(r1) // q) * q
            vr[q] = float(np.var(r1[:n].reshape(-1, q).sum(1)) / (q * np.var(r1)))
        out[a] = dict(row=row, vr=vr)
        print(f"  {a}: " + "  ".join(f"{k} 前半{v[0]:+.3f}・後半{v[1]:+.3f}" for k, v in row.items())
              + " ｜分散比 " + "・".join(f"{ {4: '1時間', 16: '4時間', 96: '1日'}[q]}{v:.2f}" for q, v in vr.items()))
    SUM["TN03"] = out


def sec_tp01(ctx):
    X = ctx["X"]
    print("\n==== TP01 偶然の目安：時刻と向きをランダムに選んで入った場合（それぞれ300回。安定度の中央値・上位5%、コストなしと往復0.05%）")
    rng = np.random.default_rng(0)
    out = {}
    for per_day, H in ((1, 4), (1, 16), (1, 48), (3, 4), (3, 16)):
        st0, st1, a1 = [], [], []
        for _ in range(300):
            res = {}
            for a in COINS:
                x = X[a]
                nd = x["nd"]
                pos = np.zeros(len(x["C"]))
                for _k in range(per_day):
                    s = rng.integers(0, END - H + 1, nd)
                    sd = rng.choice([-1.0, 1.0], nd)
                    b = np.arange(nd) * DAY + s
                    for o in range(H):
                        pos[b + o] = sd
                res[a] = res_pos(x, a, pos * window(x))
            m0 = R1.stats(total(res, 0.0, idx=ctx["idx"]))
            m1 = R1.stats(total(res, 1.0, idx=ctx["idx"]))
            st0.append(m0["安定度"])
            st1.append(m1["安定度"])
            a1.append(m1["年平均%"])
        st0, st1, a1 = map(np.array, (st0, st1, a1))
        out[(per_day, H)] = dict(s0_med=np.median(st0), s0_95=np.quantile(st0, 0.95), s1_med=np.median(st1), s1_95=np.quantile(st1, 0.95),
                                 a1_med=np.median(a1))
        o = out[(per_day, H)]
        print(f"  1銘柄1日{per_day}回・{H * 0.25:g}時間：コストなしの安定度 中央値{o['s0_med']:+.2f}・上位5%{o['s0_95']:+.2f} ／ 往復0.05%で 中央値{o['s1_med']:+.2f}"
              f"・上位5%{o['s1_95']:+.2f}（年平均の中央値{o['a1_med']:+.1f}%）")
    SUM["TP01"] = out


def sec_ta(ctx):
    X, idx = ctx["X"], ctx["idx"]
    nan = None
    # TA01 東京
    v = {}
    for rl in (2, 4):
        for xb, xl in ((32, "17時"), (56, "23時"), (80, "翌5時")):
            for sl in (True, False):
                def fn(mask, rl=rl, xb=xb, sl=sl):
                    out = {}
                    for a in COINS:
                        x = X[a]
                        hi, lo = x["M"]["H"][:, :rl].max(1), x["M"]["L"][:, :rl].min(1)
                        out[a] = brk(x, a, rl, xb - 1, xb, hi, lo, lo if sl else nan, hi if sl else nan, mask=mask if mask is None else mask[a])
                    return out
                v[f"東京{rl * 15}分のレンジ・決済{xl}・損切り{'あり' if sl else 'なし'}"] = fn
    run_variants(ctx, "TA01", "東京の寄り付き（日本時間9時）のレンジのブレイク（先に約定した方。損切りはレンジの反対側）", v)
    # TA02 ロンドン
    uk = uk_summer(idx)
    v = {}
    for rl in (2, 4):
        for xb, xl in ((56, "23時"), (80, "翌5時")):
            for sl in (True, False):
                def fn(mask, rl=rl, xb=xb, sl=sl):
                    out = {}
                    s0 = np.where(uk, 28, 32)
                    for a in COINS:
                        x = X[a]
                        cols = s0[:, None] + np.arange(rl)[None, :]
                        hi = np.take_along_axis(x["M"]["H"], cols, 1).max(1)
                        lo = np.take_along_axis(x["M"]["L"], cols, 1).min(1)
                        out[a] = brk(x, a, s0 + rl, xb - 1, xb, hi, lo, lo if sl else nan, hi if sl else nan, mask=None if mask is None else mask[a])
                    return out
                v[f"ロンドン{rl * 15}分・決済{xl}・損切り{'あり' if sl else 'なし'}"] = fn
    run_variants(ctx, "TA02", "ロンドンの寄り付き（日本時間16時、冬は17時）のレンジのブレイク", v)
    # TA03 ニューヨーク
    us = VX.summer(idx)
    v = {}
    for rl in (2, 4):
        for sl in (True, False):
            def fn(mask, rl=rl, sl=sl):
                out = {}
                s0 = np.where(us, 54, 58)
                xb = np.where(us, 80, 84)
                for a in COINS:
                    x = X[a]
                    cols = s0[:, None] + np.arange(rl)[None, :]
                    hi = np.take_along_axis(x["M"]["H"], cols, 1).max(1)
                    lo = np.take_along_axis(x["M"]["L"], cols, 1).min(1)
                    out[a] = brk(x, a, s0 + rl, xb - 1, xb, hi, lo, lo if sl else nan, hi if sl else nan, mask=None if mask is None else mask[a])
                return out
            v[f"ニューヨーク{rl * 15}分・米国株の終わりに決済・損切り{'あり' if sl else 'なし'}"] = fn
    run_variants(ctx, "TA03", "ニューヨークの寄り付き（米国株の始まり）のレンジのブレイク", v)
    print("    ---- 実行の確認：決済を米国株の終わりの代わりに、翌朝8時15分（日本時間）にする（スワップの時刻をまたぐ）")
    for rl in (2, 4):
        out = {}
        s0 = np.where(us, 54, 58)
        for a in COINS:
            x = X[a]
            cols = s0[:, None] + np.arange(rl)[None, :]
            hi = np.take_along_axis(x["M"]["H"], cols, 1).max(1)
            lo = np.take_along_axis(x["M"]["L"], cols, 1).min(1)
            out[a] = brk(x, a, s0 + rl, np.where(us, 80, 84) - 1, 93, hi, lo, lo, hi)
        show(ctx, "TA03", f"ニューヨーク{rl * 15}分・損切りあり・決済は翌朝8時15分（入るのは米国株の終わりまで）", out,
             memo="実行の確認。スワップの時刻をまたぐ")
    # TA04 アジア時間
    v = {}
    for sl in (True, False):
        for buf in (0.0, 0.1):
            def fn(mask, sl=sl, buf=buf):
                out = {}
                for a in COINS:
                    x = X[a]
                    hi, lo = x["M"]["H"][:, :28].max(1), x["M"]["L"][:, :28].min(1)
                    b = buf * x["atr"]
                    out[a] = brk(x, a, 28, END - 1, END, hi + b, lo - b, lo if sl else nan, hi if sl else nan, mask=None if mask is None else mask[a])
                return out
            v[f"アジア時間のレンジ・余裕ATR×{buf}・損切り{'あり' if sl else 'なし'}"] = fn
    run_variants(ctx, "TA04", "アジア時間（日本時間9〜16時）のレンジを、16時以降に抜けたらその向き（翌5時に決済）", v)


def sec_tb(ctx):
    X = ctx["X"]
    v = {}
    for thr in (0.0, 0.25, 0.5):
        for lo in (False, True):
            def fn(mask, thr=thr, lo=lo):
                out = {}
                for a in COINS:
                    x = X[a]
                    M = x["M"]
                    ra = M["C"][:, 31] - x["prevC"]
                    sd = np.where(np.abs(ra) >= thr * x["atr"], np.sign(ra), 0.0)
                    if lo:
                        sd = np.clip(sd, 0, None)
                    if mask is not None:
                        sd = sd * mask[a]
                    P = np.zeros((x["nd"], DAY))
                    P[:, 32:END] = np.nan_to_num(sd)[:, None]
                    out[a] = res_pos(x, a, P.ravel())
                return out
            v[f"9〜17時の向き（下限ATR×{thr}）・{'買いだけ' if lo else '買い＋売り'}"] = fn
    run_variants(ctx, "TB01", "アジア時間（9〜17時）の向きに、17時〜翌5時を持つ（スワップの前に決済）", v)
    v = {}
    for N in (1, 2, 4, 8):
        for thr in (0.0, 0.25):
            def fn(mask, N=N, thr=thr):
                out = {}
                nb = 4 * N
                for a in COINS:
                    x = X[a]
                    r = x["M"]["C"][:, nb - 1] - x["prevC"]
                    sd = np.where(np.abs(r) >= thr * x["atr"], np.sign(r), 0.0)
                    if mask is not None:
                        sd = sd * mask[a]
                    P = np.zeros((x["nd"], DAY))
                    P[:, nb:END] = np.nan_to_num(sd)[:, None]
                    out[a] = res_pos(x, a, P.ravel())
                return out
            v[f"最初の{N}時間の向き（下限ATR×{thr}）"] = fn
    run_variants(ctx, "TB02", "1日（9時〜）の最初のN時間の向きに、その後〜翌5時を持つ", v)


def sec_tc(ctx):
    X = ctx["X"]

    def sides(x, filt, lo):
        al = np.ones(len(x["C"]), bool)
        ash = np.ones(len(x["C"]), bool)
        if filt:
            s = x["sdir"][x["bd"]]
            al, ash = s > 0, s < 0
        if lo:
            ash = np.zeros(len(x["C"]), bool)
        return al, ash
    v = {}
    for N in (20, 48, 96):
        for filt in (False, True):
            for lo in (False, True):
                def fn(mask, N=N, filt=filt, lo=lo):
                    out = {}
                    for a in COINS:
                        x = X[a]
                        Hs, Ls = pd.Series(x["H"]), pd.Series(x["L"])
                        hiN, loN = Hs.rolling(N).max().shift(1).to_numpy(), Ls.rolling(N).min().shift(1).to_numpy()
                        hiM, loM = Hs.rolling(N // 2).max().shift(1).to_numpy(), Ls.rolling(N // 2).min().shift(1).to_numpy()
                        al, ash = sides(x, filt, lo)
                        w = window(x) & day_mask(x, None if mask is None else mask[a])
                        pos = chan_pos(x["C"], np.nan_to_num(hiN, nan=np.inf), np.nan_to_num(loN, nan=-np.inf),
                                       np.nan_to_num(hiM, nan=np.inf), np.nan_to_num(loM, nan=-np.inf), al, ash, w)
                        out[a] = res_pos(x, a, pos)
                    return out
                v[f"{N}本のドンチャン・{'S1の向き' if filt else '向きの絞りなし'}・{'買いだけ' if lo else '買い＋売り'}"] = fn
    run_variants(ctx, "TC01", "15分足のドンチャン・ブレイク（N本の高値・安値を終値で抜けたら入り、N/2本の反対側か翌5時で決済）", v)
    v = {}
    for f_, s_ in ((12, 48), (24, 96)):
        for filt in (False, True):
            for lo in (False, True):
                def fn(mask, f_=f_, s_=s_, filt=filt, lo=lo):
                    out = {}
                    for a in COINS:
                        x = X[a]
                        Cs = pd.Series(x["C"])
                        d = (Cs.ewm(span=f_, adjust=False).mean() - Cs.ewm(span=s_, adjust=False).mean()).shift(1).to_numpy()
                        sg = np.sign(np.nan_to_num(d))
                        al, ash = sides(x, filt, lo)
                        sg = np.where(sg > 0, al * 1.0, np.where(sg < 0, -1.0 * ash, 0.0))
                        out[a] = res_pos(x, a, sg * window(x) * day_mask(x, None if mask is None else mask[a]))
                    return out
                v[f"平均{f_}/{s_}本・{'S1の向き' if filt else '向きの絞りなし'}・{'買いだけ' if lo else '買い＋売り'}"] = fn
    run_variants(ctx, "TC02", "15分足の短い平均と長い平均の向き（9時〜翌5時だけ持つ）", v)
    v = {}
    for L in (4, 8, 24):
        for H in (1, 4):
            for filt in (False, True):
                def fn(mask, L=L, H=H, filt=filt):
                    out = {}
                    step = 4 * H
                    for a in COINS:
                        x = X[a]
                        C = x["C"]
                        sg = np.sign(np.r_[np.full(4 * L + 1, np.nan), C[:-4 * L - 1] * 0 + (C[4 * L:-1] - C[:-4 * L - 1])])
                        sg = np.nan_to_num(sg[:len(C)])
                        t = np.arange(len(C))
                        reb = t - (x["bod"] % step)
                        pos = sg[reb]
                        if filt:
                            s = x["sdir"][x["bd"]]
                            pos = np.where(pos * s > 0, pos, 0.0)
                        out[a] = res_pos(x, a, pos * window(x) * day_mask(x, None if mask is None else mask[a]))
                    return out
                v[f"直近{L}時間の向きに{H}時間・{'S1の向き' if filt else '絞りなし'}"] = fn
    run_variants(ctx, "TC03", "直近L時間の値動きの向きに、次のH時間を持つ（9時〜翌5時）", v)


def zscore(x):
    lr = np.r_[0.0, np.diff(np.log(x["C"]))]
    sd = pd.Series(lr).rolling(672, min_periods=200).std().shift(1).to_numpy()
    return lr, lr / sd


def sec_td(ctx):
    X = ctx["X"]
    v = {}
    for k in (2.5, 3.0, 4.0):
        for H in (4, 16):
            for rev in (True, False):
                def fn(mask, k=k, H=H, rev=rev):
                    out = {}
                    for a in COINS:
                        x = X[a]
                        lr, z = zscore(x)
                        sig = np.where(np.abs(np.nan_to_num(z)) > k, np.sign(lr), 0.0) * (-1.0 if rev else 1.0)
                        w = window(x) & day_mask(x, None if mask is None else mask[a])
                        out[a] = res_pos(x, a, hold_after(sig, H, w))
                    return out
                v[f"{k}倍・{H * 0.25:g}時間・{'戻り（逆）' if rev else '同じ向き'}"] = fn
    run_variants(ctx, "TD01", "15分の値動きが、ふだんのk倍を超えた後（戻る向きと、同じ向き）", v, family=lambda k: k.split("・")[-1])


def sec_tg(ctx):
    X = ctx["X"]
    v = {}
    for L in (4, 16):
        for thr in (0.54, 0.58):
            for H in (4, 16):
                def fn(mask, L=L, thr=thr, H=H):
                    out = {}
                    for a in COINS:
                        x = X[a]
                        tb = pd.Series(x["TB"]).rolling(L).sum() / pd.Series(x["V"]).rolling(L).sum().replace(0, np.nan)
                        tb = tb.to_numpy()
                        sig = np.where(tb > thr, 1.0, np.where(tb < 1 - thr, -1.0, 0.0))
                        w = window(x) & day_mask(x, None if mask is None else mask[a])
                        out[a] = res_pos(x, a, hold_after(np.nan_to_num(sig), H, w))
                    return out
                v[f"直近{L * 0.25:g}時間・比率{thr}・{H * 0.25:g}時間"] = fn
    run_variants(ctx, "TG01", "成行の買いの比率が高ければ買い・低ければ売り", v)
    v = {}
    for k in (3.0, 5.0):
        for H in (4, 16):
            for fol in (True, False):
                def fn(mask, k=k, H=H, fol=fol):
                    out = {}
                    for a in COINS:
                        x = X[a]
                        vr = x["V"] / pd.Series(x["V"]).rolling(672, min_periods=200).mean().shift(1).to_numpy()
                        sig = np.where(np.nan_to_num(vr) > k, np.sign(x["C"] - x["O"]), 0.0) * (1.0 if fol else -1.0)
                        w = window(x) & day_mask(x, None if mask is None else mask[a])
                        out[a] = res_pos(x, a, hold_after(sig, H, w))
                    return out
                v[f"出来高{k:g}倍・{H * 0.25:g}時間・{'同じ向き' if fol else '逆'}"] = fn
    run_variants(ctx, "TG02", "出来高が急に増えた15分足の向き（同じ向きと逆）", v, family=lambda k: k.split("・")[-1])


def sec_th(ctx):
    X = ctx["X"]
    b, e = X["BTC"], X["ETH"]
    lb, zb = zscore(b)
    le, ze = zscore(e)
    v = {}
    for k in (2.0, 3.0):
        for H in (1, 4, 8):
            def fn(mask, k=k, H=H):
                lag = np.nan_to_num(ze) * np.sign(lb) < k / 2
                sig = np.where((np.abs(np.nan_to_num(zb)) > k) & lag, np.sign(lb), 0.0)
                w = window(e) & day_mask(e, None if mask is None else mask["ETH"])
                return {"BTC": empty(b), "ETH": res_pos(e, "ETH", hold_after(sig, H, w))}
            v[f"BTCが{k:g}倍・ETHが半分未満・{H * 0.25:g}時間"] = fn
    run_variants(ctx, "TH01", "BTCが15分で大きく動き、ETHが追いついていない時、ETHをその向きに（ETHだけ）", v)


def sec_tj(ctx):
    X, idx = ctx["X"], ctx["idx"]
    us = VX.summer(idx)
    wd = idx.dayofweek.to_numpy()
    v = {}
    for gmin in (0.0, 0.005):
        for fade in (True, False):
            def fn(mask, gmin=gmin, fade=fade):
                out = {}
                for a in COINS:
                    x = X[a]
                    O, H, L, C = x["O"], x["H"], x["L"], x["C"]
                    rows = []
                    for d in np.flatnonzero(wd == 4):
                        if d + 3 >= x["nd"] or not np.isfinite(x["atr"][d + 3]):
                            continue
                        fb = d * DAY + (84 if us[d] else 88)
                        sb = (d + 2) * DAY + (88 if us[d + 2] else 92)
                        xbar = (d + 3) * DAY + END
                        pf, ps = C[fb - 1], O[sb]
                        gap = (ps - pf) / pf
                        if abs(gap) < max(gmin, 1e-9):
                            continue
                        sd = -np.sign(gap) if fade else np.sign(gap)
                        e, xp, xb = ps, C[xbar - 1], xbar - 1
                        if fade:
                            for j in range(sb, xbar):
                                if (sd > 0 and H[j] >= pf) or (sd < 0 and L[j] <= pf):
                                    xp, xb = (max(O[j], pf) if sd > 0 else min(O[j], pf)) if j > sb else pf, j
                                    break
                        rows.append((d + 3, sd, e, xp, sb, xb))
                    if mask is not None:
                        rows = [r for r in rows if mask[a][r[0]]]
                    if not rows:
                        out[a] = empty(x)
                        continue
                    dd, sd, e, xp, eb, xb = map(np.array, zip(*rows))
                    r = res_trades(x, a, dd.astype(int), sd, e, xp, np.ones(len(dd)), eb, xb)
                    r["ret"] = sd * (xp - e) / e
                    out[a] = r
                return out
            v[f"窓{gmin * 100:g}%以上・{'埋める向き' if fade else '続く向き'}"] = fn
    run_variants(ctx, "TJ01", "CMEの窓：金曜の終わり〜日曜の始まりの値の差を、埋める向き・続く向きに（月曜20時UTC＝火曜5時まで）", v,
                 family=lambda k: k.split("・")[-1])
    v = {}
    for fb, fl in ((0, "9時"), (32, "17時"), (64, "1時")):
        for when in ("前", "後"):
            for sd in (1.0, -1.0):
                def fn(mask, fb=fb, when=when, sd=sd):
                    out = {}
                    for a in COINS:
                        x = X[a]
                        P = np.zeros((x["nd"], DAY))
                        if when == "後":
                            P[:, fb:fb + 4] = sd
                        elif fb == 0:
                            P[:, 92:96] = sd
                        else:
                            P[:, fb - 4:fb] = sd
                        if mask is not None:
                            P *= np.asarray(mask[a])[:, None]
                        out[a] = res_pos(x, a, P.ravel() * np.isfinite(x["atr"][x["bd"]]))
                    return out
                v[f"{fl}の{when}1時間・{'買い' if sd > 0 else '売り'}"] = fn
    run_variants(ctx, "TJ02", "資金調達の時刻（日本時間1時・9時・17時）の前後1時間を持つ", v)


def sec_te(ctx):
    X = ctx["X"]
    print("\n==== TE01 第1回の形（前半で一番良い設定）に「静かな日だけ」をかける")
    q = {a: X[a]["q"] for a in COINS}
    out = {}
    for pid, bk in BEST.items():
        fn = BUILD[pid][bk]
        coins = ("ETH",) if pid == "TH01" else COINS
        base = SUM[pid]["r"]
        r = show(ctx, "TE01", f"{pid}「{bk}」＋静かな日だけ", fn(q), coins=coins)
        out[pid] = (base, r)
        print(f"      静かな日だけにすると 安定度{base['安定度']:.2f}→{r['安定度']:.2f}（前半{base['_h1s']:.2f}→{r['_h1s']:.2f}・後半{base['_h2s']:.2f}→{r['_h2s']:.2f}）")
    SUM["TE01"] = out


SECTIONS = {"tn": sec_tn03, "tp": sec_tp01, "ta": sec_ta, "tb": sec_tb, "tc": sec_tc, "td": sec_td, "tg": sec_tg, "th": sec_th, "tj": sec_tj,
            "te": sec_te}


def main(cache, only=None):
    names = list(SECTIONS) if not only else only.split(",")
    if not only:
        print(f"（古い「{GROUP}」の行を {drop(GROUP)} 行消して、計算し直す）")
    X = load(cache)
    idx = X["BTC"]["idx"]
    D = R1.prep(cache)
    parts = R7.s1_parts(D, COINS, D["BTC"])
    per, T = R7.build(parts, R7.size_rules(parts, COINS)["S4案"])
    s4 = R1.total(per)
    s4r = R1.stats(s4)
    R1.line("S4案（比べる相手）", s4r)
    ctx = dict(X=X, idx=idx, s4=s4, s4r=s4r)
    for n in names:
        SECTIONS[n](ctx)
    print("\n==== まとめ（前半で一番良い形の判定）")
    for pid, o in SUM.items():
        if "verdict" not in o:
            continue
        r = o["r"]
        print(f"  {pid} {o['verdict']:12s}「{o['best']}」：年{r['年平均%']:+.2f}% 落込{r['最大の落ち込み%']:.1f}% 安定度{r['安定度']:.2f}"
              f"（前半{r['_h1s']:.2f}・後半{r['_h2s']:.2f}）・倍のコストで年{r['_r2']['年平均%']:+.2f}%・1回{r['_gross']:+.3f}%・年{r['_n']:.0f}回"
              f"｜{o['n']}通り中{o['npos']}通りがプラス｜S4案と月の相関{o['corr']:+.2f}・足すと{o['mix']['安定度']:.2f}")
    return SUM


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
