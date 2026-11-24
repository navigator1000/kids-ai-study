#!/usr/bin/env python3
"""計画書（project/research/plan.md）第5回・前半：統計・機械学習（N）の12件（BTC・ETH）。

  翌日（9時〜翌9時）の上げ下げを予想するもの（毎月はじめに、その前日までのデータだけで学び直す。先読みなし）
    N01 ロジスティック回帰／N02 決定木／N03 k近傍法／N04 ランダムフォレスト／N12 小さなニューラルネット
  確率・統計の道具
    N05 マルコフ連鎖（上げ下げの並びごとの、翌日に上げる確率）／N06 隠れマルコフモデル（相場の状態を2〜3つに分ける）
    N07 周期の発見（フーリエ分析）／N08 自己相関／N09 ハースト指数（トレンドが続きやすいか）
    N11 遺伝的アルゴリズム（ルールを自動で作らせる。過剰最適化の見本）／N13 GARCH（翌日の値動きの大きさを予想して量を決める）
評価は2019年1月から（2017年8月〜2018年のデータを最初の学習に残すため）。前半は2019〜2022年、後半は2023年〜。
比べる相手（A01・S1・E05など）も同じ期間で計算し直す。
使い方は2通り：(a) 予想の向きに9時〜翌9時を持つ（買い×1・売り×0.5）
              (b) 値幅ブレイク（A01）の向き・休む日・量の決め方に使う
共通: Vantage CFD（往復0.05%、買いのスワップ年25〜35%、売り0%）、1R＝ATR（日足）1本分＝資金の0.5%（単利）。
必要なもの: pip install scikit-learn hmmlearn arch
結果は results.csv（グループ「計画・第5回」、計画IDつき）と plan.csv。
使い方: python3 project/tools/round5_lab.py <キャッシュ> [ml,markov,hmm,fourier,acf,hurst,ga,garch のどれか（カンマ区切り）]
"""
import logging
import sys
import warnings

import numpy as np
import pandas as pd

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
import more_methods_lab as MM  # noqa: E402
import portfolio_lab as PF  # noqa: E402
import round1_lab as R1  # noqa: E402
import round2_lab as R2  # noqa: E402
import round2b_lab as R2B  # noqa: E402
import round3_lab as R3  # noqa: E402
import round4_lab as R4  # noqa: E402
import round4b_lab as R4B  # noqa: E402
from registry import SPLIT, drop, metrics_row  # noqa: E402

COINS = ("BTC", "ETH")
GROUP = "計画・第5回"
DAY = 96
R3.GROUP = GROUP
START19 = pd.Timestamp("2019-01-01")
MM.START = START19                                    # 評価は2019年から（最初の学習に1年以上のデータを残す）
R1.YEARS = (MM.END - MM.START).days / 365.25
warnings.filterwarnings("ignore")
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

BASE = ["前日の動き", "2日前の動き", "3日前の動き", "5日の動き", "10日の動き", "20日の動き", "60日の動き", "ボラの比（10日÷60日）",
        "ボラの1年の順位", "前日の値幅", "前日の引けの位置", "20日の中の位置", "5日平均との差", "20日平均との差", "50日平均との差",
        "BTCの前日の動き", "BTCの5日の動き"]
DOW = [f"{c}曜日" for c in "月火水木金土"]
LAGS = [f"{k}日前" for k in range(1, 21)]
SUMMARY = {}                                          # 計画ID → README・plan.csv に書くための数字


# ---------------------------------------------------------------- 材料
def build(D):
    """日ごとの材料（その日の朝9時にわかる値だけ）と、その日の値動き（9時〜翌9時の対数の騰落率）。"""
    assert (D["BTC"]["idx"] == D["ETH"]["idx"]).all()
    F, RET = {}, {}
    for a in COINS:
        x = D[a]
        idx = x["idx"]
        c, h, lo = (pd.Series(x[k], index=idx) for k in ("dC", "dH", "dL"))
        lr = np.log(c).diff()
        v10, v30, v60 = (lr.rolling(n).std() for n in (10, 30, 60))
        f = pd.DataFrame(index=idx)
        f["前日の動き"] = lr / v30
        f["2日前の動き"] = lr.shift(1) / v30
        f["3日前の動き"] = lr.shift(2) / v30
        for n in (5, 10, 20):
            f[f"{n}日の動き"] = lr.rolling(n).sum() / (v30 * np.sqrt(n))
        f["60日の動き"] = lr.rolling(60).sum() / (v60 * np.sqrt(60))
        f["ボラの比（10日÷60日）"] = v10 / v60
        f["ボラの1年の順位"] = v10.rolling(365, min_periods=120).rank(pct=True)
        f["前日の値幅"] = np.log(h / lo) / v30
        f["前日の引けの位置"] = ((c - lo) / (h - lo)).fillna(0.5)
        f["20日の中の位置"] = (c - lo.rolling(20).min()) / (h.rolling(20).max() - lo.rolling(20).min())
        for n in (5, 20, 50):
            f[f"{n}日平均との差"] = np.log(c / c.rolling(n).mean()) / v30
        for k in range(1, 21):
            f[f"{k}日前"] = lr.shift(k - 1) / v30
        F[a] = f.shift(1)                             # その日の朝にわかる値にする
        RET[a] = lr
    for a in COINS:
        F[a]["BTCの前日の動き"] = F["BTC"]["前日の動き"].to_numpy()
        F[a]["BTCの5日の動き"] = F["BTC"]["5日の動き"].to_numpy()
        for k, nm in enumerate(DOW):
            F[a][nm] = (F[a].index.dayofweek == k).astype(float)
    return F, RET


def months():
    m = pd.date_range(START19, MM.END, freq="MS")
    return list(zip(m, list(m[1:]) + [MM.END + pd.Timedelta(days=1)]))


def walk(ctx, make, cols, start=START19):
    """毎月はじめに、その前日までの BTC・ETH のデータで学び直し、その月の各日の「上がる確率」を出す。"""
    F, RET = ctx["F"], ctx["RET"]
    X = {a: F[a][cols].to_numpy(float) for a in COINS}
    ok = {a: ~np.isnan(X[a]).any(1) for a in COINS}
    y = {a: (RET[a].to_numpy() > 0).astype(int) for a in COINS}
    yok = {a: RET[a].notna().to_numpy() for a in COINS}
    idx = F["BTC"].index
    out = {a: np.full(len(idx), np.nan) for a in COINS}
    for m0, m1 in months():
        if m0 < start:
            continue
        tr = {a: ok[a] & yok[a] & (idx < m0) for a in COINS}
        model = make().fit(np.vstack([X[a][tr[a]] for a in COINS]), np.concatenate([y[a][tr[a]] for a in COINS]))
        for a in COINS:
            te = ok[a] & (idx >= m0) & (idx < m1)
            if te.any():
                out[a][te] = model.predict_proba(X[a][te])[:, 1]
    return out


class Avg:
    """いくつかのモデルの確率の平均（ニューラルネットの初期値の運を減らす）。"""
    def __init__(self, makers):
        self.ms = [m() for m in makers]

    def fit(self, X, y):
        for m in self.ms:
            m.fit(X, y)
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.ms], axis=0)


def side_from_p(p, m=0.0):
    p = np.asarray(p, float)
    with np.errstate(invalid="ignore"):
        return np.where(p > 0.5 + m, 1.0, np.where(p < 0.5 - m, -1.0, 0.0))


def conf_side(p, q=0.8, win=365, minp=60):
    """予想の自信（確率と50%の差）が、それまでの1年の上位 1-q に入る日だけ、その向きにする。"""
    p = np.asarray(p, float)
    s = pd.Series(np.abs(p - 0.5))
    th = s.rolling(win, min_periods=minp).quantile(q).shift(1).to_numpy()
    with np.errstate(invalid="ignore"):
        return np.where(s.to_numpy() > th, np.sign(p - 0.5), 0.0)


def hitrate(ctx, sides):
    """向きを決めた日のうち、当たった割合（%）。前半・後半。"""
    out = []
    for lo, hi in ((START19, SPLIT), (SPLIT, MM.END + pd.Timedelta(days=1))):
        n = k = 0
        for a in COINS:
            s, r, idx = np.asarray(sides[a]), ctx["RET"][a].to_numpy(), ctx["F"][a].index
            sel = (idx >= lo) & (idx < hi) & (s != 0) & ~np.isnan(r)
            n += sel.sum()
            k += (np.sign(r[sel]) == s[sel]).sum()
        out.append(100 * k / max(n, 1))
    return out


# ---------------------------------------------------------------- 売買の形
def gross(ctx, sides):
    """同じ売買を、コストとスワップなしで回した時の年平均（%）。予想の力とコストの重さを分けて見るため。"""
    saved = {a: PF.SPEC[a] for a in COINS}
    try:
        for a in COINS:
            PF.SPEC[a] = (0.0, 0.0, 0.0)
        per, _ = R3.both(ctx["D"], R4B.daytrade(lambda x, a: sides[a]))
    finally:
        for a in COINS:
            PF.SPEC[a] = saved[a]
    return metrics_row(R1.total(per))["年平均%"]


def dt(ctx, pid, name, sides, memo="", hit=True):
    """日ごとの向き（その朝にわかる値）で9時〜翌9時を持つ。買い×1・売り×0.5。"""
    g = gross(ctx, sides)
    if hit:
        h1, h2 = hitrate(ctx, sides)
        memo = (memo + "。" if memo else "") + f"当たった割合：前半{h1:.1f}%・後半{h2:.1f}%"
    memo += f"。コストとスワップなしなら年{g:+.1f}%"
    r = R3.show(pid, name, *R3.both(ctx["D"], R4B.daytrade(lambda x, a: sides[a])), memo=memo, tf="日足")
    r["_gross"] = g
    if hit:
        r["_hit"] = (h1, h2)
        print(f"      当たった割合：前半{h1:.1f}%・後半{h2:.1f}%｜コストとスワップなしなら年{g:+.2f}%")
    return r


def vbo_dir(ctx, pid, name, dirs, memo=""):
    """値幅ブレイク（k=0.8・買い×1＋売り×0.5）の向きを、5日平均の代わりに dirs（日ごとの+1・-1・0）で決める。"""
    per, Ts = {}, []
    for a in COINS:
        s, T = R2.run(R2.periods(ctx["D"][a], DAY, 0), a, ma=False, extra_ok=lambda P_, side, a=a: side * dirs[a] > 0)
        per[a] = s
        Ts.append(T)
    return R3.show(pid, name, per, pd.concat(Ts, ignore_index=True), memo=memo, tf="15分足")


def scaled(ctx, pid, name, per, T, m, memo=""):
    """値幅ブレイクの毎日の損益に、日ごとの倍率 m（0なら休む）をかける。"""
    per2 = {a: per[a] * m[a] for a in COINS}
    T = T.reset_index(drop=True)
    keep = np.zeros(len(T), bool)
    for a in COINS:
        sel = (T["coin"] == a).to_numpy()
        pos = ctx["D"][a]["idx"].get_indexer(pd.DatetimeIndex(T.loc[sel, "day"]))
        keep[sel] = np.asarray(m[a])[pos] > 0
    return R3.show(pid, name, per2, T[keep], memo=memo, tf="15分足")


def comp(ctx, pid, res):
    R3.component(pid, res, ctx["v0"], ctx["base"])
    return R3.OUT[pid]


def keep_sum(pid, **kw):
    SUMMARY.setdefault(pid, {}).update(kw)


def brief(r):
    return dict(a=r["年平均%"], dd=r["最大の落ち込み%"], s=r["安定度"], s1=r["_h1s"], s2=r["_h2s"], a1=r["前半の年平均%"],
                a2=r["後半の年平均%"], neg=r["マイナスの年"], n=r["_n"], g=r.get("_gross"), hit=r.get("_hit"))


# ---------------------------------------------------------------- N01〜N04・N12 予想のモデル
def ml_item(ctx, pid, title, models, cols, margins=(0.0, 0.02)):
    print(f"\n==== {pid} {title}：前日までの材料から、翌日（9時〜翌9時）の上げ下げを予想（毎月はじめに学び直す）")
    res, probs, of = {}, {}, {}
    for key, mk in models.items():
        P = walk(ctx, mk, cols)
        probs[key] = P
        for m in list(margins) + ["自信"]:
            if m == "自信":
                lab, sides = key + "・自信が強い日（それまでの1年の上位20%）だけ", {a: conf_side(P[a]) for a in COINS}
            else:
                lab = key + ("" if m == 0 else f"・確率{50 + 100 * m:.0f}%超か{50 - 100 * m:.0f}%未満の日だけ")
                sides = {a: side_from_p(P[a], m) for a in COINS}
            res[lab] = dt(ctx, pid, f"{title}（{lab}）の予想の向きに9時〜翌9時", sides)
            of[lab] = key
    o = comp(ctx, pid, res)
    key = of[o["best"]]
    f = vbo_dir(ctx, pid, f"値幅ブレイク：向きを{title}（{key}）の予想で決める", {a: side_from_p(probs[key][a]) for a in COINS})
    keep_sum(pid, best=o["best"], verdict=o["verdict"], r=brief(o["r"]), hit=o["r"]["_hit"], mix=brief(o["mix"]), corr=o["corr"],
             npos=o["npos"], n=o["n"], vbo=brief(f), hits={k: v["_hit"] for k, v in res.items()},
             rows={k: brief(v) for k, v in res.items()})
    return probs


def training(ctx, cols, end=SPLIT):
    X, y = [], []
    for a in COINS:
        f = ctx["F"][a][cols]
        ok = f.notna().all(1).to_numpy() & ctx["RET"][a].notna().to_numpy() & (f.index < end)
        X.append(f.to_numpy(float)[ok])
        y.append((ctx["RET"][a].to_numpy()[ok] > 0).astype(int))
    return np.vstack(X), np.concatenate(y)


def sec_ml(ctx):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier, export_text

    cols = BASE + DOW
    # 上げた日の割合（予想の物差し）
    for lo, hi, half in ((START19, SPLIT, "前半"), (SPLIT, MM.END + pd.Timedelta(days=1), "後半")):
        u = [((ctx["RET"][a] > 0) & (ctx["RET"][a].index >= lo) & (ctx["RET"][a].index < hi)).sum() /
             ((ctx["RET"][a].index >= lo) & (ctx["RET"][a].index < hi)).sum() * 100 for a in COINS]
        print(f"    上げた日の割合（{half}）：BTC{u[0]:.1f}%・ETH{u[1]:.1f}%")
        keep_sum("基準", **{f"up_{half}": u})
    ones = {a: np.ones(len(ctx["F"][a])) for a in COINS}
    keep_sum("基準", always_long=brief(dt(ctx, "基準", "毎日買い（9時〜翌9時、2019年から）", ones)))

    lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    ml_item(ctx, "N01", "ロジスティック回帰", {"材料17個＋曜日": lr}, cols, margins=(0.0, 0.01, 0.02))
    m = lr().fit(*training(ctx, cols))
    coef = pd.Series(m[-1].coef_[0], index=cols)
    top = coef.reindex(coef.abs().sort_values(ascending=False).index)[:6]
    print("    2022年末までで学んだ時の、効きの大きい材料（標準化した係数）：" + "・".join(f"{k}{v:+.3f}" for k, v in top.items()))
    keep_sum("N01", coef=top.round(3).to_dict())

    trees = {f"深さ{d}": (lambda d=d: DecisionTreeClassifier(max_depth=d, min_samples_leaf=100, random_state=0)) for d in (2, 3, 4)}
    ml_item(ctx, "N02", "決定木", trees, cols, margins=(0.0,))
    t = DecisionTreeClassifier(max_depth=2, min_samples_leaf=100, random_state=0).fit(*training(ctx, cols))
    txt = export_text(t, feature_names=cols, show_weights=True, decimals=2)
    print("    2022年末までで学んだ深さ2の木：\n" + "\n".join("      " + s for s in txt.splitlines()))
    keep_sum("N02", tree=txt)

    knn = {f"k={k}": (lambda k=k: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))) for k in (5, 20, 50)}
    ml_item(ctx, "N03", "k近傍法", knn, BASE, margins=(0.0,))

    rf = lambda: RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=50, max_features="sqrt", n_jobs=4, random_state=0)
    ml_item(ctx, "N04", "ランダムフォレスト", {"木200本・深さ4": rf}, cols, margins=(0.0, 0.02))

    mlp = lambda: Avg([lambda s=s: make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(16,), alpha=1e-2, max_iter=300, early_stopping=True, n_iter_no_change=20, random_state=s)) for s in (0, 1, 2)])
    ml_item(ctx, "N12", "小さなニューラルネット", {"過去20日の動き・中間16個": mlp}, LAGS, margins=(0.0, 0.02))


# ---------------------------------------------------------------- N05 マルコフ連鎖
def markov_p(r, n, win):
    """直近 n 日の上げ下げの並びごとに、翌日に上げた割合（前日までに数えたもの）。win>0 なら直近 win 日だけで数える。"""
    s = (np.nan_to_num(r) > 0).astype(np.int64)
    nd = len(s)
    st = np.full(nd, -1)
    for d in range(n, nd):
        v = 0
        for k in range(1, n + 1):
            v = v * 2 + s[d - k]
        st[d] = v
    up, cnt = np.zeros(2 ** n), np.zeros(2 ** n)
    tu = tc = 0.0
    p, base = np.full(nd, np.nan), np.full(nd, np.nan)
    for d in range(nd):
        if st[d] >= 0 and cnt[st[d]] >= 20:
            p[d], base[d] = up[st[d]] / cnt[st[d]], tu / tc
        if st[d] >= 0:
            up[st[d]] += s[d]
            cnt[st[d]] += 1
            tu, tc = tu + s[d], tc + 1
            if win and d - win >= 0 and st[d - win] >= 0:
                o = st[d - win]
                up[o] -= s[d - win]
                cnt[o] -= 1
                tu, tc = tu - s[d - win], tc - 1
    return p, base


def sec_markov(ctx):
    pid = "N05"
    print("\n==== N05 マルコフ連鎖：直近1〜3日の上げ下げの並びごとに、翌日に上げる確率を数える（前日までのデータだけ）")
    tab = {}
    for a in COINS:
        r = ctx["RET"][a]
        s = (r > 0).astype(int)
        prev = s.shift(1)
        for half, sel in (("前半", (r.index >= START19) & (r.index < SPLIT)), ("後半", (r.index >= SPLIT) & (r.index <= MM.END))):
            pu, pdn, b = s[sel & (prev == 1)].mean() * 100, s[sel & (prev == 0)].mean() * 100, s[sel].mean() * 100
            tab[(a, half)] = (pu, pdn, b)
            print(f"    {a} {half}：前日に上げた→翌日も上げ {pu:.1f}%／前日に下げた→翌日は上げ {pdn:.1f}%（全体 {b:.1f}%）")
    keep_sum(pid, tab=tab)
    res = {}
    for n in (1, 2, 3):
        for rel, win in ((False, 0), (True, 0), (True, 365)):
            sides = {}
            for a in COINS:
                p, b = markov_p(ctx["RET"][a].to_numpy(), n, win)
                q = p - (b if rel else 0.5)
                with np.errstate(invalid="ignore"):
                    sides[a] = np.where(q > 0.02, 1.0, np.where(q < -0.02, -1.0, 0.0))
            lab = f"{n}日の並び" + ("・全体の上げやすさとの差" if rel else "・50%との差") + (f"・直近{win}日で数える" if win else "")
            res[lab] = dt(ctx, pid, f"マルコフ連鎖（{lab}）が2ポイント以上の時だけ9時〜翌9時", sides)
    o = comp(ctx, pid, res)
    keep_sum(pid, best=o["best"], verdict=o["verdict"], r=brief(o["r"]), hit=o["r"]["_hit"], npos=o["npos"], n=o["n"],
             mix=brief(o["mix"]), rows={k: brief(v) for k, v in res.items()})


# ---------------------------------------------------------------- N06 隠れマルコフモデル
def hmm_probs(X, idx, K, seeds=(0, 1)):
    """毎月はじめに、その前日までのデータで隠れマルコフモデルを学び直す。
    返す値：各日の「その日の状態」の確率（前日までのデータから。状態は値幅の小さい順）と、平均の騰落率が一番高い状態の番号。"""
    from hmmlearn.hmm import GaussianHMM
    n = len(X)
    valid = ~np.isnan(X).any(1)
    first = int(np.argmax(valid))
    pred, bull = np.full((n, K), np.nan), np.full(n, -1)
    for m0, m1 in months():
        tr = valid & (idx < m0)
        mu, sd = X[tr].mean(0), X[tr].std(0)
        Z = (np.where(valid[:, None], X, 0.0) - mu) / sd
        best = None
        for s in seeds:
            m = GaussianHMM(n_components=K, covariance_type="diag", n_iter=100, tol=1e-3, random_state=s)
            try:
                m.fit(Z[tr])
                sc = m.score(Z[tr])
            except Exception:                         # noqa: BLE001
                continue
            if best is None or sc > best[0]:
                best = (sc, m)
        m = best[1]
        means, var, A = m.means_, m.covars_, m.transmat_
        var = np.array([np.diag(v) for v in var]) if var.ndim == 3 else var
        order = np.argsort(means[:, 1])
        te = (idx >= m0) & (idx < m1)
        last = int(np.nonzero(te)[0].max()) + 1
        a = m.startprob_.copy()
        P = np.zeros((last, K))
        for t in range(first, last):
            pr = a @ A if t > first else m.startprob_
            P[t] = pr
            if valid[t]:
                lb = -0.5 * (np.log(2 * np.pi * var) + (Z[t] - means) ** 2 / var).sum(1)
                w = pr * np.exp(lb - lb.max())
                a = w / w.sum()
            else:
                a = pr
        pred[te] = P[te[:last]][:, order]
        bull[te] = int(np.argmax(means[order, 0]))
    return pred, bull


def sec_hmm(ctx):
    pid = "N06"
    D = ctx["D"]
    print("\n==== N06 隠れマルコフモデル：その日の騰落率と値幅から、相場の状態を2つ・3つに自動で分ける（毎月学び直し、前日までで判定）")
    st, bull = {}, {}
    for K in (2, 3):
        for a in COINS:
            x = D[a]
            X = np.column_stack([ctx["RET"][a].to_numpy(), np.log(x["dH"] / x["dL"])])
            p, b = hmm_probs(X, x["idx"], K)
            st[(K, a)] = np.where(np.isnan(p).any(1), -1, np.nanargmax(np.nan_to_num(p, nan=-1), axis=1))
            bull[(K, a)] = b
    # 状態ごとの値幅ブレイクの成績（3つに分けた時）
    T0 = ctx["T0"].reset_index(drop=True)
    stv = np.full(len(T0), -1)
    for a in COINS:
        sel = (T0["coin"] == a).to_numpy()
        stv[sel] = st[(3, a)][D[a]["idx"].get_indexer(pd.DatetimeIndex(T0.loc[sel, "day"]))]
    rows = {}
    for k, nm in enumerate(("静か", "ふつう", "荒い")):
        for half, sel in (("前半", T0.day < SPLIT), ("後半", T0.day >= SPLIT)):
            t = T0[(stv == k) & sel.to_numpy()]
            rows[(nm, half)] = (len(t), t.R.mean() if len(t) else np.nan)
        share = np.mean([(st[(3, a)][(D[a]["idx"] >= START19) & (D[a]["idx"] <= MM.END)] == k).mean() for a in COINS]) * 100
        rows[(nm, "日の割合")] = share
        print(f"    3状態の「{nm}」（日の{share:.0f}%）：値幅ブレイクの1回あたり 前半{rows[(nm, '前半')][1]:+.3f}R（{rows[(nm, '前半')][0]}回）"
              f"・後半{rows[(nm, '後半')][1]:+.3f}R（{rows[(nm, '後半')][0]}回）")
    keep_sum(pid, by_state=rows)
    out = {}
    per0, T0b = ctx["per0"], ctx["T0"]
    out["2状態・静かな状態の日だけ"] = scaled(ctx, pid, "値幅ブレイク：隠れマルコフ（2状態）で静かな状態の日だけ", per0, T0b,
                                      {a: (st[(2, a)] == 0).astype(float) for a in COINS})
    out["3状態・静かな状態の日だけ"] = scaled(ctx, pid, "値幅ブレイク：隠れマルコフ（3状態）で静かな状態の日だけ", per0, T0b,
                                      {a: (st[(3, a)] == 0).astype(float) for a in COINS})
    out["3状態・荒い状態の日は休む"] = scaled(ctx, pid, "値幅ブレイク：隠れマルコフ（3状態）で荒い状態の日は休む", per0, T0b,
                                      {a: ((st[(3, a)] >= 0) & (st[(3, a)] != 2)).astype(float) for a in COINS})
    res = {}
    for K in (2, 3):
        res[f"{K}状態"] = R3.show(pid, f"隠れマルコフ（{K}状態）で上げの状態の間だけ買い持ち",
                                *R3.both(D, R4B.holdwhile(lambda x, a, K=K: (st[(K, a)] >= 0) & (st[(K, a)] == bull[(K, a)]))))
    o = comp(ctx, pid, res)
    keep_sum(pid, filt={k: brief(v) for k, v in out.items()}, best=o["best"], verdict=o["verdict"], r=brief(o["r"]),
             rows={k: brief(v) for k, v in res.items()}, mix=brief(o["mix"]))


# ---------------------------------------------------------------- N07 周期（フーリエ分析）
def peaks(v, nperm=1000, seed=0):
    """強い周期の上位3つ（日）と、一番強い周期・7日の周期が偶然でも出る確率。"""
    v = v - v.mean()
    N = len(v)
    pw = np.abs(np.fft.rfft(v)) ** 2
    pw[0] = 0
    k = np.argsort(pw)[::-1][:3]
    k7 = int(round(N / 7))
    rng = np.random.default_rng(seed)
    mx, p7 = [], []
    for _ in range(nperm):
        q = np.abs(np.fft.rfft(rng.permutation(v))) ** 2
        mx.append(q[1:].max())
        p7.append(q[k7])
    return N / k, (np.array(mx) >= pw.max()).mean(), (np.array(p7) >= pw[k7]).mean()


def fourier_side(r, idx, N, K):
    """毎月はじめに直近 N 日の騰落率をフーリエ分解し、強い周期 K 個をのばした値の向きに賭ける。"""
    r = np.nan_to_num(r)
    side = np.zeros(len(r))
    for m0, m1 in months():
        e = int(idx.searchsorted(m0))
        if e < N:
            continue
        w = r[e - N:e]
        X = np.fft.rfft(w - w.mean())
        pw = np.abs(X) ** 2
        pw[0] = 0
        top = np.argsort(pw)[-K:]
        days = np.nonzero((idx >= m0) & (idx < m1))[0]
        t = N + (days - e)
        xh = sum((2 / N) * np.abs(X[k]) * np.cos(2 * np.pi * k * t / N + np.angle(X[k])) for k in top)
        side[days] = np.sign(xh)
    return side


def sec_fourier(ctx):
    pid = "N07"
    print("\n==== N07 周期の発見：騰落率と値動きの大きさをフーリエ分析（前半・後半で別々に）。強い周期は偶然で出る程度か")
    tab = {}
    for a in COINS:
        r = ctx["RET"][a]
        for half, lo, hi in (("前半", START19, SPLIT), ("後半", SPLIT, MM.END + pd.Timedelta(days=1))):
            v = r[(r.index >= lo) & (r.index < hi)].dropna().to_numpy()
            per, p, p7 = peaks(v)
            pa, pab, pa7 = peaks(np.abs(v))
            tab[(a, half)] = dict(per=per, p=p, p7=p7, pa=pa, pab=pab, pa7=pa7)
            print(f"    {a} {half}：上げ下げの強い周期 {per[0]:.1f}・{per[1]:.1f}・{per[2]:.1f}日（偶然でも出る確率{p * 100:.0f}%、7日周期{p7 * 100:.0f}%）"
                  f"／値動きの大きさ {pa[0]:.0f}・{pa[1]:.0f}・{pa[2]:.0f}日（{pab * 100:.0f}%、7日周期{pa7 * 100:.1f}%）")
    keep_sum(pid, tab=tab)
    res = {}
    for N in (256, 512):
        for K in (1, 3):
            res[f"直近{N}日・強い周期{K}つ"] = dt(ctx, pid, f"周期（直近{N}日の強い周期{K}つ）をのばした向きに9時〜翌9時",
                                            {a: fourier_side(ctx["RET"][a].to_numpy(), ctx["D"][a]["idx"], N, K) for a in COINS})
    o = comp(ctx, pid, res)
    keep_sum(pid, best=o["best"], verdict=o["verdict"], r=brief(o["r"]), npos=o["npos"], n=o["n"], rows={k: brief(v) for k, v in res.items()})


# ---------------------------------------------------------------- N08 自己相関
def acf(v, L):
    v = v - v.mean()
    s = (v * v).sum()
    return np.array([(v[k:] * v[:-k]).sum() / s for k in range(1, L + 1)])


def sec_acf(ctx):
    pid = "N08"
    print("\n==== N08 自己相関：今日の動きと、1〜30日前の動きの関係（前半・後半で別々に）")
    tab, rho1 = {}, {}
    for a in COINS:
        r = ctx["RET"][a]
        for half, lo, hi in (("前半", START19, SPLIT), ("後半", SPLIT, MM.END + pd.Timedelta(days=1))):
            v = r[(r.index >= lo) & (r.index < hi)].dropna().to_numpy()
            ac, aa = acf(v, 30), acf(np.abs(v), 30)
            thr = 1.96 / np.sqrt(len(v))
            sig = [k + 1 for k in range(30) if abs(ac[k]) > thr]
            tab[(a, half)] = dict(ac=ac, aa=aa, thr=thr, sig=sig)
            if half == "前半":
                rho1[a] = ac
            print(f"    {a} {half}：1日前{ac[0]:+.3f}・7日前{ac[6]:+.3f}（偶然の範囲±{thr:.3f}）。範囲を超えた日数：{sig or 'なし'}"
                  f"／値動きの大きさ（絶対値）は1日前{aa[0]:+.2f}・7日前{aa[6]:+.2f}・30日前{aa[29]:+.2f}")
    keep_sum(pid, tab=tab)
    res = {}
    for k in (1, 7):
        for W in (90, 365):
            sides = {}
            for a in COINS:
                r = ctx["RET"][a]
                rho = r.rolling(W).corr(r.shift(k)).shift(1).to_numpy()
                sides[a] = np.sign(np.nan_to_num(rho)) * np.sign(r.shift(k).fillna(0).to_numpy())
            res[f"{k}日前・直近{W}日の相関の向き"] = dt(ctx, pid, f"自己相関（{k}日前・直近{W}日）の向きに9時〜翌9時", sides)
        sides = {a: np.sign(rho1[a][k - 1]) * np.sign(ctx["RET"][a].shift(k).fillna(0).to_numpy()) for a in COINS}
        res[f"{k}日前・前半の相関の向きで固定"] = dt(ctx, pid, f"自己相関（{k}日前・前半の向きで固定）で9時〜翌9時", sides)
    o = comp(ctx, pid, res)
    keep_sum(pid, best=o["best"], verdict=o["verdict"], r=brief(o["r"]), npos=o["npos"], n=o["n"], rows={k: brief(v) for k, v in res.items()})


# ---------------------------------------------------------------- N09 ハースト指数
def hurst_roll(c, win=100, lags=np.arange(2, 21)):
    """直近 win 日の終値から、ずらし幅ごとの値動きのばらつきの伸び方でハースト指数を出す（0.5より上＝続きやすい）。"""
    lc = np.log(c)
    H = np.full(len(lc), np.nan)
    ll = np.log(lags)
    llc = ll - ll.mean()
    for t in range(win, len(lc) + 1):
        w = lc[t - win:t]
        if np.isnan(w).any():
            continue
        y = np.log([np.std(w[g:] - w[:-g]) for g in lags])
        H[t - 1] = (llc * (y - y.mean())).sum() / (llc ** 2).sum()
    return H


def sec_hurst(ctx):
    pid = "N09"
    D, RET = ctx["D"], ctx["RET"]
    print("\n==== N09 ハースト指数（直近100日）：0.5より上なら順張り（10日の向き）、下なら逆張り（前日の反対）")
    H, Hm, mom, rev = {}, {}, {}, {}
    for a in COINS:
        h = hurst_roll(D[a]["dC"])
        H[a] = np.r_[np.nan, h[:-1]]                  # その朝にわかる値
        Hm[a] = pd.Series(H[a]).rolling(365, min_periods=100).median().to_numpy()
        r = RET[a]
        mom[a] = np.sign(r.rolling(10).sum().shift(1).fillna(0).to_numpy())
        rev[a] = -np.sign(r.shift(1).fillna(0).to_numpy())
        sel = (D[a]["idx"] >= START19) & (D[a]["idx"] <= MM.END)
        print(f"    {a}：ハースト指数の中央値 {np.nanmedian(H[a][sel]):.2f}、0.5より上の日 {np.nanmean(H[a][sel] > 0.5) * 100:.0f}%")
        keep_sum(pid, **{f"H_{a}": (float(np.nanmedian(H[a][sel])), float(np.nanmean(H[a][sel] > 0.5) * 100))})
    with np.errstate(invalid="ignore"):
        res = {"0.5で切り替え": dt(ctx, pid, "ハースト指数0.5で順張り・逆張りを切り替え", {a: np.where(H[a] > 0.5, mom[a], np.where(H[a] < 0.5, rev[a], 0)) for a in COINS}),
               "1年の中央値で切り替え": dt(ctx, pid, "ハースト指数の1年の中央値で順張り・逆張りを切り替え",
                                    {a: np.where(H[a] > Hm[a], mom[a], np.where(H[a] <= Hm[a], rev[a], 0)) for a in COINS}),
               "0.5より上の時だけ順張り": dt(ctx, pid, "ハースト指数が0.5より上の時だけ順張り", {a: np.where(H[a] > 0.5, mom[a], 0) for a in COINS}),
               "0.5より下の時だけ逆張り": dt(ctx, pid, "ハースト指数が0.5より下の時だけ逆張り", {a: np.where(H[a] < 0.5, rev[a], 0) for a in COINS})}
        ref = {"いつも順張り": dt(ctx, pid, "参考：いつも順張り（10日の向き）", mom),
               "いつも逆張り": dt(ctx, pid, "参考：いつも逆張り（前日の反対）", rev)}
        o = comp(ctx, pid, res)
        filt = {"0.5より上の日だけ": scaled(ctx, pid, "値幅ブレイク：ハースト指数が0.5より上の日だけ", ctx["per0"], ctx["T0"],
                                       {a: (H[a] > 0.5).astype(float) for a in COINS}),
                "0.5より下の日だけ": scaled(ctx, pid, "値幅ブレイク：ハースト指数が0.5より下の日だけ", ctx["per0"], ctx["T0"],
                                       {a: (H[a] < 0.5).astype(float) for a in COINS})}
    keep_sum(pid, best=o["best"], verdict=o["verdict"], r=brief(o["r"]), npos=o["npos"], n=o["n"], rows={k: brief(v) for k, v in res.items()},
             ref={k: brief(v) for k, v in ref.items()}, filt={k: brief(v) for k, v in filt.items()})


# ---------------------------------------------------------------- N11 遺伝的アルゴリズム
def sec_ga(ctx, seeds=(1, 2, 3, 4, 5), pop_n=80, gens=40):
    pid = "N11"
    F, D = ctx["F"], ctx["D"]
    print("\n==== N11 遺伝的アルゴリズム：買い・売りの条件（材料と境目、最大3つずつ）を前半（2019〜2022年）の安定度が高くなるように自動で作り、後半で確かめる")
    cols = BASE
    idx = D["BTC"]["idx"]
    tr = (idx >= START19) & (idx < SPLIT)
    Xa = {a: F[a][cols].to_numpy(float) for a in COINS}
    Q = np.nanquantile(np.vstack([Xa[a][tr] for a in COINS]), np.arange(1, 10) / 10, axis=0)
    PL, PS = {}, {}
    for a in COINS:
        d0 = np.arange(len(D[a]["dO"])) * DAY
        PL[a] = R2B.window(D[a], a, d0, d0 + DAY, np.ones(len(d0)), np.ones(len(d0)))[0].to_numpy()
        PS[a] = R2B.window(D[a], a, d0, d0 + DAY, -np.ones(len(d0)), np.full(len(d0), 0.5))[0].to_numpy()
    mon = ((idx.year - 2019) * 12 + idx.month - 1).to_numpy()
    nF = len(cols)

    def mask(conds, a):
        m = np.ones(len(idx), bool)
        with np.errstate(invalid="ignore"):
            for f, op, q in conds:
                m &= (Xa[a][:, f] > Q[q, f]) if op else (Xa[a][:, f] < Q[q, f])
        return m

    def pnl(ind):
        tot, n = np.zeros(len(idx)), 0
        ml = {a: mask(ind[0], a) if ind[0] else np.zeros(len(idx), bool) for a in COINS}
        ms = {a: mask(ind[1], a) if ind[1] else np.zeros(len(idx), bool) for a in COINS}
        for a in COINS:
            tot += ml[a] * PL[a] + ms[a] * PS[a]
            n += ml[a][tr].sum() + ms[a][tr].sum()
        return tot, n, ml, ms

    def fit(ind):
        tot, n, _, _ = pnl(ind)
        if n / 4 < 20:
            return -9.0
        mm = np.bincount(mon[tr], weights=tot[tr])
        return mm.mean() / mm.std() * np.sqrt(12) if mm.std() > 0 else -9.0

    def rc(g):
        return (int(g.integers(nF)), int(g.integers(2)), int(g.integers(9)))

    def ri(g):
        L = [rc(g) for _ in range(int(g.integers(1, 4)))]
        S = [rc(g) for _ in range(int(g.integers(1, 4)))] if g.random() < 0.5 else []
        return (L, S)

    def cross(p1, p2, g):
        c = (list(p1[0] if g.random() < 0.5 else p2[0]), list(p1[1] if g.random() < 0.5 else p2[1]))
        return c if (c[0] or c[1]) else (list(p1[0]) or [rc(g)], list(p1[1]))

    def mut(ind, g):
        L, S = list(ind[0]), list(ind[1])
        side = L if (g.random() < 0.5) else S
        if not side:
            if g.random() < 0.5:
                side.append(rc(g))
            return (L, S)
        i = int(g.integers(len(side)))
        f, op, q = side[i]
        u = g.random()
        if u < 0.4:
            side[i] = (f, op, int(np.clip(q + g.choice([-1, 1]), 0, 8)))
        elif u < 0.6:
            side[i] = (int(g.integers(nF)), op, q)
        elif u < 0.7:
            side[i] = (f, 1 - op, q)
        elif u < 0.85 and len(side) < 3:
            side.append(rc(g))
        elif len(side) > 1 or (L and S):
            side.pop(i)
        return (L, S) if (L or S) else ind

    def desc(conds):
        return "かつ".join(f"「{cols[f]}」{'>' if op else '<'}{Q[q, f]:+.2f}（下から{(q + 1) * 10}%）" for f, op, q in conds) or "なし"

    rows = []
    for seed in seeds:
        g = np.random.default_rng(seed)
        pop = [ri(g) for _ in range(pop_n)]
        fits = [fit(i) for i in pop]
        for _ in range(gens):
            order = np.argsort(fits)[::-1]
            new = [pop[i] for i in order[:4]]
            while len(new) < pop_n:
                c1, c2 = (pop[max(g.choice(pop_n, 3, replace=False), key=lambda i: fits[i])] for _ in range(2))
                ch = cross(c1, c2, g)
                if g.random() < 0.7:
                    ch = mut(ch, g)
                new.append(ch)
            pop, fits = new, [fit(i) for i in new]
        b = int(np.argmax(fits))
        best = pop[b]
        _, _, ml, ms = pnl(best)
        per, Ts = {}, []
        for a in COINS:
            per[a] = pd.Series(ml[a] * PL[a] + ms[a] * PS[a], index=idx)
            Ts.append(pd.DataFrame({"day": np.r_[idx[ml[a]], idx[ms[a]]],
                                    "R": np.r_[PL[a][ml[a]] / 0.5, PS[a][ms[a]] / 0.25], "hold": 1}))
        T = pd.concat(Ts, ignore_index=True)
        T = T[(T.day >= MM.START) & (T.day <= MM.END)]
        rule = f"買い：{desc(best[0])}／売り：{desc(best[1])}"
        r = R3.show(pid, f"遺伝的アルゴリズムで作った規則（種{seed}）", per, T, memo=f"前半で選んだ規則。{rule}")
        print(f"      前半（学習）の安定度{fits[b]:.2f} → 後半{r['_h2s']:.2f}。{rule}")
        rows.append(dict(seed=seed, fit=fits[b], r=brief(r), rule=rule))
    keep_sum(pid, rows=rows)


# ---------------------------------------------------------------- N13 GARCH
def garch_sig(lr, idx, start=pd.Timestamp("2018-03-01")):
    """毎月はじめに GARCH(1,1)（t分布）を学び直し、各日の値動きの大きさ（1日の標準偏差、%）を前日までのデータで予想する。"""
    from arch import arch_model
    r = 100 * lr
    ok = ~np.isnan(r)
    first = int(np.argmax(ok))
    sig = np.full(len(r), np.nan)
    m = pd.date_range(start, MM.END, freq="MS")
    for m0, m1 in zip(m, list(m[1:]) + [MM.END + pd.Timedelta(days=1)]):
        tr = ok & (idx < m0)
        res = arch_model(r[tr], mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False).fit(disp="off", show_warning=False)
        mu, om, al, be = (res.params[k] for k in ("mu", "omega", "alpha[1]", "beta[1]"))
        e = np.where(ok, r - mu, 0.0)
        te = (idx >= m0) & (idx < m1)
        last = int(np.nonzero(te)[0].max()) + 1
        s2 = np.zeros(last)
        s2[first] = np.var(r[tr])
        for t in range(first + 1, last):
            s2[t] = om + al * e[t - 1] ** 2 + be * s2[t - 1]
        sig[te] = np.sqrt(s2[te[:last]])
    return sig


def sec_garch(ctx):
    pid = "N13"
    D = ctx["D"]
    print("\n==== N13 GARCH：翌日の値動きの大きさを予想して、量や休む日を決める")
    sig = {a: garch_sig(ctx["RET"][a].to_numpy(), D[a]["idx"]) for a in COINS}
    cors = {}
    for a in COINS:
        x = D[a]
        sel = (x["idx"] >= START19) & (x["idx"] <= MM.END)
        rngd = np.log(x["dH"] / x["dL"]) * 100
        cand = {"GARCH": sig[a], "ATR（14日）": x["atr"] / x["dO"] * 100,
                "10日の実現ボラ": ctx["RET"][a].rolling(10).std().shift(1).to_numpy() * 100}
        cors[a] = {k: np.corrcoef(v[sel & ~np.isnan(v)], rngd[sel & ~np.isnan(v)])[0, 1] for k, v in cand.items()}
        print(f"    {a}：その日の値幅との相関 " + "・".join(f"{k}{v:.2f}" for k, v in cors[a].items()))
    keep_sum(pid, cors=cors)
    ms, mq, m3 = {}, {}, {}
    for a in COINS:
        x = D[a]
        s = pd.Series(sig[a])
        atrp = pd.Series(x["atr"] / x["dO"] * 100)
        c = (atrp / s).rolling(365, min_periods=100).median().shift(1)
        ms[a] = (atrp / (c * s)).clip(1 / 3, 3).fillna(1.0).to_numpy()
        med = s.rolling(365, min_periods=100).median().shift(1)
        mq[a] = np.where(med.isna(), 1.0, (s < med).astype(float))
        pct = s.rolling(365, min_periods=100).apply(lambda v: (v[:-1] < v[-1]).mean(), raw=True).to_numpy()
        m3[a] = np.where(np.isnan(pct), 1.0, np.array([1.5, 1.0, 0.5])[np.searchsorted([1 / 3, 2 / 3], np.nan_to_num(pct, nan=0.5))])
    per0, T0 = ctx["per0"], ctx["T0"]
    out = {"GARCHで1Rを決める": scaled(ctx, pid, "値幅ブレイク：1Rを、ATRの代わりにGARCHの予想で決める", per0, T0, ms),
           "GARCHで静かな日だけ": scaled(ctx, pid, "値幅ブレイク：GARCHの予想が1年の中央値より小さい日だけ", per0, T0, mq),
           "GARCHの順位で3段階": scaled(ctx, pid, "値幅ブレイク：GARCHの予想の1年の順位で3段階（1.5・1・0.5倍）", per0, T0, m3),
           "参考：E05（10日の実現ボラで静かな日だけ）": scaled(ctx, pid, "参考：値幅ブレイク：10日の実現ボラが1年の中央値より小さい日だけ（E05）",
                                                 per0, T0, ctx["Mq"]),
           "参考：E07（ATRの順位で3段階）": scaled(ctx, pid, "参考：値幅ブレイク：ATRの1年の順位で3段階（E07）", per0, T0, ctx["M3"]),
           "S1の静かな日をGARCHで判定": scaled(ctx, pid, "S1の静かな日を、10日の実現ボラの代わりにGARCHで判定", ctx["vb5"], ctx["Tb5"], mq)}
    keep_sum(pid, rows={k: brief(v) for k, v in out.items()})


# ---------------------------------------------------------------- 本体
SECTIONS = {"ml": sec_ml, "markov": sec_markov, "hmm": sec_hmm, "fourier": sec_fourier, "acf": sec_acf, "hurst": sec_hurst,
            "ga": sec_ga, "garch": sec_garch}


def main(cache, only=None):
    names = list(SECTIONS) if not only else only.split(",")
    if not only:
        print(f"（古い「{GROUP}」の行を {drop(GROUP)} 行消して、計算し直す）")
    D = R1.prep(cache)
    F, RET = build(D)
    per0, T0 = R2.combo(D)
    v0 = R1.total(per0)
    base = R1.stats(v0, T0)
    R1.line("A01（基準・2019年から）", base)
    vb5, Tb5 = R4.vbo_var(D, "btc5", False)
    Mq = {a: R4.mult(D[a], "静かな日だけ") for a in COINS}
    M3 = {a: R4.mult(D[a], "3段階") for a in COINS}
    ctx = dict(D=D, F=F, RET=RET, per0=per0, T0=T0, v0=v0, base=base, vb5=vb5, Tb5=Tb5, Mq=Mq, M3=M3)
    a01 = R3.show("基準", "A01 値幅ブレイク（2019年から）", per0, T0, tf="15分足")
    s1 = scaled(ctx, "基準", "S1 ETHはBTCの向き＋静かな日だけ（2019年から）", vb5, Tb5, Mq)
    keep_sum("基準", A01=brief(a01), S1=brief(s1))
    for n in names:
        SECTIONS[n](ctx)
    print("\n==== まとめ（部品としての判定）")
    for pid, o in R3.OUT.items():
        r = o["r"]
        print(f"  {pid} {o['verdict']:12s} 前半で一番良い形「{o['best']}」：年{r['年平均%']:+.2f}% 落込{r['最大の落ち込み%']:.1f}% "
              f"安定度{r['安定度']:.2f}（前半{r['_h1s']:.2f}・後半{r['_h2s']:.2f}）｜{o['n']}通り中{o['npos']}通りがプラス")
    return SUMMARY


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
