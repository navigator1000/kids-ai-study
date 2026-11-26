#!/usr/bin/env python3
"""ランキングのページ（Artifact「BTC・ETH 検証ランキング」）に載せるデータを作る。

入力:
  project/research/results.csv   検証の記録（計画書の検証は、グループが「計画・」で始まる行）
  project/research/plan.csv      計画書（ルール・調べる設定・結果のまとめ）
  <capture>/*.json               capture_series.py で集めた、行ごとの月ごとの損益（資金に対する%）
出力（<out> フォルダ）:
  logics.json  1行＝1つの検証結果。数字・月ごとの損益・前提の印・「同じ計画IDで一番良い設定」の印
  items.json   1行＝計画書の1件。ルール・設定・結果のまとめ・合格の段階
  refs.json    偶然の目安（O12 コイン投げの1,000回）
どれも {rows, meta} の形で、meta にこのスクリプトの全文を入れる（ページの「出どころ」で読める）。

数字の決まり:
  ・年平均・落ち込み・安定度・負けた年は results.csv の値（毎日の損益から計算したもの）
  ・前半（2018〜2022年）・後半（2023年〜）・最近12か月の安定度と落ち込みは、月ごとの損益から計算する
    （安定度＝月の損益の平均÷ばらつき×√12。落ち込みは月末の値で測るので、毎日で測るより少し小さく出る）
  ・前半・後半の年平均は results.csv の値
  ・「一番良い設定」は、同じ計画IDの行のうち前提がそろったもの（Vantage・2018年〜・BTC+ETH）から、
    前半の安定度が一番高いものを選ぶ（後半の成績を見て選ばない）
使い方: python3 project/tools/ranking_data.py <capture フォルダ> <out フォルダ>
"""
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "research"))
H2 = "2023-01"
ROUND = {"計画・第1回": "第1回", "計画・第2回": "第2回・前半", "計画・第2回後半": "第2回・後半", "計画・第3回": "第3回・前半",
         "計画・第3回後半": "第3回・後半", "計画・第4回": "第4回・前半", "計画・第4回後半": "第4回・後半", "計画・第5回": "第5回・前半",
         "計画・第5回後半": "第5回・後半", "計画・第6回": "第6回・前半", "計画・第6回後半": "第6回・後半",
         "計画・第7回": "第7回"}
CANDS = {"計画・第2回\tA01 A01（基準・確認）": ("A01", "値幅ブレイクの基本形", "5日平均の向きに、始値±前日の値幅×0.8の逆指値。翌朝9時に決済"),
         "計画・第4回\tM08 S1 ETHはBTCの向き＋静かな日だけ": ("S1", "ETHもBTCの向き＋静かな日だけ", "第一候補（第4回）"),
         "計画・第4回\tM08 S2 ETHはBTCの向き＋3段階＋C29×0.25": ("S2", "S1＋量を3段階＋C29×0.25", "利益は大きいが、毎日の持ち替えで手間が増える"),
         "計画・第6回\tA15 S1：直近2日が全敗なら翌日は2.0倍（実際に取った取引で判定）": ("S4案", "S1＋直近2日が全敗なら翌日は2倍", "全期間の結果を見た後に見つけた形。前向きテストで確かめる"),
         "計画・第5回後半\tP05 国内現物：S1の買いだけ（往復0.15%）": ("国内現物S1", "S1の買いだけ（国内現物・往復0.15%）", "SBI VCトレードを想定。スワップなし")}


def num(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def stab(m):
    m = np.asarray(m, float)
    return round(float(m.mean() / m.std(ddof=1) * math.sqrt(12)), 2) if len(m) > 2 and m.std(ddof=1) > 0 else None


def dd(m):
    eq = np.cumsum(np.r_[0.0, np.asarray(m, float)])
    return round(float((np.maximum.accumulate(eq) - eq).max()), 2) if len(m) else None


def stage(text):
    t = str(text)
    for s in ("3次合格", "2次合格", "1次合格"):
        if s in t:
            return s
    if "不合格" in t or "採用しない" in t:
        return "不合格"
    if "合格" in t:
        return "合格"
    return "—"


def main(capdir, out):
    os.makedirs(out, exist_ok=True)
    cap = {}
    for f in sorted(glob.glob(os.path.join(capdir, "*.json"))):
        cap.update(json.load(open(f, encoding="utf-8")))
    res = pd.read_csv(os.path.join(ROOT, "results.csv"))
    res = res[res["グループ"].astype(str).str.startswith("計画")].reset_index(drop=True)
    plan = pd.read_csv(os.path.join(ROOT, "plan.csv"), dtype=str).fillna("")
    pinfo = {r.ID: r for r in plan.itertuples()}

    rows = []
    for i, r in res.iterrows():
        k = f"{r['グループ']}\t{r['名前']}"
        pid = str(r["計画ID"]) if str(r["計画ID"]) != "nan" else ""
        name = str(r["名前"])
        if pid and name.startswith(pid + " "):
            name = name[len(pid) + 1:]
        s = cap.get(k)
        months = sorted(s) if s else []
        vals = [s[m] for m in months] if s else []
        ser = pd.Series(vals, index=months, dtype=float)
        h1, h2 = ser[ser.index < H2], ser[ser.index >= H2]
        r12 = ser.iloc[-12:] if len(ser) >= 12 else ser.iloc[0:0]
        cost = str(r["コストの前提"])
        per = str(r["期間"])
        ref = any(w in name for w in ("参考", "基準", "確かめ")) or pid in ("基準", "K", "Q")
        mix = name.startswith("値幅ブレイク＋") and "×0.5" in name
        basis = cost.startswith("Vantage") and per.startswith("2018-01") and str(r["銘柄"]) == "BTC,ETH" and bool(s)
        neg = str(r["マイナスの年"])
        cand = CANDS.get(k)
        rows.append(dict(
            key=f"r{i:04d}", pid=pid, round=ROUND.get(r["グループ"], r["グループ"]), name=name,
            cat=pinfo[pid].分類 if pid in pinfo else "その他", cand=cand[0] if cand else None, cand_title=cand[1] if cand else None,
            cand_note=cand[2] if cand else None, item=pinfo[pid].名前 if pid in pinfo else None,
            annual=num(r["年平均%"]), dd=num(r["最大の落ち込み%"]), eff=num(r["効率"]), stab=num(r["安定度"]),
            neg_years=neg if neg not in ("nan", "") else None, neg_n=0 if neg == "なし" else (len(neg.split(",")) if neg not in ("nan", "") else None),
            worst=num(r["最悪の年%"]), posm=num(r["プラスの月%"]), trades=num(r["年の取引回数"]),
            a_h1=num(r["前半の年平均%"]), a_h2=num(r["後半の年平均%"]),
            s_h1=stab(h1), s_h2=stab(h2), s_r12=stab(r12), dd_h1=dd(h1), dd_h2=dd(h2), dd_r12=dd(r12),
            a_r12=round(float(r12.sum()), 2) if len(r12) == 12 else None,
            e_h1=None, e_h2=None, e_r12=None,
            period=per if per != "nan" else None, cost=cost, tf=str(r["時間足"]), size=str(r["1回の量"]),
            memo=str(r["メモ"]) if str(r["メモ"]) != "nan" else None,
            basis=basis, ref=ref, mix=mix, best=False, best_any=False,
            m0=months[0] if months else None, m=",".join(f"{v:.3f}" for v in vals) if vals else None))
    df = pd.DataFrame(rows)
    # 効率（年平均÷落ち込み）を期間ごとに
    for a, d, e in (("a_h1", "dd_h1", "e_h1"), ("a_h2", "dd_h2", "e_h2"), ("a_r12", "dd_r12", "e_r12")):
        df[e] = [round(x / y, 2) if x is not None and y and y > 0 and not (isinstance(x, float) and math.isnan(x)) else None
                 for x, y in zip(df[a], df[d])]
    # 同じ計画IDで、前提がそろった行から、前半の安定度が一番高いものを「一番良い設定」に（best）。
    # 前提がそろわない行も含めて選んだものは best_any（「前提がそろったものだけ」を外した時に使う）
    ok = df.basis & ~df.ref & ~df.mix & df.s_h1.notna()
    for pid, g in df[ok].groupby("pid"):
        df.loc[g.s_h1.idxmax(), "best"] = True
    ok2 = ~df.ref & ~df.mix & (df.s_h1.notna() | df.stab.notna())
    for pid, g in df[ok2].groupby("pid"):
        df.loc[g.s_h1.fillna(g.stab).idxmax(), "best_any"] = True
    for c in ("A01", "S1", "S2", "S4案", "国内現物S1"):
        assert (df.cand == c).sum() == 1, c
    items = []
    for p in plan.itertuples():
        if p.状態 != "済" or not (df.pid == p.ID).any():
            continue
        items.append(dict(pid=p.ID, cat=p.分類, title=p.名前, rule=p.ルール, setting=p.調べる設定 if p.調べる設定 != "—" else "",
                          tf=p.足, result=p.結果, stage=stage(p.結果), n_rows=int((df.pid == p.ID).sum())))
    refs = [dict(ref="coin", label="コイン投げ（時刻も向きもランダム）", a_med=-0.97, a_best=1.76, s_med=-0.34, s_best=0.57),
            dict(ref="long", label="いつも買い（時刻だけランダム）", a_med=-0.40, a_best=3.27, s_med=-0.10, s_best=0.69)]
    code = open(os.path.abspath(__file__), encoding="utf-8").read()
    meta = [dict(label="ranking_data.py", value=code, type="code", lang="python"),
            dict(label="Input", value="project/research/results.csv・plan.csv と、capture_series.py で集めた月ごとの損益")]
    clean = lambda recs: json.loads(json.dumps(recs, ensure_ascii=False, allow_nan=True).replace("NaN", "null"))
    for name, recs in (("logics", df.to_dict("records")), ("items", items), ("refs", refs)):
        recs = [{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in rec.items()} for rec in recs]
        with open(os.path.join(out, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump({"rows": clean(recs), "meta": meta}, f, ensure_ascii=False, separators=(",", ":"))
    print(f"logics {len(df)}行（曲線あり{df.m.notna().sum()}・前提がそろった{int(df.basis.sum())}・一番良い設定{int(df.best.sum())}）・"
          f"items {len(items)}件 → {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
