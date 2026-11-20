#!/usr/bin/env python3
"""検証結果の記録（project/research/results.csv）。同じ「グループ＋名前」は上書きする。

使い方（スクリプトから）:
    from registry import record, metrics_row
    record(group="分散", name="値幅ブレイク4時間帯", coins="BTC,ETH", timeframe="1時間足", **metrics_row(daily, trades_per_year=...))
一覧: python3 project/tools/registry.py   （効率＝年平均÷最大の落ち込み の高い順）
"""
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "research"))
PATH = os.path.join(ROOT, "results.csv")
FIELDS = ["記録日", "グループ", "名前", "銘柄", "時間足", "期間", "コストの前提", "1回の量",
          "年平均%", "最大の落ち込み%", "効率", "マイナスの年", "最悪の年%", "プラスの月%", "最長連続マイナス月",
          "高値回復の最長日数", "年の取引回数", "前半の年平均%", "後半の年平均%", "メモ"]
SPLIT = pd.Timestamp("2023-01-01")


def _metrics(x):
    eq = x.cumsum()
    yrs = max((x.index[-1] - x.index[0]).days / 365.25, 0.5)
    yearly = x.groupby(x.index.year).sum()
    monthly = x.resample("ME").sum()
    run = neg_run = 0
    for v in monthly:
        run = run + 1 if v < 0 else 0
        neg_run = max(neg_run, run)
    peak = eq.cummax()
    longest = cur = 0
    for u in (eq < peak - 1e-9):
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    return dict(annual=x.sum() / yrs, dd=(peak - eq).max(), yearly=yearly, pos_m=(monthly > 0).mean(), neg_run=neg_run, longest=longest)


def metrics_row(daily, trades_per_year=np.nan):
    """毎日の損益（資金に対する％、単利）から記録用の値を作る。"""
    daily = daily.sort_index()
    m = _metrics(daily)
    a, b = daily[daily.index < SPLIT], daily[daily.index >= SPLIT]
    neg = [int(y) for y, v in m["yearly"].items() if v < 0]
    return {"年平均%": round(m["annual"], 2), "最大の落ち込み%": round(m["dd"], 2),
            "効率": round(m["annual"] / m["dd"], 2) if m["dd"] > 0 else np.nan,
            "マイナスの年": ",".join(str(y) for y in neg) or "なし", "最悪の年%": round(m["yearly"].min(), 2),
            "プラスの月%": round(m["pos_m"] * 100, 1), "最長連続マイナス月": m["neg_run"], "高値回復の最長日数": m["longest"],
            "年の取引回数": round(trades_per_year, 1) if trades_per_year == trades_per_year else "",
            "前半の年平均%": round(_metrics(a)["annual"], 2) if len(a) > 30 else "",
            "後半の年平均%": round(_metrics(b)["annual"], 2) if len(b) > 30 else "",
            "期間": f"{daily.index[0]:%Y-%m}〜{daily.index[-1]:%Y-%m}"}


def record(**row):
    os.makedirs(ROOT, exist_ok=True)
    row.setdefault("記録日", date.today().isoformat())
    row.setdefault("コストの前提", "Vantage CFD（往復0.05%・買いスワップ年25〜35%・売り0%）")
    row.setdefault("1回の量", "ATR1本分の値動き＝資金の0.5%（単利）")
    df = pd.read_csv(PATH) if os.path.exists(PATH) else pd.DataFrame(columns=FIELDS)
    df = df[~((df["グループ"] == row["グループ"]) & (df["名前"] == row["名前"]))]
    df = pd.concat([df, pd.DataFrame([{k: row.get(k, "") for k in FIELDS}])], ignore_index=True)
    df.to_csv(PATH, index=False)


def show(top=None):
    df = pd.read_csv(PATH)
    df = df.sort_values("効率", ascending=False)
    cols = ["グループ", "名前", "年平均%", "最大の落ち込み%", "効率", "マイナスの年", "プラスの月%", "後半の年平均%"]
    print(df[cols].head(top).to_string(index=False) if top else df[cols].to_string(index=False))


if __name__ == "__main__":
    show(int(sys.argv[1]) if len(sys.argv) > 1 else None)
