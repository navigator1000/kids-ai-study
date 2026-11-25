#!/usr/bin/env python3
"""ランキングのページ用：検証のスクリプトを「results.csv に書かない」形でもう一度動かし、記録されるはずの行ごとに
月ごとの損益（資金に対する%）を集める。

しくみ: registry の metrics_row を包んで、計算した毎日の損益とその数字を覚えておく。record が呼ばれたら、
同じ数字（年平均・落ち込み・安定度・期間）を持つ損益を探して、その行の月ごとの損益として保存する。
drop と record は何も書かない（results.csv は変わらない）。スクリプトは1つずつ別のプロセスで動かす
（モジュールを読み込んだ時の設定が混ざらないように）。

使い方: python3 project/tools/capture_series.py <キャッシュ> <モジュール名> [関数名（main）] <出力のJSON>
"""
import importlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import registry  # noqa: E402

_orig = registry.metrics_row
_recent = []
CAPT = {}


def _key(r):
    return (r.get("年平均%"), r.get("最大の落ち込み%"), r.get("安定度"), r.get("期間"))


def metrics_row(daily, trades_per_year=np.nan):
    r = _orig(daily, trades_per_year)
    try:
        m = daily.sort_index().resample("ME").sum()
        _recent.append((_key(r), m))
        del _recent[:-300]
    except Exception:  # noqa: BLE001
        pass
    return r


def record(**row):
    k = _key(row)
    m = next((s for kk, s in reversed(_recent) if kk == k), None)
    CAPT[f"{row.get('グループ')}\t{row.get('名前')}"] = None if m is None else {f"{i:%Y-%m}": round(float(v), 4) for i, v in m.items()}


registry.metrics_row = metrics_row
registry.record = record
registry.drop = lambda group, tf=None: 0

if __name__ == "__main__":
    cache, modname = sys.argv[1], sys.argv[2]
    fn = sys.argv[3] if len(sys.argv) > 4 else "main"
    out = sys.argv[-1]
    mod = importlib.import_module(modname)
    getattr(mod, fn)(cache)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(CAPT, f, ensure_ascii=False)
    print(f"{modname}.{fn}: {len(CAPT)}行（曲線あり{sum(v is not None for v in CAPT.values())}行）→ {out}")
