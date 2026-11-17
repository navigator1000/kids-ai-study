#!/usr/bin/env python3
"""毎朝のトレード計画（日足の確定後＝日本時間9時過ぎに実行）。

strategy_config.json のルールに従い、銘柄ごとに次を出力する。
  - 今日の行動（新規買い／保有継続／見送り）と価格（買値・損切り・利確・時間切れ日）
  - 1回の損失を risk_jpy 円に抑える数量
  - 見送りの場合、買いシグナルが出る価格
  - 運用開始日（live_since）以降の成績（ルール通りに売買した場合を自動で記録）
  - このルールの過去検証の成績
使い方: python3 project/tools/daily_plan.py [キャッシュ用ディレクトリ]
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
import strategy_lab as S  # noqa: E402
from marketdata import _get, load_all  # noqa: E402

NAMES = {"BTC": "ビットコイン", "ETH": "イーサリアム", "Gold": "ゴールド（金先物）"}
UNIT = {"BTC": "BTC", "ETH": "ETH", "Gold": "オンス"}
CLOSE_JST = {"BTC": "毎朝9:00", "ETH": "毎朝9:00", "Gold": "毎朝7:00（夏時間は6:00）"}


def live_price(asset):
    try:
        if asset in ("BTC", "ETH"):
            return float(_get(f"https://data-api.binance.vision/api/v3/ticker/price?symbol={asset}USDT")["price"])
        m = _get("https://query1.finance.yahoo.com/v8/finance/chart/GC=F?range=1d&interval=5m")
        return float(m["chart"]["result"][0]["meta"]["regularMarketPrice"])
    except Exception:
        return None


def usdjpy():
    m = _get("https://query1.finance.yahoo.com/v8/finance/chart/JPY=X?range=1d&interval=1h")
    return float(m["chart"]["result"][0]["meta"]["regularMarketPrice"])


def exit_date(asset, start, hold):
    if asset == "Gold":
        return (start + pd.offsets.BDay(hold - 1)).date()
    return (start + pd.Timedelta(days=hold - 1)).date()


def usd(x):
    return f"${x:,.0f}" if x >= 100 else f"${x:,.2f}"


def stats_line(R):
    R = np.asarray([r for r in R if not np.isnan(r)])
    if len(R) < 2:
        return "データ不足"
    s = S.summarize(R)
    return f"{s['n']}回 勝率{s['win']*100:.0f}% 平均{s['exp']:+.2f}R 最大{s['maxl']}連敗"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache", nargs="?", default=None, help="価格データのキャッシュ用ディレクトリ")
    ap.add_argument("--asof", help="過去のこの日付（YYYY-MM-DD）の足までで計画を再現する（学習用）")
    ap.add_argument("--live", help="運用開始日を上書きする（YYYY-MM-DD）")
    o = ap.parse_args()
    cfg = json.load(open(f"{HERE}/strategy_config.json"))
    rule = cfg["rule"]
    S.HOLD = rule["hold_days"]
    risk_jpy = cfg["risk_jpy"]
    live = pd.Timestamp(o.live or cfg["live_since"])
    data = load_all(o.cache)
    if o.asof:
        data = {k: v[v.index <= pd.Timestamp(o.asof)] for k, v in data.items()}
    fx = usdjpy()
    now = pd.Timestamp.now(tz="Asia/Tokyo")
    head = f"{o.asof}の足まで（再現）" if o.asof else f"{now:%Y-%m-%d %H:%M} JST"
    print(f"# トレード計画 {head}（ドル円 {fx:.2f}）")
    print(f"ルール: {cfg['name']} ｜ 1回の許容損失 {risk_jpy:,}円 ＝ 1R\n")

    live_all, bt_all = [], []
    for a in cfg["assets"]:
        df = data[a]
        P = S.prep(df)
        N = len(df)
        i = N - 1
        c, sma, hh, atr = P["c"][i], P["sma"][rule["sma"]][i], P["hh"][i], P["atr"][i]
        px = c if o.asof else (live_price(a) or c)
        args = (1, rule["sma"], "bo", rule["stop_atr"], rule["target_R"], cfg["cost"][a])
        bt = [x for x in S.run(P, *args) if not x["open"]]
        sp = pd.Timestamp(cfg["split"][a])
        i0 = max(int(np.searchsorted(P["t"], live)), 200)
        lt = S.run(P, *args, i0=i0) if i0 < N else []
        bt_all += [dict(x, asset=a) for x in bt]
        live_all += [dict(x, asset=a) for x in lt]
        pos = lt[-1] if lt and lt[-1]["open"] else None

        print(f"## {a}（{NAMES[a]}）")
        print(f"- 確定終値 {usd(c)}（{P['t'][i]:%m/%d}足、{CLOSE_JST[a]}確定） / 現在値 {usd(px)}")
        trend = "上（上昇トレンド）" if c > sma else "下（下降トレンド）"
        print(f"- 200日線 {usd(sma)} → 終値は{trend} ｜ 直近20日高値 {usd(hh)} ｜ ATR(14) {usd(atr)}")

        if pos:
            held = pos["bars"]
            ed = exit_date(a, pos["t"], rule["hold_days"])
            unreal = (c - pos["fill"]) / pos["risk"]
            print(f"- **判定: 保有継続**（{pos['t']:%m/%d} {usd(pos['fill'])}で買い）")
            print(f"  - 損切り {usd(pos['stop'])} ／ 利確 {usd(pos['tgt'])} ／ 時間切れ {ed}の終値（保有{held}日目）")
            print(f"  - 終値ベースの含み損益 {unreal:+.2f}R（{unreal*risk_jpy:+,.0f}円）")
        elif S.signal(P, i, 1, rule["sma"], "bo"):
            risk = rule["stop_atr"] * atr
            units = risk_jpy / (risk * fx)
            ed = exit_date(a, P["t"][i] + pd.Timedelta(days=1), rule["hold_days"])
            print("- **判定: 新規買い**（①終値が200日線より上 ②終値が直近20日高値を更新 の両方が成立）")
            print(f"  - 買値: 成行（現在値 {usd(px)} 付近）")
            print(f"  - 損切り: 約定価格 − {usd(risk)}（ATR×{rule['stop_atr']:g}）→ 現在値なら {usd(px - risk)}")
            print(f"  - 利確: 約定価格 + {usd(risk*rule['target_R'])}（損切り幅×{rule['target_R']:g}）→ 現在値なら {usd(px + risk*rule['target_R'])}")
            print(f"  - 時間切れ: {ed}ごろの終値で手仕舞い（{rule['hold_days']}日）")
            print(f"  - 数量（損失{risk_jpy:,}円）: {units:.4f} {UNIT[a]}（約{units*px*fx/10000:,.1f}万円分）")
        else:
            ok1, ok2 = c > sma, c > hh
            trig = max(sma, P["h"][-20:].max())
            print(f"- **判定: 見送り**（①200日線より上 {'○' if ok1 else '×'} ②20日高値更新 {'○' if ok2 else '×'}）")
            print(f"  - 次の買いシグナル: 終値が {usd(trig)} を上回ったとき"
                  f"（現在値から{(trig/px-1)*100:+.1f}%、ATR{(trig-px)/atr:.1f}本分）")
        R_in = [x["R"] for x in bt if x["t"] < sp]
        R_out = [x["R"] for x in bt if x["t"] >= sp]
        print(f"- 過去検証: 学習期間 {stats_line(R_in)} ／ 検証期間({sp:%Y}年〜) {stats_line(R_out)}")
        if bt:
            last = bt[-3:]
            ex = "、".join(f"{x['t']:%Y/%m/%d}買い→{x['exit_t']:%m/%d}{x['why']} {x['R']:+.1f}R" for x in last)
            print(f"- 直近の過去事例: {ex}")
        print()

    closed = sorted([x for x in live_all if not x["open"]], key=lambda x: x["exit_t"])
    print(f"## 運用成績（{live:%Y-%m-%d}以降、ルール通りに売買した場合）")
    if closed:
        R = np.array([x["R"] for x in closed])
        print(f"- 決済{len(R)}回 勝ち{(R > 0).sum()}回 合計{R.sum():+.2f}R（{R.sum()*risk_jpy:+,.0f}円）")
        for x in closed[-10:]:
            print(f"  - {x['asset']} {x['t']:%m/%d}買い {usd(x['fill'])} → {x['exit_t']:%m/%d} {x['why']} {usd(x['ex'])} {x['R']:+.2f}R")
    else:
        print("- 決済済みの取引はまだありません")
    opens = [x for x in live_all if x["open"]]
    print(f"- 保有中 {len(opens)}件")

    R = pd.Series([x["R"] for x in sorted(bt_all, key=lambda x: x["exit_t"])])
    eq = R.cumsum()
    yrs = (pd.Timestamp.now() - pd.Timestamp("2001-08-01")).days / 365.25
    print("\n## 参考: このルールの過去検証（3銘柄合計）")
    print(f"- {len(R)}回 勝率{(R > 0).mean()*100:.0f}% 平均{R.mean():+.2f}R ／ 最大{S.summarize(R)['maxl']}連敗"
          f" ／ 最大の落ち込み {(eq - eq.cummax()).min():.1f}R（{(eq - eq.cummax()).min()*risk_jpy:,.0f}円）")
    print(f"- 年あたり取引回数の目安: 金 約{sum(1 for x in bt_all if x['asset']=='Gold')/yrs:.1f}回"
          f" ＋ 暗号資産 約{sum(1 for x in bt_all if x['asset']!='Gold')/((pd.Timestamp.now()-pd.Timestamp('2018-03-01')).days/365.25):.1f}回")
    print(f"- {cfg['unseen_validation']}")


if __name__ == "__main__":
    main()
