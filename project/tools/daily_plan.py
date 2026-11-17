#!/usr/bin/env python3
"""毎朝のトレード計画（日足の確定後＝日本時間9時過ぎに実行）。

strategy_config.json のルールと口座の銘柄仕様（Vantage）に従い、銘柄ごとに次を出力する。
  - 今日の行動（新規買い／保有継続／前日に決済／見送り）と価格
    損切り・利確は「約定価格からの幅」で示す（判定に使う価格とVantageの価格は少しずれるため）
  - 1回の損失を risk_jpy 円に抑えるロット数（最小ロットで超える場合は実際の損失額）
  - 見送りの場合、買いシグナルが出る価格
  - 運用開始日（live_since）以降の成績（ルール通りに売買した場合を自動で記録、コスト・スワップ込み）
  - このルールの過去検証の成績（コスト・スワップ込み）
使い方: python3 project/tools/daily_plan.py [キャッシュ用ディレクトリ] [--asof YYYY-MM-DD] [--live YYYY-MM-DD]
"""
import argparse
import json
import math
import sys

import numpy as np
import pandas as pd

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
import strategy_lab as S  # noqa: E402
from marketdata import _get, load_all  # noqa: E402

NAMES = {"BTC": "ビットコイン", "ETH": "イーサリアム", "Gold": "ゴールド"}
BASIS = {"BTC": "Binance BTCUSDT", "ETH": "Binance ETHUSDT", "Gold": "金先物 GC=F"}


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


def usd(x):
    return f"${x:,.0f}" if abs(x) >= 1000 else f"${x:,.2f}"


def nights(x, last_t=None):
    end = x["exit_t"] if "exit_t" in x else last_t
    return max((end - x["t"]).days + (1 if x.get("why") == "時間切れ" else 0.5), 0)


def net_R(x, spec, last_t=None, last_close=None):
    """スワップ込みのR。保有中は終値ベースの含み損益。"""
    gross = x["R"] if not x["open"] else (last_close - x["fill"]) / x["risk"]
    swap = spec["swap_long_pct_year"] / 36500 * nights(x, last_t) * x["fill"] / x["risk"]
    return gross - swap


def lots_for(spec, risk_usd_per_unit, fx, risk_jpy):
    lot = risk_jpy / (risk_usd_per_unit * fx) / spec["contract"]
    step = spec["lot_step"]
    lot = max(math.floor(lot / step + 0.5) * step, spec["min_lot"])     # 最も近い単位に丸める
    return lot, lot * spec["contract"] * risk_usd_per_unit * fx


def exit_morning(asset, start, hold):
    """期限の足が確定する日（＝その朝のレポートで決済）。"""
    last_bar = start + (pd.offsets.BDay(hold - 1) if asset == "Gold" else pd.Timedelta(days=hold - 1))
    return (last_bar + pd.Timedelta(days=1)).date()


def stats_line(R, yrs):
    R = np.asarray(R)
    if len(R) < 2:
        return "データ不足"
    s = S.summarize(R)
    return f"年{len(R)/yrs:.1f}回 勝率{s['win']*100:.0f}% 平均{s['exp']:+.2f}R 最大{s['maxl']}連敗"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache", nargs="?", default=None, help="価格データのキャッシュ用ディレクトリ")
    ap.add_argument("--asof", help="過去のこの日付（YYYY-MM-DD）の足までで計画を再現する（学習用）")
    ap.add_argument("--live", help="運用開始日を上書きする（YYYY-MM-DD）")
    o = ap.parse_args()
    cfg = json.load(open(f"{HERE}/strategy_config.json"))
    rule, risk_jpy = cfg["rule"], cfg["risk_jpy"]
    live = pd.Timestamp(o.live or cfg["live_since"])
    data = load_all(o.cache)
    if o.asof:
        data = {k: v[v.index <= pd.Timestamp(o.asof)] for k, v in data.items()}
    fx = usdjpy()
    now = pd.Timestamp.now(tz="Asia/Tokyo")
    head = f"{o.asof}の足まで（再現）" if o.asof else f"{now:%Y-%m-%d %H:%M} JST"
    print(f"# トレード計画 {head}（ドル円 {fx:.2f}）")
    print(f"ルール: {cfg['name']} ｜ 1R＝{risk_jpy:,}円 ｜ 口座: {cfg['broker']}")
    print("買い条件: ①終値が200日線より上 ②終値が直近20日高値を更新 → 朝のレポート後に成行買い。"
          f"利確は損切り幅×{rule['target_R']:g}、損切り幅と期限は銘柄別\n")

    live_all, bt_all = [], []
    for a, spec in cfg["assets"].items():
        df = data[a]
        P = S.prep(df)
        N, i = len(df), len(df) - 1
        c, sma, hh, atr = P["c"][i], P["sma"][rule["sma"]][i], P["hh"][i], P["atr"][i]
        px = c if o.asof else (live_price(a) or c)
        S.HOLD = spec["hold_days"]
        args = (1, rule["sma"], "bo", spec["stop_atr"], rule["target_R"], cfg["cost"][a])
        bt = [x for x in S.run(P, *args) if not x["open"]]
        sp = pd.Timestamp(cfg["split"][a])
        i0 = max(int(np.searchsorted(P["t"], live)), 200)
        lt = S.run(P, *args, i0=i0) if i0 < N else []
        for x in bt:
            x["Rn"] = net_R(x, spec)
        for x in lt:
            x["Rn"] = net_R(x, spec, P["t"][i], c)
        bt_all += [dict(x, asset=a) for x in bt]
        live_all += [dict(x, asset=a) for x in lt]
        pos = lt[-1] if lt and lt[-1]["open"] else None
        just_closed = [x for x in lt if not x["open"] and x["exit_t"] == P["t"][i]]

        print(f"## {spec['symbol']}（{NAMES[a]}）｜ 損切りATR×{spec['stop_atr']:g}・最大{spec['hold_days']}日")
        print(f"- 判定用の価格（{BASIS[a]}）: 確定終値 {usd(c)}（{P['t'][i]:%m/%d}足） / 現在値 {usd(px)}")
        print(f"- 200日線 {usd(sma)}（終値は{'上' if c > sma else '下'}） ｜ 直近20日高値 {usd(hh)} ｜ ATR(14) {usd(atr)}")
        for x in just_closed:
            print(f"- **前日に決済**: {x['t']:%m/%d}の買い → {x['why']} {x['Rn']:+.2f}R"
                  + ("（期限到来：アプリで保有が残っていれば今朝決済）" if x["why"] == "時間切れ" else "（アプリの損切り/利確注文で決済済みのはず）"))
        if pos:
            ed = exit_morning(a, pos["t"], spec["hold_days"])
            print(f"- **判定: 保有継続**（{pos['t']:%m/%d}買い・判定価格 {usd(pos['fill'])}、{pos['bars']}日目）")
            print(f"  - 損切り 約定価格−{usd(pos['risk'])} ／ 利確 約定価格+{usd(pos['risk']*rule['target_R'])}"
                  f" ／ 期限 {ed:%m/%d}朝のレポートで決済")
            print(f"  - 含み損益（終値・スワップ込み）{pos['Rn']:+.2f}R")
        elif S.signal(P, i, 1, rule["sma"], "bo"):
            risk = spec["stop_atr"] * atr
            lot, actual = lots_for(spec, risk, fx, risk_jpy)
            ed = exit_morning(a, P["t"][i] + pd.Timedelta(days=1), spec["hold_days"])
            over = " ※最小ロットのため1Rを超えます" if lot <= spec["min_lot"] and actual > risk_jpy * 1.25 else ""
            print("- **判定: 新規買い**（①②とも成立）")
            print(f"  - 注文: {spec['symbol']} 成行買い {lot:.2f}ロット（損切りになった場合の損失 約{actual:,.0f}円）{over}")
            print(f"  - 損切り: 約定価格 − {usd(risk)}（ATR×{spec['stop_atr']:g}） ／ 利確: 約定価格 + {usd(risk*rule['target_R'])}（損切り幅×{rule['target_R']:g}）")
            print(f"  - 目安（{usd(px)}で約定した場合）: 損切り {usd(px - risk)} ／ 利確 {usd(px + risk*rule['target_R'])}")
            print(f"  - 期限: {ed:%m/%d}朝のレポートで、残っていれば決済（最大{spec['hold_days']}日）")
        else:
            trig = max(sma, P["h"][-20:].max())
            print(f"- **判定: 見送り**（①200日線より上 {'○' if c > sma else '×'} ②20日高値更新 {'○' if c > hh else '×'}）")
            print(f"  - 次の買いシグナル: 終値が {usd(trig)} を上回ったとき（現在値から{(trig/px-1)*100:+.1f}%、ATR{(trig-px)/atr:.1f}本分）")
        yin = (sp - P["t"][200]).days / 365.25
        yout = (P["t"][-1] - sp).days / 365.25
        print(f"- 過去検証（コスト・スワップ込み）: 学習期間 {stats_line([x['Rn'] for x in bt if x['t'] < sp], yin)}"
              f" ／ 検証期間({sp:%Y}年〜) {stats_line([x['Rn'] for x in bt if x['t'] >= sp], yout)}")
        if bt:
            ex = "、".join(f"{x['t']:%Y/%m/%d}買い→{x['exit_t']:%m/%d}{x['why']} {x['Rn']:+.1f}R" for x in bt[-3:])
            print(f"- 直近の過去事例: {ex}")
        print()

    closed = sorted([x for x in live_all if not x["open"]], key=lambda x: x["exit_t"])
    print(f"## 運用成績（{live:%Y-%m-%d}以降、ルール通りに売買した場合・コスト・スワップ込み）")
    if closed:
        R = np.array([x["Rn"] for x in closed])
        print(f"- 決済{len(R)}回 勝ち{(R > 0).sum()}回 合計{R.sum():+.2f}R（1R＝{risk_jpy:,}円なら{R.sum()*risk_jpy:+,.0f}円）")
        for x in closed[-10:]:
            print(f"  - {x['asset']} {x['t']:%m/%d}買い → {x['exit_t']:%m/%d} {x['why']} {x['Rn']:+.2f}R")
    else:
        print("- 決済済みの取引はまだありません")
    print(f"- 保有中 {sum(1 for x in live_all if x['open'])}件")

    srt = sorted(bt_all, key=lambda x: x["exit_t"])
    R = pd.Series([x["Rn"] for x in srt])
    eq = R.cumsum()
    since = pd.Timestamp("2018-03-01")   # 3銘柄がそろって検証できる期間
    common = [x for x in srt if x["t"] >= since]
    yrs = (pd.Timestamp.now() - since).days / 365.25
    print("\n## 参考: 過去検証（3銘柄合計・コスト・スワップ込み）")
    print(f"- {len(R)}回 勝率{(R > 0).mean()*100:.0f}% 平均{R.mean():+.2f}R ／ 最大{S.summarize(R)['maxl']}連敗"
          f" ／ 最大の落ち込み {(eq - eq.cummax()).min():.1f}R")
    print(f"- 3銘柄がそろう{since:%Y}年以降: 年{len(common)/yrs:.1f}回、年{sum(x['Rn'] for x in common)/yrs:+.1f}R")
    print(f"- {cfg['unseen_validation']}")


if __name__ == "__main__":
    main()
