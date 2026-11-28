#!/usr/bin/env python3
"""検証の計画書を扱う道具。正本は project/research/plan.csv（1行1パターン）。plan.md の一覧はここで作る。

使い方（--plan daytrade をつけると、デイトレードの計画書 daytrade_plan.csv・daytrade_plan.md を扱う）:
  python3 project/tools/plan_tool.py md                  plan.md の一覧を plan.csv から作り直す
  python3 project/tools/plan_tool.py status              分類ごとの進み具合
  python3 project/tools/plan_tool.py next [件数]         次に検証する候補（優先「高」→「中」→「低」の「未」と「一部済」）
  python3 project/tools/plan_tool.py set ID 状態 [結果]   状態（済・一部済・未・データ待ち・見送り）と結果を書き、plan.md も作り直す
"""
import csv
import os
import sys
from datetime import date

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "research"))
CSV = os.path.join(ROOT, "plan.csv")
MD = os.path.join(ROOT, "plan.md")
MARK = "<!-- ここから下は project/tools/plan_tool.py が plan.csv から作る（手で直さない） -->"
FIELDS = ["ID", "分類", "名前", "ルール", "調べる設定", "足", "性格", "優先", "状態", "必要なデータ", "結果"]
STATES = ["済", "一部済", "未", "データ待ち", "見送り"]
ORDER = {"高": 0, "中": 1, "低": 2}
PLANS = {"main": ("plan.csv", "plan.md", FIELDS),
         "daytrade": ("daytrade_plan.csv", "daytrade_plan.md", FIELDS[:6] + ["実行"] + FIELDS[6:])}


def use(name):
    """扱う計画書を切り替える（main：plan.csv、daytrade：daytrade_plan.csv）。"""
    global CSV, MD, FIELDS
    c, m, f = PLANS[name]
    CSV, MD, FIELDS = os.path.join(ROOT, c), os.path.join(ROOT, m), list(f)


def load():
    with open(CSV, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save(rows):
    with open(CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def cell(s):
    return (s or "").replace("|", "／").replace("\n", " ").strip()


def counts(rows):
    c = {s: 0 for s in STATES}
    for r in rows:
        c[r["状態"]] = c.get(r["状態"], 0) + 1
    return c


def cats(rows):
    out = []
    for r in rows:
        if r["分類"] not in out:
            out.append(r["分類"])
    return out


def render(rows):
    head = open(MD, encoding="utf-8").read().split(MARK)[0] if os.path.exists(MD) else "# 検証の計画書\n\n"
    lines = [f"最終更新：{date.today().isoformat()}" if ln.startswith("最終更新：") else ln for ln in head.rstrip().splitlines()]
    c = counts(rows)
    out = ["\n".join(lines), "", MARK, "", f"## 8. 一覧（{len(rows)}件）", "",
           "進み具合：" + "・".join(f"{s}{c[s]}" for s in STATES), "",
           "| 分類 | 件数 | " + " | ".join(STATES) + " |", "|---|---|" + "---|" * len(STATES)]
    for k in cats(rows):
        cc = counts([r for r in rows if r["分類"] == k])
        out.append(f"| {k} | {sum(cc.values())} | " + " | ".join(str(cc[s]) for s in STATES) + " |")
    for k in cats(rows):
        out += ["", f"### {k}", "", "| ID | 名前 | ルールの要点（調べる設定） | 優先 | 状態・結果 |", "|---|---|---|---|---|"]
        for r in (x for x in rows if x["分類"] == k):
            how = "・".join(v for v in (r.get("足", ""), r.get("実行", "")) if v and v != "—")
            rule = (f"〔{how}〕" if how else "") + r["ルール"]
            if r["調べる設定"] not in ("", "—"):
                rule += f"（{r['調べる設定']}）"
            st = r["状態"]
            if r["状態"] == "データ待ち" and r["必要なデータ"]:
                st += f"：{r['必要なデータ']}"
            if r["結果"]:
                st += f"：{r['結果']}"
            out.append(f"| {r['ID']} | {cell(r['名前'])} | {cell(rule)} | {r['優先']} | {cell(st)} |")
    with open(MD, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def main(argv):
    if "--plan" in argv:
        i = argv.index("--plan")
        use(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    rows = load()
    cmd = argv[1] if len(argv) > 1 else "status"
    if cmd == "md":
        render(rows)
        print(f"{os.path.basename(MD)} を作り直した（{len(rows)}件）")
    elif cmd == "status":
        c = counts(rows)
        print(f"{len(rows)}件 ｜ " + " ".join(f"{s}{c[s]}" for s in STATES))
        for k in cats(rows):
            cc = counts([r for r in rows if r["分類"] == k])
            print(f"  {k:26s} " + " ".join(f"{s}{cc[s]:3d}" for s in STATES))
    elif cmd == "next":
        n = int(argv[2]) if len(argv) > 2 else 20
        todo = [r for r in rows if r["状態"] in ("未", "一部済") and r["優先"] in ORDER]
        todo.sort(key=lambda r: (ORDER[r["優先"]], r["ID"]))
        for r in todo[:n]:
            print(f"  {r['ID']} ［{r['優先']}・{r['状態']}］ {r['名前']}：{r['ルール']}")
    elif cmd == "set":
        if len(argv) < 4:
            sys.exit(__doc__)
        rid, st = argv[2], argv[3]
        if st not in STATES:
            sys.exit(f"状態は {'・'.join(STATES)} のどれか")
        hit = [r for r in rows if r["ID"] == rid]
        if not hit:
            sys.exit(f"{rid} は一覧にない")
        hit[0]["状態"] = st
        if len(argv) > 4:
            hit[0]["結果"] = argv[4]
        save(rows)
        render(rows)
        print(f"{rid} を「{st}」にした")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
