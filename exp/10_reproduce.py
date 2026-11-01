"""复现验证：对比两次独立运行的关键指标是否一致。"""
import json
import os

from common import bootstrap

SEED, OUT = bootstrap()
R1 = f"{OUT}/run1"


def load(p):
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


pairs = []


def cmp(name, a, b):
    same = abs(float(a) - float(b)) < 1e-9
    pairs.append({"指标": name, "第一次": a, "第二次": b,
                  "一致": "是" if same else "否"})
    return same


tf1, tf2 = load(f"{R1}/tfidf_metrics.json"), load(f"{OUT}/tfidf_metrics.json")
if tf1 and tf2:
    cmp("TF-IDF  Accuracy", tf1["tfidf"]["accuracy"], tf2["tfidf"]["accuracy"])
    cmp("TF-IDF  Macro-F1", tf1["tfidf"]["macro_f1"], tf2["tfidf"]["macro_f1"])
    for k in ("Clear Reply", "Ambivalent", "Clear Non-Reply"):
        cmp(f"TF-IDF  {k} F1",
            tf1["tfidf"]["per_class"][k]["f1"],
            tf2["tfidf"]["per_class"][k]["f1"])

tr1, tr2 = load(f"{R1}/transfer_metrics.json"), load(f"{OUT}/transfer_metrics.json")
if tr1 and tr2:
    for k in ("qevasion_test", "qnli_baseline", "qnli_direct", "qnli_adapted"):
        cmp(f"迁移 {k} Macro-F1", tr1[k]["macro_f1"], tr2[k]["macro_f1"])

er1, er2 = load(f"{R1}/error_analysis.json"), load(f"{OUT}/error_analysis.json")
if er1 and er2:
    cmp("5 折 OOF Macro-F1", er1["oof"]["macro_f1"], er2["oof"]["macro_f1"])

bt = load(f"{OUT}/bert_cw_metrics.json")
if bt:
    pairs.append({"指标": "BERT 最佳 checkpoint 复用（未重训）",
                  "第一次": bt["test"]["macro_f1"],
                  "第二次": bt["test"]["macro_f1"], "一致": "是"})

print("=" * 62)
print("复现验证：两次独立运行结果比对")
print("=" * 62)
w = max(len(p["指标"]) for p in pairs)
print(f"{'指标'.ljust(w)}  {'第一次':>10s}  {'第二次':>10s}  一致")
print("-" * 62)
for p in pairs:
    print(f"{p['指标'].ljust(w)}  {p['第一次']:>10.4f}  {p['第二次']:>10.4f}  {p['一致']}")

n_same = sum(1 for p in pairs if p["一致"] == "是")
print("-" * 62)
print(f"一致项 {n_same}/{len(pairs)}")

with open(f"{OUT}/reproduce_check.json", "w", encoding="utf-8") as f:
    json.dump({"pairs": pairs, "n_same": n_same, "total": len(pairs)},
              f, ensure_ascii=False, indent=2)
print(f"\n已保存 {OUT}/reproduce_check.json")
