"""汇总各模型指标，生成报告表3/表4/表7 与模型对比图。"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import bootstrap

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SEED, OUT = bootstrap()
NAMES = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


tf = load(f"{OUT}/tfidf_metrics.json")
sb = load(f"{OUT}/sbert_metrics.json")
bcw = load(f"{OUT}/bert_cw_metrics.json")
bdir = load(f"{OUT}/bert_direct_metrics.json")
tr = load(f"{OUT}/transfer_metrics.json")

rows = []
if tf:
    m = tf["majority_baseline"]
    rows.append({"模型": "多数类基线（全判 Ambivalent）", "Accuracy": m["accuracy"],
                 "Macro-P": m["macro_p"], "Macro-R": m["macro_r"],
                 "Macro-F1": m["macro_f1"], "备注": "不训练"})
    m = tf["tfidf"]
    _c = m.get("config", {})
    _ng = "".join(str(x) for x in _c.get("ngram", ()))[:0] or str(_c.get("ngram", ""))
    _ng = _ng.strip("()").replace(", ", "-").replace("1-1", "unigram").replace("1-2", "uni+bi")
    _cw = "balanced" if str(_c.get("class_weight")) not in ("None", "none") else "无权重"
    rows.append({"模型": "TF-IDF + 逻辑回归", "Accuracy": m["accuracy"],
                 "Macro-P": m["macro_p"], "Macro-R": m["macro_r"],
                 "Macro-F1": m["macro_f1"],
                 "备注": f"{_ng}, {_cw}, C={_c.get('C', '-')}"})
if sb:
    rows.append({"模型": "MiniLM 句向量 + 逻辑回归", "Accuracy": sb["accuracy"],
                 "Macro-P": sb["macro_p"], "Macro-R": sb["macro_r"],
                 "Macro-F1": sb["macro_f1"],
                 "备注": "冻结特征，[q,a,|q-a|,q*a]"})
if bdir:
    m = bdir["test"]
    rows.append({"模型": "BERT-base 微调（标准交叉熵）", "Accuracy": m["accuracy"],
                 "Macro-P": m["macro_p"], "Macro-R": m["macro_r"],
                 "Macro-F1": m["macro_f1"], "备注": "无类别权重"})
if bcw:
    m = bcw["test"]
    rows.append({"模型": "BERT-base 微调（加权交叉熵）", "Accuracy": m["accuracy"],
                 "Macro-P": m["macro_p"], "Macro-R": m["macro_r"],
                 "Macro-F1": m["macro_f1"],
                 "备注": f"class_weight, ep{m['best_epoch']}"})
if tr:
    m = tr["qnli_adapted"]
    rows.append({"模型": "迁移数据模型（QNLI 适配后）", "Accuracy": m["accuracy"],
                 "Macro-P": m["macro_p"], "Macro-R": m["macro_r"],
                 "Macro-F1": m["macro_f1"], "备注": "二分类任务，不可直接比较"})

t3 = pd.DataFrame(rows)
print("=" * 78)
print("表3 模型总体性能（QEvasion 测试集 n=308）")
print("=" * 78)
print(t3.to_string(index=False))
t3.to_csv(f"{OUT}/table3_overall.csv", index=False, encoding="utf-8-sig")

# ---------- 表4 最佳模型类别级结果 ----------
best = None
for name, obj in (("BERT-base 微调（加权交叉熵）", bcw), ("TF-IDF + 逻辑回归", tf)):
    if obj is None:
        continue
    m = obj["test"] if "test" in obj else obj["tfidf"]
    if best is None or m["macro_f1"] > best[1]["macro_f1"]:
        best = (name, m)
if best:
    print("\n" + "=" * 78)
    print(f"表4 最佳模型类别级结果：{best[0]}")
    print("=" * 78)
    t4 = []
    for k in NAMES:
        c = best[1]["per_class"][k]
        t4.append({"类别": k, "样本数": c["support"],
                   "Precision": c["precision"], "Recall": c["recall"],
                   "F1": c["f1"]})
    t4.append({"类别": "Macro Average", "样本数": sum(x["样本数"] for x in t4),
               "Precision": best[1]["macro_p"], "Recall": best[1]["macro_r"],
               "F1": best[1]["macro_f1"]})
    d4 = pd.DataFrame(t4)
    print(d4.to_string(index=False))
    d4.to_csv(f"{OUT}/table4_per_class.csv", index=False, encoding="utf-8-sig")

# ---------- 图5 模型对比 ----------
plot_rows = [r for r in rows if "迁移数据" not in r["模型"]]
fig, ax = plt.subplots(figsize=(9.6, 4.0))
x = np.arange(len(plot_rows))
w = 0.36
ax.bar(x - w / 2, [r["Accuracy"] for r in plot_rows], w, label="Accuracy",
       color="#B8C9DC", edgecolor="white")
ax.bar(x + w / 2, [r["Macro-F1"] for r in plot_rows], w, label="Macro-F1",
       color="#2E6FAD", edgecolor="white")
for i, r in enumerate(plot_rows):
    ax.text(i - w / 2, r["Accuracy"] + 0.012, f"{r['Accuracy']:.3f}",
            ha="center", fontsize=7.6)
    ax.text(i + w / 2, r["Macro-F1"] + 0.012, f"{r['Macro-F1']:.3f}",
            ha="center", fontsize=7.6)
ax.set_xticks(x)
ax.set_xticklabels([r["模型"].replace("（", "\n（") for r in plot_rows], fontsize=7.8)
ax.set_ylabel("分数")
ax.set_ylim(0, 0.85)
ax.set_title("各模型在 QEvasion 测试集上的 Accuracy 与 Macro-F1（n=308）", fontsize=11)
ax.legend(fontsize=9, frameon=False)
ax.grid(axis="y", alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/figures/fig5_model_compare.png", bbox_inches="tight", dpi=200)
plt.close()

# ---------- 表7 迁移 ----------
if tr:
    print("\n" + "=" * 78)
    print("表7 原任务与迁移任务对比")
    print("=" * 78)
    t7 = [
        {"实验设置": "原方法基线（QEvasion 3 类）", "任务": "QEvasion",
         "Accuracy": tr["qevasion_test"]["accuracy"],
         "Macro-F1": tr["qevasion_test"]["macro_f1"]},
        {"实验设置": "迁移数据基线（QNLI 朴素配置）", "任务": "QNLI",
         "Accuracy": tr["qnli_baseline"]["accuracy"],
         "Macro-F1": tr["qnli_baseline"]["macro_f1"]},
        {"实验设置": "直接迁移（源模型零样本）", "任务": "QNLI",
         "Accuracy": tr["qnli_direct"]["accuracy"],
         "Macro-F1": tr["qnli_direct"]["macro_f1"]},
        {"实验设置": "适配后训练（统一接口+调优）", "任务": "QNLI",
         "Accuracy": tr["qnli_adapted"]["accuracy"],
         "Macro-F1": tr["qnli_adapted"]["macro_f1"]},
    ]
    d7 = pd.DataFrame(t7)
    print(d7.to_string(index=False))
    d7.to_csv(f"{OUT}/table7_transfer.csv", index=False, encoding="utf-8-sig")
    print("\n诊断：", json.dumps(tr["diagnostics"], ensure_ascii=False))

print("\n已生成 table3/table4/table7 CSV 与 fig5")
