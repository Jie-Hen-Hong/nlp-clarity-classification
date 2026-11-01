"""错误分析：分组交叉验证、长度分桶、典型误判样本。

说明：官方测试集缺少 president 字段（308 条全为空），因此“按总统分组”的分析
改在训练集上以按 title 分组的 5 折交叉验证（out-of-fold 预测）完成。
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import GroupKFold
from common import bootstrap

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SEED, OUT = bootstrap()
NAMES = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]
SHORT = ["CR", "AMB", "CNR"]

tr = pd.read_parquet("processed/qev_train.parquet")
te = pd.read_parquet("processed/qev_test.parquet")


def qa_text(df):
    return (df["text_a"] + " || " + df["text_b"]).tolist()


def met(y, p, labels=(0, 1, 2)):
    pr, rc, f, s = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    return {"accuracy": round(float(accuracy_score(y, p)), 4),
            "macro_f1": round(float(f.mean()), 4),
            "per_class_f1": {NAMES[i]: round(float(f[i]), 4) for i in range(3)},
            "n": int(len(y))}


# ---------------- 5 折分组交叉验证（按访谈标题分组，避免同一访谈泄漏） ----------------
gkf = GroupKFold(n_splits=5)
oof = np.full(len(tr), -1)
fold_rows = []
for k, (i_tr, i_va) in enumerate(gkf.split(tr, tr["label"], groups=tr["group_title"]), 1):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                          lowercase=True, strip_accents="unicode")
    Xa = vec.fit_transform(qa_text(tr.iloc[i_tr]))
    Xb = vec.transform(qa_text(tr.iloc[i_va]))
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                             random_state=SEED)
    clf.fit(Xa, tr.iloc[i_tr]["label"].values)
    pred = clf.predict(Xb)
    oof[i_va] = pred
    m = met(tr.iloc[i_va]["label"].values, pred)
    m["fold"] = k
    fold_rows.append(m)
    print(f"fold{k}: n={m['n']:4d} Acc={m['accuracy']:.4f} F1={m['macro_f1']:.4f}")

oof_m = met(tr["label"].values, oof)
print(f"\nOOF 汇总: Acc={oof_m['accuracy']:.4f} Macro-F1={oof_m['macro_f1']:.4f}")
print(f"各折 Macro-F1: {[r['macro_f1'] for r in fold_rows]}")
print(f"均值={np.mean([r['macro_f1'] for r in fold_rows]):.4f} "
      f"标准差={np.std([r['macro_f1'] for r in fold_rows]):.4f}")

tr_oof = tr.copy()
tr_oof["pred"] = oof

# ---------------- 按总统分组 ----------------
print("\n" + "=" * 70)
print("按总统分组（OOF 预测）")
pres_rows = []
for p, g in tr_oof.groupby("group_president"):
    m = met(g["label"].values, g["pred"].values)
    m["president"] = p.split()[-1] if p else "(空)"
    pres_rows.append(m)
    print(f"  {m['president']:10s} n={m['n']:4d} Acc={m['accuracy']:.4f} "
          f"F1={m['macro_f1']:.4f}  CNR_F1={m['per_class_f1']['Clear Non-Reply']:.4f}")

# ---------------- 按回答长度分桶 ----------------
bins = [0, 100, 300, 10 ** 6]
labels_b = ["短 (<100 词)", "中 (100-300)", "长 (>300)"]
tr_oof["len_bucket"] = pd.cut(tr_oof["a_words"], bins=bins, labels=labels_b)
print("\n按回答长度分桶（OOF 预测）")
len_rows = []
for b, g in tr_oof.groupby("len_bucket", observed=True):
    m = met(g["label"].values, g["pred"].values)
    m["bucket"] = str(b)
    m["avg_words"] = round(float(g["a_words"].mean()), 1)
    m["cnr_share"] = round(float((g["label"] == 2).mean()), 4)
    len_rows.append(m)
    print(f"  {str(b):14s} n={m['n']:4d} 平均{m['avg_words']:6.0f}词 "
          f"Acc={m['accuracy']:.4f} F1={m['macro_f1']:.4f} "
          f"CNR占比={m['cnr_share']:.3f}")

# ---------------- 图：分组表现 ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
ax = axes[0]
x = np.arange(len(pres_rows))
ax.bar(x - 0.2, [r["macro_f1"] for r in pres_rows], 0.38,
       label="Macro-F1", color="#2E6FAD")
ax.bar(x + 0.2, [r["per_class_f1"]["Clear Non-Reply"] for r in pres_rows], 0.38,
       label="Clear Non-Reply F1", color="#D98A3D")
ax.set_xticks(x)
ax.set_xticklabels([r["president"] for r in pres_rows], fontsize=9)
ax.set_ylabel("分数"); ax.set_ylim(0, 0.7)
ax.set_title("(a) 按总统分组的表现（OOF）", fontsize=11)
ax.legend(fontsize=8, frameon=False)
ax.grid(axis="y", alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
x = np.arange(len(len_rows))
ax.bar(x - 0.2, [r["macro_f1"] for r in len_rows], 0.38,
       label="Macro-F1", color="#2E6FAD")
ax.bar(x + 0.2, [r["accuracy"] for r in len_rows], 0.38,
       label="Accuracy", color="#9DB7D4")
ax.set_xticks(x)
ax.set_xticklabels([r["bucket"] for r in len_rows], fontsize=8.5)
ax.set_ylabel("分数"); ax.set_ylim(0, 0.8)
ax.set_title("(b) 按回答长度分桶的表现（OOF）", fontsize=11)
ax.legend(fontsize=8, frameon=False)
ax.grid(axis="y", alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/figures/fig4_group_perf.png", bbox_inches="tight", dpi=200)
plt.close()

# ---------------- 典型误判样本（测试集） ----------------
pf = pd.read_csv(f"{OUT}/pred_tfidf.csv")
wrong = pf[pf["gold_label"] != pf["pred_tfidf"]].copy()
wrong["a_words"] = wrong["text_b"].str.split().str.len()
# 同一回答可能对应多个问题，先去重，避免选出重复样本
wrong["ans_key"] = wrong["text_b"].astype(str).str[:200]
wrong = wrong.drop_duplicates(subset=["ans_key", "gold_label", "pred_tfidf"])

picks = []
used_keys, used_ans = set(), set()


def take(cond, typ, k=1):
    cand = wrong[cond].copy()
    cand = cand[~cand["ans_key"].isin(used_ans)]
    cand = cand.sort_values("a_words", ascending=False)
    for _, row in cand.head(k).iterrows():
        picks.append({"idx": row.name, "type": typ})
        used_ans.add(row["ans_key"])


take((wrong["gold_label"] == 1) & (wrong["pred_tfidf"] == 0),
     "隐式回答被读成明确回答（长而空泛）", 2)
take((wrong["gold_label"] == 0) & (wrong["pred_tfidf"] == 1),
     "明确回答被读成模糊回答（部分/半回答）", 2)
take((wrong["gold_label"] == 2) & (wrong["pred_tfidf"] == 1),
     "明确拒答被读成模糊回答（主题转移）", 1)
take((wrong["gold_label"] == 1) & (wrong["pred_tfidf"] == 2),
     "模糊回答被读成拒答（边界模糊）", 1)
take((wrong["gold_label"] == 0) & (wrong["pred_tfidf"] == 2),
     "明确回答被读成拒答（表述迂回）", 1)

samples = []
for p in picks:
    r = pf.loc[p["idx"]]
    samples.append({
        "q": str(r["text_a"])[:120],
        "a": str(r["text_b"])[:200],
        "gold": r["gold_name"], "pred": r["pred_tfidf_name"],
        "a_words": int(len(str(r["text_b"]).split())),
        "type": p["type"],
    })

print("\n" + "=" * 70)
print(f"典型误判样本（共 {len(samples)} 条）")
for i, s in enumerate(samples, 1):
    print(f"\n[{i}] {s['type']}  ({s['a_words']} 词)")
    print(f"  问：{s['q'][:90]}")
    print(f"  答：{s['a'][:150]}...")
    print(f"  真实={s['gold']}  预测={s['pred']}")

# ---------------- 长度与正确率的关系（图5） ----------------
pf2 = pf.copy()
pf2["a_words"] = pf2["text_b"].str.split().str.len()
pf2["correct"] = (pf2["gold_label"] == pf2["pred_tfidf"]).astype(int)
pf2["bucket"] = pd.cut(pf2["a_words"], bins=bins, labels=labels_b)
rate = pf2.groupby("bucket", observed=True)["correct"].mean()
print("\n测试集各长度桶准确率：")
print(rate.round(4).to_string())

out = {
    "oof": oof_m, "folds": fold_rows,
    "fold_mean_f1": round(float(np.mean([r["macro_f1"] for r in fold_rows])), 4),
    "fold_std_f1": round(float(np.std([r["macro_f1"] for r in fold_rows])), 4),
    "by_president": pres_rows, "by_length": len_rows,
    "test_length_accuracy": {str(k): round(float(v), 4) for k, v in rate.items()},
    "error_samples": samples,
    "note": "测试集 president 字段为空，分组分析基于训练集 OOF",
}
with open(f"{OUT}/error_analysis.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
tr_oof[["uid", "group_president", "a_words", "label", "pred"]].to_csv(
    f"{OUT}/oof_pred.csv", index=False, encoding="utf-8-sig")
print(f"\n已保存 {OUT}/error_analysis.json 与 fig4")
