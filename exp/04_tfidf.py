"""基线A：TF-IDF + 逻辑回归。

划分策略：官方 train/test 保持不变；train 内部按 title 分组切出 dev（避免同一访谈泄漏），
仅在 dev 上选超参，最后用 train+dev 重训并在 test 上报告一次。
"""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from split_util import group_split_by_ratio
from common import bootstrap

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SEED, OUT = bootstrap()
NAMES = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]
SHORT = ["CR", "AMB", "CNR"]


def make_text(df, mode):
    if mode == "qa":
        return (df["text_a"] + " || " + df["text_b"]).tolist()
    return df["text_b"].tolist()


def evaluate(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_p": round(float(p.mean()), 4),
        "macro_r": round(float(r.mean()), 4),
        "macro_f1": round(float(f.mean()), 4),
        "weighted_f1": round(float(np.average(f, weights=s)), 4),
        "per_class": {NAMES[i]: {"precision": round(float(p[i]), 4),
                                 "recall": round(float(r[i]), 4),
                                 "f1": round(float(f[i]), 4),
                                 "support": int(s[i])} for i in range(3)},
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


tr = pd.read_parquet("processed/qev_train.parquet")
te = pd.read_parquet("processed/qev_test.parquet")
y_tr, y_te = tr["label"].values, te["label"].values

fit_df, dev_df, dev_titles = group_split_by_ratio(tr, "group_title", 0.2, SEED)
print(f"train={len(fit_df)}  dev={len(dev_df)}  test={len(te)}")
print(f"dev 覆盖 {len(dev_titles)} 个访谈标题，占 {len(dev_df)/len(tr)*100:.1f}% 样本")

# ---------------- 多数类基线 ----------------
maj = int(pd.Series(y_tr).mode()[0])
maj_pred = np.full(len(y_te), maj)
majority = evaluate(y_te, maj_pred)
print("\n[多数类基线] 全部判为", NAMES[maj], majority)

# ---------------- 超参搜索（仅 dev） ----------------
results = []
t0 = time.time()
for mode in ("qa", "a_only"):
    for ngram in ((1, 1), (1, 2)):
        for cw in (None, "balanced"):
            for C in (1.0, 4.0):
                vec = TfidfVectorizer(ngram_range=ngram, min_df=2,
                                      sublinear_tf=True, lowercase=True,
                                      strip_accents="unicode")
                Xtr = vec.fit_transform(make_text(fit_df, mode))
                Xdv = vec.transform(make_text(dev_df, mode))
                clf = LogisticRegression(C=C, class_weight=cw, max_iter=2000,
                                         random_state=SEED)
                clf.fit(Xtr, fit_df["label"].values)
                dv_pred = clf.predict(Xdv)
                m = evaluate(dev_df["label"].values, dv_pred)
                results.append({
                    "text": mode, "ngram": str(ngram),
                    "class_weight": str(cw), "C": C,
                    "dim": int(Xtr.shape[1]),
                    "dev_acc": m["accuracy"], "dev_macro_f1": m["macro_f1"],
                })
                print(f"  {mode:7s} ng={str(ngram):6s} cw={str(cw):8s} C={C:<4} "
                      f"dim={Xtr.shape[1]:<7d} dev F1={m['macro_f1']:.4f} "
                      f"Acc={m['accuracy']:.4f}")

print(f"\n搜索耗时 {time.time()-t0:.1f}s")

best = max(results, key=lambda r: r["dev_macro_f1"])
print("最佳配置:", best)

# ---------------- 用 train+dev 重训，在 test 上评估 ----------------
t0 = time.time()
vec = TfidfVectorizer(ngram_range=eval(best["ngram"]) if best["ngram"].startswith("(")
                      else tuple(int(x) for x in best["ngram"].strip("()").split(",")),
                      min_df=2, sublinear_tf=True, lowercase=True,
                      strip_accents="unicode")
Xtr = vec.fit_transform(make_text(tr, best["text"]))
Xte = vec.transform(make_text(te, best["text"]))
clf = LogisticRegression(C=best["C"],
                         class_weight=None if best["class_weight"] == "None"
                         else best["class_weight"],
                         max_iter=2000, random_state=SEED)
clf.fit(Xtr, y_tr)
te_pred = clf.predict(Xte)
train_time = time.time() - t0
tfidf_metrics = evaluate(y_te, te_pred)
tfidf_metrics.update({"model": "TF-IDF + LR", "config": best,
                      "feature_dim": int(Xtr.shape[1]),
                      "fit_seconds": round(train_time, 2),
                      "n_train": int(len(tr)), "n_test": int(len(te))})
print("\n[TF-IDF + LR] test:", json.dumps(
    {k: tfidf_metrics[k] for k in ("accuracy", "macro_p", "macro_r", "macro_f1",
                                   "weighted_f1", "feature_dim")},
    ensure_ascii=False))

# ---------------- 保存预测 ----------------
pred_df = pd.DataFrame({
    "uid": te["uid"], "text_a": te["text_a"], "text_b": te["text_b"],
    "gold_label": y_te, "gold_name": [NAMES[i] for i in y_te],
    "pred_tfidf": te_pred, "pred_tfidf_name": [NAMES[i] for i in te_pred],
})
pred_df.to_csv(f"{OUT}/pred_tfidf.csv", index=False, encoding="utf-8-sig")

# ---------------- 混淆矩阵（图） ----------------
fig, ax = plt.subplots(figsize=(5.2, 4.4))
cm = np.array(tfidf_metrics["confusion"])
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(SHORT); ax.set_yticklabels(SHORT)
for i in range(3):
    for j in range(3):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() * 0.55 else "#333333",
                fontsize=12)
ax.set_xlabel("预测标签"); ax.set_ylabel("真实标签")
ax.set_title(f"TF-IDF + LR 混淆矩阵（test n={len(te)}）", fontsize=11)
plt.colorbar(im, ax=ax, fraction=0.045)
plt.tight_layout()
plt.savefig(f"{OUT}/figures/cm_tfidf.png", bbox_inches="tight", dpi=200)
plt.close()

# ---------------- 特征可解释性：各类别最具判别力的词 ----------------
feat = np.array(vec.get_feature_names_out())
interp = {}
for ci in range(3):
    top = np.argsort(clf.coef_[ci])[::-1][:20]
    interp[NAMES[ci]] = feat[top].tolist()

# ---------------- 固定消融：在 test 上展示 ngram / class_weight / 输入方式的影响 ----------------
ablation = []
print("\n固定消融（train+dev 重训，test 评估；仅作对照，不用于选参）")
for mode in ("qa", "a_only"):
    for ngram in ((1, 1), (1, 2)):
        for cw in (None, "balanced"):
            v = TfidfVectorizer(ngram_range=ngram, min_df=2, sublinear_tf=True,
                                lowercase=True, strip_accents="unicode")
            Xa = v.fit_transform(make_text(tr, mode))
            Xb = v.transform(make_text(te, mode))
            c = LogisticRegression(C=1.0, class_weight=cw, max_iter=2000,
                                   random_state=SEED)
            c.fit(Xa, y_tr)
            m = evaluate(y_te, c.predict(Xb))
            row = {"text": mode, "ngram": str(ngram),
                   "class_weight": "None" if cw is None else "balanced",
                   "dim": int(Xa.shape[1]),
                   "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                   "macro_r": m["macro_r"],
                   "cr_f1": m["per_class"]["Clear Reply"]["f1"],
                   "cnr_f1": m["per_class"]["Clear Non-Reply"]["f1"]}
            ablation.append(row)
            print(f"  {mode:7s} ng={str(ngram):6s} cw={row['class_weight']:8s} "
                  f"Acc={m['accuracy']:.4f} F1={m['macro_f1']:.4f} "
                  f"CR_F1={row['cr_f1']:.4f} CNR_F1={row['cnr_f1']:.4f}")

with open(f"{OUT}/tfidf_metrics.json", "w", encoding="utf-8") as f:
    json.dump({"majority_baseline": majority, "tfidf": tfidf_metrics,
               "search_log": results, "top_features": interp,
               "ablation_test": ablation}, f, ensure_ascii=False, indent=2)

print("\n各粗类前 12 个正权重词：")
for k, v in interp.items():
    print(f"  {k:16s} {', '.join(v[:12])}")

print(f"\n已保存 {OUT}/tfidf_metrics.json, {OUT}/pred_tfidf.csv, 混淆矩阵图")
print(f"总耗时 {time.time()-t0:.1f}s")
