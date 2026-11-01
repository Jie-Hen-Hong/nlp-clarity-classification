"""跨场景迁移实验：QEvasion（3 类回避检测）→ QNLI（2 类答案蕴含）。

四组对照（对应报告表7）：
  1. 原方法基线      QEvasion 上训练并在 QEvasion 测试集评估
  2. 迁移数据基线    QNLI 上从零训练、QNLI 验证集评估（朴素配置：unigram、无权重）
  3. 直接迁移        QEvasion 的 3 类模型直接预测 QNLI，把 CR 折成 entailment
  4. 适配后训练      统一字段适配器 + 与 QEvasion 相同的调优配方在 QNLI 上重训
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from common import bootstrap
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

SEED, OUT = bootstrap()
NAMES3 = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]


def met(y, p, labels):
    pr, rc, f, s = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    return {"accuracy": round(float(accuracy_score(y, p)), 4),
            "macro_p": round(float(pr.mean()), 4),
            "macro_r": round(float(rc.mean()), 4),
            "macro_f1": round(float(f.mean()), 4),
            "n": int(len(y)),
            "confusion": confusion_matrix(y, p, labels=labels).tolist()}


def qa_text(a, b):
    return (a + " || " + b).tolist()


# ---------------- 数据 ----------------
qev_tr = pd.read_parquet("processed/qev_train.parquet")
qev_te = pd.read_parquet("processed/qev_test.parquet")

qnli_tr = pd.read_parquet("data/qnli_train.parquet")
qnli_va = pd.read_parquet("data/qnli_validation.parquet")

# 统一字段适配器：QNLI (question, sentence, label) → (text_a, text_b, label)
def adapt_qnli(df):
    return pd.DataFrame({
        "text_a": df["question"].astype("string").fillna(""),
        "text_b": df["sentence"].astype("string").fillna(""),
        "label": df["label"].astype(int),
    })

qn_tr = adapt_qnli(qnli_tr)
qn_va = adapt_qnli(qnli_va)
print(f"QEvasion train={len(qev_tr)} test={len(qev_te)}")
print(f"QNLI train={len(qn_tr)} valid={len(qn_va)}  "
      f"标签分布={qn_tr['label'].value_counts().to_dict()}")
print(f"QNLI 平均长度：句子 {qn_tr['text_b'].str.split().str.len().mean():.1f} 词, "
      f"问题 {qn_tr['text_a'].str.split().str.len().mean():.1f} 词")

rows = []
t0 = time.time()

# ---------------- 1. 原方法基线（QEvasion 上） ----------------
vec_src = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                          lowercase=True, strip_accents="unicode")
Xsrc = vec_src.fit_transform(qa_text(qev_tr["text_a"], qev_tr["text_b"]))
clf_src = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                             random_state=SEED)
clf_src.fit(Xsrc, qev_tr["label"].values)
p_src = clf_src.predict(vec_src.transform(qa_text(qev_te["text_a"], qev_te["text_b"])))
m1 = met(qev_te["label"].values, p_src, [0, 1, 2])
rows.append({"setting": "原方法基线（QEvasion）", "task": "QEvasion 3 类",
             **m1})
print(f"\n[1] 原方法基线  Acc={m1['accuracy']:.4f} Macro-F1={m1['macro_f1']:.4f}")

# ---------------- 2. 迁移数据基线（QNLI 上朴素配置） ----------------
vec_t = TfidfVectorizer(ngram_range=(1, 1), min_df=2, sublinear_tf=True,
                        lowercase=True, strip_accents="unicode")
Xt = vec_t.fit_transform(qa_text(qn_tr["text_a"], qn_tr["text_b"]))
clf_t = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
clf_t.fit(Xt, qn_tr["label"].values)
p_t = clf_t.predict(vec_t.transform(qa_text(qn_va["text_a"], qn_va["text_b"])))
m2 = met(qn_va["label"].values, p_t, [0, 1])
rows.append({"setting": "迁移数据基线（QNLI 朴素配置）", "task": "QNLI 2 类", **m2})
print(f"[2] 迁移数据基线  Acc={m2['accuracy']:.4f} Macro-F1={m2['macro_f1']:.4f}")

# ---------------- 3. 直接迁移（源模型零样本迁到 QNLI） ----------------
Xq = vec_src.transform(qa_text(qn_va["text_a"], qn_va["text_b"]))
prob = clf_src.predict_proba(Xq)
# 折叠规则：P(Clear Reply) 视为 entailment 概率；其余两类合并视为 not_entailment
p_direct = np.where(prob[:, 0] >= (1 - prob[:, 0]), 0, 1)
# 等价于 argmax 落在 CR 即 0，否则 1
p_direct = np.where(prob.argmax(1) == 0, 0, 1)
m3 = met(qn_va["label"].values, p_direct, [0, 1])
rows.append({"setting": "直接迁移（源模型零样本）", "task": "QNLI 2 类", **m3})
print(f"[3] 直接迁移      Acc={m3['accuracy']:.4f} Macro-F1={m3['macro_f1']:.4f}")
print(f"    预测为正类比例={float((p_direct==0).mean()):.3f} "
      f"(真实={float((qn_va['label'].values==0).mean()):.3f})")

# ---------------- 4. 适配后训练（统一接口 + 调优配方） ----------------
vec_a = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                        lowercase=True, strip_accents="unicode")
Xa = vec_a.fit_transform(qa_text(qn_tr["text_a"], qn_tr["text_b"]))
clf_a = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                           random_state=SEED)
clf_a.fit(Xa, qn_tr["label"].values)
p_a = clf_a.predict(vec_a.transform(qa_text(qn_va["text_a"], qn_va["text_b"])))
m4 = met(qn_va["label"].values, p_a, [0, 1])
rows.append({"setting": "适配后训练（统一接口+调优）", "task": "QNLI 2 类", **m4})
print(f"[4] 适配后训练    Acc={m4['accuracy']:.4f} Macro-F1={m4['macro_f1']:.4f}")

# ---------------- 5. 跨域词汇覆盖率诊断 ----------------
src_vocab = set(vec_src.vocabulary_)
qn_tokens = set()
for a, b in zip(qn_va["text_a"].head(2000), qn_va["text_b"].head(2000)):
    qn_tokens.update((a + " " + b).lower().split())
cover = len(qn_tokens & src_vocab) / max(1, len(qn_tokens))
print(f"\n[诊断] QNLI 词汇中被 QEvasion 词表覆盖的比例：{cover:.3f}")
print(f"[诊断] QEvasion 词表大小={len(src_vocab)}，QNLI 适配后词表={len(vec_a.vocabulary_)}")

# ---------------- 6. 迁移后的类别级结果 ----------------
detail = {
    "qevasion_test": m1, "qnli_baseline": m2,
    "qnli_direct": m3, "qnli_adapted": m4,
    "diagnostics": {
        "src_vocab": len(src_vocab),
        "qnli_vocab_adapted": len(vec_a.vocabulary_),
        "cross_domain_token_coverage": round(cover, 4),
        "qnli_positive_rate_true": round(float((qn_va["label"].values == 0).mean()), 4),
        "qnli_positive_rate_direct": round(float((p_direct == 0).mean()), 4),
    },
    "seconds": round(time.time() - t0, 1),
}

pd.DataFrame(rows)[["setting", "task", "accuracy", "macro_p", "macro_r",
                    "macro_f1", "n"]].to_csv(
    f"{OUT}/transfer_table.csv", index=False, encoding="utf-8-sig")

qd = pd.DataFrame({
    "question": qn_va["text_a"], "sentence": qn_va["text_b"],
    "gold": qn_va["label"], "pred_direct": p_direct, "pred_adapted": p_a,
})
qd.to_csv(f"{OUT}/pred_qnli.csv", index=False, encoding="utf-8-sig")

with open(f"{OUT}/transfer_metrics.json", "w", encoding="utf-8") as f:
    json.dump(detail, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 70)
print(f"{'设置':32s}{'Acc':>8s}{'Macro-P':>9s}{'Macro-R':>9s}{'Macro-F1':>10s}")
print("=" * 70)
for r in rows:
    print(f"{r['setting']:32s}{r['accuracy']:>8.4f}{r['macro_p']:>9.4f}"
          f"{r['macro_r']:>9.4f}{r['macro_f1']:>10.4f}")
print(f"\n总耗时 {time.time()-t0:.1f}s；已保存 {OUT}/transfer_table.csv 等")
