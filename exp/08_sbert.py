"""基线B-1：预训练句向量（all-MiniLM-L6-v2）+ 逻辑回归。

编码方式：question 与 answer 分别编码，取 last_hidden_state 的 attention-mask 加权均值，
再做 L2 归一化；句对特征按 [q, a, |q-a|, q*a] 拼接后送入逻辑回归。
"""
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from transformers import AutoModel, AutoTokenizer
from common import bootstrap

MODEL_DIR = "models/minilm"
NAMES = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]
SEED, OUT = bootstrap()
MAX_LEN = 256

torch.manual_seed(SEED)
torch.set_num_threads(16)
tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModel.from_pretrained(MODEL_DIR, local_files_only=True)
model.eval()
print(f"模型加载完成，隐层维度 = {model.config.hidden_size}")


def encode(texts, batch=48):
    vecs = []
    t0 = time.time()
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        enc = tok(chunk, padding=True, truncation=True, max_length=MAX_LEN,
                  return_tensors="pt")
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        vecs.append(emb.numpy())
        if (i // batch) % 20 == 0:
            print(f"  {i+len(chunk)}/{len(texts)}  {time.time()-t0:.0f}s")
    return np.vstack(vecs)


tr = pd.read_parquet("processed/qev_train.parquet")
te = pd.read_parquet("processed/qev_test.parquet")

t_start = time.time()
print("编码训练集 question ...")
Q_tr = encode(tr["text_a"].tolist())
print("编码训练集 answer ...")
A_tr = encode(tr["text_b"].tolist())
print("编码测试集 ...")
Q_te = encode(te["text_a"].tolist())
A_te = encode(te["text_b"].tolist())
enc_time = time.time() - t_start
print(f"编码完成，耗时 {enc_time/60:.1f} 分钟")


def feats(q, a):
    return np.hstack([q, a, np.abs(q - a), q * a])


X_tr, X_te = feats(Q_tr, A_tr), feats(Q_te, A_te)
y_tr, y_te = tr["label"].values, te["label"].values
print(f"特征维度 {X_tr.shape[1]}（4 × {Q_tr.shape[1]}）")

t0 = time.time()
clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000,
                         random_state=SEED)
clf.fit(X_tr, y_tr)
fit_time = time.time() - t0
p = clf.predict(X_te)

pr, rc, f, s = precision_recall_fscore_support(y_te, p, labels=[0, 1, 2],
                                               zero_division=0)
m = {
    "model": "MiniLM + LR",
    "accuracy": round(float(accuracy_score(y_te, p)), 4),
    "macro_p": round(float(pr.mean()), 4),
    "macro_r": round(float(rc.mean()), 4),
    "macro_f1": round(float(f.mean()), 4),
    "weighted_f1": round(float(np.average(f, weights=s)), 4),
    "per_class": {NAMES[i]: {"precision": round(float(pr[i]), 4),
                             "recall": round(float(rc[i]), 4),
                             "f1": round(float(f[i]), 4),
                             "support": int(s[i])} for i in range(3)},
    "confusion": confusion_matrix(y_te, p, labels=[0, 1, 2]).tolist(),
    "encode_seconds": round(enc_time, 1),
    "fit_seconds": round(fit_time, 2),
    "max_len": MAX_LEN,
    "feature_dim": int(X_tr.shape[1]),
}
print("\n[TEST]", json.dumps({k: m[k] for k in
      ("accuracy", "macro_p", "macro_r", "macro_f1", "weighted_f1")},
      ensure_ascii=False))

pd.DataFrame({
    "uid": te["uid"], "gold_label": y_te,
    "gold_name": [NAMES[i] for i in y_te],
    "pred_sbert": p, "pred_sbert_name": [NAMES[i] for i in p],
}).to_csv(f"{OUT}/pred_sbert.csv", index=False, encoding="utf-8-sig")

with open(f"{OUT}/sbert_metrics.json", "w", encoding="utf-8") as f2:
    json.dump(m, f2, ensure_ascii=False, indent=2)
print(f"已保存 {OUT}/sbert_metrics.json 与 pred_sbert.csv")
