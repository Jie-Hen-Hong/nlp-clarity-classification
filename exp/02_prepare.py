"""数据清洗、标签映射与统一接口构造。

输出：
  processed/qev_train.parquet / qev_test.parquet
  outputs/label_map.json
  outputs/data_stats.json   （对应报告表1）
"""
import json
import os
import pandas as pd
from common import bootstrap

SEED, OUT = bootstrap()

RAW_TRAIN = "data/qev_train.parquet"
RAW_TEST = "data/qev_test.parquet"
OUT_DIR = "processed"
os.makedirs(OUT_DIR, exist_ok=True)

CLARITY2ID = {"Clear Reply": 0, "Ambivalent": 1, "Clear Non-Reply": 2}
ID2CLARITY = {v: k for k, v in CLARITY2ID.items()}

# 依据论文 Figure 3：Explicit 属 Clear Reply；5 个回避细类属 Ambivalent；3 个属 Clear Non-Reply
EVASION2CLARITY = {
    "Explicit": "Clear Reply",
    "Implicit": "Ambivalent",
    "Dodging": "Ambivalent",
    "General": "Ambivalent",
    "Deflection": "Ambivalent",
    "Partial/half-answer": "Ambivalent",
    "Declining to answer": "Clear Non-Reply",
    "Claims ignorance": "Clear Non-Reply",
    "Clarification": "Clear Non-Reply",
}


def load(path):
    df = pd.read_parquet(path)
    for c in ("question", "interview_answer", "clarity_label", "evasion_label",
              "president", "title"):
        df[c] = df[c].astype("string").fillna("")
    return df


def report_missing(df, name):
    print(f"\n[{name}] 原始 {len(df)} 行")
    for c in ("question", "interview_answer", "clarity_label"):
        empty = int((df[c].str.strip() == "").sum())
        print(f"  {c}: 空值 {empty}  ({empty/len(df)*100:.2f}%)")
    print(f"  inaudible=True: {int(df['inaudible'].sum())}")
    print(f"  完全重复行: {int(df.duplicated().sum())}")
    dup_qa = int(df.duplicated(subset=["question", "interview_answer"]).sum())
    print(f"  question+answer 重复: {dup_qa}")
    print(f"  不同 interview_answer: {df['interview_answer'].nunique()}")
    print(f"  不同 title: {df['title'].nunique()}")


train = load(RAW_TRAIN)
test = load(RAW_TEST)
report_missing(train, "train")
report_missing(test, "test")

print("\n" + "=" * 70)
print("细类 → 粗类 映射验证（依据论文 Figure 3）")
print("=" * 70)
ct = pd.crosstab(train["evasion_label"], train["clarity_label"])
print(ct.to_string())

mapped = train["evasion_label"].map(EVASION2CLARITY)
agree = (mapped == train["clarity_label"]).sum()
print(f"\n映射与数据自带 clarity_label 一致: {agree}/{len(train)} "
      f"({agree/len(train)*100:.2f}%)")
bad = train[mapped != train["clarity_label"]]
if len(bad):
    print("不一致样本的标签组合：")
    print(bad.groupby(["evasion_label", "clarity_label"]).size().to_string())
else:
    print("→ 完全一致，说明该映射就是数据集的实际标注口径。")

# ---------- 统一接口 ----------
def build(df, split):
    out = pd.DataFrame({
        "uid": [f"{split}_{i}" for i in range(len(df))],
        "text_a": df["question"].str.strip(),
        "text_b": df["interview_answer"].str.strip(),
        "clarity_label": df["clarity_label"],
        "label": df["clarity_label"].map(CLARITY2ID),
        "evasion_label": df["evasion_label"],
        "group_president": df["president"],
        "group_title": df["title"],
        "q_words": df["question"].str.split().str.len(),
        "a_words": df["interview_answer"].str.split().str.len(),
        "a_chars": df["interview_answer"].str.len(),
    })
    return out


tr = build(train, "train")
te = build(test, "test")

before = len(tr)
tr = tr[(tr["text_a"] != "") & (tr["text_b"] != "")].reset_index(drop=True)
print(f"\ntrain 丢弃空文本: {before - len(tr)} 条")

tr["label"] = tr["label"].astype(int)
te["label"] = te["label"].astype(int)

tr.to_parquet(f"{OUT_DIR}/qev_train.parquet", index=False)
te.to_parquet(f"{OUT_DIR}/qev_test.parquet", index=False)

stats = {
    "train": {
        "n": len(tr),
        "missing": int(before - len(tr)),
        "avg_q_words": round(float(tr["q_words"].mean()), 1),
        "avg_a_words": round(float(tr["a_words"].mean()), 1),
        "avg_q_chars": round(float(tr["text_a"].str.len().mean()), 1),
        "avg_a_chars": round(float(tr["text_b"].str.len().mean()), 1),
        "label_dist": tr["clarity_label"].value_counts().to_dict(),
        "president_dist": tr["group_president"].value_counts().to_dict(),
        "n_titles": int(tr["group_title"].nunique()),
    },
    "test": {
        "n": len(te),
        "missing": int(len(test) - len(te)),
        "avg_q_words": round(float(te["q_words"].mean()), 1),
        "avg_a_words": round(float(te["a_words"].mean()), 1),
        "avg_q_chars": round(float(te["text_a"].str.len().mean()), 1),
        "avg_a_chars": round(float(te["text_b"].str.len().mean()), 1),
        "label_dist": te["clarity_label"].value_counts().to_dict(),
        "president_dist": te["group_president"].value_counts().to_dict(),
        "n_titles": int(te["group_title"].nunique()),
    },
    "mapping_check": {"agree": int(agree), "total": int(len(train)),
                      "rate": round(float(agree / len(train)), 4)},
    "dataset_sha": "3afc18f0b582b3cfdb927822cff57ddc6e871f9c",
}

with open(f"{OUT}/label_map.json", "w", encoding="utf-8") as f:
    json.dump({"clarity2id": CLARITY2ID, "evasion2clarity": EVASION2CLARITY},
              f, ensure_ascii=False, indent=2)
with open(f"{OUT}/data_stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 70)
print("表1 数据统计")
print("=" * 70)
print(f"{'':14s}{'样本数':>8s}{'缺失':>6s}{'平均问长':>10s}{'平均答长':>10s}")
for k, v in stats.items():
    if k in ("train", "test"):
        print(f"{k:14s}{v['n']:>8d}{v['missing']:>6d}"
              f"{v['avg_q_words']:>10.1f}{v['avg_a_words']:>10.1f}")
print(f"\n保存完成：processed/, {OUT}/label_map.json, {OUT}/data_stats.json\n[环境] 随机种子 = {SEED}（本步不含随机性，与种子无关）")
