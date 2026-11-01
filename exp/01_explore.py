"""探查 QEvasion 原始数据结构与标签取值。"""
import json
import pandas as pd
from common import bootstrap

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 80)

SEED, OUT = bootstrap()
PROBE = {}


def probe(path, name):
    df = pd.read_parquet(path)
    print(f"\n{'='*70}\n{name}  shape={df.shape}")
    print("columns:", list(df.columns))
    print("\n-- dtypes --")
    print(df.dtypes)
    print("\n-- head 2 --")
    print(df.head(2).to_string()[:1500])
    info = {"shape": list(df.shape), "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "null": {c: int(df[c].isna().sum()) for c in df.columns}}
    for c in df.columns:
        is_text = (pd.api.types.is_object_dtype(df[c])
                   or pd.api.types.is_string_dtype(df[c]))
        if is_text and not pd.api.types.is_numeric_dtype(df[c]):
            vals = df[c].astype(str)
            uniq = vals.nunique()
            info.setdefault("nunique", {})[c] = int(uniq)
            if uniq <= 20:
                info.setdefault("values", {})[c] = vals.value_counts().to_dict()
                print(f"\n-- unique[{c}] = {uniq} --")
                print(df[c].astype(str).value_counts().to_string())
    PROBE[name] = info
    return df


tr = probe("data/qev_train.parquet", "train")
te = probe("data/qev_test.parquet", "test")

print("\n" + "=" * 70)
print("长度统计（字符数）")
for col in ("question", "interview_answer"):
    if col in tr.columns:
        s = tr[col].astype(str).str.len()
        print(f"{col}: mean={s.mean():.1f} median={s.median():.0f} "
              f"p95={s.quantile(.95):.0f} max={s.max()}")

print("\n完全重复行 train:", int(tr.duplicated().sum()))
print("question+answer 重复 train:",
      int(tr.duplicated(subset=["question", "interview_answer"]).sum()))
print("train/test 的 answer 交集:",
      len(set(tr["interview_answer"]) & set(te["interview_answer"])))

with open(f"{OUT}/raw_probe.json", "w", encoding="utf-8") as f:
    json.dump(PROBE, f, ensure_ascii=False, indent=2)
print(f"\nsaved {OUT}/raw_probe.json")
