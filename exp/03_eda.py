"""探索性数据分析：分布图、长度分布、分组统计、泄漏风险检查。"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import bootstrap

SEED, OUT = bootstrap()

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110

FIG = f"{OUT}/figures"
os.makedirs(FIG, exist_ok=True)

CLR = {"Clear Reply": "#2E6FAD", "Ambivalent": "#D98A3D", "Clear Non-Reply": "#8C8C8C"}

tr = pd.read_parquet("processed/qev_train.parquet")
te = pd.read_parquet("processed/qev_test.parquet")
ORDER = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]

# ============ 图1  标签分布 ============
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))

ax = axes[0]
vc = tr["clarity_label"].value_counts().reindex(ORDER)
vt = te["clarity_label"].value_counts().reindex(ORDER)
x = np.arange(3)
w = 0.38
b1 = ax.bar(x - w / 2, vc.values, w, label=f"训练集 (n={len(tr)})",
            color=[CLR[k] for k in ORDER], edgecolor="white", linewidth=0.8)
b2 = ax.bar(x + w / 2, vt.values, w, label=f"测试集 (n={len(te)})",
            color=[CLR[k] for k in ORDER], alpha=0.45,
            edgecolor="white", linewidth=0.8, hatch="//")
for b in list(b1) + list(b2):
    h = b.get_height()
    ax.text(b.get_x() + b.get_width() / 2, h + 25, f"{int(h)}\n{h/ (len(tr) if b in b1 else len(te))*100:.1f}%",
            ha="center", va="bottom", fontsize=7.6)
ax.set_xticks(x)
ax.set_xticklabels(["Clear Reply\n明确回答", "Ambivalent\n模糊回答", "Clear Non-Reply\n明确不回答"])
ax.set_ylabel("样本数")
ax.set_title("(a) 粗类 clarity_label 分布", fontsize=11)
ax.legend(fontsize=8.5, frameon=False)
ax.set_ylim(0, 2400)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25)

ax = axes[1]
ve = tr["evasion_label"].value_counts().sort_values()
colors = ["#2E6FAD" if k == "Explicit" else
          ("#8C8C8C" if k in ("Declining to answer", "Claims ignorance", "Clarification")
           else "#D98A3D") for k in ve.index]
ax.barh(range(len(ve)), ve.values, color=colors, edgecolor="white", linewidth=0.8)
ax.set_yticks(range(len(ve)))
ax.set_yticklabels(ve.index, fontsize=8.5)
for i, v in enumerate(ve.values):
    ax.text(v + 12, i, str(v), va="center", fontsize=8)
ax.set_xlabel("样本数")
ax.set_title("(b) 细类 evasion_label 分布（仅训练集有标签）", fontsize=11)
ax.set_xlim(0, 1200)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", alpha=0.25)

plt.tight_layout()
plt.savefig(f"{FIG}/fig1_label_dist.png", bbox_inches="tight", dpi=200)
plt.close()

# ============ 图2  长度分布 ============
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

ax = axes[0]
data = [tr.loc[tr["clarity_label"] == k, "a_words"].values for k in ORDER]
bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                medianprops=dict(color="black", linewidth=1.2))
for patch, k in zip(bp["boxes"], ORDER):
    patch.set_facecolor(CLR[k])
    patch.set_alpha(0.75)
ax.set_xticklabels(["Clear\nReply", "Ambivalent", "Clear\nNon-Reply"], fontsize=9)
ax.set_ylabel("回答词数 (words)")
ax.set_title("(a) 回答长度按清晰度分组（箱线图，已隐藏离群点）", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25)
for i, k in enumerate(ORDER):
    m = tr.loc[tr["clarity_label"] == k, "a_words"].median()
    ax.text(i + 1.33, m, f"{m:.0f}", va="center", ha="left", fontsize=8,
            color="#333333")
ax.plot([0.72, 3.28], [0, 0], alpha=0)
ax.set_xlim(0.45, 3.75)

ax = axes[1]
bins = np.arange(0, 260, 10)
ax.hist(tr["q_words"], bins=bins, color="#7A9EC4", edgecolor="white", linewidth=0.6)
ax.axvline(tr["q_words"].median(), color="#C0504D", linestyle="--", linewidth=1.2,
           label=f"中位数 {tr['q_words'].median():.0f}")
ax.axvline(tr["q_words"].mean(), color="#4F6228", linestyle=":", linewidth=1.2,
           label=f"均值 {tr['q_words'].mean():.1f}")
ax.set_xlabel("问题词数 (words)")
ax.set_ylabel("样本数")
ax.set_title("(b) 问题长度分布", fontsize=11)
ax.legend(fontsize=8.5, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25)
ax.set_xlim(0, 160)

plt.tight_layout()
plt.savefig(f"{FIG}/fig2_length.png", bbox_inches="tight", dpi=200)
plt.close()

# ============ 图3  分组与泄漏 ============
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))

ax = axes[0]
pv = pd.crosstab(tr["group_president"], tr["clarity_label"])[ORDER]
pv = pv.loc[pv.sum(axis=1).sort_values().index]
left = np.zeros(len(pv))
for k in ORDER:
    ax.barh(range(len(pv)), pv[k].values, left=left, color=CLR[k],
            label=k, edgecolor="white", linewidth=0.7)
    left += pv[k].values
ax.set_yticks(range(len(pv)))
ax.set_yticklabels([i.replace("Donald J. ", "").replace("Joseph R. ", "")
                    for i in pv.index], fontsize=9)
for i, tot in enumerate(pv.sum(axis=1).values):
    ax.text(tot + 15, i, str(int(tot)), va="center", fontsize=8.5)
ax.set_xlabel("样本数")
ax.set_title("(a) 训练集按总统的标签构成", fontsize=11)
ax.legend(fontsize=8, frameon=False, loc="lower right")
ax.set_xlim(0, 1500)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", alpha=0.25)

ax = axes[1]
reuse = tr.groupby("text_b").size().value_counts().sort_index()
ax.bar(reuse.index.astype(str), reuse.values, color="#7A9EC4",
       edgecolor="white", linewidth=0.8)
for i, (k, v) in enumerate(reuse.items()):
    ax.text(i, v + 6, str(v), ha="center", fontsize=8)
ax.set_xlabel("同一回答被使用的次数")
ax.set_ylabel("回答数")
ax.set_title("(b) 回答复用情况（泄漏风险检查）", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.savefig(f"{FIG}/fig3_group_leak.png", bbox_inches="tight", dpi=200)
plt.close()

# ============ 文本统计 ============
print("=" * 70)
print("分组统计")
print("=" * 70)
print(pd.crosstab(tr["group_president"], tr["clarity_label"], margins=True).to_string())
print("\n测试集 president 分布：")
print(te["group_president"].value_counts().to_string())
print(f"测试集 title 数：{te['group_title'].nunique()}")

print("\n长度统计（词数）")
print(tr.groupby("clarity_label")["a_words"].agg(["count", "mean", "median", "max"]).to_string())
print()
print(tr.groupby("clarity_label")["q_words"].agg(["count", "mean", "median"]).to_string())

# 泄漏：同一回答是否跨越 train/test
inter = set(tr["text_b"]) & set(te["text_b"])
print(f"\ntrain 与 test 共享的回答数：{len(inter)}")
print(f"训练集内部：{tr['text_b'].nunique()} 个回答覆盖 {len(tr)} 个样本 "
      f"（平均每回答 {len(tr)/tr['text_b'].nunique():.2f} 个问题）")

summary = {
    "fig1": {"train_dist": tr["clarity_label"].value_counts().to_dict(),
             "test_dist": te["clarity_label"].value_counts().to_dict()},
    "length": {
        "answer_words_by_clarity": tr.groupby("clarity_label")["a_words"]
        .agg(["mean", "median"]).round(1).to_dict(),
        "q_words": {"mean": round(float(tr["q_words"].mean()), 1),
                    "median": float(tr["q_words"].median())},
        "a_words": {"mean": round(float(tr["a_words"].mean()), 1),
                    "median": float(tr["a_words"].median()),
                    "p95": float(tr["a_words"].quantile(.95))},
    },
    "leak": {"shared_answers_train_test": len(inter),
             "answers_in_train": int(tr["text_b"].nunique()),
             "test_titles": int(te["group_title"].nunique())},
}
with open(f"{OUT}/eda_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"\n图已保存至 {FIG}/")
