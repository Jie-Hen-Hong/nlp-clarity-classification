"""扩展实验：标签层级一致性分析（大作业扩展方向）。

结构：9 个细粒度回避策略（evasion_label）→ 3 个粗粒度清晰度类别（clarity_label）。

评估协议说明：
  官方测试集 308 条只提供 clarity_label（3 类），evasion_label 全为空。
  因此细粒度相关的分析全部在训练集 3448 条上、以按访谈标题分组的 5 折交叉验证
  （out-of-fold 预测）完成，与 06_error.py 的分组分析保持同一协议。

分析内容：
  1. 层级一致性：粗类内部细类的异质程度（归一化熵、主导类占比）
  2. 细粒度分类：9 类 TF-IDF+LR 的 OOF 表现与长尾失效
  3. 层级 vs 平坦：三种把 9 类信息折回 3 类的方案对比
  4. 错误三分解：粗类错 / 粗类对但细类错 / 完全正确
"""
import json
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
from sklearn.model_selection import GroupKFold
from common import bootstrap

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SEED, OUT = bootstrap()
COARSE = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]
EVA9 = ["Explicit", "Dodging", "Implicit", "General", "Deflection",
        "Declining to answer", "Claims ignorance", "Clarification",
        "Partial/half-answer"]
E2C = {"Explicit": 0,
       "Dodging": 1, "Implicit": 1, "General": 1, "Deflection": 1,
       "Partial/half-answer": 1,
       "Declining to answer": 2, "Claims ignorance": 2, "Clarification": 2}
SHORT9 = ["Explicit", "Dodging", "Implicit", "General", "Deflection",
          "Declining", "Ignorance", "Clarify", "Partial"]

# 与 04_tfidf.py 在 dev 上选出的最优配置一致：仅用回答、bigram、类别权重均衡
CFG = dict(text="a_only", ngram=(1, 2), cw="balanced", C=1.0,
           min_df=2, sublinear_tf=True)

tr = pd.read_parquet("processed/qev_train.parquet")
te = pd.read_parquet("processed/qev_test.parquet")
tr = tr.assign(y9=tr["evasion_label"].map(lambda s: EVA9.index(str(s))))
n = len(tr)
print(f"训练集 n={n}，种子={SEED}，特征配置={CFG}")


def feat(df):
    return df["text_b"].tolist()


def met(y, p, labels, names):
    pr, rc, f, s = precision_recall_fscore_support(
        y, p, labels=labels, zero_division=0)
    return {"accuracy": round(float(accuracy_score(y, p)), 4),
            "macro_p": round(float(pr.mean()), 4),
            "macro_r": round(float(rc.mean()), 4),
            "macro_f1": round(float(f.mean()), 4),
            "per_class": {names[j]: {"precision": round(float(pr[k]), 4),
                                     "recall": round(float(rc[k]), 4),
                                     "f1": round(float(f[k]), 4),
                                     "support": int(s[k])}
                          for k, j in enumerate(labels)},
            "confusion": confusion_matrix(y, p, labels=labels).tolist()}


# ================= 1. 层级一致性 =================
print("\n" + "=" * 74)
print("[1] 层级结构：3 个粗类由 9 个细类聚合而成")
print("=" * 74)
rows = []
for ci, cname in enumerate(COARSE):
    sub = tr[tr["label"] == ci]
    members = [e for e in EVA9 if E2C[e] == ci]
    cnt = {e: int((tr["evasion_label"] == e).sum()) for e in members}
    shares = np.array([cnt[e] for e in members], dtype=float) / len(sub)
    k = len(members)
    ent = float(-(shares * np.log(shares)).sum())
    rows.append({"coarse": cname, "n": int(len(sub)), "share": round(len(sub) / n, 4),
                 "n_fine": k, "members": members, "counts": cnt,
                 "norm_entropy": round(ent / np.log(k), 4) if k > 1 else 0.0,
                 "dominance": round(float(shares.max()), 4),
                 "dominant_fine": members[int(np.argmax(shares))]})
    r = rows[-1]
    print(f"{cname:16s} n={r['n']:5d} ({r['share']*100:5.1f}%) 细类数={k} "
          f"归一化熵={r['norm_entropy']:.4f} 主导={r['dominant_fine']}"
          f"({r['dominance']*100:.1f}%)")

fine_stat = []
print("\n细类明细：")
for e in EVA9:
    ntr = int((tr["evasion_label"] == e).sum())
    coarse = COARSE[E2C[e]]
    share_in = ntr / int((tr["label"] == E2C[e]).sum())
    fine_stat.append({"fine": e, "coarse": coarse, "n_train": ntr,
                      "share_all": round(ntr / n, 4),
                      "share_in_coarse": round(share_in, 4)})
    print(f"  {e:22s} -> {coarse:16s} n={ntr:5d} "
          f"占全集 {ntr/n*100:5.2f}%  占所属粗类 {share_in*100:5.1f}%")

# ================= 2. 5 折分组交叉验证 =================
print("\n" + "=" * 74)
print("[2] 按访谈标题分组的 5 折交叉验证（out-of-fold）")
print("=" * 74)
gkf = GroupKFold(n_splits=5)
oof_flat = np.full(n, -1)
oof_fine = np.full(n, -1)
oof_hard = np.full(n, -1)
oof_soft = np.full(n, -1)
oof_fine_nw = np.full(n, -1)
oof_hard_nw = np.full(n, -1)
oof_soft_nw = np.full(n, -1)
oof_fine_conf = np.zeros((n, 9))
fold_sizes = []

t0 = time.time()
for k, (i_tr, i_va) in enumerate(
        gkf.split(tr, tr["label"], groups=tr["group_title"]), 1):
    fold_sizes.append(int(len(i_va)))
    vec = TfidfVectorizer(ngram_range=CFG["ngram"], min_df=CFG["min_df"],
                          sublinear_tf=CFG["sublinear_tf"], lowercase=True,
                          strip_accents="unicode")
    Xa = vec.fit_transform(feat(tr.iloc[i_tr]))
    Xb = vec.transform(feat(tr.iloc[i_va]))

    c3 = LogisticRegression(C=CFG["C"], class_weight=CFG["cw"],
                            max_iter=2000, random_state=SEED)
    c3.fit(Xa, tr.iloc[i_tr]["label"].values)
    oof_flat[i_va] = c3.predict(Xb)

    # 9 类（带类别权重，与主实验配方一致）
    c9 = LogisticRegression(C=CFG["C"], class_weight=CFG["cw"],
                            max_iter=2000, random_state=SEED)
    c9.fit(Xa, tr.iloc[i_tr]["y9"].values)
    p9 = c9.predict(Xb)
    oof_fine[i_va] = p9
    oof_hard[i_va] = np.array([E2C[EVA9[i]] for i in p9])

    proba = c9.predict_proba(Xb)
    coarse_p = np.zeros((Xb.shape[0], 3))
    for j, cls in enumerate(c9.classes_):
        coarse_p[:, E2C[EVA9[cls]]] += proba[:, j]
    for j, cls in enumerate(c9.classes_):
        oof_fine_conf[i_va, cls] = proba[:, j]
    oof_soft[i_va] = coarse_p.argmax(1)

    # 9 类（不加类别权重）：用于检验概率求和是否因权重而失真
    c9n = LogisticRegression(C=CFG["C"], class_weight=None,
                             max_iter=2000, random_state=SEED)
    c9n.fit(Xa, tr.iloc[i_tr]["y9"].values)
    p9n = c9n.predict(Xb)
    oof_fine_nw[i_va] = p9n
    oof_hard_nw[i_va] = np.array([E2C[EVA9[i]] for i in p9n])
    proba_n = c9n.predict_proba(Xb)
    coarse_pn = np.zeros((Xb.shape[0], 3))
    for j, cls in enumerate(c9n.classes_):
        coarse_pn[:, E2C[EVA9[cls]]] += proba_n[:, j]
    oof_soft_nw[i_va] = coarse_pn.argmax(1)

    m3 = met(tr.iloc[i_va]["label"].values, oof_flat[i_va], [0, 1, 2], COARSE)
    m9 = met(tr.iloc[i_va]["y9"].values, p9, list(range(9)), EVA9)
    print(f"fold{k}: n={len(i_va):4d}  平坦3类 F1={m3['macro_f1']:.4f} | "
          f"细粒度9类 F1={m9['macro_f1']:.4f}")

print(f"交叉验证耗时 {time.time()-t0:.1f}s")

y3 = tr["label"].values
y9 = tr["y9"].values
assert (oof_flat >= 0).all() and (oof_fine >= 0).all()

# ================= 3. 四组评估 =================
print("\n" + "=" * 74)
print("[3] 四组方案对比（同一批 OOF 预测）")
print("=" * 74)
m_flat = met(y3, oof_flat, [0, 1, 2], COARSE)
m_hard = met(y3, oof_hard, [0, 1, 2], COARSE)
m_soft = met(y3, oof_soft, [0, 1, 2], COARSE)
m_fine = met(y9, oof_fine, list(range(9)), EVA9)
m_hard_nw = met(y3, oof_hard_nw, [0, 1, 2], COARSE)
m_soft_nw = met(y3, oof_soft_nw, [0, 1, 2], COARSE)

print(f"{'方案':40s}{'Acc':>8s}{'Macro-F1':>10s}")
print("-" * 58)
for nm, m in (("A 平坦 3 类（直接监督）", m_flat),
              ("B 层级·硬映射（9类加权）", m_hard),
              ("C 层级·概率聚合（9类加权）", m_soft),
              ("D 细粒度 9 类（不作粗类评估）", m_fine),
              ("E 层级·硬映射（9类不加权）", m_hard_nw),
              ("F 层级·概率聚合（9类不加权）", m_soft_nw)):
    print(f"{nm:40s}{m['accuracy']:>8.4f}{m['macro_f1']:>10.4f}")

print("\n细粒度 9 类各类表现：")
for e in EVA9:
    d = m_fine["per_class"][e]
    print(f"  {e:22s} P={d['precision']:.4f} R={d['recall']:.4f} "
          f"F1={d['f1']:.4f} n={d['support']:5d}")

print("\n各 3 类方案的类别级 F1：")
for nm, m in (("平坦", m_flat), ("硬映射·加权", m_hard), ("概率聚合·加权", m_soft),
              ("硬映射·无权", m_hard_nw), ("概率聚合·无权", m_soft_nw)):
    print(f"  {nm:14s} " + "  ".join(
        f"{s} {m['per_class'][c]['f1']:.4f}"
        for c, s in zip(COARSE, ["CR", "AMB", "CNR"])))

# ================= 4. 错误三分解 =================
print("\n" + "=" * 74)
print("[4] 错误分解（以细粒度模型为准）")
print("=" * 74)
exact = int((oof_fine == y9).sum())
cr_fw = int(((oof_hard == y3) & (oof_fine != y9)).sum())
cw = int((oof_hard != y3).sum())
decomp = {"n": n, "exact_right": exact, "coarse_right_fine_wrong": cr_fw,
          "coarse_wrong": cw,
          "exact_acc": round(exact / n, 4),
          "hier_acc": round((exact + cr_fw) / n, 4)}
print(f"  完全正确（粗类与细类都对） {exact:5d}  {exact/n*100:5.1f}%")
print(f"  粗类对、细类错             {cr_fw:5d}  {cr_fw/n*100:5.1f}%")
print(f"  粗类判错                   {cw:5d}  {cw/n*100:5.1f}%")
print(f"  → 层级准确率 {decomp['hier_acc']:.4f}，细粒度精确率 {decomp['exact_acc']:.4f}")

# 各粗类内部的细类混淆情况
inner = {}
print("\n各粗类内部：粗类判对的样本中，细类也判对的比例")
for ci, cname in enumerate(COARSE):
    mask = (y3 == ci) & (oof_hard == ci)
    tot = int(mask.sum())
    fine_ok = int((oof_fine[mask] == y9[mask]).sum())
    inner[cname] = {"n_coarse_right": tot, "fine_right": fine_ok,
                    "inner_acc": round(fine_ok / tot, 4) if tot else 0.0}
    print(f"  {cname:16s} 粗类判对 {tot:5d} 条，其中细类也对 {fine_ok:5d} 条 "
          f"→ 内部精度 {inner[cname]['inner_acc']:.4f}")

# 错误归属：真实类别的误判样本，落在同粗类内部还是跨粗类
print("\n误判去向归属（真实类 i 的错误样本，预测落在哪一层）")
attrib = {}
for i, e in enumerate(EVA9):
    mask = y9 == i
    errs = oof_fine[mask]
    errs = errs[errs != i]
    if len(errs) == 0:
        attrib[e] = {"n_err": 0, "inner": 0, "outer": 0,
                     "inner_share": 0.0, "outer_share": 0.0}
        continue
    inner_n = int(sum(1 for x in errs if E2C[EVA9[x]] == E2C[e]))
    outer_n = int(len(errs) - inner_n)
    attrib[e] = {"n_err": int(len(errs)), "inner": inner_n, "outer": outer_n,
                 "inner_share": round(inner_n / len(errs), 4),
                 "outer_share": round(outer_n / len(errs), 4)}
    print(f"  {e:22s} 错 {len(errs):4d} 条：同粗类内 {inner_n:4d} "
          f"({inner_n/len(errs)*100:5.1f}%)  跨粗类 {outer_n:4d} "
          f"({outer_n/len(errs)*100:5.1f}%)")

# ================= 5. 长尾效应 =================
xs = [int((tr["evasion_label"] == e).sum()) for e in EVA9]
ys_ = [m_fine["per_class"][e]["f1"] for e in EVA9]
r_tail = float(np.corrcoef(np.log(xs), ys_)[0, 1])
r_pearson = float(np.corrcoef(xs, ys_)[0, 1])
print(f"\n细类样本量 vs F1：pearson r={r_pearson:.4f}，log-样本量 r={r_tail:.4f}")

# ================= 6. 保存 =================
out = {"protocol": {
    "note": "官方测试集 evasion_label 全为空，细粒度分析在训练集 3448 条上以按访谈标题分组的 5 折 OOF 完成",
    "config": {"text": CFG["text"], "ngram": str(CFG["ngram"]),
               "class_weight": CFG["cw"], "C": CFG["C"]},
    "n": n, "n_folds": 5, "seed": SEED},
    "hierarchy": rows, "fine_stat": fine_stat,
    "flat3": m_flat, "hier_hard": m_hard, "hier_soft": m_soft, "fine9": m_fine,
    "hier_hard_nw": m_hard_nw, "hier_soft_nw": m_soft_nw,
    "decomp": decomp, "inner_coarse": inner, "attrib": attrib,
    "fold_sizes": fold_sizes,
    "tail_corr": round(r_pearson, 4), "tail_corr_log": round(r_tail, 4),
    "seconds": round(time.time() - t0, 1)}
with open(f"{OUT}/hierarchy_metrics.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

pd.DataFrame({
    "uid": tr["uid"], "text_a": tr["text_a"], "text_b": tr["text_b"],
    "a_words": tr["a_words"], "group_title": tr["group_title"],
    "gold_3": y3, "gold_9": y9,
    "gold_9_name": [EVA9[i] for i in y9],
    "pred_flat3": oof_flat, "pred_9": oof_fine,
    "pred_9_name": [EVA9[i] for i in oof_fine],
    "pred_hard3": oof_hard, "pred_soft3": oof_soft,
}).to_csv(f"{OUT}/pred_hierarchy.csv", index=False, encoding="utf-8-sig")

# ================= 7. 图 =================
# 图1 粗类内部构成
fig, ax = plt.subplots(figsize=(7.4, 3.0))
groups = [[0], [1, 2, 3, 4, 8], [5, 6, 7]]
ramps = [["#185FA5"], list(plt.cm.Oranges(np.linspace(0.82, 0.28, 5))),
         list(plt.cm.Greens(np.linspace(0.82, 0.28, 3)))]
left = np.zeros(3)
for gi, idxs in enumerate(groups):
    for j, ix in enumerate(idxs):
        e = EVA9[ix]
        frac = np.zeros(3)
        frac[gi] = (int((tr["evasion_label"] == e).sum())
                    / int((tr["label"] == gi).sum()))
        ax.barh(range(3), frac, left=left, color=ramps[gi][j], label=e,
                edgecolor="white", linewidth=1.0)
        left += frac
ax.set_yticks(range(3))
ax.set_yticklabels([f"{c}\n（{len(g)} 个细类）" for c, g in zip(COARSE, groups)],
                   fontsize=8.5)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=8.5)
ax.set_xlabel("粗类内部的细类构成比例", fontsize=9.5)
ax.set_title("三个粗类由不同数量的细粒度回避策略聚合而成", fontsize=11)
ax.legend(fontsize=7.5, ncol=3, loc="upper center",
          bbox_to_anchor=(0.5, -0.34), frameon=False,
          title="按粗类分组：蓝=Clear Reply　橙=Ambivalent　绿=Clear Non-Reply",
          title_fontsize=7.5)
plt.tight_layout()
plt.savefig(f"{OUT}/figures/hier_composition.png", bbox_inches="tight", dpi=200)
plt.close()

# 图2 9 类混淆矩阵
cm = np.array(m_fine["confusion"])
fig, ax = plt.subplots(figsize=(6.4, 5.4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(9)); ax.set_yticks(range(9))
ax.set_xticklabels(SHORT9, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(SHORT9, fontsize=8)
for i in range(9):
    for j in range(9):
        if cm[i, j]:
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7.5,
                    color="white" if cm[i, j] > cm.max() * 0.45 else "#333333")
ax.set_xlabel("预测细类"); ax.set_ylabel("真实细类")
ax.set_title(f"9 类细粒度分类混淆矩阵（OOF, n={n}）", fontsize=11)
plt.colorbar(im, ax=ax, fraction=0.045)
plt.tight_layout()
plt.savefig(f"{OUT}/figures/hier_cm9.png", bbox_inches="tight", dpi=200)
plt.close()

# 图3 错误三分解
fig, ax = plt.subplots(figsize=(6.6, 2.4))
parts = [("完全正确", exact, "#1D9E75"),
         ("粗类对·细类错", cr_fw, "#EF9F27"),
         ("粗类判错", cw, "#E24B4A")]
leftp = 0
for name, val, colr in parts:
    ax.barh([0], [val], left=leftp, color=colr, edgecolor="white",
            linewidth=1.2, label=f"{name}  {val} ({val/n*100:.1f}%)")
    leftp += val
ax.set_yticks([]); ax.set_xlim(0, n)
ax.set_xlabel(f"训练集样本数（OOF，n={n}）")
ax.set_title("细粒度模型的预测结果分解", fontsize=11)
ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.42),
          frameon=False)
plt.tight_layout()
plt.savefig(f"{OUT}/figures/hier_decomp.png", bbox_inches="tight", dpi=200)
plt.close()

# 图4 错误归属：每个细类的误判落在同粗类内还是跨粗类
fig, ax = plt.subplots(figsize=(7.6, 4.2))
order = (["Explicit"]
         + ["Dodging", "Implicit", "General", "Deflection", "Partial/half-answer"]
         + ["Declining to answer", "Claims ignorance", "Clarification"])
labels_show = ["Explicit", "Dodging", "Implicit", "General", "Deflection",
               "Partial", "Declining", "Ignorance", "Clarify"]
inn = np.array([attrib[e]["inner_share"] * 100 for e in order])
out = np.array([attrib[e]["outer_share"] * 100 for e in order])
ypos = np.arange(len(order))[::-1]
ax.barh(ypos, inn, color="#1D9E75", label="同粗类内部混淆", edgecolor="white",
        linewidth=0.8)
ax.barh(ypos, out, left=inn, color="#E24B4A", label="跨粗类混淆",
        edgecolor="white", linewidth=0.8)
for yi, iv, ov in zip(ypos, inn, out):
    if iv > 6:
        ax.text(iv / 2, yi, f"{iv:.0f}%", ha="center", va="center",
                fontsize=7.5, color="white")
    if ov > 6:
        ax.text(iv + ov / 2, yi, f"{ov:.0f}%", ha="center", va="center",
                fontsize=7.5, color="white")
ax.set_yticks(ypos)
ax.set_yticklabels(labels_show, fontsize=9)
ax.set_xlim(0, 100)
ax.set_xlabel("该细类全部误判样本的构成比例")
ax.axhline(7.5, color="#B4B2A9", lw=0.7, ls="--")
ax.axhline(2.5, color="#B4B2A9", lw=0.7, ls="--")
ax.text(102, 8, "Clear Reply", fontsize=7.5, color="#5F5E5A", va="center")
ax.text(102, 5, "Ambivalent", fontsize=7.5, color="#5F5E5A", va="center")
ax.text(102, 1, "Clear Non-Reply", fontsize=7.5, color="#5F5E5A", va="center")
ax.set_title("误判去向的两个模式：内部混淆与跨粗类混淆", fontsize=10.5)
ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.13),
          ncol=2, frameon=False)
plt.tight_layout()
plt.savefig(f"{OUT}/figures/hier_error_attribution.png", bbox_inches="tight",
            dpi=200)
plt.close()

# 图5 长尾
fig, ax = plt.subplots(figsize=(5.8, 3.6))
ax.scatter(xs, ys_, s=52, color="#378ADD", edgecolor="white", linewidth=0.9, zorder=3)
for xi, yi, e in zip(xs, ys_, SHORT9):
    ax.annotate(e, (xi, yi), fontsize=7, xytext=(5, 4),
                textcoords="offset points", color="#5F5E5A")
ax.set_xlabel("训练集细类样本量")
ax.set_ylabel("OOF F1")
ax.set_title(f"细类规模与分类性能（r = {r_pearson:.3f}）", fontsize=11)
ax.grid(alpha=0.25, linewidth=0.6)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/figures/hier_tail.png", bbox_inches="tight", dpi=200)
plt.close()

print(f"\n已保存 {OUT}/hierarchy_metrics.json, pred_hierarchy.csv 与 5 张图")
