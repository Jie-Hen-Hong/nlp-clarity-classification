"""基线B：bert-base-uncased 句对微调。

用法：
  python exp/05_bert.py --tag direct            # 标准交叉熵
  python exp/05_bert.py --tag cw                # 加权交叉熵（balanced）
"""
import argparse
import json
import os
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from split_util import group_split_by_ratio
from common import bootstrap

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

NAMES = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]
SHORT = ["CR", "AMB", "CNR"]
MODEL_DIR = "models/bert-base-uncased"
SEED_ENV = int(os.environ.get("NLP_SEED", 8))

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="direct")
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--max_len", type=int, default=160)
ap.add_argument("--batch", type=int, default=24)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--seed", type=int, default=SEED_ENV,
                help="随机种子，默认取环境变量 NLP_SEED，否则 8")
ap.add_argument("--use_cw", action="store_true")
args = ap.parse_args()
SEED, OUT = bootstrap(args.seed)
print(f"[环境] 随机种子 = {SEED}   输出目录 = {OUT}")


class _Tee:
    """把标准输出同时写到控制台和日志文件，便于事后用 make_shots.py 生成截图。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()

    def isatty(self):
        return False

    def writable(self):
        return True

    @property
    def encoding(self):
        return "utf-8"


_log_path = f"{OUT}/log_bert_{args.tag}.txt"
_log_fh = open(_log_path, "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _log_fh)
sys.stderr = _Tee(sys.__stderr__, _log_fh)
print(f"[环境] 本次训练日志同时写入 {_log_path}")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(args.threads)
torch.set_num_interop_threads(1) if torch.get_num_interop_threads() > 1 else None

dev_name = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={dev_name}  torch={torch.__version__}")
if dev_name == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

# ---------------- 数据 ----------------
tr_all = pd.read_parquet("processed/qev_train.parquet")
te = pd.read_parquet("processed/qev_test.parquet")

rng = np.random.RandomState(args.seed)
tr, dv, dev_titles = group_split_by_ratio(tr_all, "group_title", 0.2, args.seed)
print(f"train={len(tr)}  dev={len(dv)}  test={len(te)}  "
      f"(dev 占 {len(dv)/len(tr_all)*100:.1f}% 样本, {len(dev_titles)} 个标题)")

tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)


class PairDS(Dataset):
    def __init__(self, df, with_label=True):
        self.a = df["text_a"].tolist()
        self.b = df["text_b"].tolist()
        self.y = df["label"].tolist() if with_label else None

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        enc = tok(self.a[i], self.b[i], truncation=True,
                  max_length=args.max_len, padding="max_length",
                  return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.y is not None:
            item["labels"] = torch.tensor(self.y[i], dtype=torch.long)
        return item


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


ds_tr = PairDS(tr)
ds_dv = PairDS(dv)
ds_te = PairDS(te, with_label=True)
dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, collate_fn=collate,
                   num_workers=0)
dl_dv = DataLoader(ds_dv, batch_size=32, collate_fn=collate, num_workers=0)
dl_te = DataLoader(ds_te, batch_size=32, collate_fn=collate, num_workers=0)

# ---------------- 模型 ----------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR, num_labels=3, local_files_only=True).to(dev_name)

if args.use_cw:
    cnt = np.bincount(tr["label"].values, minlength=3).astype(float)
    w = cnt.sum() / (3 * cnt)
    class_weight = torch.tensor(w, dtype=torch.float, device=dev_name)
    print("class_weight =", {NAMES[i]: round(float(w[i]), 3) for i in range(3)})
else:
    class_weight = None

loss_fn = nn.CrossEntropyLoss(weight=class_weight)

no_decay = ("bias", "LayerNorm.weight")
opt = torch.optim.AdamW([
    {"params": [p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
    {"params": [p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
], lr=args.lr)

total_steps = len(dl_tr) * args.epochs
warmup = int(total_steps * 0.1)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: s / max(1, warmup) if s < warmup
    else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))

use_amp = dev_name == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)


def run_eval(dl):
    model.eval()
    preds, golds, tot_loss = [], [], 0.0
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(dev_name) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(**batch)
            logits = out.logits.float()
            tot_loss += float(loss_fn(logits, batch["labels"])) * len(batch["labels"])
            preds += logits.argmax(-1).cpu().tolist()
            golds += batch["labels"].cpu().tolist()
    return np.array(golds), np.array(preds), tot_loss / len(dl.dataset)


def metrics(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    return {"accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_p": round(float(p.mean()), 4),
            "macro_r": round(float(r.mean()), 4),
            "macro_f1": round(float(f.mean()), 4),
            "weighted_f1": round(float(np.average(f, weights=s)), 4),
            "per_class": {NAMES[i]: {"precision": round(float(p[i]), 4),
                                     "recall": round(float(r[i]), 4),
                                     "f1": round(float(f[i]), 4),
                                     "support": int(s[i])} for i in range(3)}}


# ---------------- 训练 ----------------
hist = []
best = {"macro_f1": -1}
t_start = time.time()
for ep in range(1, args.epochs + 1):
    model.train()
    t0 = time.time()
    run_loss, n = 0.0, 0
    for step, batch in enumerate(dl_tr, 1):
        batch = {k: v.to(dev_name) for k, v in batch.items()}
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"],
                           token_type_ids=batch.get("token_type_ids")).logits
            loss = loss_fn(logits.float(), batch["labels"])
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        run_loss += float(loss) * len(batch["labels"])
        n += len(batch["labels"])
        if step % 100 == 0:
            print(f"  ep{ep} step {step}/{len(dl_tr)} loss={run_loss/n:.4f}")
    y_dv, p_dv, dv_loss = run_eval(dl_dv)
    m = metrics(y_dv, p_dv)
    dt = time.time() - t0
    hist.append({"epoch": ep, "train_loss": round(run_loss / n, 4),
                 "dev_loss": round(dv_loss, 4), **m, "seconds": round(dt, 1)})
    print(f"ep{ep}: train_loss={run_loss/n:.4f} dev_loss={dv_loss:.4f} "
          f"dev Acc={m['accuracy']:.4f} Macro-F1={m['macro_f1']:.4f}  ({dt:.0f}s)")
    if m["macro_f1"] > best["macro_f1"]:
        best = {**m, "epoch": ep, "dev_loss": dv_loss}
        torch.save(model.state_dict(), f"{OUT}/bert_{args.tag}_best.pt")
        print(f"  → 保存最佳模型（ep{ep}, Macro-F1={m['macro_f1']:.4f}）")

total_time = time.time() - t_start
print(f"\n训练完成，总耗时 {total_time/60:.1f} 分钟；最佳 epoch={best['epoch']}")

# ---------------- 用最佳模型评估 test ----------------
model.load_state_dict(torch.load(f"{OUT}/bert_{args.tag}_best.pt",
                                 map_location=dev_name))
y_te, p_te, te_loss = run_eval(dl_te)
te_m = metrics(y_te, p_te)
cm = confusion_matrix(y_te, p_te, labels=[0, 1, 2])
te_m.update({"model": f"BERT-base ({args.tag})", "confusion": cm.tolist(),
             "test_loss": round(te_loss, 4), "best_epoch": best["epoch"],
             "train_seconds": round(total_time, 1),
             "config": {"max_len": args.max_len, "batch": args.batch,
                        "lr": args.lr, "epochs": args.epochs,
                        "use_class_weight": args.use_cw,
                        "seed": args.seed, "device": dev_name}})
print("\n[TEST]", json.dumps({k: te_m[k] for k in
      ("accuracy", "macro_p", "macro_r", "macro_f1", "weighted_f1")},
      ensure_ascii=False))

pd.DataFrame({
    "uid": te["uid"], "text_a": te["text_a"], "text_b": te["text_b"],
    "gold_label": y_te, "gold_name": [NAMES[i] for i in y_te],
    "pred_bert": p_te, "pred_bert_name": [NAMES[i] for i in p_te],
}).to_csv(f"{OUT}/pred_bert_{args.tag}.csv", index=False, encoding="utf-8-sig")

with open(f"{OUT}/bert_{args.tag}_metrics.json", "w", encoding="utf-8") as f:
    json.dump({"test": te_m, "history": hist, "best_dev": best},
              f, ensure_ascii=False, indent=2)

# ---------------- 图：训练曲线 + 混淆矩阵 ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
ax = axes[0]
eps = [h["epoch"] for h in hist]
ax.plot(eps, [h["train_loss"] for h in hist], "o-", label="训练损失", color="#2E6FAD")
ax.plot(eps, [h["dev_loss"] for h in hist], "s--", label="验证损失", color="#C0504D")
ax.set_xlabel("轮次"); ax.set_ylabel("交叉熵损失"); ax.set_title("(a) 训练/验证损失", fontsize=11)
ax2 = ax.twinx()
ax2.plot(eps, [h["macro_f1"] for h in hist], "^:", label="验证 Macro-F1", color="#4F6228")
ax2.set_ylabel("验证 Macro-F1", color="#4F6228")
ax2.set_ylim(0, 0.8)
ax.legend(loc="upper right", fontsize=8, frameon=False)
ax2.legend(loc="center right", fontsize=8, frameon=False)
ax.grid(alpha=0.25)

ax = axes[1]
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(SHORT); ax.set_yticklabels(SHORT)
for i in range(3):
    for j in range(3):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12,
                color="white" if cm[i, j] > cm.max() * 0.55 else "#333333")
ax.set_xlabel("预测标签"); ax.set_ylabel("真实标签")
ax.set_title(f"(b) 测试集混淆矩阵（n={len(te)}）", fontsize=11)
plt.colorbar(im, ax=ax, fraction=0.045)
plt.tight_layout()
plt.savefig(f"{OUT}/figures/bert_{args.tag}.png", bbox_inches="tight", dpi=200)
plt.close()
print(f"已保存 {OUT}/bert_{args.tag}_metrics.json 与图表")
