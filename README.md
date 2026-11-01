# 回答清晰度分类中的标签层级效应

本仓库是《自然语言处理》期末大作业的完整代码与结果，研究 **对话问答回避检测**（response clarity classification）任务中的三个问题：基础模型能到什么水平、跨场景迁移的边界在哪、标签层级结构是否提供额外信息。

任务把每一组（问题，回答）判定为三类之一：**Clear Reply**（明确回答）、**Ambivalent**（模糊回答）、**Clear Non-Reply**（明确拒答）。数据集中的回答全部来自美国政治访谈。

## 主要结论

1. **预训练表示本身不值钱，值钱的是任务适配。** 冻结的 MiniLM 句向量（Macro-F1 0.4325）甚至不如 TF-IDF（0.4646）；让表示层参与训练后（BERT 微调）才达到 0.5141。
2. **跨场景直接迁移几乎失效。** QEvasion 训练的分类器直接预测 QNLI，准确率 0.5147，仅比二分类随机猜测高 1.5 个点。两域词汇覆盖率仅 29.4%。
3. **细粒度监督未改善粗类分类，但揭示了错误的两种性质。** 用 9 类模型折回 3 类，与直接三分类持平（0.5899 对 0.5850）。但误判归属分析显示：Ambivalent 的误判 65%~82% 发生在**同一粗类内部**（边界模糊），Clear Non-Reply 的误判 78%~93% **跨越粗类边界**（系统性混淆）。这比单一 Macro-F1 更能说明瓶颈所在。

## 实验结果

### 原任务三分类（测试集 n=308）

| 方法 | Accuracy | Macro-F1 | Clear Reply F1 | Ambivalent F1 | Clear Non-Reply F1 |
|---|---|---|---|---|---|
| 多数类基线 | 0.6688 | 0.2672 | 0.0000 | 0.8016 | 0.0000 |
| TF-IDF + LR | 0.5422 | 0.4646 | 0.2588 | 0.6700 | 0.4651 |
| MiniLM（冻结）+ LR | 0.5260 | 0.4325 | 0.4145 | 0.6369 | 0.2462 |
| BERT 微调 | **0.6006** | **0.5141** | 0.5164 | 0.6854 | 0.3404 |

> 准确率最高的其实是多数类基线（全判为 Ambivalent），但它的 Macro-F1 只有 0.2672。本任务以 Macro-F1 为准。

### 跨场景迁移（QEvasion → QNLI）

| 设置 | Accuracy | Macro-F1 |
|---|---|---|
| 原方法基线（QEvasion） | 0.5422 | 0.4257 |
| 迁移数据基线（QNLI 朴素配置） | 0.6044 | 0.6044 |
| 直接迁移（源模型零样本） | 0.5147 | 0.5034 |
| 适配后训练（统一字段 + 调优配方） | 0.6044 | 0.5925 |

### 标签层级扩展（5 折分组交叉验证，n=3448）

| 方案 | Accuracy | Macro-F1 |
|---|---|---|
| A 平坦三分类 | 0.5850 | **0.5047** |
| B 层级·硬映射（9 类 argmax → 3 类） | 0.5899 | 0.5025 |
| C 层级·概率聚合（9 类概率按粗类求和） | 0.6003 | 0.3360 |
| D 细粒度 9 类 | 0.2718 | 0.2577 |

层级结构：Clear Reply 由 1 个细类（Explicit）构成，Ambivalent 由 5 个，Clear Non-Reply 由 3 个。Ambivalent 的内部归一化熵为 0.9094，Clear Non-Reply 为 0.9847。

## 数据

| 数据集 | 训练 | 测试/验证 | 说明 |
|---|---|---|---|
| QEvasion | 3448 | 308 | 三分类清晰度 + 九分类回避策略 |
| QNLI | 104743 | 5463 | 二分类蕴含关系，用于迁移实验 |

`data/` 目录已包含预处理后的 parquet 文件，可直接运行。原始数据来自 HuggingFace：

- QEvasion: <https://huggingface.co/datasets/ailsntua/QEvasion>
- QNLI: <https://huggingface.co/datasets/glue>（配置名 `qnli`）

**注意**：官方测试集 308 条只提供三分类标签，`evasion_label` 全部为空。因此细粒度分析在训练集上以按访谈标题分组的 5 折交叉验证完成（见 `exp/11_hierarchy.py`）。

## 环境

纯 CPU 运行，无需 GPU。

```bash
pip install -r requirements.txt
```

核心依赖：Python 3.13、scikit-learn 1.9.0、torch 2.15.0+cpu、transformers 5.22.0、pandas 3.0.2。

预训练权重需单独下载，放到 `models/` 下（仓库不包含，因为体积较大）：

```
models/
├── bert-base-uncased/     # 约 421 MB
└── minilm/                # all-MiniLM-L6-v2，约 88 MB
```

```python
from huggingface_hub import snapshot_download
snapshot_download("google-bert/bert-base-uncased", local_dir="models/bert-base-uncased")
snapshot_download("sentence-transformers/all-MiniLM-L6-v2", local_dir="models/minilm")
```

## 运行

所有脚本通过环境变量 `NLP_SEED` 控制随机种子，通过 `common.py` 统一输出目录。种子为 42 时输出到 `outputs/`，其余种子输出到 `outputs_seedN/`。

```bash
# 设定随机种子（本项目报告使用 8）
export NLP_SEED=8          # Windows PowerShell: $env:NLP_SEED=8

cd exp
python 01_explore.py       # 探查原始数据结构
python 02_prepare.py       # 清洗、9→3 标签映射、按标题分组划分
python 03_eda.py           # 分布与泄漏检查，出图
python 04_tfidf.py         # 基线 A：TF-IDF + 逻辑回归
python 08_sbert.py         # 基线 B-1：MiniLM 冻结 + 逻辑回归
python 05_bert.py --tag cw --use_cw --epochs 3   # 基线 B-2：BERT 微调
python 06_error.py         # 错误分析：5 折交叉验证、长度分桶、误判样本
python 07_transfer.py      # 跨场景迁移：四组对照
python 09_summary.py       # 汇总指标与对比图
python 11_hierarchy.py     # 扩展：标签层级一致性分析
```

全部跑完约 1.5 小时（CPU），其中 BERT 微调占约 50 分钟（3 轮，每轮约 17 分钟）。

05 脚本支持的命令行参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--tag` | `direct` | 输出文件名后缀 |
| `--epochs` | 2 | 训练轮次 |
| `--max_len` | 160 | 最大序列长度 |
| `--batch` | 24 | batch size |
| `--lr` | 3e-5 | 学习率 |
| `--use_cw` | 关闭 | 是否使用类别权重 |
| `--seed` | 读 `NLP_SEED` | 随机种子 |

报告采用 `--tag cw --use_cw --epochs 3`。

## 目录结构

```
├── exp/                    实验脚本
│   ├── common.py           统一工作目录与输出目录
│   ├── split_util.py       按访谈标题分组切分
│   ├── 01_explore.py       原始数据探查
│   ├── 02_prepare.py       清洗与标签映射
│   ├── 03_eda.py           探索性数据分析
│   ├── 04_tfidf.py         基线 A
│   ├── 05_bert.py          基线 B-2（BERT 微调）
│   ├── 06_error.py         错误分析
│   ├── 07_transfer.py      跨场景迁移
│   ├── 08_sbert.py         基线 B-1（MiniLM 冻结）
│   ├── 09_summary.py       汇总与图表
│   └── 11_hierarchy.py     扩展：标签层级分析
├── data/                   预处理后的数据（parquet）
├── models/                 预训练权重（需自行下载）
├── outputs/
│   ├── *.json              各实验的完整指标
│   ├── *.csv               逐样本预测结果
│   └── figures/            论文插图
├── docs/                   论文正文（docx / pdf）
├── requirements.txt
└── 复现说明.md             环境配置与常见问题的详细说明
```

## 复现性说明

- 所有涉及随机性的环节（划分、分类器、BERT 初始化与数据顺序）统一使用同一随机种子。
- 由于 308 条测试集样本量较小，单次划分的指标存在波动。同一配置下种子 42 与种子 8 的 BERT Macro-F1 分别为 0.5752 与 0.5141，相差 0.06。因此本文的相对结论主要依据交叉验证结果（n=3448），测试集数字仅作参照。
- 分类器使用 `random_state` 固定，`n_jobs` 未启用并行，避免多线程带来的非确定性。

## 引用

```bibtex
@inproceedings{thomas-etal-2024-never,
  title = "{``}I Never Said That{''}: A dataset, taxonomy and baselines on response clarity classification",
  author = "Thomas, Konstantinos and Filandrianos, Giorgos and Lymperaiou, Maria and Zerva, Chrysoula and Stamou, Giorgos",
  booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
  year = "2024",
  pages = "5204--5233"
}
```
