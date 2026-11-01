# 预训练权重

本目录用于存放预训练模型权重，仓库不包含这些文件（体积较大）。

按 README 中的说明下载：

```python
from huggingface_hub import snapshot_download
snapshot_download("google-bert/bert-base-uncased", local_dir="models/bert-base-uncased")
snapshot_download("sentence-transformers/all-MiniLM-L6-v2", local_dir="models/minilm")
```

下载后目录结构：

```
models/
├── bert-base-uncased/    config.json, model.safetensors, tokenizer.json, vocab.txt
└── minilm/               同上
```
