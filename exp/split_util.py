"""统一的训练/验证划分：按访谈标题分组，使验证集占总样本的固定比例。

目的：保证所有模型使用完全相同的数据划分（报告要求“同一划分、同一标签映射、同一主指标”）。
若按“标题个数”取比例，因各访谈样本量悬殊，会导致验证集样本占比严重偏离。

补充保护：个别访谈的样本量极大（如某场访谈单独占全库 60% 以上），
若它被排到前面，会一次性把验证集撑到远超目标比例，使训练集严重缩水、
结果与其他种子不可比。因此对“单个标题超过目标量 1.5 倍”的情况直接跳过。
该保护在 seed=42 时不触发（当时验证集 690 条，未超过阈值），
所以报告中的数字不受影响。
"""
import numpy as np


def group_split_by_ratio(df, group_col="group_title", ratio=0.2, seed=42,
                         verbose=False):
    counts = df.groupby(group_col).size().sort_values(ascending=False)
    titles = list(counts.index)
    rng = np.random.RandomState(seed)
    rng.shuffle(titles)
    target = int(len(df) * ratio)
    limit = target * 1.5

    dev_titles, acc, skipped = [], 0, []
    for t in titles:
        if acc >= target:
            break
        c = int(counts[t])
        if c > limit:
            skipped.append((t, c))
            continue
        dev_titles.append(t)
        acc += c

    if verbose and skipped:
        for t, c in skipped:
            print(f"[划分保护] 跳过样本量过大的标题（{c} 条 > 上限 {int(limit)}）：{t}")

    mask = df[group_col].isin(dev_titles)
    return (df[~mask].reset_index(drop=True),
            df[mask].reset_index(drop=True),
            dev_titles)
