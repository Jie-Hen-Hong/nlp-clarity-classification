"""从真实训练日志中抽取关键行，渲染成终端风格截图（用于报告图 6）。

日志与输出目录跟随 NLP_SEED（种子 42 → outputs/，其它 → outputs_seedN/），
运行前请确保对应目录下已有 log_bert_cw.txt（即 05_bert.py 的运行时输出）。
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import bootstrap
from make_console import render

SEED, RES = bootstrap()
LOG = f"{RES}/log_bert_cw.txt"
IMG = f"{RES}/figures/shot_train.png"

KEEP = re.compile(r"(class_weight|^train=|^  ep|^ep\d|训练完成|^\[TEST\]|已保存)")


def main():
    if not os.path.exists(LOG):
        raise SystemExit(f"找不到训练日志：{LOG}\n"
                         f"请先用 05_bert.py 训练，并把输出重定向到该文件。")
    with open(LOG, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    picked = [ln for ln in lines if KEEP.search(ln)]
    # 只保留最后一条 train= 配置行与全部 ep 行
    cfg_line = [ln for ln in picked if ln.startswith("train=")]
    body = [ln for ln in picked if not ln.startswith("train=")]
    text = "\n".join((cfg_line[-1:] + body))
    os.makedirs(f"{RES}/figures", exist_ok=True)
    render(text, IMG, title="python exp/05_bert.py --tag cw --use_cw --epochs 3",
           width_px=1200, font_size=16, highlight=("ep", "[TEST]", "训练完成", "已保存"))
    print("lines:", len(cfg_line[-1:] + body))
    print(f"saved: {IMG}")


if __name__ == "__main__":
    main()
