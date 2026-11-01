"""统一的环境引导：切换工作目录到项目根，并从环境变量读取种子与输出目录。

设计目的
--------
1. 在 PyCharm 里点击“运行”时，工作目录默认是脚本所在的 exp/，
   而所有脚本都用相对路径 data/ processed/ outputs/。
   bootstrap() 会先把工作目录切到项目根（C:\\Temp\\nlp），因此不必手工配置 Working directory。
2. 种子与输出目录都用环境变量控制，一行不改就能整条流水线换种子：
       NLP_SEED=8
   输出目录默认跟着种子走（种子 42 → outputs/，其它种子 → outputs_seed8/），
   因此换种子绝不会覆盖写报告时用的那套结果。也可用 NLP_OUT 显式指定。
   默认 NLP_SEED=42、NLP_OUT=outputs，与写报告时的那次运行完全一致。

用法（放在每个脚本开头，import 之后）：
    from common import bootstrap
    SEED, OUT = bootstrap()
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bootstrap(seed=None, out=None):
    """切换工作目录并返回 (seed, out_dir)。"""
    os.chdir(ROOT)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

    seed = int(seed if seed is not None else os.environ.get("NLP_SEED", 42))

    # 输出目录：显式指定 > 环境变量 > 按种子自动分目录
    if out is None:
        out = os.environ.get("NLP_OUT") or (
            "outputs" if seed == 42 else f"outputs_seed{seed}")
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    return seed, out


def describe(seed, out):
    """打印当前配置，便于在日志里确认这次跑的是哪个种子。"""
    print(f"[环境] 项目根 = {ROOT}")
    print(f"[环境] 随机种子 = {seed}   输出目录 = {out}")
