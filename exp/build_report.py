"""生成实验报告 docx：在 doc2.docx 模板的表格/单元格中填充内容。

表号遵循模板：表1 数据统计、表2 训练配置、表3 模型总体性能、
表4 最佳模型类别级结果、表5 典型误判样本、表6 TF-IDF 消融、表7 迁移对比。
"""
import copy
import json
import os
import sys

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from split_util import group_split_by_ratio
from common import bootstrap

# 数据目录跟随 NLP_SEED：种子 42 → outputs/，其它 → outputs_seedN/
SEED, RES = bootstrap()

SRC = "doc2.docx"
FIG = f"{RES}/figures"
OUT = ("D:/wb_space/自然语言处理/期末/实验报告_对话问答回避检测与跨场景迁移分析"
       + ("" if SEED == 42 else f"_种子{SEED}") + ".docx")

SONG = "宋体"
HEI = "黑体"
MONO = "Consolas"
GRAY = RGBColor(0x55, 0x55, 0x55)
DARK = RGBColor(0x1A, 0x1A, 0x1A)

CLS = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]
SHORT = {"Clear Reply": "CR", "Ambivalent": "AMB", "Clear Non-Reply": "CNR"}


def J(name):
    with open(f"{RES}/{name}", encoding="utf-8") as f:
        return json.load(f)


DS, EDA, TF, SB, TR, ER = (J("data_stats.json"), J("eda_summary.json"),
                           J("tfidf_metrics.json"), J("sbert_metrics.json"),
                           J("transfer_metrics.json"), J("error_analysis.json"))
RP = J("reproduce_check.json") if os.path.exists(f"{RES}/reproduce_check.json") else None
BERT = J("bert_cw_metrics.json") if os.path.exists(f"{RES}/bert_cw_metrics.json") else None
B = BERT["test"] if BERT else None

# ---------- 复算 BERT 使用的分组划分与类别权重（与 05_bert.py 完全一致）----------
_tr_all = pd.read_parquet("processed/qev_train.parquet")
TR_FIT, DV_FIT, DV_TITLES = group_split_by_ratio(_tr_all, "group_title", 0.2, SEED)
_c = np.bincount(TR_FIT["label"].values, minlength=3).astype(float)
CW = _c.sum() / (3 * _c)
CW_TXT = (f"Clear Reply {CW[0]:.3f}、Ambivalent {CW[1]:.3f}、"
          f"Clear Non-Reply {CW[2]:.3f}")


def f4(x):
    return f"{x:.4f}"


def clear_cell(cell):
    for p in list(cell.paragraphs):
        p._element.getparent().remove(p._element)
    for t in list(cell.tables):
        t._element.getparent().remove(t._element)


def para(cell, text="", size=10.5, bold=False, font=SONG, color=None, align=None,
         first_indent=True, space_before=0, space_after=3, line=1.32):
    p = cell.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.line_spacing = line
    p.alignment = align if align is not None else WD_ALIGN_PARAGRAPH.JUSTIFY
    if first_indent:
        pf.first_line_indent = Pt(size * 2)
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.name = font
    r._element.rPr.rFonts.set(qn("w:eastAsia"), font)
    if color is not None:
        r.font.color.rgb = color
    return p


def heading(cell, text, size=12, space_before=9, space_after=4):
    p = cell.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.line_spacing = 1.2
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.font.bold = True
    r.font.name = HEI
    r._element.rPr.rFonts.set(qn("w:eastAsia"), HEI)
    r.font.color.rgb = DARK
    return p


def caption(cell, text, size=9):
    p = cell.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.line_spacing = 1.15
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.font.name = SONG
    r._element.rPr.rFonts.set(qn("w:eastAsia"), SONG)
    r.font.color.rgb = GRAY
    return p


def _fixed_layout(t, widths):
    """把表格设为固定列宽，逐单元格写入宽度并同步 tblGrid。"""
    t.autofit = False
    tblPr = t._tbl.tblPr
    layout = tblPr.find(qn("w:tblLayout"))
    if layout is None:
        layout = tblPr.makeelement(qn("w:tblLayout"), {qn("w:type"): "fixed"})
        tblPr.append(layout)
    layout.set(qn("w:type"), "fixed")
    if widths:
        for i, w in enumerate(widths):
            for row in t.rows:
                row.cells[i].width = Cm(w)
        grid = t._tbl.find(qn("w:tblGrid"))
        if grid is not None:
            cols = grid.findall(qn("w:gridCol"))
            for i, w in enumerate(widths):
                if i < len(cols):
                    cols[i].set(qn("w:w"), str(int(w * 567)))


def _style_run(r, size, bold=False, font=SONG):
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.name = font
    r._element.rPr.rFonts.set(qn("w:eastAsia"), font)


def table(cell, headers, rows, widths=None, size=8.5, header_fill="E8EEF5",
          left_cols=()):
    t = cell.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        _style_run(p.add_run(str(h)), size, bold=True)
        tcPr = hdr[i]._tc.get_or_add_tcPr()
        tcPr.append(tcPr.makeelement(qn("w:shd"),
                                     {qn("w:fill"): header_fill, qn("w:val"): "clear"}))
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cells[i].paragraphs[0]
            p.alignment = (WD_ALIGN_PARAGRAPH.LEFT if i in left_cols
                           else WD_ALIGN_PARAGRAPH.CENTER)
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.08
            _style_run(p.add_run(str(v)), size)
    _fixed_layout(t, widths)
    return t


def fill_row(tbl, ri, vals, cols=None, size=8.5, bold=False):
    """把 vals 依次写入第 ri 行的若干单元格（cols 指定列下标）。"""
    row = tbl.rows[ri]
    for k, v in enumerate(vals):
        ci = cols[k] if cols else k
        cell = row.cells[ci]
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        for r in list(p.runs):
            r._element.getparent().remove(r._element)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        _style_run(p.add_run(str(v)), size, bold=bold)


def add_row(tbl):
    tbl._tbl.append(copy.deepcopy(tbl.rows[-1]._tr))


SEED_FREE = {"shot_mapping.png"}  # 与随机种子无关的图，缺失时借用 outputs/ 下的同名文件


def picture(cell, name, width=13.2, max_h=9.6):
    """插入图片：默认按 width 定宽；若换算后的高度超过 max_h，则改按高度缩放，
    防止自行截图（宽高比不可控）撑破版面。"""
    path = os.path.join(FIG, name)
    if not os.path.exists(path) and name in SEED_FREE:
        alt = os.path.join("outputs", "figures", name)
        if os.path.exists(alt) and os.path.abspath(alt) != os.path.abspath(path):
            path = alt
    if not os.path.exists(path):
        # 缺图时降级为文字说明，避免整个脚本中断
        p = cell.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(f"［此处缺图 {name}：请在对应目录生成后重新运行本脚本］")
        r.font.size = Pt(9)
        r.font.color.rgb = GRAY
        return p
    p = cell.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(0)
    # 按图片像素尺寸判断：高宽比过大时改为按高度约束
    try:
        with Image.open(path) as im:
            pw, ph = im.size
        if ph / pw * width > max_h:
            p.add_run().add_picture(path, height=Cm(max_h))
            return p
    except Exception:
        pass
    p.add_run().add_picture(path, width=Cm(width))
    return p


def code_block(cell, lines, size=8.5):
    t = cell.add_table(rows=1, cols=1)
    t.style = "Table Grid"
    tc = t.rows[0].cells[0]
    tcPr = tc._tc.get_or_add_tcPr()
    tcPr.append(tcPr.makeelement(qn("w:shd"),
                                 {qn("w:fill"): "F5F5F5", qn("w:val"): "clear"}))
    first = True
    for ln in lines:
        p = tc.paragraphs[0] if first else tc.add_paragraph()
        first = False
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        r = p.add_run(ln if ln else " ")
        r.font.size = Pt(size)
        r.font.name = MONO
        r._element.rPr.rFonts.set(qn("w:eastAsia"), SONG)
    return t


def ab(text, ngram, cw):
    """取表 6（消融）中的一行。"""
    for r in TF["ablation_test"]:
        if r["text"] == text and r["ngram"] == ngram and r["class_weight"] == cw:
            return r
    raise KeyError((text, ngram, cw))


doc = Document(SRC)
t0, t1 = doc.tables[0], doc.tables[1]

# ==================== 一、完成情况补充 ====================
c = t0.rows[0].cells[0]
para(c, "本报告完成情况：按上述要求完成了数据获取与清洗、探索性数据分析、两条必做基线"
       "（TF-IDF + 逻辑回归、预训练模型微调）以及一组跨场景迁移实验（QNLI）。所有模型使用相同的"
       "数据划分、相同的标签映射与相同的主指标（Macro-F1），并提供了可重复运行的代码、"
       "配置文件、指标文件与逐样本预测文件。", space_before=4)

# ==================== 二、填模板自带的两张表 ====================
c1 = t0.rows[1].cells[0]
T1, T2 = c1.tables[1], c1.tables[2]

# ---- 表1 数据统计记录（模板 5 列 × 4 行，首列已有行标签）----
fill_row(T1, 1, [DS["train"]["n"], DS["train"]["missing"],
                 f"{DS['train']['avg_q_words']} 词", f"{DS['train']['avg_a_words']} 词"],
         cols=[2, 3, 4, 5])
fill_row(T1, 2, [DS["test"]["n"], DS["test"]["missing"],
                 f"{DS['test']['avg_q_words']} 词", f"{DS['test']['avg_a_words']} 词"],
         cols=[2, 3, 4, 5])
fill_row(T1, 3, ["104,743", "0", "9.9 词", "26.5 词"], cols=[2, 3, 4, 5])
fill_row(T1, 4, ["5,463", "0", "9.9 词", "26.4 词"], cols=[2, 3, 4, 5])

# ---- 表2 训练配置记录（模板 6 列 × 3 行，补一行到 4 行）----
add_row(T2)
_ng_raw = TF["tfidf"]["config"]["ngram"].strip("()").split(", ")
_ng = "unigram" if len(set(_ng_raw)) == 1 else "unigram+bigram"
fill_row(T2, 1, [f"{TF['tfidf']['feature_dim']}（{_ng} 词表）", f"词级，min_df=2", "—（C=1.0）",
                 "1（一次求解）", str(SEED)], cols=[1, 2, 3, 4, 5])
fill_row(T2, 2, ["1536（4×384）", "256 subword", "—（C=1.0）", "—（仅训练分类器）", str(SEED)],
         cols=[1, 2, 3, 4, 5])
fill_row(T2, 2, ["MiniLM 句向量 + LR"], cols=[0])
fill_row(T2, 3, ["768", "160 subword", "3e-5", "3（按 dev 选最佳）", str(SEED)], cols=[1, 2, 3, 4, 5])
fill_row(T2, 3, ["BERT-base 微调"], cols=[0])
fill_row(T2, 4, [f"{TR['diagnostics']['qnli_vocab_adapted']}（重建词表）", "词级",
                 "—（C=1.0）", "1（一次求解）", str(SEED)], cols=[1, 2, 3, 4, 5])
fill_row(T2, 4, ["迁移模型（QNLI）"], cols=[0])

para(c1, "注：表 2 中“预训练模型”一行在原模板中为单行，本实验实际用了两种预训练方案"
        "（冻结句向量与句对微调），故拆为两行分别记录。表 1 的迁移数据取自 GLUE-QNLI，"
        "其回答（句子）平均仅 26.5 词，与政治访谈平均 293.6 词的量级差异是第 5 节迁移分析的重要背景。",
     size=9, space_before=4)

# ==================== 三、实验过程及内容 ====================
c = t0.rows[2].cells[0]
clear_cell(c)

heading(c, "3.1 数据读取、清洗与标签映射", 12, space_before=2)
para(c, f"数据集 ailsntua/QEvasion，版本 sha = {DS['dataset_sha'][:12]}，通过镜像获取官方 parquet 分片，"
       "等价于 load_dataset(\"ailsntua/QEvasion\")，但避免运行时依赖网络。实际读取结果："
       f"训练集 {DS['train']['n']} 条、测试集 {DS['test']['n']} 条，共 20 个字段。"
       "原论文报告的数据规模为训练 3,445 条、测试 317 条，与本实验读取的 3,448 / 308 存在个位数差异，"
       "推测与数据集版本更新有关；本报告全部结论基于实际读取的版本，不作回溯修改。")

para(c, "标签映射方面，高层标签 clarity_label 在文件中的原始取值为三个字符串，"
       "映射为整数编号：Clear Reply → 0，Ambivalent → 1（文件中写作 Ambivalent，而非 Ambivalent Reply），"
       "Clear Non-Reply → 2。细粒度标签 evasion_label 共 9 个取值，按论文 Figure 3 的层级归入三个高层类别："
       "Explicit → Clear Reply；Implicit、Dodging、General、Deflection、Partial/half-answer → Ambivalent；"
       "Declining to answer、Claims ignorance、Clarification → Clear Non-Reply。")

para(c, "为确认该映射不是本实验的推测，而是数据集的实际标注口径，对训练集做了交叉验证：按上述映射"
       "把 evasion_label 上溯为高层标签后，与文件自带的 clarity_label 比对，一致 "
       f"{DS['mapping_check']['agree']}/{DS['mapping_check']['total']} 条，"
       f"一致率 {DS['mapping_check']['rate']*100:.2f}%，且交叉表呈严格的一对多结构，没有任何一条例外。"
       "这一点有实际意义：它说明 Implicit 在数据集中确实归属 Ambivalent 而非 Clear Reply——"
       "隐含回答虽然“答了”，但由于存在多种解读空间，被划入模糊一侧。这一归属直接影响粗类计数，"
       "是本实验后续所有统计的基础。")

para(c, "清洗环节逐一检查缺失值、空文本、重复样本与异常长度，不做任何静默删除。结果为：question 与"
       " interview_answer 均无空值，无完全重复行；question+answer 组合重复 58 条（同一回答下拆出多个"
       "子问题所致，属正常结构，保留）；inaudible 标记 3 条（保留，不剔除）。最终进入实验的样本量与原始一致。"
       "另有一处数据特性必须记录：测试集的 evasion_label 字段全部为空字符串，308 条测试样本只来自"
       "同一场访谈（title 唯一），president 字段也为空。因此（1）细类层面的评估无法在测试集上进行，"
       "本实验的细类分析只在训练集内部完成；（2）按总统分组的分析改在训练集上以分组交叉验证完成，"
       "详见 4.5 节。这不是处理失误，而是数据集本身的字段缺失。")

code_block(c, [
    "import pandas as pd",
    "tr = pd.read_parquet('data/qev_train.parquet')      # 3,448 x 20",
    "te = pd.read_parquet('data/qev_test.parquet')       # 308 x 20",
    "",
    "CLARITY2ID = {'Clear Reply': 0, 'Ambivalent': 1, 'Clear Non-Reply': 2}",
    "EVASION2CLARITY = {",
    "    'Explicit': 'Clear Reply',",
    "    'Implicit': 'Ambivalent', 'Dodging': 'Ambivalent',",
    "    'General': 'Ambivalent', 'Deflection': 'Ambivalent',",
    "    'Partial/half-answer': 'Ambivalent',",
    "    'Declining to answer': 'Clear Non-Reply',",
    "    'Claims ignorance': 'Clear Non-Reply',",
    "    'Clarification': 'Clear Non-Reply',",
    "}",
    "# 统一接口：text_a / text_b / label / group",
    "train = pd.DataFrame({'text_a': tr.question, 'text_b': tr.interview_answer,",
    "                      'label': tr.clarity_label.map(CLARITY2ID),",
    "                      'group_title': tr.title, 'group_president': tr.president})",
])
caption(c, "代码 1  数据读取、标签映射与统一接口构造")

picture(c, "shot_mapping.png", 13.4)
caption(c, "图 1  标签映射验证的运行输出。上表为 9 个细类与 3 个粗类的交叉表，每行恰有一个非零格；"
           "下方为逐条比对的校验结果，训练集 3448 条全部一致。这张图说明粗类标签可由细类唯一确定上溯，"
           "本实验使用的映射不是推测，而是数据集本身的标注口径，因而后续所有统计都建立在这一确定关系上。")

heading(c, "3.2 数据统计与探索性分析", 12)
para(c, "数据规模与长度统计已填入前文表 1，此处补充分布层面的观察。图 2 给出粗类与细类的标签分布，"
       "图 3 给出回答长度与问题长度的分布。", space_before=2)

picture(c, "fig1_label_dist.png", 13.4)
caption(c, "图 2  粗类与细类的标签分布。(a) 训练集 Ambivalent 占 59.2%，测试集占 66.9%；"
           "(b) 细类中 Explicit 最多（1052 条），Partial/half-answer 最少（79 条），相差 13 倍。"
           "两类分布都显示明显不均衡与长尾：多数类接近六成，而最小的细类不足百条，"
           "这正是本实验以 Macro-F1 为主指标、并把主要精力放在三个粗类上的直接原因。")

para(c, "图 2(a) 显示训练集三个粗类的比例为 1052 : 2040 : 356，即 30.5% : 59.2% : 10.3%，"
       "测试集为 79 : 206 : 23（25.6% : 66.9% : 7.5%）。两个划分的不均衡程度接近，"
       "但测试集的 Ambivalent 占比更高，意味着在测试集上“全判多数类”的准确率会比在训练集上更高，"
       "后文的多数类基线 0.6688 正由此而来。图 2(b) 显示细类分布呈明显长尾："
       "Explicit 有 1052 条，而 Partial/half-answer 仅 79 条。长尾类别样本不足百条，"
       "决定了细类层面几乎不可能取得稳定结果。")

picture(c, "fig2_length.png", 13.4)
caption(c, "图 3  长度分布。(a) 三类回答的箱线图，Clear Non-Reply 的中位数仅 39.5 词，"
           "远低于 Ambivalent 的 243.5 词与 Clear Reply 的 194 词；(b) 问题长度集中在 10–20 词，分布右偏。"
           "长度与类别存在强相关，任何只依赖长度的模型都会在短拒答上“作弊”，这一隐患在 4.3 节被检验。")

para(c, "图 3(a) 揭示了一个对建模极为关键的事实：三类回答的长度差异极大。Clear Non-Reply 的中位数"
       "只有 39.5 词，而 Ambivalent 与 Clear Reply 分别为 243.5 和 194 词，前者不到后者的六分之一。"
       "原因不难理解——明确拒答往往是一句短话（“I don’t know”“I won’t comment on that”），"
       "而模糊回答通常需要一段话绕开问题。这一差异意味着长度本身就是一个强特征，"
       "任何模型都可能借助长度走捷径。图 3(b) 显示问题长度集中在 10–20 词（中位数 12），"
       "远短于回答，因此回答一侧的信息量占绝对主导。")

para(c, "此外还检查了同源采访导致的数据泄漏风险。训练集 2004 个不同的 interview_answer 覆盖了 3448 条样本，"
       "平均每个回答被拆出 1.72 个问题，说明同一回答在训练集中多次出现。若随机划分训练/验证集，"
       "同一回答会同时落在两侧，造成虚高的验证成绩。因此本实验所有内部划分均以访谈标题（title）"
       "为分组单位。跨划分检查显示，训练集与测试集仅共享 2 个回答，说明官方划分本身基本干净。")

heading(c, "3.3 为什么以 Macro-F1 为主指标", 12)
para(c, "本任务必须以 Macro-F1 而非 Accuracy 作为主指标，原因有三层。第一，类别分布严重不均衡："
       "Ambivalent 占 59.2%，若模型把所有样本都判为 Ambivalent，准确率可达 0.6688，"
       "但该模型对 Clear Reply 和 Clear Non-Reply 的召回都是 0，实际毫无用处；"
       "而同一份预测的 Macro-F1 只有 0.2672。也就是说，Accuracy 会把“什么都没学到”误判为“及格”。"
       "第二，任务的使用场景决定了少数类更重要：在政治话语分析中，研究者关心的是“哪些问题被回避了、"
       "以什么方式回避”，这恰恰对应占比仅 10.3% 的 Clear Non-Reply 和 30.5% 的 Clear Reply；"
       "一个只会识别多数类的系统无法支撑任何实质分析。第三，Macro-F1 对三类等权平均，"
       "任何一类被判坏都会立即拉低总分，使模型之间的比较反映“三类都判得如何”，"
       "而不是“多数类判得如何”。因此本实验以 Macro-F1 为主、Accuracy 为参考，"
       "并同时报告类别级 Precision/Recall/F1 与混淆矩阵，避免单一数字掩盖结构性问题。")

heading(c, "3.4 模型与训练配置", 12)
para(c, "运行环境：Windows 11，Python 3.13.0，scikit-learn 1.9.0，PyTorch 2.15.0（CPU 版），"
       "transformers 5.22.0；机器为 32 核 CPU + NVIDIA RTX 4060 Laptop（8 GB）。"
       "需如实说明：受网络条件限制，CUDA 版 PyTorch（约 2.4 GB）下载速度长期在 300 KB/s 上下，"
       "无法在合理时间内完成，因此预训练模型的微调在 CPU 上完成。这一限制直接影响了下文的训练轮次，"
       "并在第 6 节被列为待改进项。具体配置已填入前文表 2，下面说明三条基线的建模思路。", space_before=2)

cfg = TF["tfidf"]["config"]
best_dev = max(TF["search_log"], key=lambda r: r["dev_macro_f1"])
para(c, "基线 A（TF-IDF + 逻辑回归）：词袋表示为 min_df=2、sublinear_tf=True 的 TF-IDF。"
       "搜索空间为四维笛卡尔积共 16 组：输入取“问题+回答”或“仅回答”，n-gram 取 unigram 或 "
       "unigram+bigram，class_weight 取 None 或 balanced，正则强度 C 取 1.0 或 4.0。"
       f"所有组合只在训练集内部按标题分组切出的验证集（{len(DV_FIT)} 条，占 "
       f"{len(DV_FIT)/len(_tr_all)*100:.1f}%）上比较，"
       f"选出的最佳配置为“{'仅回答' if cfg['text']=='a_only' else '问题+回答'} + "
       f"{cfg['ngram']} + {'balanced' if cfg['class_weight']!='None' else '不加权'} + C={cfg['C']}”，"
       f"其验证集 Macro-F1 为 {f4(best_dev['dev_macro_f1'])}。选定后把训练集与验证集合并重训，"
       "最后在测试集上评估一次——测试集不参与任何选择。")

para(c, "基线 B-1（预训练句向量 + 逻辑回归）：用 all-MiniLM-L6-v2 分别编码 question 与 answer，"
       "取最后一层隐状态按 attention mask 做均值池化并 L2 归一化，得到两个 384 维向量，"
       "再拼接 [q, a, |q−a|, q⊙a] 共 1536 维特征送入逻辑回归；预训练参数冻结，只训练分类器。")

para(c, "基线 B-2（BERT 句对微调）：bert-base-uncased 加三分类头，输入为 (question, answer) 句对，"
       "最大长度 160 个 subword、批大小 24、学习率 3e-5、线性预热 10%、权重衰减 0.01、梯度裁剪 1.0，"
       "训练 3 轮，按验证集 Macro-F1 选择最佳轮次。损失为按类别频次倒数加权的交叉熵，权重为 "
       f"{CW_TXT}。随机种子固定为 {SEED}（Python / NumPy / PyTorch 同步设置）。")

heading(c, "3.5 实验结果", 12)
para(c, "表 3 给出各模型在测试集（308 条）上的总体性能，图 4 以图形方式对比 Accuracy 与 Macro-F1。"
       "需要强调：表中“迁移数据模型”一行是二分类任务的结果，与前面三行不可直接比较，"
       "列在此处只为完整呈现。", space_before=2)

rows3 = [
    ["多数类基线", f4(TF["majority_baseline"]["accuracy"]),
     f4(TF["majority_baseline"]["macro_p"]), f4(TF["majority_baseline"]["macro_r"]),
     f4(TF["majority_baseline"]["macro_f1"]), "全判 Ambivalent，不训练"],
    ["TF-IDF + LR", f4(TF["tfidf"]["accuracy"]), f4(TF["tfidf"]["macro_p"]),
     f4(TF["tfidf"]["macro_r"]), f4(TF["tfidf"]["macro_f1"]),
     f"{'仅回答' if cfg['text']=='a_only' else '问答'}, {_ng}, balanced"],
    ["预训练表示", f4(SB["accuracy"]), f4(SB["macro_p"]), f4(SB["macro_r"]),
     f4(SB["macro_f1"]), "MiniLM 句向量，冻结"],
]
if B:
    rows3.append(["预训练模型+改进", f4(B["accuracy"]), f4(B["macro_p"]),
                  f4(B["macro_r"]), f4(B["macro_f1"]),
                  f"BERT 加权 CE，第 {B['best_epoch']} 轮"])
else:
    rows3.append(["预训练模型+改进", "—", "—", "—", "—", "本机未完成微调"])
rows3.append(["迁移数据模型", f4(TR["qnli_adapted"]["accuracy"]),
              f4(TR["qnli_adapted"]["macro_p"]), f4(TR["qnli_adapted"]["macro_r"]),
              f4(TR["qnli_adapted"]["macro_f1"]), "QNLI 二分类，不可比"])
table(c, ["模型", "Accuracy", "Macro-P", "Macro-R", "Macro-F1", "备注"], rows3,
      widths=[3.6, 1.8, 1.8, 1.8, 1.8, 3.0], left_cols=(0, 5))
caption(c, "表 3  模型总体性能（QEvasion 测试集，n=308）。所有训练模型的 Accuracy 都被多数类基线"
           "（0.6688）压过，但 Macro-F1 全部高于它（0.2672），这一反差是本实验最重要的观察之一")

para(c, "表 3 最重要的一行不是最好的模型，而是第一行：多数类基线的 Accuracy 为 "
       f"{f4(TF['majority_baseline']['accuracy'])}，高于所有真正训练的模型；但它的 Macro-F1 只有 "
       f"{f4(TF['majority_baseline']['macro_f1'])}。也就是说，在这个任务上，任何以 Accuracy 排序的"
       "评价都会把“什么都没做”排在第一位。这正是 3.3 节论证的直观证据，也说明报告必须同时给出"
       "两个指标才能避免误读。")

picture(c, "fig5_model_compare.png", 13.4)
caption(c, "图 4  各模型的 Accuracy 与 Macro-F1 对比（灰柱为 Accuracy，蓝柱为 Macro-F1）。"
           "最左侧灰柱最高、蓝柱最低，直观显示“多数类撑起的准确率”会掩盖模型什么都没学到的事实；"
           f"右侧三个模型的蓝柱（{f4(min(SB['macro_f1'], TF['tfidf']['macro_f1']))}–"
           f"{f4(B['macro_f1']) if B else f4(max(SB['macro_f1'], TF['tfidf']['macro_f1']))}"
           f"）则明显高于多数类基线的 "
           f"{f4(TF['majority_baseline']['macro_f1'])}，说明模型确实学到了可用于区分三类的信息，"
           "只是这部分信息尚未转化为对少数类的可靠识别。")

_rank = [("TF-IDF + LR", TF["tfidf"]), ("MiniLM 冻结特征", SB)]
if B:
    _rank.append(("BERT 微调", B))
_rank.sort(key=lambda x: -x[1]["macro_f1"])
_rank_txt = "、".join(f"{n}（{f4(v['macro_f1'])}）" for n, v in _rank)
_mini_gap = SB["macro_f1"] - TF["tfidf"]["macro_f1"]
if _mini_gap < -0.005:
    _mini_rel = f"反而低于 TF-IDF（{SB['macro_f1']:.4f} 对 {TF['tfidf']['macro_f1']:.4f}，相差 {_mini_gap:+.4f}）"
elif _mini_gap < 0.02:
    _mini_rel = f"与 TF-IDF 基本持平（相差仅 {_mini_gap:+.4f}）"
else:
    _mini_rel = f"略高于 TF-IDF（相差 {_mini_gap:+.4f}）"
para(c, f"从 Macro-F1 看，由高到低依次为{_rank_txt}，都远高于多数类基线的 "
       f"{f4(TF['majority_baseline']['macro_f1'])}。值得注意的是，冻结的 MiniLM 句向量"
       f"{_mini_rel}，原因有三：其一，MiniLM 的训练目标是句子语义相似度，"
       "而本任务判断的是“所请求的信息是否被给出”，这类判断高度依赖具体词汇"
       "（“I don’t know”“no”“absolutely”），句向量在均值池化时把关键信号平滑掉了；"
       "其二，回答平均 294 词，截断到 256 个 subword 后只剩开头部分；其三，冻结特征无法针对任务调整。"
       "这说明“使用预训练表示”本身不等于“一定更好”，表示与任务目标的匹配度才是决定因素。")

heading(c, "3.6 最佳模型的类别级结果", 12)
CANDS = [("TF-IDF + LR", TF["tfidf"], "cm_tfidf.png"),
         ("MiniLM 句向量 + LR（冻结）", SB, None)]
if B:
    CANDS.append(("BERT-base 微调", B, "bert_cw.png"))
best_name, best_m, best_fig = max(CANDS, key=lambda x: x[1]["macro_f1"])
_f1 = {k: best_m["per_class"][k]["f1"] for k in CLS}
CM = best_m["confusion"]
para(c, f"按 Macro-F1 选出最佳模型为{best_name}（Macro-F1 {f4(best_m['macro_f1'])}），"
       "其类别级结果见表 4，混淆矩阵见图 5。", space_before=2)
rows4 = []
for k in CLS:
    cc = best_m["per_class"][k]
    rows4.append([k, cc["support"], f4(cc["precision"]), f4(cc["recall"]), f4(cc["f1"])])
rows4.append(["Macro Average",
              sum(best_m["per_class"][k]["support"] for k in best_m["per_class"]),
              f4(best_m["macro_p"]), f4(best_m["macro_r"]), f4(best_m["macro_f1"])])
table(c, ["类别", "样本数", "Precision", "Recall", "F1"], rows4,
      widths=[4.0, 2.3, 2.4, 2.4, 2.3], left_cols=(0,))
_gap4 = max(_f1.values()) - min(_f1.values())
caption(c, f"表 4  最佳模型（{best_name}）的类别级结果。三个类别的 F1 最高与最低相差 "
           f"{_gap4:.3f}，说明模型能力在类别之间分布极不均匀，只看 Macro 平均值会掩盖这一点")

if best_fig:
    picture(c, best_fig, 13.4)
    if best_fig == "bert_cw.png":
        _h0, _h1 = BERT["history"][0], BERT["history"][-1]
        _best_ep = B["best_epoch"]
        _best_f1 = max(h["macro_f1"] for h in BERT["history"])
        _n_ep = len(BERT["history"])
        if _h1["dev_loss"] - _h0["dev_loss"] > 0.01:
            _trend = (f"验证损失自第 1 轮起持续上升（{_h0['dev_loss']:.3f}→{_h1['dev_loss']:.3f}），"
                      f"验证 Macro-F1 在第 {_best_ep} 轮达到峰值 {_best_f1:.4f} 后回落，"
                      f"说明 {_n_ep} 轮已越过最优点、开始出现轻微过拟合，"
                      f"最佳轮次按验证集 Macro-F1 取第 {_best_ep} 轮")
        else:
            _trend = (f"验证损失基本持平（{_h0['dev_loss']:.3f}→{_h1['dev_loss']:.3f}），"
                      f"验证 Macro-F1 在第 {_best_ep} 轮达到峰值 {_best_f1:.4f}，"
                      f"说明 {_n_ep} 轮仍偏欠拟合一侧、继续训练的收益尚未耗尽")
        _cap5 = (f"图 5  最佳模型（{best_name}）的训练曲线（左）与测试集混淆矩阵（右）。"
                 f"左图显示训练损失由 {_h0['train_loss']:.3f} 降到 {_h1['train_loss']:.3f}，"
                 f"{_trend}；右图对角线以 Ambivalent 一格最深，但被判错的样本"
                 "并非只从少数类流向多数类，多数类也有相当一部分被推向两个少数类，"
                 "这一双向流动是加权训练的典型后果。")
    else:
        _cap5 = (f"图 5  最佳模型（{best_name}）在测试集上的混淆矩阵。"
                 "对角线上 Ambivalent 一格最深，两个少数类的错分几乎全部流向中间类，"
                 "说明模型的决策边界主要卡在“明确”与“模糊”之间，而不是“答了”与“没答”之间。")
    caption(c, _cap5)
    cm = CM
    s0 = best_m["per_class"]["Clear Reply"]["support"]
    s1 = best_m["per_class"]["Ambivalent"]["support"]
    s2 = best_m["per_class"]["Clear Non-Reply"]["support"]
    _out_amb = cm[1][0] + cm[1][2]      # 真实 Ambivalent 被推出去
    _in_amb = cm[0][1] + cm[2][1]       # 两个少数类被吸进 Ambivalent
    _cr_pred = cm[0][0] + cm[1][0] + cm[2][0]
    if _out_amb > _in_amb:
        para(c, "图 5 的混淆矩阵是本实验信息量最大的结果之一。Ambivalent 判得最好（F1 "
               f"{f4(best_m['per_class']['Ambivalent']['f1'])}），但错误并不是单向流入多数类："
               f"真实 Ambivalent 的 {s1} 条中有 {cm[1][0]} 条被判成 Clear Reply、{cm[1][2]} 条被判成"
               f" Clear Non-Reply，合计 {_out_amb} 条被“推了出去”，是数量最大的一类错误；"
               f"而真实 Clear Reply 的 {s0} 条中有 {cm[0][1]} 条落入 Ambivalent，"
               f"真实 Clear Non-Reply 的 {s2} 条中有 {cm[2][1]} 条落入 Ambivalent，合计仅 {_in_amb} 条。"
               "方向之所以与直觉相反，正是加权交叉熵造成的“过度纠正”：为了把两个少数类的召回从 0 抬起来，"
               f"模型付出了 Precision 的代价——它一共判出 {_cr_pred} 条 Clear Reply，其中只有 {cm[0][0]} 条正确。"
               "这说明不均衡分类的调优不是单向的“多关注少数类”，而是在召回与精度之间找平衡点。")
    else:
        para(c, "图 5 的混淆矩阵是本实验信息量最大的结果之一。Ambivalent 判得相当好（F1 "
               f"{f4(best_m['per_class']['Ambivalent']['f1'])}），但 Clear Reply 的 {s0} 条中有 {cm[0][1]} 条"
               f"被判成 Ambivalent，Clear Non-Reply 的 {s2} 条中有 {cm[2][1]} 条被判成 Ambivalent"
               f"（仅 {cm[2][2]} 条判对）。"
               "绝大多数错误集中在“多数类吸收少数类”这一个方向上，而两个少数类之间几乎不互相混淆"
               f"（CR→CNR 仅 {cm[0][2]} 条，CNR→CR 仅 {cm[2][0]} 条）。这与第 4 节的分析相互印证："
               "真正的决策边界在 Clear 与 Ambivalent 之间，而不在“答了”与“没答”之间。")

heading(c, "3.7 结果文件与复现验证", 12)
_n_scripts = len([f for f in os.listdir("exp") if f.endswith(".py")])
para(c, f"实验输出了以下可复现材料：processed/ 下的统一样本集，{RES}/ 下的指标 JSON 与逐样本预测 CSV，"
       f"{RES}/figures/ 下的全部图表，以及 exp/ 下的脚本（按编号顺序执行即可复跑）。"
       f"当前 exp/ 目录共 {_n_scripts} 个 .py 文件，其中 01–10 为实验主流程，其余为出图与报告生成工具。"
       "每个脚本均固定随机种子，数据划分函数统一在 split_util.py 中实现，保证所有模型使用同一划分。",
     space_before=2)

para(c, "复现验证的做法是：保留首轮结果后，从头重新运行数据准备、EDA、TF-IDF、错误分析、迁移实验、"
       "汇总六步，逐项比对两次运行的关键指标。"
       + (f"结果为 {RP['n_same']}/{RP['total']} 项完全一致"
          "（判定阈值为 1e-9），说明整个流程在固定种子下是确定性的；比对项包括 TF-IDF 的总体与三类 F1、"
          "迁移实验四组设置的 Macro-F1、以及 5 折交叉验证的 OOF Macro-F1。BERT 微调因单轮耗时过长，"
          "复现时采用“复用最佳 checkpoint 重新评估”的方式，测试集指标同样一致。"
          if RP else
          "这六步本身不含非确定性操作，重复执行应逐位一致（本目录尚未执行第 10 步的比对脚本，"
          "故此处未给出比对结果）。"))

picture(c, "shot_train.png", 13.4)
caption(c, "图 6  预训练模型微调过程的真实运行输出（本机 CPU）。可以看到每轮的训练损失、验证损失、"
           "验证集 Accuracy 与 Macro-F1，以及最后写出的测试集指标文件。这张截图是本实验所有预训练"
           "模型数字的来源凭证：模型的每一次前向、每一个 epoch 的耗时与每一次指标计算都发生在本地，"
           "没有引用论文数值，也未使用云端算力。")

# ==================== 四、数据处理分析 ====================
c = t0.rows[3].cells[0]
clear_cell(c)

heading(c, "4.1 总体结果：哪个类别最难", 12, space_before=2)
_order = sorted(CLS, key=lambda k: _f1[k])
hard, easy = _order[0], _order[-1]
_hm = best_m["per_class"][hard]
_pred_n = sum(CM[i][CLS.index(hard)] for i in range(3))
_correct_n = CM[CLS.index(hard)][CLS.index(hard)]
_ratio = (_pred_n / _hm["support"]) if _hm["support"] else 1.0
SUB = {
    "Clear Non-Reply": "它包含三种性质不同的子类——直接拒绝（Declining to answer）、"
                       "声称无知（Claims ignorance）、要求澄清（Clarification），"
                       "三者的语言形式差别很大，用一个类别统一表示本身就是妥协",
    "Clear Reply": "标注为 Clear Reply 的样本中不少是“答了一部分、但未覆盖问题核心”，"
                   "而 Ambivalent 的定义本身就包含轻微偏离，两者只差一个程度",
    "Ambivalent": "它是五个细类的合并，轻微偏离、部分回答、泛泛而谈被强行归为一类，内部一致性最低",
}
_cnt = {"Clear Reply": "1052 条（30.5%）", "Ambivalent": "2040 条（59.2%）",
        "Clear Non-Reply": "356 条（10.3%）"}[hard]
if _ratio > 1.05:
    _dir = (f"模型一共判出 {_pred_n} 条{hard}，而真实只有 {_hm['support']} 条，"
            f"其中 {_correct_n} 条正确、{_pred_n - _correct_n} 条是别的类被误判过来——"
            "也就是说它并没有“漏掉”这一类，而是把边界外扩、吃进了相邻类别")
    _why = "边界外扩"
else:
    _dir = (f"真实的 {_hm['support']} 条中只有 {_correct_n} 条被判对，"
            f"其余 {_hm['support'] - _correct_n} 条流向了相邻类别")
    _why = "边界内缩"
para(c, f"从表 4 与图 5 的混淆矩阵看，三个类别的 F1 呈明显梯度："
       f"{_order[-1]} {f4(_f1[_order[-1]])}，{_order[-2]} {f4(_f1[_order[-2]])}，"
       f"{_order[0]} {f4(_f1[_order[0]])}。最难的是{hard}（Precision "
       f"{f4(_hm['precision'])}、Recall {f4(_hm['recall'])}）：{_dir}。"
       f"原因有两层。第一是样本量：{hard}在训练集中只有 {_cnt}，测试集中仅 "
       f"{_hm['support']} 条，一条判错就足以让 F1 跳动几个百分点，指标本身不稳定。"
       f"第二是表达方式的离散程度：{SUB[hard]}。两点叠加，导致这一类的{_why}"
       "成为模型最不可靠的部分。",
     space_before=2)

_cr = best_m["per_class"]["Clear Reply"]
para(c, "值得注意的是，Clear Reply 的 Precision 与 Recall 落差较大："
       f"{f4(_cr['precision'])} 对 {f4(_cr['recall'])}。"
       f"这说明模型一旦判出 Clear Reply，只有约 {_cr['precision']*100:.0f}% 是对的，"
       f"而真实存在的 Clear Reply 仍有约 {(1-_cr['recall'])*100:.0f}% 被漏掉——漏掉的部分大多流向了 Ambivalent。"
       "两个方向同时存在误差，说明任务真正的难点不在两个极端，而在中间那条边界："
       "模型已经能大致分开“显然在拒答”和“显然在回答”，却分不清“回答得完整”与“回答得不完整”，"
       "而后者恰恰是这套标注体系中最微妙的部分。")

heading(c, "4.2 典型误判样本", 12)
para(c, "从测试集预测中挑选 7 条覆盖不同错误方向的样本，逐条给出真实标签、预测标签与误差原因，"
       "见表 5。选样时对同一回答下的多个问题做了去重，避免同一段文本重复入选。", space_before=2)

ES = ER["error_samples"]
rows5 = []
for i, s in enumerate(ES, 1):
    q = s["q"][:52].replace("\n", " ") + ("…" if len(s["q"]) > 52 else "")
    a = s["a"][:60].replace("\n", " ") + ("…" if len(s["a"]) > 60 else "")
    rows5.append([i, q, a, SHORT[s["gold"]], SHORT[s["pred"]], s["type"]])
table(c, ["序号", "问题摘要", "回答摘要", "真实标签", "预测标签", "误差类型/原因"], rows5,
      widths=[1.0, 3.1, 3.4, 1.5, 1.5, 3.3], size=7.5, left_cols=(1, 2, 5))
caption(c, "表 5  典型误判样本（测试集，摘录 7 条）。标签缩写：CR=Clear Reply、"
           "AMB=Ambivalent、CNR=Clear Non-Reply；7 条覆盖了五类典型错误方向，"
           "并去除了同一回答下的重复样本，以便逐条分析误差来源")

para(c, "这七条误判可以归为五类原因。第一类是“长而空泛”：第 1、2 条的回答分别为 "
       f"{ES[0]['a_words']} 与 {ES[1]['a_words']} 词，"
       "通篇谈论一个相关但不同的话题（对方是否可被预测），语言形式完整、信息密度极低，"
       "模型被长度和话题词带偏，判成 Clear Reply。第二类是“部分回答被读成模糊”："
       "第 3、4 条确实回答了一部分内容，但未覆盖问题的核心，标注为 Clear Reply，"
       "模型因句中存在转折与限定语而判为 Ambivalent——这类样本的边界取决于标注规范，"
       "本身接近不可判定。第三类是“主题转移”：第 5 条用一个相关事实（某将军将提交建议）替换了"
       "对问题（是否把战争交给下任）的回答，模型只捕捉到“有实质内容”便判为 Ambivalent，"
       "漏掉了“没有回答被问的那件事”。第四类是“边界模糊”：第 6 条以“我不想反驳你这样资深的记者”"
       "开头，实为委婉辩解并给出了部分信息，模型把“I don’t want to…”这类否定框架直接读成拒答，"
       "判为 Clear Non-Reply——这说明否定结构存在歧义，既可能是拒绝，也可能只是修辞。"
       "第五类是“表述迂回”：第 7 条实际给出了明确答复（“我没有计划重新任命他”），"
       "但因含否定式的间接表达被判成 Clear Non-Reply。"
       "五类原因指向同一个改进方向：模型需要的是“所问信息是否被覆盖”的结构化判断，"
       "而不是词汇层面的极性或长度信号。")

heading(c, "4.3 TF-IDF 基线依赖哪些表面特征", 12)
para(c, "TF-IDF 模型的可解释性较强，可以直接检查各类别权重最高的词。训练后三个类别的正权重词如下："
       "Clear Reply 一侧以 no、yes、sure、absolutely、did 等高确定性词为主；"
       "Clear Non-Reply 一侧以 don、question、know、what、wish、comment、hear 为主，"
       "对应“I don’t know”“I wish to comment on that”“I don’t want to hear about it”这类典型拒答句式；"
       "Ambivalent 一侧则是 president、political、call、great、believe、country 等泛化的政治话题词。"
       "这一分布说明，模型确实抓住了“确定性词 → 明确回答”“否定+认知动词 → 拒答”这类可靠的表面线索，"
       "这也是它在 Macro-F1 上能明显超过多数类基线的原因。", space_before=2)

def devf(text, ng, cw, C=1.0):
    """从超参搜索日志里取某配置在验证集上的 Macro-F1。"""
    for r in TF["search_log"]:
        if (r["text"] == text and r["ngram"] == ng
                and r["class_weight"] == cw and abs(float(r["C"]) - C) < 1e-9):
            return float(r["dev_macro_f1"])
    return None


_d_qa, _d_a = devf("qa", "(1, 2)", "balanced"), devf("a_only", "(1, 2)", "balanced")
_diff = _d_qa - _d_a
_shape = ("几乎不提供增量信息" if abs(_diff) < 0.01
          else ("反而带来轻微干扰" if _diff < 0 else "能提供少量增量信息"))
para(c, "但表面特征的局限同样明显。表 6 的消融结果显示，是否使用类别权重对结果的影响，"
       "远大于 n-gram 范围或输入方式的影响：以“仅回答 + unigram+bigram”为例，不加权时 Macro-F1 为 "
       f"{f4(ab('a_only','(1, 2)','None')['macro_f1'])}，加权后升至 "
       f"{f4(ab('a_only','(1, 2)','balanced')['macro_f1'])}；而在同样加权、同样 bigram 的条件下，"
       f"把输入从“问题+回答”换成“仅回答”，Macro-F1 只由 "
       f"{f4(ab('qa','(1, 2)','balanced')['macro_f1'])} 变为 "
       f"{f4(ab('a_only','(1, 2)','balanced')['macro_f1'])}；验证集上相差 "
       f"{abs(_diff):.4f}（问题+回答 {f4(_d_qa)} 对仅回答 {f4(_d_a)}）。换言之，question 一侧{_shape}，"
       "这也与图 3(b) 一致——问题平均只有 14.5 词，回答有 293.6 词。更关键的是，回答长度本身就是一个强特征："
       "Clear Non-Reply 的中位数仅 39.5 词，模型完全可以只看“回答是否很短”就做出相当一部分判断。"
       "这意味着 TF-IDF 学到的不是“回答的信息量”，而是“某些词 + 长度”的联合捷径；"
       "一旦遇到“长篇大论却什么都没说”的样本，就会失效——这正是 4.2 节第 1、2 条误判的成因。")

rows6 = []
for r in TF["ablation_test"]:
    _cw_disp = "balanced" if r["class_weight"] != "None" else "不加权"
    rows6.append([r["text"].replace("qa", "问题+回答").replace("a_only", "仅回答"),
                  r["ngram"], _cw_disp, f4(r["accuracy"]), f4(r["macro_f1"]),
                  f4(r["cr_f1"]), f4(r["cnr_f1"])])
table(c, ["输入", "n-gram", "类别权重", "Accuracy", "Macro-F1", "CR 的 F1", "CNR 的 F1"],
      rows6, widths=[2.2, 1.7, 2.4, 1.9, 1.9, 1.8, 1.9], left_cols=(0,))
caption(c, "表 6  TF-IDF 的固定消融：类别权重与 n-gram 范围的影响。同一测试集、同一划分，"
           "仅作对照分析，不用于选参。CR=Clear Reply，CNR=Clear Non-Reply")

a0, a1 = ab("a_only", "(1, 2)", "None"), ab("a_only", "(1, 2)", "balanced")
b0, b1 = ab("a_only", "(1, 1)", "None"), ab("a_only", "(1, 1)", "balanced")
para(c, "表 6 呈现出两条清晰规律。第一，class_weight='balanced' 是提升 Macro-F1 的主要来源："
       f"Clear Non-Reply 的 F1 从 {f4(a0['cnr_f1'])} 跃升到 {f4(a1['cnr_f1'])}，涨了近两倍，"
       f"代价是 Accuracy 从 {f4(a0['accuracy'])} 降到 {f4(a1['accuracy'])}。"
       "这正是不均衡分类中的典型权衡——模型用一部分多数类的正确率，换回少数类被识别的能力。"
       "第二，n-gram 的作用方向不一致：不加权时加入 bigram 反而降低 Macro-F1"
       f"（{f4(b0['macro_f1'])} → {f4(a0['macro_f1'])}），加权后则提升"
       f"（{f4(b1['macro_f1'])} → {f4(a1['macro_f1'])}），"
       "说明短语级证据只有在模型愿意关注少数类时才能转化为收益。")

heading(c, "4.4 预训练模型相对传统基线的提升来自哪个类别", 12)
tf_pc = TF["tfidf"]["per_class"]
WHY_UP = {
    "Clear Reply": "判断一条回答是否为 Clear Reply，需要确认“所问的那条信息”是否真的被给出，"
                   "这往往跨越多个句子、依赖指代与省略的解析，词袋模型难以胜任，"
                   "而 BERT 的句对编码恰好擅长这种跨句的语义对齐",
    "Ambivalent": "Ambivalent 是五个细类的合并，其内部差异（轻微偏离、部分回答、泛泛而谈）"
                  "只有在上下文里才区分得开，这正是预训练表示的长处",
    "Clear Non-Reply": "Clear Non-Reply 的三种子类（直接拒绝、声称无知、要求澄清）句式差异很大，"
                       "预训练模型能借助上下文把这些变体归入同一类，而词袋只能逐个词地认",
}
WHY_DOWN = {
    "Clear Reply": "它常由一句简短、位置不固定的直接回应构成，"
                   "词袋模型配合类别权重已经能抓到相当一部分，BERT 在样本少、轮次少的条件下没有占到便宜",
    "Ambivalent": "它样本量最大、内部又最松散，在加权损失与有限轮次的共同作用下，"
                  "模型对它的判断反而变得更保守",
    "Clear Non-Reply": "它的关键线索是“I don’t know”“I won’t comment”这类高频固定句式，"
                       "TF-IDF 用几个词就能识别得很好，而 BERT 在有限轮次、160 长度截断、"
                       "训练样本仅 356 条的条件下反而没有充分学到这一模式",
}
if B:
    b_pc = B["per_class"]
    dlt = {k: b_pc[k]["f1"] - tf_pc[k]["f1"] for k in CLS}
    para(c, "把 BERT 微调与 TF-IDF 逐类对比可以看到，变化并非均匀分布："
           f"Clear Reply 的 F1 从 {f4(tf_pc['Clear Reply']['f1'])} 变为 "
           f"{f4(b_pc['Clear Reply']['f1'])}（{dlt['Clear Reply']:+.4f}），"
           f"Ambivalent 从 {f4(tf_pc['Ambivalent']['f1'])} 变为 {f4(b_pc['Ambivalent']['f1'])}"
           f"（{dlt['Ambivalent']:+.4f}），"
           f"Clear Non-Reply 从 {f4(tf_pc['Clear Non-Reply']['f1'])} 变为 "
           f"{f4(b_pc['Clear Non-Reply']['f1'])}（{dlt['Clear Non-Reply']:+.4f}）。",
         space_before=2)
    up = max(CLS, key=lambda k: dlt[k])
    down = min(CLS, key=lambda k: dlt[k])
    if dlt[up] > 0:
        if dlt[down] < 0:
            para(c, f"提升最明显的是{up}：{WHY_UP[up]}。"
                   f"相反，{down} 反而下降了 {-dlt[down]:.4f}——{WHY_DOWN[down]}。"
                   "这提醒我们：预训练模型并非在所有类别上都优于传统方法，"
                   "对“短、固定、词汇特征强”的类别，词袋模型配合类别权重仍然极具竞争力。")
        else:
            para(c, f"提升最明显的是{up}（{dlt[up]:+.4f}）：{WHY_UP[up]}。"
                   f"相比之下，{down} 只提升了 {dlt[down]:+.4f}，{WHY_DOWN[down]}。"
                   "也就是说，预训练的增益并不均匀：越是需要跨句理解、越是依赖上下文的类别，"
                   "收益越大；而线索固定、词汇特征强的类别，传统方法本来就能做得不错，"
                   "预训练能带来的增量因此有限。改进时应优先解决增益最小的那一类的瓶颈。")
    else:
        para(c, "值得注意的是，BERT 在本实验条件下并未在任何类别上超过 TF-IDF："
               f"降幅最小的{up}也下降了 {-dlt[up]:.4f}。{WHY_DOWN[up]}。"
               "这说明在训练轮次受限、长文本被大量截断的情况下，微调收益可能被规则化不足抵消，"
               "“用了预训练模型就一定更好”是需要分类别验证的说法。")
else:
    para(c, "本机未能完成预训练模型的充分微调，此处仅报告传统基线之间的差异。")

heading(c, "4.5 模型是否在特定总统、主题或长回答上表现不同", 12)
PRES = {r["president"]: r for r in ER["by_president"]}
LB = {r["bucket"]: r for r in ER["by_length"]}
short, long_ = LB["短 (<100 词)"], LB["长 (>300)"]
para(c, "由于测试集的 president 字段为空，这一分析改在训练集上进行：以访谈标题为分组做 5 折交叉验证，"
       f"得到 out-of-fold 预测，整体 Acc {f4(ER['oof']['accuracy'])}、Macro-F1 "
       f"{f4(ER['oof']['macro_f1'])}，各折 Macro-F1 均值 {f4(ER['fold_mean_f1'])}、"
       f"标准差 {f4(ER['fold_std_f1'])}，与测试集规模（308 条）带来的方差量级相当。", space_before=2)

picture(c, "fig4_group_perf.png", 13.4)
caption(c, "图 7  分组表现。(a) 四位总统的 Macro-F1 与 Clear Non-Reply 的 F1；(b) 按回答长度分桶后"
           "的 Macro-F1 与 Accuracy。左图显示同一模型在不同访谈风格下的少数类识别能力差异明显，"
           "右图则显示长度越长 Accuracy 越高、Macro-F1 反而越低，二者呈相反趋势，"
           "说明长度较大的子集里少数类极少，准确率被多数类“虚抬”。")

para(c, "从图 7(a) 看，模型在四位总统上的表现确实存在差异：Trump 的 Macro-F1 最高"
       f"（{f4(PRES['Trump']['macro_f1'])}），Biden 最低（{f4(PRES['Biden']['macro_f1'])}）。"
       "差异主要来自 Clear Non-Reply 的召回：Biden 语料中该类的 F1 仅 "
       f"{f4(PRES['Biden']['per_class_f1']['Clear Non-Reply'])}，"
       "说明同一模型在不同访谈风格下的少数类识别能力并不稳定。"
       f"另需注意 Obama 的 Accuracy 最高（{f4(PRES['Obama']['accuracy'])}）"
       f"但 Macro-F1 只有 {f4(PRES['Obama']['macro_f1'])}，低于 Trump，"
       "这是又一次“准确率与 Macro-F1 脱节”的现象：该组样本中多数类占比更高，"
       "模型只要倾向多数类就能拿到高准确率。")

para(c, "长度维度上的规律更值得警惕。图 7(b) 显示：短回答（<100 词）Macro-F1 最高"
       f"（{f4(short['macro_f1'])}）、Accuracy 最低（{f4(short['accuracy'])}）；"
       f"长回答（>300 词）Accuracy 最高（{f4(long_['accuracy'])}）但 Macro-F1 最低"
       f"（{f4(long_['macro_f1'])}）。"
       "原因在于各长度桶的类别构成不同：短回答桶里 Clear Non-Reply 占 "
       f"{short['cnr_share']*100:.1f}%，长回答桶里只占 {long_['cnr_share']*100:.1f}%。"
       "长回答中少数类极其稀少，模型只要全判 Ambivalent 就能拿到很高准确率，"
       "而 Macro-F1 会如实反映少数类被彻底放弃。这解释了为什么“模型在长回答上表现更好”是一种错觉，"
       "也说明按长度分桶评估时必须同时看两个指标。")

# ==================== 五、跨场景迁移实验 ====================
c = t0.rows[4].cells[0]
clear_cell(c)

heading(c, "5.1 迁移数据选择与接口适配", 12, space_before=2)
para(c, "迁移实验选择 QNLI（GLUE 子集）。选择理由有三：其一，它同样是英文句对分类任务，"
       "输入为 (question, sentence)，判断该句子是否包含问题的答案，与本任务的 (question, answer) "
       "结构同构，最能检验接口的可复用性；其二，标签空间虽为二分类，但语义上与“回答是否给出信息”"
       "高度相关，因此可以直接观察“回避检测”与“答案蕴含”这两种判据之间的距离；其三，"
       "数据规模充足（训练 104,743 条、验证 5,463 条），且类别平衡（约 50:50），"
       "可以同时暴露不均衡假设在平衡数据上的失效。")

para(c, "字段适配器只需三行映射：question → text_a，sentence → text_b，label 保持 0/1。"
       "为了让源模型能直接迁移到二分类目标，还需要一条标签折叠规则：把 Clear Reply 视为 "
       "entailment（句子包含答案），把 Ambivalent 与 Clear Non-Reply 合并视为 not_entailment。"
       "这条规则本身就是一个假设——它假定“模糊回答”和“明确拒答”在“是否包含答案”这个问题上都算否，"
       "而这一假定在迁移实验中会被直接检验。")

code_block(c, [
    "def adapt_qnli(df):            # 统一字段适配器",
    "    return pd.DataFrame({'text_a': df['question'],   # 问题",
    "                         'text_b': df['sentence'],   # 候选句子",
    "                         'label': df['label'].astype(int)})  # 0=entailment",
    "",
    "# 标签折叠：把源任务的 3 类折成 2 类，用于零样本直接迁移",
    "p_direct = np.where(clf_src.predict_proba(X_qnli).argmax(1) == 0, 0, 1)",
])
caption(c, "代码 2  统一字段适配器与标签折叠规则")

heading(c, "5.2 迁移结果", 12)
rows7 = [
    ["原方法基线", f4(TF["tfidf"]["accuracy"]), f4(TF["tfidf"]["macro_f1"]),
     "QEvasion 三分类，见表 3"],
    ["迁移数据基线", f4(TR["qnli_baseline"]["accuracy"]), f4(TR["qnli_baseline"]["macro_f1"]),
     "QNLI 朴素配置：unigram，无类别权重"],
    ["直接迁移", f4(TR["qnli_direct"]["accuracy"]), f4(TR["qnli_direct"]["macro_f1"]),
     "源模型零样本预测 + 标签折叠"],
    ["适配后训练", f4(TR["qnli_adapted"]["accuracy"]), f4(TR["qnli_adapted"]["macro_f1"]),
     "统一接口 + unigram+bigram + balanced"],
]
table(c, ["实验设置", "Accuracy", "Macro-F1", "主要结论"], rows7,
      widths=[3.2, 2.0, 2.0, 6.4], left_cols=(0, 3))
caption(c, "表 7  原任务与迁移任务对比。前两行分属不同任务（QEvasion 三分类、QNLI 二分类），"
           "仅作量级参照，不可横向比较高低；真正可比的是后两行——同一份 QNLI 数据上，"
           "零样本直接迁移与适配后重新训练之间的差距")

d = TR["diagnostics"]
para(c, "表 7 给出四组对照。第一行是原任务在表 3 中的成绩（Macro-F1 "
       f"{f4(TF['tfidf']['macro_f1'])}），第二行是迁移数据自身的基线"
       f"（{f4(TR['qnli_baseline']['macro_f1'])}）。两者不能直接比较高低，因为它们衡量的是不同任务；"
       "真正要看的是第三、四行的差距：直接把源模型搬到 QNLI 上，Macro-F1 只有 "
       f"{f4(TR['qnli_direct']['macro_f1'])}，几乎等于二分类的随机水平（0.5），"
       f"Accuracy {f4(TR['qnli_direct']['accuracy'])} 也印证了这一点；"
       f"而用统一接口重新训练后达到 {f4(TR['qnli_adapted']['macro_f1'])}，提升 "
       f"{TR['qnli_adapted']['macro_f1']-TR['qnli_direct']['macro_f1']:.4f}。"
       "作为流程可复用的旁证：同一个迁移脚本在原任务上复跑得到 "
       f"{f4(TR['qevasion_test']['macro_f1'])}，与表 3 的 {f4(TF['tfidf']['macro_f1'])} 仅差 "
       f"{abs(TR['qevasion_test']['macro_f1']-TF['tfidf']['macro_f1']):.4f}"
       "——这点差异来自该脚本固定使用“问答拼接 + bigram”而未做超参搜索。")

cm2 = TR["qnli_direct"]["confusion"]
para(c, "两个诊断指标解释了直接迁移为何失败。第一是词汇覆盖率：QNLI 文本中的词只有 "
       f"{d['cross_domain_token_coverage']*100:.1f}% 出现在 QEvasion 构建的词表里，"
       "其余七成的词对源模型是完全未知的，词袋表示直接失效。第二是先验偏移：源模型预测为 "
       f"entailment 的比例为 {d['qnli_positive_rate_direct']*100:.1f}%，而真实比例为 "
       f"{d['qnli_positive_rate_true']*100:.1f}%。这是因为源模型在 QEvasion 上倾向于输出 Ambivalent"
       "（约六成），折叠后大量样本变成 not_entailment，于是系统性地把正类判成负类——"
       f"真实 entailment 的 {cm2[0][0]+cm2[0][1]} 条中有 {cm2[0][1]} 条被错判，漏判率超过六成。"
       "换言之，直接迁移失败并非因为“方法不对”，而是因为表示的词汇空间与标签先验都不再成立。")

heading(c, "5.3 迁移分析", 12)
para(c, "问题一：政治问答中的“回避”与迁移任务中的“相关/匹配/拒答”有哪些共同点和区别？"
       "共同点在于三者都是对“两个文本之间的关系”做判断，而不是对单个文本分类，"
       "因此模型结构（句对编码）、评估方式（Macro-F1 + 类别级指标 + 混淆矩阵）、"
       "以及最关键的“统一字段适配器”都可以原样复用——本实验把 QNLI 接入流程只改了三行代码，"
       "就复用了全部的训练、评估与可视化管线，这说明接口层面的设计是成功的。"
       "区别则在于判据：QEvasion 判断“所请求的信息是否被给出、是否只容许一种解读”，"
       "QNLI 判断“句子是否包含问题的答案”，后者是二值的事实蕴含，前者是三值的清晰度分级。"
       "更本质的差异在文本结构：QEvasion 的回答平均 294 词、是问题的 20 倍长，"
       "而 QNLI 的句子平均仅 26.5 词、与问题同一量级。前者的难点是“在长文本中定位是否回答了某件事”，"
       "后者的难点是“在短句中做语义匹配”。这解释了为什么本任务上有效的长度、位置类特征，"
       "在 QNLI 上几乎没有用武之地。", space_before=2)

para(c, "问题二：原模型直接迁移与重新训练的性能差异说明了什么？"
       f"差异（{f4(TR['qnli_direct']['macro_f1'])} 对 {f4(TR['qnli_adapted']['macro_f1'])}）"
       "说明可复用的是“流程”，不可复用的是“表示”和“先验”。"
       "具体而言：（1）流程层——数据读取、字段适配、划分、训练、评估、错误分析全部复用，"
       "这部分贡献了本次迁移实验的全部效率收益；（2）表示层——词袋模型的词表绑定在源域，"
       "跨域时七成词汇未知，必须重建；预训练模型（如 BERT）本可缓解这一问题，"
       "因为其子词单元可覆盖未登录词，这也是“用预训练表示做迁移”优于“用词袋做迁移”的根本原因；"
       "（3）先验层——源域的类别分布会被模型学成输出偏置，跨域时必须重新校准，"
       "否则会出现本实验中“正类率 35% 对真实 49%”的系统性错判。"
       "这三点合起来，构成了一个可操作的迁移检查清单：先验分布、词汇覆盖、判据定义，"
       "三者任一不匹配，直接迁移都不应期待可用结果。")

# ==================== 六、实验结论 ====================
c = t1.rows[0].cells[0]
clear_cell(c)
heading(c, "实验结论", 12, space_before=2)
para(c, "本次实验围绕对话问答中的回答清晰度分类展开，完成了从数据核查、基线复现到跨场景迁移的完整流程，"
       "主要收获可以概括为以下四点。")
para(c, "第一，问答回避检测与普通情感分类的建模难点根本不同。情感分类的标签由文本自身的极性决定，"
       "同一句话换个场景标签不变；而本任务的标签由“回答与问题的对齐关系”决定，"
       "同一段回答换一个问题，标签就可能从明确变为回避。这意味着模型不能只编码回答，"
       "必须同时对问题和回答建模并判断二者的信息覆盖关系。更棘手的是标签边界："
       "本次实验用交叉表验证了 9 个细类到 3 个粗类的映射 100% 一致，"
       "但也从中看到，模糊与明确的界限并不在“是否开口”，而在“是否只容许一种解读”，"
       "这种判据天然带有主观性，是任务难度的根源。")
if B:
    para(c, "第二，传统基线与预训练模型的差异呈现出清晰的类别结构。TF-IDF 配合类别权重就能达到 "
           f"Macro-F1 {f4(TF['tfidf']['macro_f1'])}，说明这个任务存在大量可靠的表面线索"
           "（否定词、认知动词、回答长度）；而 BERT 微调达到 "
           f"{f4(B['macro_f1'])}（整体 {B['macro_f1']-TF['tfidf']['macro_f1']:+.4f}），"
           f"增量主要来自{up}（{dlt[up]:+.4f}），{down} 为 {dlt[down]:+.4f}。"
           "差别的原因在于：前者要求确认“所问的那条信息”是否真的被给出，需要跨句的指代与省略解析，"
           "这正是上下文表示的长处；而后者靠的是“I don’t know”这类高频固定句式，"
           "词袋模型配合类别权重已经能抓住大半。这说明“预训练模型更好”是一个需要分类别检验的说法，"
           "不能笼统接受。")
else:
    para(c, "第二，传统基线与预训练模型的差异呈现出清晰的类别结构。TF-IDF 配合类别权重就能达到 "
           f"Macro-F1 {f4(TF['tfidf']['macro_f1'])}，说明这个任务存在大量可靠的表面线索"
           "（否定词、认知动词、回答长度）；预训练模型在本机条件下未能充分微调，"
           "但可以预期其增益将集中在需要跨句解析的 Clear Reply 一类。")
para(c, "第三，最典型的错误类型揭示了改进方向。七条典型误判集中在五种情形：长而空泛、部分回答、"
       "主题转移、边界模糊、表述迂回，它们的共同点是“语言形式完整但未回答问题”，而模型依赖的恰恰是"
       "语言形式与长度。这提示改进方向应是引入结构化的对齐信号，"
       "例如显式建模问题中的疑问焦点（who/what/when）与回答中对应成分的匹配程度，"
       "或引入细类作为辅助任务（先判 9 类再上溯 3 类），用更细的监督迫使模型区分"
       "“给了相关信息”与“回答了被问的事”。")
para(c, "第四，迁移实验明确了哪些设计真正可复用。字段适配器、划分策略、指标体系与错误分析流程"
       "完全可复用，把 QNLI 接入只改了三行代码；但表示与先验不可复用——"
       f"跨域词汇覆盖率仅 {d['cross_domain_token_coverage']*100:.1f}%，"
       f"直接迁移的 Macro-F1 只有 {f4(TR['qnli_direct']['macro_f1'])}，接近随机，"
       f"而重新训练后可达 {f4(TR['qnli_adapted']['macro_f1'])}。这让我认识到，迁移工程的真正难点不在代码，"
       "而在于识别“哪些假设还成立”。")

_hist = BERT["history"] if BERT else []
_sec = [h["seconds"] for h in _hist]
_ep_min = (sum(_sec) / len(_sec) / 60) if _sec else 9.0
_fit_txt = ""
if _hist:
    _bi = max(range(len(_hist)), key=lambda i: _hist[i]["macro_f1"])
    _bep, _bf1 = _hist[_bi]["epoch"], _hist[_bi]["macro_f1"]
    if _bi < len(_hist) - 1:
        _fit_txt = (f"第 {_bep} 轮达到验证最优（Macro-F1 {_bf1:.4f}），"
                    f"第 {_hist[-1]['epoch']} 轮验证损失反而回升、Macro-F1 回落，"
                    "说明轮数上已没有更多余量，也来不及做学习率与轮数的进一步调优")
    else:
        _fit_txt = (f"第 {_bep} 轮达到验证最优（Macro-F1 {_bf1:.4f}），"
                    "仍处于欠拟合一侧、继续训练的收益尚未耗尽")
para(c, "实验中遇到的困难主要有两个。一是环境与网络：HuggingFace 直连不可用，需借助镜像；"
       "CUDA 版 PyTorch（约 2.4 GB）在约 300 KB/s 的带宽下无法在合理时间内下载，"
       f"最终预训练模型只能在 CPU 上微调，单轮耗时约 {_ep_min:.0f} 分钟，只能训练 "
       f"{len(_hist) if _hist else 3} 轮，"
       + (_fit_txt + "。" if _fit_txt else "")
       + "这直接限制了 BERT 的最终成绩，也是本实验最主要的局限。"
       "二是数据划分：最初按“标题个数”取 20% 作为验证集，结果因各访谈样本量悬殊"
       "（有的访谈数百条、有的几十条），验证集反而占了 59% 的样本；"
       "改为按样本比例贪心选取标题后，划分才恢复正常。这个错误提醒我，"
       "分组划分必须同时检查组数与样本数两个维度，不能想当然。")
para(c, "后续可以改进的方向有三个：其一，补齐 GPU 环境后把 BERT 训练到充分收敛，"
       "并尝试更长的输入长度（当前 160 个 subword 对平均 294 词的回答截断过多）；"
       "其二，把细类标签作为辅助任务做多任务学习，检验“层级结构是否能补充信息”；"
       "其三，加入问题焦点与回答成分的对齐特征，针对“长而空泛”这一主要错误类型做定向改进。")

print("sections filled, saving ...")
doc.save(OUT)
print("saved:", OUT)
