"""把真实运行日志渲染成终端风格图片，供报告插入。"""
import sys

from PIL import Image, ImageDraw, ImageFont

BG = (30, 30, 30)
FG = (220, 220, 220)
ACCENT = (106, 190, 120)
DIM = (150, 150, 150)


def render(text, out_path, title="", width_px=1180, font_size=13, max_lines=None,
           highlight=()):
    font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", font_size)
    bold = ImageFont.truetype("C:/Windows/Fonts/consolab.ttf", font_size)
    # Consolas 无中文字形，含中文的行改用等宽感较强的雅黑，避免出现方框
    cjk = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", font_size, index=0)
    cjk_bold = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", font_size, index=0)

    def has_cjk(s):
        return any("\u3000" <= ch <= "\u9fff" for ch in s)

    lines = text.replace("\r", "").rstrip("\n").split("\n")
    if max_lines:
        lines = lines[:max_lines]

    pad = 18
    line_h = int(font_size * 1.55)
    head_h = 30 if title else 0
    height = pad * 2 + head_h + line_h * len(lines)

    img = Image.new("RGB", (width_px, height), BG)
    d = ImageDraw.Draw(img)

    if title:
        d.rectangle([0, 0, width_px, head_h], fill=(45, 45, 45))
        for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
            d.ellipse([12 + i * 20, 9, 22 + i * 20, 19], fill=c)
        d.text((84, (head_h - font_size) / 2 - 1), title, font=font, fill=DIM)

    y = pad + head_h
    for ln in lines:
        color = FG
        f = font
        is_bold = False
        if any(h in ln for h in highlight):
            color = ACCENT
            f = bold
            is_bold = True
        elif ln.startswith("$"):
            color = (135, 206, 235)
            f = bold
            is_bold = True
        if has_cjk(ln):
            f = cjk_bold if is_bold else cjk
        d.text((pad, y), ln, font=f, fill=color)
        y += line_h
    img.save(out_path)
    print(f"保存 {out_path}  ({img.size[0]}x{img.size[1]})")


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    title = sys.argv[3] if len(sys.argv) > 3 else ""
    n = int(sys.argv[4]) if len(sys.argv) > 4 else None
    with open(src, encoding="utf-8") as f:
        render(f.read(), dst, title, max_lines=n)
