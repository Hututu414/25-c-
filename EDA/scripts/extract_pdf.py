# -*- coding: utf-8 -*-
"""提取题目 PDF 全文"""
import fitz
from pathlib import Path

pdf_path = Path(r"D:\Users\TtT20\source\repos\数学建模\国赛\25年c题\C题\C题.pdf")
out = Path(r"D:\Users\TtT20\source\repos\数学建模\国赛\25年c题\EDA\output\C题_题目全文.txt")

doc = fitz.open(str(pdf_path))
lines = []
for i, page in enumerate(doc):
    lines.append(f"===== PAGE {i+1} =====")
    lines.append(page.get_text())
out.write_text("\n".join(lines), encoding="utf-8")
print("pages:", len(doc))
print("written:", out)
