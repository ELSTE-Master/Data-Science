"""Figure for the slide "Multiple testing can be dangerous!".

Probability of at least one false rejection among k independent tests
run at alpha = 5% when all null hypotheses are true: 1 - (1 - alpha)^k.
Highlights k = 3 (our three crust comparisons) and k = 20 (the jelly-bean
colours of xkcd 882). Output: img/alpha_k.png (transparent PNG via TikZ).
"""
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEX = HERE / "tex"
IMG = HERE / "img"
ALPHA = 0.05

def alpha_k(k):
    return 1 - (1 - ALPHA) ** k

# ---- TikZ: x = k in 0..50, y = probability 0..1 ------------------------------
XMIN, XMAX, YMIN, YMAX = 0, 50, 0, 1
W, H = 11.0, 7.5
def X(k): return (k - XMIN) / (XMAX - XMIN) * W
def Y(p): return (p - YMIN) / (YMAX - YMIN) * H

out = [r"""\documentclass[tikz,border=4pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usetikzlibrary{arrows.meta}
\definecolor{txt}{RGB}{85,85,85}
\definecolor{curve}{HTML}{EB078E}
\definecolor{ours}{HTML}{6A5ACD}
\definecolor{jelly}{RGB}{0,128,0}
\begin{document}
\begin{tikzpicture}[x=1cm,y=1cm,line cap=round,line join=round]
\useasboundingbox (-1.7,-1.3) rectangle (13.4,8.2);
"""]
out.append(r"\draw[txt, line width=0.8pt, -{Stealth[length=6pt]}] (0,0) -- (%.2f,0);" % (W + 0.5))
out.append(r"\draw[txt, line width=0.8pt, -{Stealth[length=6pt]}] (0,0) -- (0,%.2f);" % (H + 0.5))
for k in range(10, 51, 10):
    out.append(r"\draw[txt] (%.3f,0) -- (%.3f,-0.12); \node[txt, font=\large, below] at (%.3f,-0.12) {%d};" % (X(k), X(k), X(k), k))
for p in [0.25, 0.5, 0.75, 1.0]:
    out.append(r"\draw[txt] (0,%.3f) -- (-0.12,%.3f); \node[txt, font=\large, left] at (-0.12,%.3f) {%d\%%};" % (Y(p), Y(p), Y(p), round(p * 100)))
out.append(r"\node[txt, font=\large] at (%.2f,-0.95) {number of tests $k$};" % (W / 2))
out.append(r"\node[txt, font=\large, rotate=90] at (-1.3,%.2f) {$\alpha_k$ = Pr(at least one false rejection)};" % (H / 2))
out.append(r"\draw[txt, dash pattern=on 4pt off 3pt, line width=0.7pt] (0,%.3f) -- (%.2f,%.3f) node[right, font=\large, text=txt] {$\alpha = 5\%%$};" % (Y(ALPHA), W, Y(ALPHA)))
grid = np.linspace(1, 50, 200)
pts = " ".join("(%.3f,%.3f)" % (X(k), Y(alpha_k(k))) for k in grid)
out.append(r"\draw[curve, line width=2.2pt] plot coordinates {%s};" % pts)
for k in [1, 2, 5, 10, 50]:
    out.append(r"\fill[curve] (%.3f,%.3f) circle (0.09);" % (X(k), Y(alpha_k(k))))
    if k < 5:
        continue                      # 5% and 10% would crowd the k = 3 label
    out.append(r"\node[curve, font=\normalsize, anchor=south east] at (%.3f,%.3f) {%d\%%};" % (X(k) - 0.05, Y(alpha_k(k)) + 0.1, round(100 * alpha_k(k))))
out.append(r"\fill[ours] (%.3f,%.3f) circle (0.13);" % (X(3), Y(alpha_k(3))))
out.append(r"\node[ours, font=\large, anchor=west] at (%.3f,%.3f) {$k = 3$ (our crust comparisons): %d\%%};" % (X(3) + 0.3, Y(alpha_k(3)) - 0.15, round(100 * alpha_k(3))))
out.append(r"\fill[jelly] (%.3f,%.3f) circle (0.13);" % (X(20), Y(alpha_k(20))))
out.append(r"\node[jelly, font=\large, anchor=north west] at (%.3f,%.3f) {$k = 20$ (jelly-bean colours): %d\%%};" % (X(20) + 0.25, Y(alpha_k(20)) - 0.1, round(100 * alpha_k(20))))
out.append(r"\node[curve, font=\large, anchor=south west] at (%.3f,%.3f) {$\alpha_k = 1 - (1-\alpha)^k$};" % (X(30), Y(0.97) - 0.05))
out.append("\\end{tikzpicture}\n\\end{document}\n")

TEX.mkdir(exist_ok=True)
(TEX / "alpha_k.tex").write_text("\n".join(out), encoding="utf-8")
subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "alpha_k.tex"], cwd=TEX, check=True, stdout=subprocess.DEVNULL)
subprocess.run(["pdftocairo", "-png", "-transp", "-singlefile", "-scale-to-x", "2400", "-scale-to-y", "-1",
                str(TEX / "alpha_k.pdf"), str(IMG / "alpha_k")], check=True)
for ext in ("aux", "log"):
    (TEX / f"alpha_k.{ext}").unlink(missing_ok=True)
print("Saved", IMG / "alpha_k.png")
