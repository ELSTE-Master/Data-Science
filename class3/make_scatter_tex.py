"""Figure for the case-study slide "the data": SiO2 vs crustal thickness.

Grey: all 8498 cells of Farner & Lee (2017). Coloured: the Class 3 random
subsample (60 cells per class). Black line: mean SiO2 per 2-km thickness bin
(full data). Output: img/crust_scatter.png (transparent) via TikZ.
"""
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FULL = HERE.parent / "Data" / "arc_magmas_crust.csv"
SAMPLE = HERE.parent / "Data" / "arc_magmas_crust_sample.csv"
TEX = HERE / "tex"
IMG = HERE / "img"
COLORS = {"thin": "6A5ACD", "medium": "06BCF1", "thick": "8BB174"}

full = pd.read_csv(FULL)
sub = pd.read_csv(SAMPLE)

XMIN, XMAX, YMIN, YMAX = 18, 72, 40, 80
W, H = 13.0, 7.0
def X(v): return (v - XMIN) / (XMAX - XMIN) * W
def Y(v): return (v - YMIN) / (YMAX - YMIN) * H

bins = np.arange(20, 73, 2)
full["bin"] = pd.cut(full["crust_km"], bins)
means = full.groupby("bin", observed=True)["sio2"].agg(["mean", "count"])
means = means[means["count"] >= 20]
centers = [iv.mid for iv in means.index]

out = [r"""\documentclass[tikz,border=4pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usetikzlibrary{arrows.meta}
\definecolor{txt}{RGB}{85,85,85}
\definecolor{cthin}{HTML}{6A5ACD}
\definecolor{cmedium}{HTML}{06BCF1}
\definecolor{cthick}{HTML}{8BB174}
\begin{document}
\begin{tikzpicture}[x=1cm,y=1cm,line cap=round,line join=round]
\useasboundingbox (-1.5,-1.3) rectangle (14.0,8.3);
"""]
# class bands (very light) and boundaries
out.append(r"\fill[cthin!8] (%.3f,0) rectangle (%.3f,%.2f);" % (X(XMIN), X(30), H))
out.append(r"\fill[cmedium!8] (%.3f,0) rectangle (%.3f,%.2f);" % (X(30), X(43), H))
out.append(r"\fill[cthick!10] (%.3f,0) rectangle (%.3f,%.2f);" % (X(43), X(XMAX), H))
for v in (30, 43):
    out.append(r"\draw[txt, dash pattern=on 4pt off 3pt, line width=0.7pt] (%.3f,0) -- (%.3f,%.2f);" % (X(v), X(v), H))
for name, lo, hi in (("thin", XMIN, 30), ("medium", 30, 43), ("thick", 43, XMAX)):
    out.append(r"\node[c%s, font=\small\bfseries] at (%.3f,%.2f) {%s crust};" % (name, (X(lo)+X(hi))/2, H + 0.3, name))
# all cells (grey)
for x, y in zip(full["crust_km"], full["sio2"]):
    if XMIN <= x <= XMAX and YMIN <= y <= YMAX:
        out.append(r"\fill[gray!40] (%.3f,%.3f) circle (0.028);" % (X(x), Y(y)))
# subsample (coloured)
for name, col in COLORS.items():
    d = sub[sub.crust_class == name]
    for x, y in zip(d["crust_km"], d["sio2"]):
        out.append(r"\fill[c%s, fill opacity=0.85] (%.3f,%.3f) circle (0.085);" % (name, X(x), Y(y)))
# binned mean (full data)
pts = " ".join("(%.3f,%.3f)" % (X(c), Y(m)) for c, m in zip(centers, means["mean"]))
out.append(r"\draw[black, line width=1.8pt] plot[smooth, tension=0.5] coordinates {%s};" % pts)
# axes
out.append(r"\draw[txt, line width=0.8pt] (0,0) -- (%.2f,0);" % W)
out.append(r"\draw[txt, line width=0.8pt] (0,0) -- (0,%.2f);" % H)
for v in range(20, 71, 10):
    out.append(r"\draw[txt] (%.3f,0) -- (%.3f,-0.12); \node[txt, font=\small, below] at (%.3f,-0.12) {%d};" % (X(v), X(v), X(v), v))
for v in range(40, 81, 10):
    out.append(r"\draw[txt] (0,%.3f) -- (-0.12,%.3f); \node[txt, font=\small, left] at (-0.12,%.3f) {%d};" % (Y(v), Y(v), Y(v), v))
out.append(r"\node[txt, font=\small] at (%.2f,-0.95) {crustal thickness below the cell (km)};" % (W/2))
out.append(r"\node[txt, font=\small, rotate=90] at (-1.15,%.2f) {SiO$_2$ (wt.\%%)};" % (H/2))
# legend (bottom right, on top of the band)
lx, ly = 8.9, 0.45
out.append(r"\fill[white, opacity=0.85, rounded corners=3pt] (%.2f,%.2f) rectangle (%.2f,%.2f);" % (lx-0.3, ly-0.3, lx+4.3, ly+1.35))
out.append(r"\fill[gray!40] (%.2f,%.2f) circle (0.06); \node[txt, font=\footnotesize, anchor=west] at (%.2f,%.2f) {all 8498 cells};" % (lx, ly+1.05, lx+0.25, ly+1.05))
out.append(r"\fill[cthin] (%.2f,%.2f) circle (0.085); \fill[cmedium] (%.2f,%.2f) circle (0.085); \fill[cthick] (%.2f,%.2f) circle (0.085); \node[txt, font=\footnotesize, anchor=west] at (%.2f,%.2f) {our subsample, 60 per group};" % (lx-0.1, ly+0.55, lx+0.15, ly+0.55, lx+0.4, ly+0.55, lx+0.65, ly+0.55))
out.append(r"\draw[black, line width=1.8pt] (%.2f,%.2f) -- (%.2f,%.2f); \node[txt, font=\footnotesize, anchor=west] at (%.2f,%.2f) {mean SiO$_2$ per 2 km (all cells)};" % (lx-0.2, ly+0.05, lx+0.4, ly+0.05, lx+0.65, ly+0.05))
out.append("\\end{tikzpicture}\n\\end{document}\n")
TEX.mkdir(exist_ok=True)
(TEX / "crust_scatter.tex").write_text("\n".join(out), encoding="utf-8")
subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "crust_scatter.tex"], cwd=TEX, check=True, stdout=subprocess.DEVNULL)
subprocess.run(["pdftocairo", "-png", "-transp", "-singlefile", "-scale-to-x", "2400", "-scale-to-y", "-1",
                str(TEX / "crust_scatter.pdf"), str(IMG / "crust_scatter")], check=True)
print("bins used:", len(centers), "| Saved", IMG / "crust_scatter.png")
