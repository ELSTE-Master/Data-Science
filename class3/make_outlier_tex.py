"""Figure for the slide "What if one value were wrong?".

Takes the Class 3 subsample, replaces the smallest SiO2 value of the
medium-crust group by a value v, and plots the p-value of Welch's t-test
and of the Mann-Whitney-Wilcoxon test (thin vs medium, two-sided) as a
function of v. Outputs (transparent PNG via TikZ):
  img/outlier_pvalues_welch.png  (Welch only)
  img/outlier_pvalues.png        (Welch + Mann-Whitney-Wilcoxon)
"""
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
CSV = HERE.parent / "Data" / "arc_magmas_crust_sample.csv"
TEX = HERE / "tex"
IMG = HERE / "img"

d = pd.read_csv(CSV)
thin = d.loc[d.crust_class == "thin", "sio2"].to_numpy()
medium = d.loc[d.crust_class == "medium", "sio2"].to_numpy()
i_min = medium.argmin()
v0 = medium[i_min]

def pvals(v):
    m = medium.copy(); m[i_min] = v
    return (stats.ttest_ind(thin, m, equal_var=False).pvalue,
            stats.mannwhitneyu(thin, m, alternative="two-sided").pvalue)

grid = np.linspace(v0, 300, 200)
curve = [pvals(v) for v in grid]
table_vals = [v0, 60, 80, 100, 150, 200, 300, 459.6, 585]
print(f"{'value':>7s} {'Welch p':>8s} {'MWW p':>7s}")
for v in table_vals:
    pw, pm = pvals(v); print(f"{v:7.1f} {pw:8.4f} {pm:7.4f}")

# ---- TikZ: x = value 40..300, y = p-value 0..0.25 -------------------------------
XMIN, XMAX, YMIN, YMAX = 40, 300, 0, 0.12
W, H = 13.0, 7.0
def X(v): return (v - XMIN) / (XMAX - XMIN) * W
def Y(p): return (min(p, YMAX) - YMIN) / (YMAX - YMIN) * H

def figure(name, with_mww):
    out = [r"""\documentclass[tikz,border=4pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usetikzlibrary{arrows.meta}
\definecolor{txt}{RGB}{85,85,85}
\definecolor{cwelch}{HTML}{6A5ACD}
\definecolor{cmww}{HTML}{EB078E}
\begin{document}
\begin{tikzpicture}[x=1cm,y=1cm,line cap=round,line join=round]
\useasboundingbox (-1.6,-1.3) rectangle (14.2,7.6);
"""]
    out.append(r"\draw[txt, line width=0.8pt, -{Stealth[length=6pt]}] (0,0) -- (%.2f,0);" % (W + 0.4))
    out.append(r"\draw[txt, line width=0.8pt, -{Stealth[length=6pt]}] (0,0) -- (0,%.2f);" % (H + 0.4))
    for v in range(50, 301, 50):
        out.append(r"\draw[txt] (%.3f,0) -- (%.3f,-0.12); \node[txt, font=\small, below] at (%.3f,-0.12) {%d};" % (X(v), X(v), X(v), v))
    for p in [0.025, 0.05, 0.075, 0.10]:
        out.append(r"\draw[txt] (0,%.3f) -- (-0.12,%.3f); \node[txt, font=\small, left] at (-0.12,%.3f) {%g\%%};" % (Y(p), Y(p), Y(p), round(p*100, 1)))
    out.append(r"\node[txt, font=\small] at (%.2f,-0.95) {value given to the smallest medium-crust observation (SiO$_2$, wt.\%%)};" % (W/2))
    out.append(r"\node[txt, font=\small, rotate=90] at (-1.25,%.2f) {p-value (thin vs medium)};" % (H/2))
    out.append(r"\draw[txt, dash pattern=on 4pt off 3pt, line width=0.7pt] (0,%.3f) -- (%.2f,%.3f) node[right, font=\small, text=txt] {$\alpha = 5\%%$};" % (Y(0.05), W, Y(0.05)))
    series = [("cwelch", 0)] if not with_mww else [("cmww", 1), ("cwelch", 0)]
    for col, idx in series:
        pts = " ".join("(%.3f,%.3f)" % (X(v), Y(c[idx])) for v, c in zip(grid, curve))
        out.append(r"\draw[%s, line width=2.2pt] plot coordinates {%s};" % (col, pts))
    pw0, pm0 = pvals(v0)
    out.append(r"\fill[cwelch] (%.3f,%.3f) circle (0.11);" % (X(v0), Y(pw0)))
    out.append(r"\node[cwelch, font=\small, anchor=south west] at (%.3f,%.3f) {original value (%.1f): p = %.1f\%%};" % (X(v0)+0.45, 0.1, v0, 100*pw0))
    out.append(r"\draw[cwelch, line width=2.2pt] (7.2,6.7) -- (8.0,6.7); \node[cwelch, font=\small, anchor=west] at (8.1,6.7) {Welch's $t$-test};")
    if with_mww:
        out.append(r"\draw[cmww, line width=2.2pt] (7.2,6.1) -- (8.0,6.1); \node[cmww, font=\small, anchor=west] at (8.1,6.1) {Mann--Whitney--Wilcoxon};")
    out.append("\\end{tikzpicture}\n\\end{document}\n")
    TEX.mkdir(exist_ok=True)
    (TEX / f"{name}.tex").write_text("\n".join(out), encoding="utf-8")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{name}.tex"], cwd=TEX, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["pdftocairo", "-png", "-transp", "-singlefile", "-scale-to-x", "2400", "-scale-to-y", "-1",
                    str(TEX / f"{name}.pdf"), str(IMG / name)], check=True)
    print("Saved", IMG / f"{name}.png")

figure("outlier_pvalues_welch", with_mww=False)
figure("outlier_pvalues", with_mww=True)
