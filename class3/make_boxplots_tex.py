
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = HERE.parent / "Data" / "arc_magmas_crust_sample.csv"   # fixed subsample, see save_df_crust_sample.py
TEX = HERE / "tex"
IMG = HERE / "img"

SEED = 2017            # jitter only
LABELS = {"thin": "Thin crust", "medium": "Medium crust", "thick": "Thick crust"}
ROW_Y = {"thin": 5.4, "medium": 8.85, "thick": 12.3}   # same rows as the diet figures

# Horizontal axis: SiO2 (wt.%) -> x (cm)
XMIN, XMAX = 40, 80
X0, X1 = 0.75, 14.4


def xpos(v):
    return X0 + (v - XMIN) / (XMAX - XMIN) * (X1 - X0)


def fivenum(x):
    """Tukey's five-number summary (what R's boxplot() uses for the box)."""
    x = np.sort(x)
    n = len(x)
    n4 = np.floor((n + 3) / 2) / 2
    d = np.array([1, n4, (n + 1) / 2, n + 1 - n4, n]) - 1
    return 0.5 * (x[np.floor(d).astype(int)] + x[np.ceil(d).astype(int)])


# ---- TikZ ---------------------------------------------------------------------
PREAMBLE = r"""\documentclass[tikz,border=0pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usetikzlibrary{shapes.geometric}
\definecolor{txt}{RGB}{85,85,85}
\definecolor{pt}{RGB}{221,28,140}
\definecolor{boxfill}{RGB}{211,211,211}
\begin{document}
\begin{tikzpicture}[x=1cm,y=1cm,
  point/.style={circle, inner sep=0pt, minimum size=0.3cm, fill=pt, fill opacity=0.3,
                draw=pt, draw opacity=0.45, line width=0.6pt},
  meanmark/.style={diamond, inner sep=0pt, minimum size=0.42cm, fill=white, draw=black, line width=0.9pt}]
\useasboundingbox (0,0) rectangle (20,15);
"""

def tikz_figure(groups, rng):
    top = max(ROW_Y[g] for g in groups) + 1.1           # top row + jitter, no title
    out = [PREAMBLE.replace("rectangle (20,15);", "rectangle (20,%.1f);" % top)]
    # Discreet axis label under the tick labels
    out.append(r"\node[txt, font=\fontsize{20}{24}\selectfont] at (%.2f,0.85) "
               r"{SiO$_2$ content (wt.\%%)};" % ((X0 + X1) / 2))
    # Axis
    out.append(r"\draw[txt, line width=0.8pt] (%.3f,3.33) -- (%.3f,3.33);" % (X0, X1))
    for v in range(XMIN, XMAX + 1, 5):
        out.append(r"\draw[txt, line width=0.8pt] (%.3f,3.33) -- (%.3f,2.85);" % (xpos(v), xpos(v)))
        out.append(r"\node[txt, font=\fontsize{24}{28}\selectfont] at (%.3f,2.0) {%d};" % (xpos(v), v))

    for name, sample in groups.items():
        y = ROW_Y[name]
        q = fivenum(sample)
        iqr = q[3] - q[1]
        lo_w = sample[sample >= q[1] - 1.5 * iqr].min()
        hi_w = sample[sample <= q[3] + 1.5 * iqr].max()
        # Whiskers and caps
        out.append(r"\draw[black, line width=0.7pt, dash pattern=on 7pt off 7pt] "
                   r"(%.3f,%.2f) -- (%.3f,%.2f) (%.3f,%.2f) -- (%.3f,%.2f);"
                   % (xpos(lo_w), y, xpos(q[1]), y, xpos(q[3]), y, xpos(hi_w), y))
        for w in (lo_w, hi_w):
            out.append(r"\draw[black, line width=0.7pt] (%.3f,%.2f) -- (%.3f,%.2f);"
                       % (xpos(w), y - 0.35, xpos(w), y + 0.35))
        # Box and median
        out.append(r"\draw[black, line width=0.7pt, fill=boxfill] (%.3f,%.2f) rectangle (%.3f,%.2f);"
                   % (xpos(q[1]), y - 0.7, xpos(q[3]), y + 0.7))
        out.append(r"\draw[black, line width=3.5pt] (%.3f,%.2f) -- (%.3f,%.2f);"
                   % (xpos(q[2]), y - 0.7, xpos(q[2]), y + 0.7))
        # Jittered points
        for v, dy in zip(sample, rng.uniform(-0.5, 0.5, len(sample))):
            out.append(r"\node[point] at (%.3f,%.3f) {};" % (xpos(v), y + dy))
        # Mean
        out.append(r"\node[meanmark] at (%.3f,%.2f) {};" % (xpos(sample.mean()), y))
        # Group label
        out.append(r"\node[txt, font=\fontsize{22}{26}\selectfont, anchor=east] at (19.6,%.2f) {%s};"
                   % (y, LABELS[name]))

    # Legend (mean / median)
    out.append(r"\node[meanmark] at (16.3,2.55) {};")
    out.append(r"\node[txt, font=\fontsize{18}{22}\selectfont, anchor=west] at (16.7,2.55) {Mean};")
    out.append(r"\draw[black, line width=3.5pt] (16.3,1.55) -- (16.3,2.05);")
    out.append(r"\node[txt, font=\fontsize{18}{22}\selectfont, anchor=west] at (16.7,1.8) {Median};")
    out.append("\\end{tikzpicture}\n\\end{document}\n")
    return "\n".join(out)


# ---- Main ---------------------------------------------------------------------
crust = pd.read_csv(CSV)
subs = {g: crust.loc[crust["crust_class"] == g, "sio2"].to_numpy() for g in LABELS}

print(f"{'group':8s} {'n':>3s} {'mean':>6s} {'sd':>5s} {'Q1/med/Q3':>18s}")
for g in LABELS:
    x = subs[g]
    print(f"{g:8s} {len(x):3d} {x.mean():6.2f} {x.std(ddof=1):5.2f} "
          f"{str(np.percentile(x, [25, 50, 75]).round(1)):>18s}")

TEX.mkdir(exist_ok=True)
for name, keys in {"crust_boxplot_2": ["thin", "medium"],
                   "crust_boxplot_3": ["thin", "medium", "thick"]}.items():
    rng = np.random.default_rng(SEED)
    (TEX / f"{name}.tex").write_text(tikz_figure({k: subs[k] for k in keys}, rng), encoding="utf-8")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{name}.tex"],
                   cwd=TEX, check=True, stdout=subprocess.DEVNULL)
    # On Windows the fresh PDF can stay locked for a moment (antivirus, indexer): retry
    for attempt in range(5):
        res = subprocess.run(["pdftocairo", "-png", "-transp", "-singlefile", "-scale-to-x", "2400",
                              "-scale-to-y", "-1", str(TEX / f"{name}.pdf"), str(IMG / name)],
                             capture_output=True, text=True)
        if res.returncode == 0:
            break
        time.sleep(1)
    else:
        raise RuntimeError(f"pdftoppm failed for {name}: {res.stderr.strip()}")
    print("Saved", IMG / f"{name}.png")  # transparent background (slides are not white)

for ext in ("aux", "log"):
    for f in TEX.glob(f"*.{ext}"):
        f.unlink()
