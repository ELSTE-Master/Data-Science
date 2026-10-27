"""World map of the Farner & Lee (2017) cells, coloured by crustal-thickness class.

Pacific-centred equirectangular projection. Coastline: Natural Earth 1:110m
(public domain), file data_aux/ne_110m_coastline.geojson.
Output: img/crust_map.png (transparent) via TikZ.
"""
import json
import subprocess
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = HERE.parent / "Data" / "arc_magmas_crust.csv"
COAST = HERE / "data_aux" / "ne_110m_coastline.geojson"
TEX = HERE / "tex"
IMG = HERE / "img"
COLORS = {"thin": "6A5ACD", "medium": "06BCF1", "thick": "8BB174"}

SC = 0.05                                  # cm per degree
LATMIN, LATMAX = -65, 72
def lonp(lon): return ((lon - 180 + 540) % 360) - 180   # Pacific-centred longitude
def X(lon_p): return (lon_p + 180) * SC
def Y(lat): return (lat - LATMIN) * SC
W, H = 360 * SC, (LATMAX - LATMIN) * SC

d = pd.read_csv(CSV)
d["lon_p"] = lonp(d["longitude"])

out = [r"""\documentclass[tikz,border=4pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\definecolor{txt}{RGB}{85,85,85}
\definecolor{coast}{RGB}{150,150,158}
\definecolor{cthin}{HTML}{6A5ACD}
\definecolor{cmedium}{HTML}{06BCF1}
\definecolor{cthick}{HTML}{8BB174}
\begin{document}
\begin{tikzpicture}[x=1cm,y=1cm,line cap=round,line join=round]
"""]
out.append(r"\useasboundingbox (-0.2,-0.2) rectangle (%.2f,%.2f);" % (W + 0.2, H + 0.2))
out.append(r"\clip (0,0) rectangle (%.2f,%.2f);" % (W, H))
# coastlines
coast = json.load(open(COAST))
for f in coast["features"]:
    pts = [(lonp(lon), lat) for lon, lat in f["geometry"]["coordinates"]]
    seg = []
    for i, (x, y) in enumerate(pts):
        if seg and abs(x - seg[-1][0]) > 180:
            if len(seg) > 1:
                out.append(r"\draw[coast, line width=0.35pt] " + " -- ".join("(%.3f,%.3f)" % (X(a), Y(b)) for a, b in seg) + ";")
            seg = []
        seg.append((x, y))
    if len(seg) > 1:
        out.append(r"\draw[coast, line width=0.35pt] " + " -- ".join("(%.3f,%.3f)" % (X(a), Y(b)) for a, b in seg) + ";")
# cells
for cls in ("medium", "thin", "thick"):
    sub = d[d.crust_class == cls]
    for x, y in zip(sub["lon_p"], sub["latitude"]):
        out.append(r"\fill[c%s] (%.3f,%.3f) circle (0.038);" % (cls, X(x), Y(y)))
# arc labels (lon_p, lat, anchor)
labels = [("Andes", 125, -27, "west"), ("Mexico", 66, 26, "east"), ("Central America", 84, 5, "east"),
          ("Lesser Antilles", 125, 14, "west"), ("Cascades", 47, 46, "east"), ("Aleutians", 12, 63, "south"),
          ("Kamchatka", -32, 60, "east"), ("Japan", -55, 34, "east"), ("Marianas", -27, 17, "west"),
          ("Sunda", -72, -11, "north"), ("Tonga", 11, -20, "west"), ("Aegean", -154, 28, "north"),
          ("Scotia", 142, -60, "west")]
for name, lon_p, lat, anc in labels:
    out.append(r"\node[txt, font=\scriptsize, anchor=%s, inner sep=1.5pt] at (%.3f,%.3f) {%s};" % (anc, X(lon_p), Y(lat), name))
# legend
lx, ly = X(-178), Y(-43)
for i, (cls, txt) in enumerate((("thin", "thin crust ($<$ 30 km)"), ("medium", "medium (30--43 km)"), ("thick", "thick ($\\geq$ 43 km)"))):
    out.append(r"\fill[c%s] (%.2f,%.2f) circle (0.09); \node[txt, font=\scriptsize, anchor=west] at (%.2f,%.2f) {%s};" % (cls, lx, ly - 0.42*i, lx + 0.2, ly - 0.42*i, txt))
out.append("\\end{tikzpicture}\n\\end{document}\n")
TEX.mkdir(exist_ok=True)
(TEX / "crust_map.tex").write_text("\n".join(out), encoding="utf-8")
subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "crust_map.tex"], cwd=TEX, check=True, stdout=subprocess.DEVNULL)
subprocess.run(["pdftocairo", "-png", "-transp", "-singlefile", "-scale-to-x", "2800", "-scale-to-y", "-1",
                str(TEX / "crust_map.pdf"), str(IMG / "crust_map")], check=True)
print("Saved", IMG / "crust_map.png")
