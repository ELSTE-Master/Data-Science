"""Schematic for the case-study slide "why would it be so?": arc cross-sections for
thin (island arc) and thick (continental arc) crust. Vertical scale 0.04 cm/km.
Writes crust_schematic.tex, compiles it and exports img/crust_schematic.png (transparent).

Geometry: the oceanic plate comes in horizontally under the ocean, bends down at the
trench and dips under the overriding plate, whose crust tapers towards the trench
(its lower-left boundary follows the top of the slab). Labels never sit on symbols.
"""
import math
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
H = 7.5 * 2.54 / 72.27          # half width of the slab (line width 15pt), in cm
XL, XR = -0.45, 9.0             # panel edges
XB = 0.3                        # trench: the incoming plate bends down here
XE, YE = 6.6, -0.3              # deep end of the slab (leaves the panel through the bottom)
VX = 5.4                        # volcano / conduit


def P(p):
    return "(%.3f,%.3f)" % p


def panel(shift, kind):
    thick = kind == "thick"
    surf = 5.2 if thick else 5.0
    moho = surf - (2.0 if thick else 1.0)          # 50 km / 25 km
    sea = 5.0 if thick else surf + 0.16            # continental arc above sea level, island arc submerged
    yh = sea - 0.6                                 # incoming plate (centre line): trench ~8 km deep
    dx, dy = XE - XB, YE - yh
    L = math.hypot(dx, dy)
    dx, dy = dx / L, dy / L                        # unit vector down the dipping slab
    nx, ny = -dy, dx                               # unit normal pointing up
    ang = math.degrees(math.atan2(dy, dx))

    def centre(t):
        return (XB + t * dx, yh + t * dy)

    def top(t):
        return (XB + t * dx + H * nx, yh + t * dy + H * ny)

    def t_top(y):                                  # t where the top of the slab is at height y
        return (y - yh - H * ny) / dy

    t_tr = 0.5                                     # leading edge of the overriding plate
    x_tr, y_tr = top(t_tr)
    x_fa = x_tr + 1.3                              # forearc slope reaches the surface here
    x_m = top(t_top(moho))[0]                      # slab top crosses the Moho depth here
    col = "cthick" if thick else "cthin"

    o = [r"\begin{scope}[shift={(%.2f,0)}]" % shift]
    # --- geology (clipped to the panel) -------------------------------------
    o.append(r"\begin{scope}")
    o.append(r"\clip (%.2f,0) rectangle (%.2f,%.2f);" % (XL, XR, surf + 1.3))
    o.append(r"\fill[cmantle] (%.2f,0) rectangle (%.2f,%.2f);" % (XL, XR, surf))
    if sea > surf:   # island arc: ocean above the crust and in the trench
        wat = [(XL, sea), (XR, sea), (XR, surf), (x_fa, surf), (x_tr, y_tr), centre(t_tr), (XB, yh), (XL, yh)]
    else:            # continental arc: ocean only on the trench side
        x_sea = x_tr + (x_fa - x_tr) * (sea - y_tr) / (surf - y_tr)
        wat = [(XL, sea), (x_sea, sea), (x_tr, y_tr), centre(t_tr), (XB, yh), (XL, yh)]
    o.append(r"\fill[cwater] " + " -- ".join(P(p) for p in wat) + " -- cycle;")
    o.append(r"\draw[cslab, line width=15pt, line cap=butt, rounded corners=0.6cm] (%.2f,%.2f) -- (%.2f,%.2f) -- (%.2f,%.2f);"
             % (XL - 0.3, yh, XB, yh, XE, YE))
    o.append(r"\fill[ccrust] " + " -- ".join(P(p) for p in [(x_tr, y_tr), (x_fa, surf), (XR, surf), (XR, moho), (x_m, moho)]) + " -- cycle;")
    o.append(r"\end{scope}")
    # outlines
    o.append(r"\draw[txt, line width=0.9pt] %s -- %s -- %s;" % (P((x_tr, y_tr)), P((x_fa, surf)), P((XR, surf))))
    o.append(r"\draw[txt, line width=0.5pt, dash pattern=on 2pt off 2pt] (%.3f,%.2f) -- (%.2f,%.2f);" % (x_m, moho, XR, moho))
    o.append(r"\node[txt, font=\scriptsize, anchor=north east] at (8.35,%.2f) {Moho};" % (moho - 0.05))
    # plate motion arrow (on the slab, no text there) and slab label beside the slab
    o.append(r"\draw[white, line width=1.1pt, -{Stealth[length=6pt]}] %s -- %s;" % (P(centre(0.95)), P(centre(1.7))))
    lx, ly = centre(3.4)
    o.append(r"\node[txt, font=\scriptsize, rotate=%.1f] at (%.3f,%.3f) {subducting oceanic plate};"
             % (ang, lx - 0.55 * nx, ly - 0.55 * ny))
    # melting in the wedge above the slab (~100 km depth)
    for (x, y, r) in ((5.05, 1.5, 0.11), (5.35, 1.78, 0.13), (4.85, 1.95, 0.09), (5.6, 2.05, 0.09)):
        o.append(r"\fill[cmagma] (%.2f,%.2f) ellipse (%.2f and %.2f);" % (x, y, r, r * 0.8))
    o.append(r"\node[txt, font=\scriptsize, anchor=west, align=left] at (6.1,1.75) {melting of the mantle\\(water from the slab)};")
    o.append(r"\node[txt, font=\scriptsize] at (7.4,%.2f) {mantle wedge};" % (moho - 0.55))
    # ascent to the base of the crust
    o.append(r"\draw[cmagma, line width=2.2pt, -{Stealth[length=7pt]}] (5.3,2.2) -- (%.2f,%.2f);" % (VX, moho - 0.05))
    # storage and dikes
    if thick:
        o.append(r"\fill[cmagma!80!white] (%.2f,%.2f) ellipse (0.95 and 0.32);" % (VX, moho + 0.38))
        for ddx, ddy in ((-0.5, 0.0), (-0.2, 0.14), (0.15, -0.1), (0.45, 0.08), (0.0, 0.0), (0.6, -0.12)):
            o.append(r"\fill[white] (%.2f,%.2f) circle (0.05);" % (VX + ddx, moho + 0.38 + ddy))
        o.append(r"\draw[cmagma, line width=1.8pt] (%.2f,%.2f) -- (%.2f,%.2f);" % (VX, moho + 0.7, VX, surf - 0.55))
        o.append(r"\fill[cmagma!80!white] (%.2f,%.2f) ellipse (0.42 and 0.17);" % (VX, surf - 0.5))
        o.append(r"\draw[cmagma, line width=1.8pt] (%.2f,%.2f) -- (%.2f,%.2f);" % (VX, surf - 0.35, VX, surf + 0.7))
        o.append(r"\node[txt, font=\scriptsize, anchor=east, align=right] at (4.9,%.2f) {long storage near the Moho:\\cooling, crystallization,\\assimilation of crust};" % (moho + 1.25))
        o.append(r"\node[txt, font=\scriptsize, anchor=west] at (5.95,%.2f) {shallow chamber};" % (surf - 0.5))
    else:
        o.append(r"\draw[cmagma, line width=1.8pt] (%.2f,%.2f) -- (%.2f,%.2f);" % (VX, moho, VX, surf - 0.5))
        o.append(r"\fill[cmagma!80!white] (%.2f,%.2f) ellipse (0.42 and 0.17);" % (VX, surf - 0.5))
        o.append(r"\fill[white] (%.2f,%.2f) circle (0.045); \fill[white] (%.2f,%.2f) circle (0.045);" % (VX - 0.15, surf - 0.5, VX + 0.17, surf - 0.46))
        o.append(r"\draw[cmagma, line width=1.8pt] (%.2f,%.2f) -- (%.2f,%.2f);" % (VX, surf - 0.4, VX, surf + 0.6))
        o.append(r"\node[txt, font=\scriptsize, anchor=east] at (4.85,%.2f) {short, shallow storage};" % (surf - 0.5))
    # volcano / mountains
    if thick:
        pk = [(2.4, 0), (2.9, 0.3), (3.4, 0.14), (3.9, 0.4), (4.4, 0.18), (4.8, 0.3), (6.0, 0.3), (6.4, 0.2),
              (6.9, 0.42), (7.4, 0.18), (7.9, 0.33), (8.4, 0)]
        o.append(r"\fill[gray!40] " + " -- ".join("(%.2f,%.2f)" % (x, surf + h) for x, h in pk) + " -- cycle;")
        o.append(r"\fill[gray!65] (4.6,%.2f) -- (5.4,%.2f) -- (6.2,%.2f) -- cycle;" % (surf, surf + 0.95, surf))
    else:
        o.append(r"\fill[gray!65] (4.5,%.2f) -- (5.4,%.2f) -- (6.3,%.2f) -- cycle;" % (surf, surf + 0.75, surf))
    o.append(r"\draw[cmagma, line width=1.8pt, -{Stealth[length=6pt]}] (%.2f,%.2f) -- (%.2f,%.2f);" % (VX, surf + 0.3, VX, surf + (1.0 if thick else 0.85)))
    # thickness bar and labels
    o.append(r"\draw[txt, {Bar[width=5pt]}-{Bar[width=5pt]}, line width=0.8pt] (8.55,%.2f) -- (8.55,%.2f);" % (moho, surf))
    o.append(r"\node[txt, font=\small, anchor=east] at (8.4,%.2f) {$\approx$ %d km};" % ((moho + surf) / 2, 50 if thick else 25))
    o.append(r"\node[txt, font=\scriptsize, anchor=west] at (2.3,%.2f) {crust};" % (surf - 0.2))
    o.append(r"\node[txt, font=\small] at (1.3,0.6) {mantle};")
    o.append(r"\node[txt, font=\scriptsize, anchor=west] at (%.2f,%.2f) {ocean};" % (XL + 0.1, sea - 0.17 if thick else sea + 0.14))
    o.append(r"\node[%s, font=\large\bfseries] at (4.5,6.55) {%s};" % (col, "Thick crust (continental arc)" if thick else "Thin crust (island arc)"))
    cap = ("long path, extensive differentiation\\\\ $\\Rightarrow$ {\\color{cpink}high SiO$_2$} (andesite, dacite)" if thick
           else "short path, little differentiation\\\\ $\\Rightarrow$ {\\color{cpink}low SiO$_2$} (basalt, basaltic andesite)")
    o.append(r"\node[txt, font=\small, align=center] at (4.5,-0.6) {%s};" % cap)
    o.append(r"\end{scope}")
    return "\n".join(o)


tex = r"""% GENERATED by make_schematic_tex.py -- edit the script, not this file.
\documentclass[tikz,border=4pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usetikzlibrary{arrows.meta}
\definecolor{cthin}{HTML}{6A5ACD}
\definecolor{cthick}{HTML}{8BB174}
\definecolor{cpink}{HTML}{EB078E}
\definecolor{ccrust}{RGB}{222,222,222}
\definecolor{cmantle}{RGB}{252,226,204}
\definecolor{cslab}{RGB}{128,138,152}
\definecolor{cwater}{RGB}{198,224,240}
\definecolor{cmagma}{RGB}{236,112,52}
\definecolor{txt}{RGB}{85,85,85}
\begin{document}
\begin{tikzpicture}[x=1cm,y=1cm,line cap=round,line join=round]
\useasboundingbox (-0.65,-1.1) rectangle (19.7,6.9);
""" + panel(0, "thin") + "\n" + panel(10.5, "thick") + "\n\\end{tikzpicture}\n\\end{document}\n"
(HERE / "crust_schematic.tex").write_text(tex)
subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "crust_schematic.tex"], cwd=HERE, check=True, stdout=subprocess.DEVNULL)
subprocess.run(["pdftocairo", "-png", "-transp", "-singlefile", "-scale-to-x", "2400", "-scale-to-y", "-1",
                str(HERE / "crust_schematic.pdf"), str(HERE / "img" / "crust_schematic")], check=True)
for ext in ("aux", "log", "pdf"):
    (HERE / f"crust_schematic.{ext}").unlink(missing_ok=True)
print("Saved img/crust_schematic.png")
