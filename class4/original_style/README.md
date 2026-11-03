# Class 4: experimental clinopyroxene regression

The canonical source is `../class_4_slides.qmd`: 41 slides, ten editable Python demonstrations, and 32 figure assets. Edit that source, then regenerate `../class_4_slides_static.qmd`. The static companion contains executed fitted-summary tables and displayed Python code; live execution requires internet access for Pyodide 0.28.1 and its packages. Existing exercises keep their original case studies and will be updated separately.

## Reproduce from the course repository root

Use Quarto, Python with NumPy, pandas, SciPy, statsmodels, Matplotlib and IPython, and R with ggplot2, svglite and digest. Install Liberation Serif for the figure typography. The existing theme loads Roboto/Roboto Slab. Tested Python packages: NumPy 2.4.6, pandas 2.3.3, SciPy 1.16.2, statsmodels 0.14.6, Matplotlib 3.10.8 and IPython 9.14.1. Use statsmodels 0.14.6 to reproduce the supplied static summary formatting. Course-wide Python/Jupyter and R dependencies remain necessary for other pages.

```sh
python class4/prepare_cpx_split.py --verify
# Point CPX_PYTHON at the activated environment's Python executable.
CPX_PYTHON="$(command -v python)" Rscript class4/original_style/generate_figures.R
python class4/original_style/verify_numerics.py
python class4/original_style/make_static.py
quarto render class4/class_4_slides.qmd
quarto render class4/class_4_slides_static.qmd
quarto render class4/class_4_intro.qmd
python slides2pdf.py --check class_4_slides
python slides2pdf.py --from-rendered --no-compress class_4_slides
```

Install DeckTape (`npm install -g decktape`) for PDF export. `DECKTAPE_CHROME_PATH` selects a Chromium executable. Omitting `--no-compress` enables the existing Ghostscript compression. The exporter writes `_site/class4/class_4_slides.pdf`; the overview download points there. A current source copy is also tracked at `class4/class_4_slides.pdf`, so partial renders cannot copy an obsolete case study. After validating a new PDF, refresh that copy:

```sh
cp _site/class4/class_4_slides.pdf class4/class_4_slides.pdf
```

A complete build uses `python render_all.py` after static regeneration. No publication step is required for local verification. The existing Class 3 static converter is unchanged.

The R entry point verifies the pinned CSV/RDA and frozen split, invokes Python helpers, then creates all needed intermediate CSVs before reading them. The CSVs, PNG alternatives, HTML summary sidecars and R session log are generated and ignored. Retained SVG figures and the portrait allow text-only rendering without R. The experimental-workflow SVG is hand-authored; the portrait's provenance is in `figures/george_box_prompt.md`.

## Verification

`verify_numerics.py` checks frozen inputs and group separation, executes all ten cells independently, checks OLS against NumPy and regenerated R results, and verifies AIC, grouped CV, test errors and mean/prediction intervals. Regenerate figures first. No exercise files are required by these checks.

Install Playwright and its Chromium browser (or set `BROWSER_EXECUTABLE` to an installed Chromium/Brave executable), plus Poppler (`pdftotext`). The checks default to `/usr/bin/brave-browser`; override that path on other platforms. Run after rendering and PDF export:

```sh
python class4/original_style/check_preview.py
CPX_URL_PREFIX=/ python class4/original_style/check_live.py
CPX_URL_PREFIX=/Data-Science/ python class4/original_style/check_live.py
```

Set `CPX_CHECK_OUTPUT` to separate directories to retain each run's evidence. Both checkers cover 1440×900 and 1024×768. Live checks execute, edit and rerun each cell and inspect all slide layouts, image loading and browser failures. Static checks inspect all slides and the actual exported PDF's 41 pages and content. Screenshots/reports are ignored and are not course assets.

## Statistical and scientific scope

Preserve the exact data, 1,402/294 training/test allocation, deterministic grouped folds, formulas and seeds. Oxides retain reported concentration units; no unit, normalization or iron convention is inferred. Grouping reduces obvious dependence/leakage but does not establish true independent experiment identities. Classical intervals and coefficient inference use working assumptions that may be imperfect for clustered experimental observations. The previously inspected test set is a teaching comparison, not a new blind model-selection evaluation. No validation for natural crystals is claimed. See `../../Data/cpx_README.md` for checksums, provenance, sharing permission and remaining metadata questions.

## Attribution and compatibility

The original lecture credits Sébastien Biass, Stéphane Guerrier, Dominique-L. Couturier and Yuming Zhang, with acknowledgments to Lionel Voirol, Wenfei Chu and Jun Wu. The adapted title credits Sébastien Biass, Stéphane Guerrier and Samuel Orso. The course clean theme (Grant McDermott) and Quarto Live extension (George Stagg/r-wasm) remain unchanged, including their notices. The Box quotation is attributed on its slide; the portrait is a generated illustration without a supplied reference photograph. Confounding figures use original vector illustrations and explicitly simulated data.

Upstream's Class 4 source declares `CC BY`, while the website footer and other slide metadata use `CC BY-NC-SA 4.0`. The existing Class 4 declaration and website notices are preserved. Maintainers should clarify their intended relationship; this migration assigns no new license. Permission to share CPX data in this teaching project was confirmed by Samuel Orso on 2026-11-03; that permission does not invent a general dataset license or resolve scientific metadata.

Compatibility changes are limited to canonical filenames, deployment paths and checks, the static PDF mapping, navigation/overview text, and `mathjax3.html`. The latter selects Reveal's MathJax 3 adapter when an older Quarto template erroneously selects its MathJax 2 adapter, preserving the generated external or embedded MathJax URL. It also supplies an empty favicon to avoid irrelevant browser requests. No statistical calculations or slide-body explanations were changed.
