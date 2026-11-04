# Class 4 original-style alternative — instructor review

Internal review deck: 20 slides after the instructor-authorized extension, Python fitting and summary slides, and R-squared explanation. See LIVE_DEMO.md for live/static rendering instructions. See EXTENSION_NOTES.md for the current sequence, source mapping and verification. Based on the local `feature/class4-cpx-regression` implementation, on separate branch `feature/class4-cpx-original-style`. Existing lecture, exercises, website configuration, data, split and Python helpers are unchanged.

## Review artifacts

- Source: `../class_4_slides_original_style.qmd`
- Self-contained local HTML: `../class_4_slides_original_style.html`
- Twenty-page local PDF: `../class_4_slides_original_style.pdf`
- R script: `generate_figures.R`; 14 ggplot figures in `figures/` (plus the SVG experimental schematic), each SVG and 240-dpi PNG.
- Numerical outputs: `coefficients.csv`, `fit_summary.csv`, `scores.csv`.
- Browser evidence: `checks/layout.json`, 20 screenshots in `checks/`.

## Initial prototype mapping to Lecture3.Rmd (historical)

| Original section | Prototype adaptation | Decision |
|---|---|---|
| Title; motivating reading-ability example | 1–2: CPX temperature and MgO scatter | Retain a concrete scientific question and large plot. Remove treatment/control testing. |
| Is our analysis comprehensive?; regression analysis | 2–3: why regression; conditional expectation | Local averages illustrate the conditional mean. Bins are descriptive, not another candidate model. |
| Linear regression; least squares | 4–5: linear mean; vertical residuals and minimization | Distinguish unknown parameters, candidate values and estimates. Gaussian errors are not required for OLS fitting. |
| Reading model; interpretation of coefficients | 6–7: simple slope, then three-predictor regression | Conditional associations, not chemical interventions. Intercept interpretation remains cautious. |
| How good is our model?; R² | 7: fitted versus observed; training R² | R² describes training fit, cannot establish adequacy; additional predictors cannot decrease it on identical fitting rows. |
| Model diagnostic; repeated model-fit plots | 8: paired training residual plots | Keep the original fit/diagnostic rhythm and consistent axes. Detailed Q–Q and influence work deferred. |
| Model updates; model selection/AIC | 7–9: two prespecified formulas and the planned held-out comparison | No interactions, centering, AIC search, normalization or formula revision. Comparison is assessment, not tuning. |
| Concluding remarks; extrapolating | 10: revision and scientific limits | Joint chemical support, study dependence and transfer to natural crystals. |

The original treatment assignment story, one-sided p-value conversion, causal conclusions, cartoons and repeated interaction models are omitted. Normality concerns inference assumptions, not whether a line can be fitted. Residual plots cannot prove independence or generalization. A small p-value does not establish predictive usefulness. Adding predictors may leave R² unchanged, rather than always increasing it. An RMSE is not a prediction-interval half-width.

## Visual references and differences

Read `data-analytics/Lecture3.Rmd`, `my-css.css`, and `fig_reading.R`; inspected `pics/points_with_mod1.png` and `pics/diag_mod1_v2.png`. The actual reading plots use serif axes, translucent colored point fills, thin frames and dashed light grids. Lecture3 also defines minimalist ggplot themes. This adaptation combines those references in a reproducible ggplot2 theme rather than copying existing raster figures.

Updated at instructor request: slide styling now uses the same course clean.scss theme and shared default, image, spacing and Class 4 CSS as the reference class_4_slides.pdf. Roboto body text, Roboto Slab dark headings, orange italic subheadings, pale background and a four-contributor title replace the initial dark title bars and Vera Sans typography. The acknowledgment footer is removed at instructor request; attribution remains in this internal document. R figures retain pink #eb078e, purple #6A5ACD, turquoise #20B2AA and blue #2b59c3, with Liberation Serif axes. Scatter markers are approximately 25% larger, retaining transparency; local-mean and evaluation markers are also slightly enlarged. Equations retain mathematical typesetting and color. The two-facet fit and residual plots keep the models in the same order and use common scales. Native Quarto Revealjs retains the course engine; no xaringan or new student R requirement.

| Aspect | Existing redesigned lecture | This alternative |
|---|---|---|
| Organizing question | Generalization to an unfamiliar publication | Model the conditional mean, fit it, interpret it, diagnose it |
| Sequence | Prediction workflow and validation receive early emphasis | Original regression sequence; evaluation is near the end |
| Visual language | Course clean theme; Python/matplotlib demonstrations | Same course clean slide theme; original statistical palette and serif ggplot axes; large static illustrations |
| Interaction | 27 slides; live Python cells and six exercise pages | Ten static representative slides; no exercises replaced |
| Scientific protocol | Frozen data, split and three models | Identical protocol and numerical results |
| Tradeoff | More operational detail on prediction and validation | Stronger conceptual continuity with the original; requires later Python practice |

## Proposed completion after approval — 180 minutes

| Minutes | Work |
|---|---|
| 0–20 | Motivation, data meaning and training-only scatter; 10-minute Python exploration using existing Exercise 1. |
| 20–45 | Conditional expectation, straight-line approximation, least squares; hand-read residuals, then fit in Python. |
| 45–65 | Coefficients and point predictions; existing Exercise 2 adapted as a separate alternative only if authorized. |
| 65–75 | Break. |
| 75–100 | Multiple regression, conditional interpretation, training R²; compare fixed formulas in Python. |
| 100–125 | Residuals, changing spread, unusual observations; a short Q–Q explanation distinguishes inference assumptions from fitting. Training-only diagnostic practical based on Exercise 3. |
| 125–155 | Define RMSE/MAE and publication grouping; execute the single planned comparison from Exercise 4, then reveal slide 9. Include equal-publication summaries. |
| 155–175 | Scientific support, natural-crystal transfer, model revision without holdout reuse; composition/reporting task based on Exercise 5. |
| 175–180 | Exit interpretation and recovery time. |

Expand the prototype to roughly 24–28 slides following the original sequence, with Python demonstrations between static R illustrations. Keep p-values to a brief optional discussion of conditional association and assumptions, not a sequence of directional tests. Retain the existing optional grouped-validation exercise outside the core. Any prospective model revision is discussed only; changing the fixed formulas/protocol requires separate authorization. No full lecture or alternative exercise set has been implemented.

## Reproduction and checks

From the repository root:

```sh
Rscript Data-Science/class4/original_style/generate_figures.R
quarto render Data-Science/class4/class_4_slides_original_style.qmd
python Data-Science/class4/original_style/check_preview.py
```

R requires ggplot2, svglite and digest. Python 3 is used to run the existing split verifier. Browser verification/export requires Playwright and `/usr/bin/brave-browser`; this session used `/tmp/elste-class4-env/bin/python`. These are development dependencies, not student dependencies. Quarto embeds figures and MathJax in the HTML; initial rendering may need the MathJax download. `R-session.txt` records the R environment.

The script checks the CSV hash and invokes the existing manifest verifier without writes. All exploratory/fitting/diagnostic figures use training rows. The least-squares display shows 12 deterministically selected records, explicitly labeled; the line is fitted to all 1,402 training observations. Evaluation appears only on slide 9. The training-mean baseline is reused unchanged on evaluation rows. Scores include both pooled and equal-reference RMSE/MAE.

Final HTML: ten slides checked at 1440×900 and 1024×768; no text, figure or equation overflow; all slide images load and no MathJax error nodes. Final PDF: ten pages; all pages rasterized and visually inspected. No clipped figures, equations or text found. Quarto renders without fenced-div warnings. The alternative is not added to the default website render list or navigation; it is a local, explicitly rendered preview.

## Rights and attribution — do not release

The original lecture names Stéphane Guerrier, Dominique-L. Couturier and Yuming Zhang, and thanks Lionel Voirol, Wenfei Chu and Jun Wu. It displays CC BY-NC-SA 4.0; existing course/slide notices differ. Preserve these records and clarify applicable terms before release. The title retains the original lecture contributors alongside the Earth Science instructor; the requested acknowledgment-footer removal does not remove the attribution record here. These internal production notes are not student slides.

Dataset rights and scientific provenance remain unresolved. See `../../Data/cpx_README.md`: the private source repository does not establish a redistribution license, original compilation attribution, concentration/normalization and iron conventions, zero semantics or reliable experimental identities. No wt% assumption is introduced. Jorgenson et al. is context, not verified compilation attribution. Course licenses do not grant dataset rights.

Do not distribute the CSV/RDA, data-bearing figures, HTML or PDF, or include them in a public PR until permission and attribution are resolved. All work here is local; no push, PR, merge, deployment or publication. The evaluation is the frozen teaching assessment, not wholly blind external research validation and not validation of natural-crystal thermometry.
