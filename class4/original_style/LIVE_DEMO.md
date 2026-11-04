# Regression Python slides

The current deck has 41 slides. A residual-diagnostics slide follows the quadratic summary at slide 22. The unchanged final test comparison is now slide 31, followed by AIC (32–33), model search (34), and cross-validation (35–36). Confidence/prediction intervals follow at 37, with formulas and executable Python at 38; confounding occupies slides 39–41. Slides 33 and 36 add editable AIC and five-fold CV examples to the seven existing Python slides. The earlier slide content is preserved; subsequent slide numbers shift by one. All ten Python slides use the frozen training data. See `MODEL_SELECTION_NOTES.md` for the new sequence, reproducibility details and scientific precautions.

## HTML

The source uses the same existing Quarto Live filter and Pyodide Run Code editor as Class 3, referenced explicitly through `../_extensions/r-wasm/live/live.lua`. Slides 7, 9, 12, 14, 20, 21, 25, 33, 36 and 38 are executable; all statistical figures remain R/ggplot2 assets. The provided `cpx_lesson.py` checks the pinned data and frozen allocation; `train, _ = load_data()` supplies the unchanged training observations to `smf.ols("T_C ~ MgO_Cpx", data=train).fit()`.

Render from the repository root:

```sh
quarto render Data-Science/class4/class_4_slides_original_style.qmd
```

Use HTTP to open the interactive HTML, rather than opening it directly as a file. A localhost preview, from the repository root:

```sh
python3 -m http.server 8000 --bind 127.0.0.1 --directory Data-Science
```

Then open `http://127.0.0.1:8000/class4/class_4_slides_original_style.html#/fit-a-simple-regression-in-python`. This is local preview only. Pyodide/Python packages are downloaded on initial startup, so internet access is required. Keep the generated `class_4_slides_original_style_files/` directory and the course Data folder beside their original relative paths. This live HTML is not self-contained.

## PDF and offline HTML

The static source is generated from the live source; never hand-edit both decks:

```sh
python3 Data-Science/class4/original_style/make_static.py
quarto render Data-Science/class4/class_4_slides_original_style_static.qmd
python3 Data-Science/class4/original_style/check_preview.py
```

The static-source generator now runs the summary code to produce its real HTML tables. Use a Python environment with NumPy, pandas, scipy, statsmodels, matplotlib and IPython installed. The last development command requires Playwright, Brave at `/usr/bin/brave-browser`, and Poppler's `pdftotext`; it checks layouts and exports the static HTML to `class_4_slides_original_style.pdf`. It does not require loading Pyodide. `class_4_slides_original_style_static.html` remains the self-contained offline preview.

The static slides present ordinary selectable, syntax-highlighted Python without line numbers or execution controls. Slides 9 and 14 also contain the actual model summaries, generated from exactly the code printed beside it. Local Python needs NumPy, pandas, scipy, statsmodels, matplotlib and IPython (the supplied helper imports these). Run copied code from `Data-Science/class4/`, with `cpx_lesson.py` there and the original data under `../Data/`.

## Verification

- Each Python slide was inserted as a separate addition; the existing slide bodies remain unchanged.
- All ten generated static Python blocks are identical to their corresponding HTML editors' initial code.
- `check_live.py` serves a temporary loopback preview and executes the editor at 1440×900 and 1024×768. Editing code, rerunning it, restoring it, and running again are checked at both sizes. The simple-model outputs reproduce intercept 707.80 and slope 27.66. The new multiple-model output reproduces intercept 576.32 and slopes 28.37, 25.10 and −18.62. The quadratic-model slide reproduces training R-squared 0.698. The MgO degree-10 model reproduces training R-squared 0.505. The current layout check covers 82 live layouts. The new AIC and CV cells reproduce the R figure values and remain editable.
- `check_preview.py` checks all 41 static slides at both sizes (82 layouts) and the final 41-page PDF, including text markers from all additions. The Python block extracted directly from PDF page 7 executes unchanged without removing line numbers; it reproduces the same coefficients.
- Dataset, split, the original three evaluated model specifications, and original figure-generation calculations are unchanged. The degree-10 MgO model is a training-only illustration. The three-oxide quadratic is also evaluated on the frozen test set following the instructor’s explicit request. Existing rights/provenance notes still apply. No commit, push, PR, merge or publication.

Previously created sharing ZIPs are snapshots from before this interactive-slide addition; a new editable package must also include `cpx_lesson.py`, the Live filter/extension, generated live resource directory, and the new static-source generator.

## Fitted-summary slide

`display(simple.summary(slim=True))` renders the genuine statsmodels summary as two native HTML tables. The PDF generator executes the same self-contained code in a fresh namespace and embeds the resulting `Summary.as_html()` output. It also saves `simple_ols_summary.html` for inspection. No values are hand-entered. The training fit has 1,402 observations, intercept 707.8019, MgO coefficient 27.6596 and R-squared 0.4472912. The summary's technical covariance note is replaced with the clearer constant-variance and uncorrelated-errors explanation in the live display and PDF. The numerical tables and Python code are unchanged. Student-facing text introduces only the Python API and distinguishes training fit from prediction accuracy.

## Three-oxide polynomial slide

Slide 19 follows the MgO polynomial slide and mirrors slide 13's colored equation and paired observed-versus-fitted plots. Its full quadratic contains an intercept, three linear terms, three squared terms, and three pairwise products: ten coefficients. The R script fits it on training observations only, verifies full rank, and writes `three_oxide_polynomial_training.csv` and `three_oxide_polynomial_coefficients.csv`. Training R-squared rises from 0.6719542 for the additive model to 0.6983379; training RMSE decreases from 88.3960 to 84.7668 °C. The instructor later requested an additional frozen-test evaluation for this fixed quadratic. The original three models’ evaluation values remain byte-identical; the additional quadratic results are stored separately. Allow about 4–5 minutes to discuss the extra terms and distinguish training fit from predictive benefit.

The slide 20 Python formula `(MgO_Cpx + Al2O3_Cpx + TiO2_Cpx)**2` expands to the three main effects and three pairwise interactions; separate `I(...**2)` terms supply the three squares. The implicit intercept makes ten coefficients, matching the R training model. Its code fits training observations only and prints training R-squared; it fits training data only. The instructor later requested adding this unchanged model to the concluding test comparison.

## Degree-10 MgO slides

Slides 23–24 now follow the R-squared comparison on slide 22 and precede the simulated overfitting example on slide 25. The real CPX illustration compares the existing MgO quadratic with a degree-10 MgO polynomial using the same 1,402 training observations. The R script uses an orthogonal polynomial basis; Python fits powers of MgO rescaled to [-1,1], with an intercept. Both span exactly the same degree-10 polynomial model. Eleven coefficients are estimated. The 1,200 plotted curve predictions from Python and R agree within 1e-6 °C. Degree-10 training R-squared is 0.5047721790 and RMSE is 108.6095294 °C; the quadratic training RMSE remains 114.5411 °C. No test predictions or complexity selection are performed for this illustration. All pre-existing slides, figures, and frozen evaluation results are unchanged. The code prints training R-squared, and its rescaling limits must be retained when predicting new observations. Allow approximately 5–7 minutes for the two added slides.

## Empirical complexity slide: TiO₂

Slide 29 now replaces the earlier MgO graph and inset with a TiO₂-only example. At the instructor’s request, MgO, Al₂O₃ and TiO₂ degree-1–10 validation curves were examined on the same already-fixed subset. TiO₂ was chosen for visual clarity as a teaching illustration, not as a claim of superior prediction or a new model-selection result. The curves are plotted as measured without smoothing. Degree 0 adds an intercept-only baseline, clarifying the initial decrease in training and validation error. At the instructor’s request, the displayed graph ends at degree 9. The complete numerical table still retains degrees 0–10, including the decline in validation RMSE from degree 9 to degree 10.

The previously approved validation-within-training rule is unchanged: sort original training validation-group keys by SHA-256 of `ELSTE-Class4-polynomial-validation-v1\n` plus the key, and reserve the first ceil(0.2 × number of groups). Linked source groups stay together internally. There are 1,177 fitting rows and 225 validation rows. The current manifest `polynomial_complexity_validation_split.csv` is byte-identical to the previous `mgo_complexity_validation_split.csv`. Neither replaces or modifies the original frozen `Data/cpx_teaching_split.csv`. The 294-row final test set is not used for these polynomial fits, scaling or errors. The original three-model test comparison is unchanged.

Learn TiO₂ limits (0–5.83) from the fitting subset and transform to [-1,1]. Apply those same limits to validation observations; all validation TiO₂ lies within that fitting range. Degree 0 is the fitting-mean model; degrees 1–10 use orthogonal polynomial terms of rescaled TiO₂. Independent Python calculations with Legendre terms reproduce all 22 RMSE values to absolute tolerance 1e-8. The reproducible table is `tio2_complexity_errors.csv`, and the current figure is `figures/tio2_complexity_errors.svg` (plus PNG).

Training RMSE decreases from 145.8634 °C at degree 0 to 126.7039 °C at degree 10. Validation RMSE falls from 194.5751 °C at degree 0 to 167.2929 °C at degree 5, and rises to 273.4944 °C at degree 9 and 246.2084 °C at degree 10. See the generated CSV for full precision. The graph uses a labelled 100–300 °C vertical axis to focus on the observed error range; no data are clipped. This illustration does not establish an optimal degree or a universal curve. The old MgO artifacts are retained as internal history and are not referenced by the current presentation. Allow 4–5 minutes for discussion. Rights/provenance restrictions remain in force.

## Coefficient p-value slide

New slide 10 follows the fitted summary. It states the two-sided hypotheses H0: beta1 = 0 and H1: beta1 != 0 for the simple MgO slope. The existing conventional OLS fit gives slope 27.6595736654, standard error 0.8217401670, t = 33.6597561806 (1,400 residual degrees of freedom), and two-sided p = 1.7807853518e-182. The slide rounds the first three to 27.66, 0.82, and 33.66 and reports p < 0.001, explaining the summary's rounded 0.000. Rejection at a stated 5% level is explicitly conditional on the OLS assumptions. This is an explanation of the conventional model output, not a claim that the compiled analyses satisfy independent, normal, constant-variance errors. The hypothesis concerns a linear mean relationship, not independence of chemistry and temperature in all possible senses. The slide distinguishes association from causation, coefficient effect size, predictive performance, and the probability of the null hypothesis. No coefficients, models, data, split, graphics, or evaluation calculations were changed. Allow approximately 4–5 minutes.

## Multiple-regression summary and interpretation

Slides 14–15 were added after the multiple-regression overview on slide 13. Slide 14 uses the same editable Python/Run Code layout as slide 9, independently loading the frozen training observations and fitting `T_C ~ MgO_Cpx + Al2O3_Cpx + TiO2_Cpx`. `display(multiple.summary(slim=True))` produces the actual compact statsmodels tables. `make_static.py` executes each summary block in a fresh namespace and writes both `simple_ols_summary.html` and `multiple_ols_summary.html`; no displayed table values are hand-entered. The existing clearer standard-error note is applied to both live summary slides and both static tables. The multiple-model summary has intercept 576.3177, slopes 28.3747, 25.1036 and -18.6177, R-squared 0.6719542 and 1,402 observations.

Slide 15 follows the style of the simple coefficient illustration on slide 8. The new R/ggplot2 `multiple_coefficient` figure varies MgO from 15 to 16 while holding Al₂O₃ = 5.20 and TiO₂ = 0.83, taken from a training record near MgO = 15. Its predicted-temperature difference is exactly the fitted MgO coefficient (28.37 °C after display rounding). The slide also interprets the Al₂O₃ and TiO₂ slopes, always holding the other included predictors fixed. These are conditional model associations, not chemical interventions or claims of physical feasibility for arbitrary compositions. Previously existing figures, slide bodies, data, frozen allocation and final test results are unchanged. Allow approximately 6–8 minutes for these two slides.

## R-squared across nested models

The R-squared comparison is now slide 22, following the quadratic summary on slide 21 and the fitting code on slide 20. It compares ordinary training R-squared on the same 1,402 observations with intercepts: baseline 0, MgO 0.4472911551, additive three-oxide model 0.6719541907, and full quadratic three-oxide model 0.6983378896. The latter retains all three linear terms and adds three squares and three pairwise interactions, giving ten coefficients. The R script reuses the already-fitted model, writes `r_squared_models.csv`, and generates the four-row `r_squared_models` plot. Values match independent Python fits. Its labelled axis remains 0–1.

The key statement remains non-decrease rather than guaranteed strict increase: the larger model contains the smaller when additional coefficients are zero. The rule concerns ordinary training R-squared for nested OLS models on the same response/rows, with an intercept; it does not guarantee improved prediction. Original coefficients, data and frozen allocation are unchanged.

## Instructor-requested quadratic prediction comparison

Slide 30 now includes the same fixed full quadratic three-oxide model, at the instructor’s explicit request. It was specified and fitted on training data before its test errors were calculated; the frozen 294 evaluation rows are used without refitting, rescaling decisions, degree selection or tuning. Test RMSE is 97.4321594245 °C and MAE is 80.0427305327 °C, displayed as 97.4 and 80.0. The original baseline, MgO and additive multiple-model values remain unchanged, and the original six-row `scores.csv` is byte-identical to the previous version.

The separate `quadratic_evaluation.csv` stores the extra row under the same scoring conventions; `evaluation_models.csv` combines the four evaluation rows used by the figure. Original internal publication-weighted calculations are also reproduced for the additional model, but these details remain absent from the student slide. Independent Python calculations reproduce the quadratic training R-squared and all four pooled/internal error metrics to absolute tolerance 1e-9. The final plot retains the existing MAE/RMSE colors, formulas and closing geological question. The degree-10 MgO and TiO₂ examples remain outside this final test comparison. No superiority beyond this observed error comparison or natural-crystal validation is claimed.

## Quadratic summary and revised ending

A new slide 21 displays the full three-oxide quadratic's actual `quadratic.summary(slim=True)` output, using the same Run Code / static-code-plus-summary pattern as slide 14. The reproducible static generator now recognizes `quadratic` as well as `simple` and `multiple`, saving `quadratic_ols_summary.html`. The three summary slides share the clarified standard-error note; the quadratic's genuine condition-number note is retained. The new slide uses scoped 18-pixel code and 17-pixel table styles so that all ten coefficients and notes fit without changing the earlier summary slides. The model and its evaluation values are unchanged.

Using the instructor's slide numbers before this edit: former slides 25–26 now follow former slide 21, and former slide 24 follows that pair. With the new summary inserted, the final sequence from slide 20 is: fitting code (20), quadratic summary (21), R-squared comparison (22), MgO degree-10 figure (23), its Python code (24), simulated overfitting (25), Box quotation (26), training/test explanation (27), schematic complexity tradeoff (28), TiO₂ empirical tradeoff (29), and prediction errors (30). The former slide 30, Model revision and scientific limits, is removed from the student presentation. Internal provenance, rights restrictions and scientific-limit documentation remain in place. There are still 30 slides because one slide was added and one removed. Retained slide bodies, figures, data, fitted specifications, frozen allocation and all score tables are unchanged. Allow 4–5 minutes for the added summary.

## Confidence and prediction intervals in Python

Slide 38 presents the classical simple-regression formulas beside a self-contained `get_prediction(...).summary_frame(alpha=0.05)` example. The colored extra 1 in the prediction half-width supplies residual variance. The output key sits below the formulas, leaving the full right column for the editable code and its five printed values. The PDF uses identical copyable code. The new example and direct classical formulas agree to 1e-8. `CPX_LIVE_CASES=interval-code` restricts `check_live.py` to executing/editing/rerunning this cell at both viewport sizes, while still checking all 41 live layouts; the default checks all ten cells. An additional check prevents the code/output or its key from overlapping the assumptions note. All earlier slide bodies are preserved.
