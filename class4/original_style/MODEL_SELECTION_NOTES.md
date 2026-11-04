# Model selection and confounding extension — internal notes

The current order places the unchanged final test comparison at 31, followed by AIC, model search, cross-validation, uncertainty intervals and the confounding story. The quadratic residual review remains at slide 22:

| Slide | Content | Suggested time |
|---|---|---|
| 22 | Additive versus full-quadratic training residuals | 3–4 min |
| 31 | Unchanged frozen final test comparison | existing |
| 32 | AIC principle | 4–5 min |
| 33 | AIC comparison and Python | 4–5 min |
| 34 | Many candidate models and forward AIC search | 4–5 min |
| 35 | Five-fold cross-validation principle | 4–5 min |
| 36 | Applied CV and Python | 5–7 min |
| 37 | Confidence versus prediction intervals | 5–6 min |
| 38 | Classical formulas and Python intervals | 5–7 min |
| 39 | Simulated ice-cream/shark association | 3–4 min |
| 40 | Seasonal reflection | 2–3 min |
| 41 | Warmer weather as a common cause | 3–4 min |

Allow approximately 39–48 minutes, including brief code demonstrations and discussion.

## Reproduce

Run from the repository root, with a Python environment containing the existing lesson dependencies (NumPy, pandas, scipy, statsmodels, matplotlib, IPython):

```sh
CPX_PYTHON=/path/to/python Rscript Data-Science/class4/original_style/generate_figures.R
/path/to/python Data-Science/class4/original_style/make_static.py
quarto render Data-Science/class4/class_4_slides_original_style.qmd
quarto render Data-Science/class4/class_4_slides_original_style_static.qmd
/path/to/python Data-Science/class4/original_style/check_preview.py
/path/to/python Data-Science/class4/original_style/check_live.py
```

The main R generator sources `generate_selection_figures.R`, which runs `prepare_selection.py`, verifies the AIC/CV calculations independently in R, and produces ggplot2 SVG/PNG figures. `generate_confounding_svg.py` uses Python's standard library to draw original vector illustrations; there are no additional package dependencies. The preview checks need Playwright, Brave and Poppler; the live check also needs internet to load the existing Pyodide runtime.

## Calculations

| Model | Regression coefficients | AIC | Delta AIC | Pooled five-fold CV RMSE (°C) |
|---|---:|---:|---:|---:|
| MgO | 2 | 17281.138779 | 832.944382 | 115.810008 |
| MgO + Al₂O₃ + TiO₂ | 4 | 16553.746082 | 105.551685 | 90.115129 |
| Full quadratic, three oxides | 10 | 16448.194397 | 0 | 90.031564 |

The quadratic retains the same intercept, three main effects, three squares and three pairwise interactions already used in the lecture. All models use the same 1,402 frozen training observations. No new test evaluations are performed.

The displayed AIC follows `statsmodels` OLS: `-2*llf + 2*k`, where k counts regression coefficients including the intercept. R's `AIC(lm)` also counts the fitted error variance and is uniformly 2 larger here; this common constant leaves the rankings and delta AIC unchanged. The generator explicitly uses the statsmodels convention and checks agreement. These are working Gaussian OLS likelihoods; the compiled data do not establish independent, constant-variance Gaussian errors. AIC is a relative criterion, not an error measured in °C or proof that the quadratic is scientifically correct.

`cv_folds(train)` deterministically balances the existing validation groups across five folds, with sizes 281, 281, 280, 280 and 280. It sorts groups by descending size, then SHA-256 of `ELSTE-Class4-cv-v1\n` plus the group key, assigning each to the least-loaded fold. Related source groups stay intact internally. This preserves the original frozen train/test allocation and does not involve repeated test-set selection. The student-facing slides avoid publication-level protocol details.

The same folds are used for all three models; each training observation receives exactly one prediction from a model fitted to the other four folds. The RMSE is the square root of all pooled out-of-fold squared errors, rather than an unweighted mean of the five fold RMSEs. Small plotted points are individual fold errors, not confidence intervals. The three-oxide and quadratic results are essentially tied on this split; no uncertainty or universal ranking is asserted. The earlier polynomial-complexity illustration uses its own pre-existing validation subset, which is unchanged.

The original final-test results (now slide 31) remain the predefined model comparison. Because those results were already shown, the new training-only AIC/CV exercise must not be represented as followed by a fresh independent final-test assessment. Do not tune candidate models using the displayed test results.

## Confounding, sources and rights

The ice-cream/shark scatterplot is simulated teaching data (seed 3642), not empirical shark records. Sales and incident counts are generated independently conditional on a three-level weather variable. The positive marginal association illustrates a common cause; it does not show ice-cream consumption causing shark incidents. Warmer beach weather can increase ice-cream demand and time spent swimming, creating more opportunities for incidents. The vector illustrations are original, with no copied reference images.

The seasonal reflection is a separate teaching simulation (seed 3643), with monthly Poisson counts around a specified summer-peaking mean `1 + 9*exp(-0.5*((month-7)/1.8)^2)`. The pink curve is that generating mean, not a fitted empirical relationship; June–August is explicitly Northern Hemisphere summer. It does not assert that all locations or shark populations share this pattern. `sharks_seasonal_simulated.csv` preserves the plotted counts. No CPX observations, models or results change. The association slide (now 39) uses integer incident-count ticks, and both vector sharks have swept fins and an asymmetric tail. The removed on-slide conceptual credit remains here.

Conceptual inspiration: [Stéphane Guerrier, Block 6](https://stephaneguerrier.github.io/ai-assisted-data-analysis/#block6). Technical references: [statsmodels OLS AIC](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.aic.html) and [cross-validation documentation](https://scikit-learn.org/stable/modules/cross_validation.html). The implementation uses the existing packages, without adding scikit-learn.

All existing dataset provenance, permission restrictions and attribution warnings in `REVIEW.md` and the data documentation remain applicable. These additions do not authorize distribution or publication. No commit, push, merge, PR, upload or publishing is part of this change. Previously generated sharing ZIPs remain historical snapshots and have not been refreshed.

## Quadratic residual review

Slide 22 follows the full-quadratic summary, using the same layout and ggplot2 palette as the earlier diagnostic slide. The new `quadratic_residuals` SVG/PNG compares additive and full-quadratic residuals against fitted temperatures, with common axes and pink zero lines. Both panels use the identical 1,402 frozen training observations. It invites students to examine curvature, changing spread, clusters and unusual points; no test observations or model changes are introduced. All pre-existing score tables and figure assets are preserved. The current ending is test comparison 31, AIC 32–33, model search 34, CV 35–36, intervals 37, formulas/code 38, association 39, seasonal reflection 40, common cause 41.

## Pointwise uncertainty intervals

Slide 37 uses the unchanged simple training regression `T_C ~ MgO_Cpx`. `predict.lm(..., interval="confidence")` and `interval="prediction"`, both at level 0.95, generate `simple_regression_intervals.csv` on 400 MgO values spanning the training range. The target prediction 1,200 °C corresponds to MgO = 17.7948549265 (reported units); the exact computed predictor and bounds are stored in `simple_regression_prediction_1200.csv`. Mean-response CI: 1193.1263–1206.8737 °C; new-individual PI: 974.6542–1425.3458 °C. The slide rounds these to whole degrees. All endpoints were independently checked against statsmodels `get_prediction(...).summary_frame(alpha=.05)` to absolute tolerance 1e-8. The generator does not use the test set for intervals.

These are pointwise classical OLS intervals under a correctly specified linear conditional mean, independent Gaussian errors with constant variance, and a new observation from the same model at the specified MgO. The compiled data do not demonstrate those assumptions. Neither natural crystals nor other experimental settings receive a coverage claim. The narrow mean band reflects coefficient uncertainty; the prediction band adds individual residual variation. Classical half-widths are `t_(n-2,0.975) * s * sqrt(h0)` and `t_(n-2,0.975) * s * sqrt(1+h0)`, where `h0 = 1/n + (x0-xbar)^2 / sum((xi-xbar)^2)`. The additional 1 is the residual-variance contribution. The student slide prioritizes a large figure, interval meanings, numerical example, explicit assumptions and a reflection question over extra formulas.

The final test comparison was moved as an intact slide after AIC/CV; its displayed models, metrics and calculations are unchanged. The teaching sequence does not make the previously viewed test results a fresh independent evaluation of any newly selected model. No exercises or pinned data changed; release restrictions and original attributions remain in force.

## Candidate models and forward search

The model-search slide (now 34) follows the three-model AIC comparison. The existing full quadratic supplies nine non-intercept candidate terms: three linear terms, three squares and three pairwise products. With each term independently included/excluded, there are 2^9 = 512 subsets, including the intercept-only and full models; every model includes an intercept. The ggplot2 figure groups these unrestricted candidates by size, using choose(9,k) for k=0,…,9. Scientific restrictions, including retaining main terms with squares/interactions, reduce that count. This is a combinatorial illustration, not an enumeration or fitting of CPX candidate models. No forward search is actually run and the frozen comparisons are unchanged.

The slide explains an intercept-only starting point, testing each allowed one-term addition, accepting the lowest AIC only when it improves the current AIC, and stopping otherwise. A forward path can miss the global minimum across allowed subsets. This leads into the existing CV slides without suggesting that AIC alone demonstrates predictive accuracy. If implementing model selection within cross-validation, repeat the selection inside each fitting fold to avoid selection leakage; this slide introduces the strategy only. Technical reference: [official stepwise AIC documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/step.html). No new student-facing reference to R or executable code is introduced.

The regenerated original CV CSVs can differ in their final floating-point digits after restoring the temporary Python environment (the reported RMSE values, full model specifications, fold allocations and all existing plot assets are unchanged). R/Python agreement remains within the existing absolute tolerance of 1e-8. No new model search or evaluation was performed for the model-search slide.

At the instructor’s latest request, the intact final test-error comparison moved from slide 36 to immediately after slide 30 (now slide 31). Slides formerly 31–35 shift to 32–36; slides 37–40 are unchanged. No slide bodies, figures, data or calculations changed in this reorder.

## Classical interval formulas and Python

New slide 38 follows the confidence/prediction-band illustration. It defines the simple-regression prediction, h0, the residual variance estimate with n−2 degrees of freedom, and the two-sided 95% t critical value. The CI half-width is t*s*sqrt(h0); the PI half-width is t*s*sqrt(1+h0). The colored extra 1 adds the estimated residual variance s². These are pointwise classical intervals, under the assumptions and experimental scope stated on slide 37.

The standalone Python cell loads the frozen training observations, refits the same simple MgO model, chooses x0 so its fitted mean is 1,200 °C, and calls `simple.get_prediction(new).summary_frame(alpha=0.05)`. It prints the fitted mean and the four confidence/prediction limits. The `mean_ci_*` and `obs_ci_*` column meanings are explained on the slide. Exact outputs: mean 1200.0, mean CI 1193.1–1206.9, individual PI 974.7–1425.3 °C (one decimal). An independent direct calculation using the displayed leverage, residual-variance and t formulas agrees with the API to 1e-8. Training data only; no data, model specifications, frozen split or evaluation scores change.

The HTML cell is editable with Run Code; the PDF contains the identical plain Python code. No exercises change. Allow 5–7 minutes for the new slide. Later slides shift to association 39, seasonal reflection 40 and common-cause explanation 41; all earlier slide content is preserved.
