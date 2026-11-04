# Six-slide extension: instructor notes

At the six-slide-extension stage the deck had 17 slides. It now has 20 after the instructor-requested Python fitting, summary, and R-squared slides; see LIVE_DEMO.md. The six-slide sequence follows residual diagnostics and precedes the original fixed-model prediction assessment. This is a local review version; no commit, push, merge, PR or publication was performed.

## Pedagogical source

Read `/home/orsos/Downloads/LiveSession_regressionI_fr_2026.pptx` (24 slides), especially slides 5–6 and 9–14: fitting versus evaluation, Box's quotation, and prediction error on observations excluded from fitting. Its business examples and visual design are not copied. The random 80/20 recipe on its slide 24 is deliberately not adopted. The pre-existing course typography, CSS, palette and ggplot theme remain in use.

Box's wording and attribution were checked against the original 1979 chapter, *Robustness in the Strategy of Scientific Model Building*, in *Robustness in Statistics*, pp. 201–236, with the quotation on p. 202. Primary scanned chapter: https://gwern.net/doc/statistics/decision/1979-box.pdf ; publisher record: https://www.sciencedirect.com/science/chapter/edited-volume/pii/B9780124381506500182 . No external illustration was copied.

## Final slide sequence

1. Linear regression (title)
2. From controlled experiments to temperature prediction
3. Motivating example: experimental clinopyroxene
4. Regression models a conditional mean
5. Linear regression: a straight-line mean
6. Least squares: choose the line
7. What does a coefficient mean?
8. Multiple regression: more chemical information
9. Model diagnostics: what remains unexplained?
10. From straight lines to curves
11. Polynomial regression
12. All models are wrong, but some are useful
13. Training and test datasets
14. Overfitting
15. Model complexity versus out-of-sample error
16. How accurate are our temperature predictions?
17. Model revision and scientific limits

## Additional teaching time

| New slide | Suggested minutes |
|---|---:|
| Straight lines to curves | 4 |
| Polynomial regression | 6 |
| Box: models and usefulness | 3 |
| Training and test datasets | 5 |
| Overfitting and conceptual question | 6 |
| Complexity versus prediction error | 5 |

About 29 minutes of explanation; allow **30–35 additional minutes** including discussion. The existing prediction-results slide follows at position 16; the scientific-limits slide closes at position 17. No new exercise set or full-session scheduling change is introduced.

## Numerical and scientific boundaries

The CPX quadratic is `T_C ~ MgO_Cpx + I(MgO_Cpx^2)`, fitted only on the original training observations. Its training RMSE is 114.5411 °C versus 114.7397 °C for the straight line. The slide reports the small change without claiming improved test predictions. It is an illustration of a model class, not a proposed physical law or a fourth model in the frozen test comparison. No CPX test predictions are computed for the quadratic.

All original CPX data and split hashes are unchanged. The existing score CSV is byte-identical to the pre-extension version; all 24 pooled/equal-reference numerical values agree with the frozen Python snapshot to absolute tolerance 1e-9. Reference-level details remain internal and do not appear in the new student slides. Original coefficient outputs are unchanged.

The overfitting example is separate simulated data with seed 2026, 18 evenly spaced predictors, a quadratic generating mean, and Gaussian noise with SD 0.3 in arbitrary units. Degrees 1, 2 and 10 fit the same records using stable orthogonal polynomial calculations; they remain ordinary degree-k polynomial fits. An independent simulated sample of 2,000 records confirms the intended contrast. Training RMSE: 0.6130, 0.2593, 0.1937. Independent simulated RMSE: 0.6912, 0.3418, 0.3882. These simulation metrics are not CPX results and are not used to tune the CPX models. The complexity/error curves are explicitly schematic, not derived from these scores, and their U-shape is not asserted to be universal.

## Reproduction and visual verification

Run `generate_figures.R`, render `../class_4_slides_original_style.qmd`, then run `check_preview.py` using the development Python environment described in REVIEW.md. The R script produces SVG and 240-dpi PNG figures and writes training-only polynomial and simulation metric CSVs. Existing training/test scatter SVG is reused byte-for-byte under `train_test.svg`; the final assessment now uses `evaluation.svg` for the three-model error comparison, retaining the equations and final question.

Browser checks cover all 17 slides at 1440×900 and 1024×768 (34 layouts), images and mathematical rendering. PDF has 17 pages. All six new slides and the revised prediction slide were visually inspected in browser screenshots and PDF page rasters for readable figures, equations and labels. Evidence remains local in `checks/` and `/tmp`.

## Release restrictions

All earlier rights/provenance warnings in REVIEW.md and ../../Data/cpx_README.md still apply. Neither the course license nor this extension establishes rights to redistribute the CPX compilation. The PowerPoint is a pedagogical reference, not a source of copied art or licensed data. Do not publish data-bearing plots, HTML or PDF before clearance. No natural-crystal validation is claimed.
