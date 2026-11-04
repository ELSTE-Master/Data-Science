# Clinopyroxene thermobarometry teaching data

This directory includes a copy of an experimental clinopyroxene dataset for the ELSTE linear-regression course.

## Files

- `cpx.rda`: the original R data file, containing an object named `cpx`.
- `cpx_teaching.csv`: a CSV export of that object for the Python-based course.

The CSV is a **format conversion only**: no rows, columns, or measurements were filtered, changed, standardized, or imputed. The export was checked against the original data (numeric values within absolute tolerance `1e-12`; grouping identifiers identical).

## Source and provenance

- Source repository: `samorso/thermobarometry-cluster` (private).
- Source file: `data/cpx.rda`.
- Source Git blob SHA: `aa0e8abcf454684cb8007292261ffcfc852d6bb0`.
- Original file SHA-256: `d6a1b74d19e91356d7b7d1adfe344f9c096747a66f03753e09fe374ba24e207a`.
- CSV SHA-256: `6a3ff5ebe8c6371a4f7d453c8380615c2640080ec813cfd9c59cb10454c9ab96`.

**Redistribution status:** The source repository does not currently document a dataset license or full original data provenance. This is an internal teaching-development copy. Confirm the underlying data rights, scientific attribution, and license before publishing the dataset on the public upstream course website or including it in an external pull request.

## Structure

There are **1,696 observations**, **12 variables**, **192 distinct `Ref` values**, and **1,684 distinct (`Ref`, `Run`) pairs**. No missing values were found.

| Variable | Meaning |
| --- | --- |
| `T_C` | Experimental temperature in degrees Celsius |
| `P_kbar` | Experimental pressure in kbar |
| `SiO2_Cpx`, `TiO2_Cpx`, `Al2O3_Cpx`, `FeO_Cpx`, `MnO_Cpx`, `MgO_Cpx`, `CaO_Cpx`, `Na2O_Cpx` | Clinopyroxene oxide measurements; confirm the concentration-unit convention against the original scientific documentation before publication |
| `Run` | Experimental run identifier (string) |
| `Ref` | Publication/study identifier (string) |

For prediction activities, `Ref` and `Run` should normally be retained for **grouping and validation**, not supplied as chemical predictors. A random row split and a held-out-publication split assess different generalization tasks. Use only features that would actually be known at the intended prediction time.

## Python

From the `Data-Science` directory:

```python
import pandas as pd

cpx = pd.read_csv("Data/cpx_teaching.csv", dtype={"Run": "string", "Ref": "string"})
```

## Verified audit and frozen Class 4 protocol

Audit of commit `63815ff3d001090cffc92f3f5e0e0773fe52909a`: CSV and R object match exactly, including identifiers. Ten numerical columns are finite; no missing values or complete-row duplicates. Temperature spans 650–1700 °C and pressure 0–30 kbar. The eight oxide totals span 95.00–104.68 (median 99.59). These totals do not establish the unit or normalization convention. Zero TiO2 (20 rows), MnO (244), Na2O (40), and pressure (32) need source interpretation.

Six pairs have identical values for all ten numerical columns. Four cross `Gaetani&Grove_1998` / `Gaetani_et_al_2003`; one crosses `Bonechi_et_al_2020a` / `Perinelli_et_al_2019`; one is within `Beard_et_al_2020`. Retain all rows pending provenance clarification. Four Run strings have surrounding whitespace. Nine Ref/Run pairs contain multiple rows, including eight in Meen_1990 at different pressures. Do not interpret all pairs as independent experiments or average them as replicates.

`cpx_teaching_split.csv` contains row_id (one-based CSV data-row position, tied to the checksum), Ref, validation_group and split (`train` or `evaluation`). Generate/check from the project root with:

```sh
python class4/prepare_cpx_split.py
python class4/prepare_cpx_split.py --verify
```

The second command never writes. Existing manifests are verified, never silently replaced.

Group Ref labels sharing identical vectors of all eight oxide columns using connected components. A component identifier joins lexicographically sorted Ref strings with `|`. Compute SHA-256 over UTF-8 `ELSTE-Class4-cpx-v1` + one NUL character + component identifier. Sort by hexadecimal digest, breaking ties by identifier; reserve the first ceil(0.20 × 190) = 38 groups for evaluation. No alternative allocations were searched.

Expected: 190 groups; training 1,402 rows / 154 Ref labels / 152 groups; evaluation 294 rows / 38 Ref labels / 38 groups. Row, Ref, validation-group and identical-chemistry overlap are absent. This does not prove that all experiments in separate groups are independent.

The prespecified models are the training-mean T_C baseline, `T_C ~ MgO_Cpx`, and `T_C ~ MgO_Cpx + Al2O3_Cpx + TiO2_Cpx`. Use original measurements, an intercept, and unweighted OLS. No pressure, identifiers or outcome-derived predictors; no imputation, normalization or formula selection. The baseline uses the training mean on both partitions. RMSE = sqrt(mean(error²)); MAE = mean(abs(error)), both in °C. Publication-balanced companions average within Ref first (square-root after averaging MSE). Evaluation is the single planned comparison in Exercise 4, not a source of tuning decisions.

Full-data structural quality/range checks preceded training-only modeling. The holdout is a teaching assessment, not a completely blind external research validation. No validation of natural crystals is established.

## Release checklist

Before public release, establish original compilation authors, per-publication scientific sources, redistribution permission/license, concentration units, FeO versus total-iron convention, normalization, zero/detection-limit conventions, row semantics, and true experimental identities. Jorgenson et al. (2022) provides scientific context, not verified attribution of this particular compilation. Existing course and slide license notices differ; maintainers must clarify them. Course licenses do not grant rights to this dataset. Keep the RDA and CSV unchanged and do not publish data-bearing HTML/PDF outputs until clearance.
