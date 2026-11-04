"""Training-only AIC and fixed five-fold CV assets for the lecture's R figures."""
from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from cpx_lesson import load_data, selection_formulas, cv_folds

os.chdir(ROOT.parent)
train, _ = load_data()
fold = cv_folds(train)
assert len(train) == 1402 and set(fold) == set(range(5))
manifest = train[['row_id', 'validation_group']].copy()
manifest['Fold'] = fold
assert manifest.groupby('validation_group').Fold.nunique().eq(1).all()
manifest.to_csv(ROOT / 'model_selection_cv_folds.csv', index=False)

aic_rows, cv_rows, fold_rows = [], [], []
for name, formula in selection_formulas().items():
    model = smf.ols(formula, data=train).fit()
    assert model.nobs == len(train)
    aic_rows.append([name, len(model.params), model.llf, model.aic])
    all_errors, seen = [], []
    for valid_fold in range(5):
        fitting, validation = train[fold.ne(valid_fold)], train[fold.eq(valid_fold)]
        assert not set(fitting.validation_group) & set(validation.validation_group)
        fitted = smf.ols(formula, data=fitting).fit()
        error = validation.T_C - fitted.predict(validation)
        all_errors.extend(error.to_numpy())
        seen.extend(validation.row_id)
        fold_rows.append([name, valid_fold, len(validation), np.sqrt(np.mean(error**2))])
    assert sorted(seen) == sorted(train.row_id) and len(set(seen)) == len(train)
    cv_rows.append([name, np.sqrt(np.mean(np.asarray(all_errors)**2))])

aic = pd.DataFrame(aic_rows, columns=['Model', 'Coefficients', 'LogLik', 'AIC'])
aic['Delta_AIC'] = aic.AIC - aic.AIC.min()
aic.to_csv(ROOT / 'aic_scores.csv', index=False)
pd.DataFrame(cv_rows, columns=['Model', 'CV_RMSE_C']).to_csv(ROOT / 'cv_scores.csv', index=False)
pd.DataFrame(fold_rows, columns=['Model', 'Fold', 'N_valid', 'RMSE_C']).to_csv(
    ROOT / 'cv_fold_scores.csv', index=False)
print(aic.round(3).to_string(index=False))
print(pd.DataFrame(cv_rows, columns=['Model', 'CV_RMSE_C']).round(3).to_string(index=False))
