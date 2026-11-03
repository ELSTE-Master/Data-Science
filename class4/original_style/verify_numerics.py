"""Check the frozen lecture, all ten independent demos, and regenerated R outputs.

Run generate_figures.R first. Never writes data or changes allocations.
"""
from pathlib import Path
import contextlib, io, os, re, sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from cpx_lesson import load_data, selection_formulas, cv_folds, comparison, fit_models
from prepare_cpx_split import expected_manifest
os.chdir(ROOT.parent)
pd.testing.assert_frame_equal(pd.read_csv('../Data/cpx_teaching_split.csv', dtype={'Ref':'string'}), expected_manifest(), check_dtype=False)
train, test = load_data()
assert (len(train), len(test)) == (1402, 294)
text = Path('class_4_slides.qmd').read_text()
assert len(re.findall(r'^## ',text,re.M))+1 == 41
cells = re.findall(r'```\{pyodide\}\n(.*?)```',text,re.S)
assert len(cells) == 10
for i, code in enumerate(cells,1):
    with contextlib.redirect_stdout(io.StringIO()): exec(compile(code,f'cell-{i}','exec'),{})
    print(f'PASS independent Python cell {i}')
models={name:smf.ols(formula,data=train).fit() for name,formula in selection_formulas().items()}
np.testing.assert_allclose(models['MgO'].params,[707.801899293883, 27.6595736654097],rtol=0,atol=1e-8)
np.testing.assert_allclose([m.rsquared for m in models.values()],[.4472911551494,.6719541907309157,.6983378896156618],atol=1e-12,rtol=0)
np.testing.assert_allclose([m.aic for m in models.values()],[17281.138779208086,16553.746082083326,16448.194396917683],atol=1e-8,rtol=0)
for model in models.values():
    np.testing.assert_allclose(model.params,np.linalg.lstsq(model.model.exog,train.T_C,rcond=None)[0],rtol=1e-9,atol=1e-8)
rcoef=pd.read_csv(ROOT/'coefficients.csv')
np.testing.assert_allclose(np.concatenate([models['MgO'].params,models['3 oxides'].params]),rcoef.Estimate,rtol=0,atol=1e-8)
rquad=pd.read_csv(ROOT/'three_oxide_polynomial_coefficients.csv')
rquad['Term']=rquad.Term.str.replace('(Intercept)','Intercept',regex=False).str.replace('^2',' ** 2',regex=False)
np.testing.assert_allclose(models['Quadratic'].params,rquad.set_index('Term').loc[models['Quadratic'].params.index,'Estimate'],rtol=0,atol=1e-8)
fold=cv_folds(train)
assert fold.value_counts().sort_index().tolist()==[281,281,280,280,280]
assert train.assign(Fold=fold).groupby('validation_group').Fold.nunique().eq(1).all()
manifest=train[['row_id','validation_group']].assign(Fold=fold).reset_index(drop=True)
pd.testing.assert_frame_equal(manifest,pd.read_csv(ROOT/'model_selection_cv_folds.csv'),check_dtype=False)
cv=[]
for formula in selection_formulas().values():
    errors=[]
    for k in range(5):
        fit=smf.ols(formula,data=train[fold.ne(k)]).fit()
        errors.extend(train[fold.eq(k)].T_C-fit.predict(train[fold.eq(k)]))
    cv.append(np.sqrt(np.mean(np.array(errors)**2)))
np.testing.assert_allclose(cv,pd.read_csv(ROOT/'cv_scores.csv').CV_RMSE_C,atol=1e-8,rtol=0)
assert np.round(cv,1).tolist()==[115.8,90.1,90.0]
scores=comparison(train,test,fit_models(train))
np.testing.assert_allclose(scores.to_numpy(),pd.read_csv(ROOT/'scores.csv').iloc[:,2:].to_numpy(),atol=1e-8,rtol=0)
errors=test.T_C-models['Quadratic'].predict(test)
np.testing.assert_allclose([np.sqrt(np.mean(errors**2)),np.mean(abs(errors))],[97.43215942450378,80.04273053265614],atol=1e-8,rtol=0)
np.testing.assert_allclose(scores.loc['Evaluation'].iloc[:,:2].to_numpy(),[[159.444930247153, 128.262113404563], [126.642840751892, 99.5291472553825], [101.180987899091, 82.7130469313594]],atol=1e-6,rtol=0)
for filename in ['simple_regression_intervals.csv','simple_regression_prediction_1200.csv']:
    grid=pd.read_csv(ROOT/filename);pred=models['MgO'].get_prediction(grid[['MgO_Cpx']]).summary_frame(alpha=.05)
    np.testing.assert_allclose(pred[['mean','mean_ci_lower','mean_ci_upper','obs_ci_lower','obs_ci_upper']],grid[['fit','CI_lower','CI_upper','PI_lower','PI_upper']],atol=1e-8,rtol=0)
    assert (pred.obs_ci_lower<pred.mean_ci_lower).all() and (pred.obs_ci_upper>pred.mean_ci_upper).all()
print('PASS: frozen inputs/split; OLS/NumPy/R agreement; AIC; grouped CV; test errors; mean and individual intervals.')
print('CV RMSE:',cv)
