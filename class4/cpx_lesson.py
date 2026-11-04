"""Class 4 setup: local Python and Pyodide; no evaluation on import."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
PREDICTORS = ['MgO_Cpx', 'Al2O3_Cpx', 'TiO2_Cpx']
FORMULAS = {'Simple': 'T_C ~ MgO_Cpx', 'Multiple': 'T_C ~ MgO_Cpx + Al2O3_Cpx + TiO2_Cpx'}
SHA = '6a3ff5ebe8c6371a4f7d453c8380615c2640080ec813cfd9c59cb10454c9ab96'
def load_data():
    folder = next(p for p in [Path('../Data'), Path('Data')] if (p/'cpx_teaching.csv').is_file())
    path = folder/'cpx_teaching.csv'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SHA
    data = pd.read_csv(path, dtype={'Ref':'string', 'Run':'string'})
    split = pd.read_csv(folder/'cpx_teaching_split.csv', dtype={'Ref':'string'})
    assert list(split.columns) == ['row_id','Ref','validation_group','split']
    assert split.row_id.tolist() == list(range(1,1697))
    assert split.Ref.tolist() == data.Ref.tolist()
    assert set(split.split) == {'train','evaluation'}
    data = data.assign(row_id=split.row_id, validation_group=split.validation_group)
    train, evaluation = data[split.split.eq('train')].copy(), data[split.split.eq('evaluation')].copy()
    assert (len(train),len(evaluation)) == (1402,294)
    for c in ['row_id','Ref','validation_group']:
        assert not set(train[c]) & set(evaluation[c])
    assert np.isfinite(data[['T_C']+PREDICTORS]).all().all()
    return train, evaluation

def fit_models(train):
    return {name:smf.ols(formula,data=train).fit() for name,formula in FORMULAS.items()}

def errors(actual,predicted):
    e = np.asarray(actual)-np.asarray(predicted)
    return {'RMSE (°C)':float(np.sqrt(np.mean(e**2))), 'MAE (°C)':float(np.mean(np.abs(e)))}

def comparison(train,evaluation,models):
    """Single planned comparison: fixed formulas and original measurements."""
    rows=[]
    for partition,frame in [('Training',train),('Evaluation',evaluation)]:
        predictions={'Baseline':np.full(len(frame),train.T_C.mean())}
        predictions.update({name:model.predict(frame[PREDICTORS]) for name,model in models.items()})
        for name,pred in predictions.items():
            row={'Partition':partition,'Model':name,**errors(frame.T_C,pred)}
            e=pd.DataFrame({'Ref':frame.Ref.to_numpy(),'error':frame.T_C.to_numpy()-np.asarray(pred)})
            groups=e.groupby('Ref').error
            row['Publication RMSE (°C)']=float(np.sqrt(groups.apply(lambda v:np.mean(v**2)).mean()))
            row['Publication MAE (°C)']=float(groups.apply(lambda v:np.mean(np.abs(v))).mean())
            rows.append(row)
    return pd.DataFrame(rows).set_index(['Partition','Model'])

def scatter(train,model=None):
    fig,ax=plt.subplots(figsize=(6,3))
    ax.scatter(train.MgO_Cpx,train.T_C,s=12,alpha=.35,color='#6A5ACD')
    if model is not None:
        grid=pd.DataFrame({'MgO_Cpx':np.linspace(train.MgO_Cpx.min(),train.MgO_Cpx.max(),100)})
        ax.plot(grid.MgO_Cpx,model.predict(grid),color='#eb078e',lw=2)
    ax.set(xlabel='MgO (reported concentration units)',ylabel='Experimental temperature (°C)')
    fig.tight_layout(); plt.show()

def residual_plot(model):
    fig,ax=plt.subplots(figsize=(6,3))
    ax.scatter(model.fittedvalues,model.resid,s=12,alpha=.35,color='#6A5ACD')
    ax.axhline(0,color='#eb078e')
    ax.set(xlabel='Fitted temperature (°C)',ylabel='Training residual (°C)')
    fig.tight_layout(); plt.show()

def evaluation_plot(evaluation, model):
    fig,ax=plt.subplots(figsize=(5,2.7))
    pred=model.predict(evaluation[PREDICTORS])
    ax.scatter(evaluation.T_C,pred,s=15,alpha=.5,color='#6A5ACD')
    ax.plot([650,1700],[650,1700],color='#eb078e')
    ax.set(xlabel='Experimental temperature (°C)',ylabel='Evaluation prediction (°C)')
    fig.tight_layout(); plt.show()

def selection_formulas():
    """The three fixed candidate models used in the selection illustrations."""
    linear = 'MgO_Cpx + Al2O3_Cpx + TiO2_Cpx'
    squares = 'I(MgO_Cpx**2) + I(Al2O3_Cpx**2) + I(TiO2_Cpx**2)'
    return {'MgO': 'T_C ~ MgO_Cpx', '3 oxides': 'T_C ~ ' + linear,
            'Quadratic': f'T_C ~ ({linear})**2 + {squares}'}

def cv_folds(train, n_folds=5):
    """Deterministic balanced folds inside training, keeping linked records together.

    Uses only existing group identities and their sizes, never temperatures.
    The original training/evaluation allocation is not modified.
    """
    counts = train.validation_group.value_counts()
    if not 2 <= n_folds <= len(counts):
        raise ValueError('Need between 2 and the number of training groups folds.')
    salt = 'ELSTE-Class4-cv-v1\n'
    ordered = sorted(counts.index, key=lambda g: (
        -int(counts[g]), hashlib.sha256((salt + str(g)).encode()).hexdigest()))
    loads = np.zeros(n_folds, dtype=int)
    assignment = {}
    for group in ordered:
        fold = int(np.argmin(loads))
        assignment[group] = fold
        loads[fold] += int(counts[group])
    return train.validation_group.map(assignment).astype(int)
