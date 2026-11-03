"""Frozen Class 4 split. Generate once; --verify never writes files."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / 'Data'
CSV_SHA256 = '6a3ff5ebe8c6371a4f7d453c8380615c2640080ec813cfd9c59cb10454c9ab96'
RDA_SHA256 = 'd6a1b74d19e91356d7b7d1adfe344f9c096747a66f03753e09fe374ba24e207a'
SALT = 'ELSTE-Class4-cpx-v1'
OXIDES = ['SiO2_Cpx', 'TiO2_Cpx', 'Al2O3_Cpx', 'FeO_Cpx', 'MnO_Cpx', 'MgO_Cpx', 'CaO_Cpx', 'Na2O_Cpx']

def expected_manifest():
    for name, checksum in [('cpx_teaching.csv', CSV_SHA256), ('cpx.rda', RDA_SHA256)]:
        if hashlib.sha256((DATA / name).read_bytes()).hexdigest() != checksum:
            raise ValueError('Pinned data checksum mismatch: ' + name)
    data = pd.read_csv(DATA / 'cpx_teaching.csv', dtype={'Ref': 'string', 'Run': 'string'})
    assert data.shape == (1696, 12) and not data.isna().any().any()
    assert np.isfinite(data.select_dtypes('number')).all().all()
    assert data.duplicated().sum() == 0
    assert data.duplicated(data.select_dtypes('number').columns).sum() == 6
    parent = {ref: ref for ref in sorted(data.Ref.unique())}
    def root(ref):
        while parent[ref] != ref:
            ref = parent[ref]
        return ref
    for _, rows in data.groupby(OXIDES, sort=False):
        refs = sorted(rows.Ref.unique())
        for ref in refs[1:]:
            a, b = root(refs[0]), root(ref)
            parent[max(a, b)] = min(a, b)
    components = {}
    for ref in parent:
        components.setdefault(root(ref), []).append(ref)
    mapping = {ref: '|'.join(sorted(refs)) for refs in components.values() for ref in refs}
    groups = sorted(set(mapping.values()), key=lambda g: (hashlib.sha256((SALT + '\0' + g).encode('utf-8')).hexdigest(), g))
    assert len(groups) == 190
    heldout = set(groups[:int(np.ceil(.2 * len(groups)))] )
    manifest = pd.DataFrame({'row_id': np.arange(1, len(data)+1), 'Ref': data.Ref,
                             'validation_group': data.Ref.map(mapping)})
    manifest['split'] = np.where(manifest.validation_group.isin(heldout), 'evaluation', 'train')
    train = manifest.split.eq('train')
    assert (int(train.sum()), int((~train).sum())) == (1402, 294)
    assert (manifest.loc[train, 'Ref'].nunique(), manifest.loc[~train, 'Ref'].nunique()) == (154, 38)
    for col in ['row_id', 'Ref', 'validation_group']:
        assert not set(manifest.loc[train, col]) & set(manifest.loc[~train, col])
    assert not set(map(tuple, data.loc[train, OXIDES].to_numpy())) & set(map(tuple, data.loc[~train, OXIDES].to_numpy()))
    return manifest

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    expected = expected_manifest()
    path = DATA / 'cpx_teaching_split.csv'
    if args.verify or path.exists():
        actual = pd.read_csv(path, dtype={'Ref': 'string'})
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    else:
        expected.to_csv(path, index=False)
    print('Verified: pinned CSV/RDA; 1696 aligned rows; 190 groups; 1402 train / 294 evaluation; no reference/group/chemistry overlap.')

if __name__ == '__main__':
    main()
