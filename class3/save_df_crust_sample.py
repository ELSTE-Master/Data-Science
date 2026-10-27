"""Draw the fixed random subsample of arc_magmas_crust.csv used in Class 3.

The full dataset (8498 cells, Farner & Lee 2017) makes every test
overwhelmingly significant and many cells come from the same volcanoes.
The slides and exercises therefore use ONE random subsample of
N_PER_CLASS cells per crustal-thickness class, drawn once with the seed
below and committed as Data/arc_magmas_crust_sample.csv.

Choice of the seed (for teaching purposes, decided with S. Guerrier,
2026-10-26): among seeds 1-4000 we kept the one for which the thin vs
medium comparison is borderline significant (Welch p ~ 4%, medium mean
above thin mean), so that the slides can show (i) the effect of a single
wrong value on a t-test and (ii) why a multiple-testing correction
matters, while the three-group story (thick > thin, thick > medium after
Sidak) still holds. The data themselves are untouched.
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CSV_IN = HERE.parent / "Data" / "arc_magmas_crust.csv"
CSV_OUT = HERE.parent / "Data" / "arc_magmas_crust_sample.csv"

N_PER_CLASS = 60
SEED = 2346

crust = pd.read_csv(CSV_IN)
sample = (
    crust.groupby("crust_class", sort=False)
    .sample(n=N_PER_CLASS, random_state=SEED)
    .sort_index()          # keep the row order of the full dataset
)
sample.to_csv(CSV_OUT, index=False)
print(sample["crust_class"].value_counts().to_string())
