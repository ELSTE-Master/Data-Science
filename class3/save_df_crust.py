from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
XLSX = ROOT / "Data" / "1-s2.0-S0012821X17302108-mmc3.xlsx"
OUT_CSV = ROOT / "Data" / "arc_magmas_crust.csv"

CUTS = [0, 30, 43, np.inf]
LABELS = ["thin", "medium", "thick"]
COLORS = {"thin": "#6A5ACD", "medium": "#06bcf1", "thick": "#8bb174"}

# ---- Build dataset ---------------------------------------------------------
raw = pd.read_excel(XLSX, sheet_name="Table 3-10 km^2 data", header=1)
raw = raw.dropna(subset=["SiO2 (wt%)", "Crustal Thickness (km)"])
raw["Arc"] = raw["Arc"].replace({"Agean": "Aegean"})

cells = (
    raw.groupby(["Arc", "Latitude", "Longitude"])
    .agg(
        sio2=("SiO2 (wt%)", "mean"),
        crust_km=("Crustal Thickness (km)", "mean"),
    )
    .reset_index()
    .rename(columns={"Arc": "arc", "Latitude": "latitude", "Longitude": "longitude"})
)
cells["crust_class"] = pd.cut(cells["crust_km"], CUTS, labels=LABELS, right=False)
cells["sio2"] = cells["sio2"].round(2)
cells["crust_km"] = cells["crust_km"].round(2)
cells[["latitude", "longitude"]] = cells[["latitude", "longitude"]].round(4)
cells = cells[["arc", "latitude", "longitude", "crust_km", "crust_class", "sio2"]]
OUT_CSV.parent.mkdir(exist_ok=True)
cells.to_csv(OUT_CSV, index=False)
