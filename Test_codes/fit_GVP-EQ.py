# One test to fit different models to GVP and earthquake magnitude-frequency data
# Also includes some exploratory analysis of earthquake depth/magnitude by tectonic setting
# Some code to retrieve plate boundaries from a KMZ file and spatially join to earthquake data

#%%
import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shapely.geometry
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.multicomp import pairwise_tukeyhsd


#%%
def fitMe(counts, bin_centers):
    # Model A: raw linear — counts ~ mag
    X_A = bin_centers.reshape(-1,1)
    y_A = counts.reshape(-1,1)
    model_A = LinearRegression().fit(X_A, y_A)
    pred_A = model_A.predict(X_A)

    # Model B: log(target only) — log(counts) ~ mag
    model_B = LinearRegression().fit(X_A, np.log(counts).reshape(-1,1))
    pred_B_log = model_B.predict(X_A).flatten()
    pred_B = np.exp(pred_B_log)

    # Model C: log-log — log(counts) ~ log(mag)
    X_C = np.log(X_A)
    y_C = np.log(counts)
    model_C = LinearRegression().fit(X_C, y_C.reshape(-1,1))
    pred_C_log = model_C.predict(X_C).flatten()
    pred_C = np.exp(pred_C_log)

    # Evaluate fits
    def stats(y_true, y_pred):
        return {"R2": r2_score(y_true, y_pred), "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))}

    resid_A = counts - pred_A.flatten()
    resid_B = counts - pred_B
    resid_C = counts - pred_C

    return model_A, pred_A, model_B, pred_B, model_C, pred_C, resid_A, resid_B, resid_C


#%% Case 1 - Eruption frequency vs. Volcanic Explosivity Index (VEI) (Smithsonian GVP data)
# Note: VEI is a discrete index (0-8), not continuous

data = pd.read_excel('/Users/seb/Downloads/GVP_Eruption_Search_Result.xlsx', sheet_name='Eruption List', header=1)
df = data[["VEI"]].dropna()
df["VEI"] = df["VEI"].astype(int)
df = df[df["VEI"] >= 3] # focus on VEI 3 and above
#%%

# Count frequency of each VEI
counts = df["VEI"].value_counts().sort_index()
vei = counts.index.values.reshape(-1, 1)
freq = counts.values

# model_A, pred_A, model_B, pred_B, model_C, pred_C, resid_A, resid_B, resid_C = fitMe(counts.values, vei.flatten())


# Log-transform frequency
log_freq = np.log(freq)

# Fit linear regression: VEI → log(frequency)
model = LinearRegression()
model.fit(vei, log_freq)

# Predict
vei_range = np.arange(vei.min(), vei.max()+1).reshape(-1, 1)
log_pred = model.predict(vei_range)
pred = np.exp(log_pred)  # back-transform

# Plot raw + regression
plt.scatter(vei, freq, label="Observed", color="blue")
plt.plot(vei_range, pred, color="red", label="Fitted (exp decay)")
plt.yscale("log")  # log scale for frequency
plt.xlabel("Volcanic Explosivity Index (VEI)")
plt.ylabel("Eruption frequency (log scale)")
plt.title("Eruption Frequency vs. VEI (GVP Data)")
plt.legend()
plt.show()

# Show regression equation
slope = model.coef_[0]
intercept = model.intercept_
print(f"log(freq) ≈ {intercept:.2f} + {slope:.2f} × VEI")


# # Plot
# plt.figure(figsize=(8,6))
# plt.scatter(vei, counts, label="Observed counts")
# plt.plot(vei, pred_A, label="Model A: counts ~ mag")
# plt.plot(vei, pred_B, label="Model B: exp(log(counts) ~ mag)")
# plt.plot(vei, pred_C, label="Model C: exp(log(counts) ~ log(mag))")
# plt.yscale("log")
# plt.xlabel("Magnitude M")
# plt.ylabel("Frequency (number of earthquakes in bin)")
# # plt.title("Magnitude-Frequency of Earthquakes (M ≥ {})".format(M_min))
# plt.legend()
# plt.grid(True, which="both", ls="--")
# plt.show()

#%%

eq = pd.read_csv('/Users/seb/Downloads/query_M4.5+_2000-2024.csv')

# Load earthquake data
# eq = pd.read_csv("path/to/your/earthquake_catalog.csv")
# Filter: keep only events with magnitude ≥ M_min so you're not hitting catalog completeness threshold
M_min = 4.5
# eq = eq[eq['mag'] >= M_min].dropna(subset=['mag'])

# Build histogram / counts per magnitude bin
bin_width = 0.1
bins = np.arange(M_min, eq['mag'].max() + bin_width, bin_width)
counts, edges = np.histogram(eq['mag'], bins=bins)
# Use bin centers
bin_centers = (edges[:-1] + edges[1:]) / 2

# Remove bins with zero counts (if any)
mask = counts > 0
bin_centers = bin_centers[mask]
counts = counts[mask]

# Now prepare three models

# Model A: raw linear — counts ~ mag
X_A = bin_centers.reshape(-1,1)
y_A = counts.reshape(-1,1)
model_A = LinearRegression().fit(X_A, y_A)
pred_A = model_A.predict(X_A)

# Model B: log(target only) — log(counts) ~ mag
model_B = LinearRegression().fit(X_A, np.log(counts).reshape(-1,1))
pred_B_log = model_B.predict(X_A).flatten()
pred_B = np.exp(pred_B_log)

# Model C: log-log — log(counts) ~ log(mag)
X_C = np.log(X_A)
y_C = np.log(counts)
model_C = LinearRegression().fit(X_C, y_C.reshape(-1,1))
pred_C_log = model_C.predict(X_C).flatten()
pred_C = np.exp(pred_C_log)

# Evaluate fits
def stats(y_true, y_pred):
    return {"R2": r2_score(y_true, y_pred), "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))}

print("A (raw):", stats(y_A.flatten(), pred_A))
print("B (semi-log):", stats(counts, pred_B))
print("C (log-log):", stats(counts, pred_C))

# Plot
plt.figure(figsize=(8,6))
plt.scatter(bin_centers, counts, label="Observed counts")
plt.plot(bin_centers, pred_A, label="Model A: counts ~ mag")
plt.plot(bin_centers, pred_B, label="Model B: exp(log(counts) ~ mag)")
plt.plot(bin_centers, pred_C, label="Model C: exp(log(counts) ~ log(mag))")
plt.yscale("log")
plt.xlabel("Magnitude M")
plt.ylabel("Frequency (number of earthquakes in bin)")
plt.title("Magnitude-Frequency of Earthquakes (M ≥ {})".format(M_min))
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()

# Residuals plot
fig, axes = plt.subplots(1,3, figsize=(15,4))
resid_A = counts - pred_A.flatten()
resid_B = counts - pred_B
resid_C = counts - pred_C
axes[0].scatter(bin_centers, resid_A); axes[0].hlines(0, bin_centers.min(), bin_centers.max(), ls="--"); axes[0].set_title("Model A residuals")
axes[1].scatter(bin_centers, resid_B); axes[1].hlines(0, bin_centers.min(), bin_centers.max(), ls="--"); axes[1].set_title("Model B residuals")
axes[2].scatter(bin_centers, resid_C); axes[2].hlines(0, bin_centers.min(), bin_centers.max(), ls="--"); axes[2].set_title("Model C residuals")
for ax in axes:
    ax.set_xlabel("Fitted Count")
    ax.set_ylabel("Residuals")
plt.tight_layout()
plt.show()


#%%
# Read KMZ file (GeoPandas reads KML, so we need to specify the layer)
boundaries = gpd.read_file("/Users/seb/Downloads/plate-boundaries.kmz", driver="KML", layer="Plate Interface")
# Add 'boundary' column based on 'description' containing 'Transform'
boundaries['boundary'] = None
boundaries.loc[boundaries['description'].str.contains('Transform', case=False, na=False), 'boundary'] = 'Transform'
boundaries.loc[boundaries['description'].str.contains('Divergent', case=False, na=False), 'boundary'] = 'Divergent'
boundaries.loc[boundaries['description'].str.contains('Convergent', case=False, na=False), 'boundary'] = 'Convergent'
boundaries.loc[boundaries['description'].str.contains('Other', case=False, na=False), 'boundary'] = 'Other'

# Cheeky way to solve antemeridian problems
clip_box = shapely.geometry.box(-179., -90, 179., 90)
boundaries = boundaries.clip(clip_box)

boundaries_proj = boundaries.to_crs(epsg=3857)  # Project to Web Mercator for buffering
boundaries_proj['geometry'] = boundaries_proj.geometry.buffer(100000, cap_style=2)  # cap_style=2 for flat ends
boundaries_buf = boundaries_proj.to_crs('EPSG:4326')  # Reproject back to WGS84 after buffering


#%%
eq = gpd.read_file("/Users/seb/Downloads/query_M4.5+_2000-2024.csv")
eq = eq[['time', 'latitude', 'longitude', 'depth', 'mag', 'place']].dropna(subset=['mag'])
eq['latitude'] = pd.to_numeric(eq['latitude'], errors='coerce')
eq['longitude'] = pd.to_numeric(eq['longitude'], errors='coerce')
eq['depth'] = pd.to_numeric(eq['depth'], errors='coerce')
eq['mag'] = pd.to_numeric(eq['mag'], errors='coerce')
eq['time'] = pd.to_datetime(eq['time'], errors='coerce')
# eq = eq.dropna(subset=['latitude', 'longitude', 'depth', 'mag'])

# Convert earthquake data to GeoDataFrame
eq = gpd.GeoDataFrame(eq, geometry=gpd.points_from_xy(eq.longitude, eq.latitude), crs="EPSG:4326")

#%%
# 4. Spatial join: assign tectonic type to earthquakes
eqT = gpd.sjoin(eq, boundaries_buf[['geometry','boundary']], how='left', predicate='intersects')

# 5. Fill in remaining earthquakes as Intraplate
eqT['boundary'] = eqT['boundary'].fillna('Intraplate')

# Count earthquakes by boundary type
eqT['boundary'].value_counts()

    
df = eqT[['time', 'latitude', 'longitude', 'depth', 'mag', 'boundary']].rename(columns={'boundary': 'tectonics'}).dropna(subset=['mag'])
df = df.loc[0:10000]

#%%

sns.scatterplot(data=df, x='mag', y='depth', hue='tectonics', size='mag', sizes=(4,200), alpha=0.3)

sns.boxplot(data=df, x='tectonics', y='depth', hue='tectonics')
sns.violinplot(data=df, x='tectonics', y='depth', hue='tectonics')
sns.violinplot(data=df, x='tectonics', y='mag', hue='tectonics')

sns.pairplot(df, hue='tectonics', vars=['mag', 'depth'])


sns.histplot(data=df[df['tectonics']=='Transform'], x='mag', hue='tectonics', multiple='dodge', bins=20)







#%%
# Group magnitudes by tectonic category
groups = [grp['mag'].values for _, grp in df.groupby('tectonics')]
labels = df['tectonics'].unique()

# Run one-way ANOVA
f_stat, p_val = stats.f_oneway(*groups)
print(f"ANOVA: F = {f_stat:.4f}, p = {p_val:.4g}")

# Summary by group
print("\nGroup summaries:")
print(df.groupby('tectonics')['mag'].describe())

# Post-hoc Tukey HSD test
tukey = pairwise_tukeyhsd(df['mag'], df['tectonics'], alpha=0.05)
print("\nTukey HSD results:")
print(tukey.summary())











#%%
# Plot plate boundaries colored by 'boundary' type, earthquakes as scatter, and add basemap
fig, ax = plt.subplots(figsize=(12, 10))
bound_type = ['Convergent', 'Transform', 'Divergent', 'Other']
eq_types = ['Convergent', 'Transform', 'Divergent', 'Other', 'Intraplate']
# eq_types = ['Convergent', 'Transform', 'Divergent', 'Other']
# colors = plt.cm.tab10(np.linspace(0, 1, len(boundary_types)))

# eq_col = ['#7F3C8D','#11A579','#3969AC','#F2B701']
eq_col = ['#7F3C8D','#11A579','#3969AC','#F2B701','#E73F74']
bound_col = ['#7F3C8D','#11A579','#3969AC','#F2B701']

# Plot earthquakes: size by magnitude, color by boundary type
for btype, color in zip(eq_types, eq_col):
    subset = eqT[eqT['boundary'] == btype]
    plt.scatter(
        subset.geometry.x,
        subset.geometry.y,
        s=subset['mag'] * 8,  # adjust scaling as needed
        color=color,
        label=btype,
        alpha=0.1,
        zorder=2
    )

# Plot plate boundaries
for btype, color in zip(bound_type, bound_col):
    boundaries[boundaries['boundary'] == btype].plot(ax=ax, color=color, linewidth=3, label=btype, zorder=11)


ax.set_title("Plate Boundaries and Earthquakes by Boundary Type")
ax.legend(title="Boundary Type")

# Add basemap
ctx.add_basemap(ax, crs=eqT.crs.to_string(), source='NASAGIBS.ASTER_GDEM_Greyscale_Shaded_Relief')

plt.tight_layout()
plt.show()