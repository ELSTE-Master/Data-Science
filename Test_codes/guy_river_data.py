# Test code to explore the Guyana river sediment data from Guy's class

#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#%%
data1 = pd.read_csv('/Users/seb/Documents/WORK/Teaching/UNIGE/Master/Data Science/Data/Guy/ORENOQ1.TXT', sep='\t', header=None)
data2 = pd.read_csv('/Users/seb/Documents/WORK/Teaching/UNIGE/Master/Data Science/Data/Guy/ORENOQ2.TXT', sep='\t', header=None)
df = pd.concat([data1, data2])

d = {
    1: 'x',
    2: 'y',
    3: 'dir',
    4: 'bathy',
    5: 'speed',
    6: 'GS_mode',
    7: 'GS_sorting',
    8: 'Clay',
    9: 'Sand',
    10: 'Si',
    11: 'Fe',
    12: 'Au',
    13: 'Pt', #platinum
    14: 'Mn',
    15: 'Al',
    16: 'Ca',
    17: 'Mg',
    18: 'C_org',
    19: 'Zn',
}

df = df.rename(columns=d).drop(columns=[0])

#%%
# df = pd.read_clipboard()
df = df.set_index('Sample', drop=True)

d = {
    'CoordX': 'x',
    'CoordY': 'y',
    'Courant': 'dir',
    'Bathy': 'bathy',
    'Vitesse': 'speed',
    'GS': 'GS_mode',
    'SIg': 'GS_sorting',
    'Clay': 'Clay',
    'Sand': 'Sand',
    'Si':  'Si',
    'Fe':  'Fe',
    'Au':  'Au',
    'Pt':  'Pt', #platinum
    'Mn':  'Mn',
    'Al':  'Al',
    'Ca':  'Ca',
    'Mg':  'Mg',
    'C Org':  'C_org',
    'Zn':  'Zn',
}


df = df.rename(columns=d)

#%%
sns.pairplot(df)

#%%
corr = df[['bathy', 'GS_mode', 'Clay', 'Sand', 'Si', 'Fe', 'Au', 'Pt', 'Mn', 'Al', 'Ca', 'Mg', 'C_org', 'Zn']].corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr,  cmap=cmap, mask=mask, vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

#%%
df_sub = df[['flux', 'depth', 'speed', 'GS_mode', 'GS_sorting', 'Fe', 'Au', 'Pt', 'Mn', 'Al', 'Ca', 'Mg', 'C_org', 'Zn']]
sns.pairplot(df_sub)




#%% PCA test

df2 = df[['bathy', 'GS_mode', 'Clay', 'Sand', 'Si', 'Fe', 'Au', 'Pt', 'Mn', 'Al', 'Ca', 'Mg', 'C_org', 'Zn']]

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df2.dropna())
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_scaled)
pca_result_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'], index=df2.dropna().index)
# Get the loadings (factors) for each column
loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2'],
    index=df2.dropna().columns
)
print("PCA Loadings (Factors):")
print(loadings)

#%%
plt.figure(figsize=(8, 8))
plt.scatter(loadings['PC1'], loadings['PC2'])

for i, var in enumerate(loadings.index):
    plt.annotate(var, (loadings['PC1'][i], loadings['PC2'][i]), fontsize=10)

plt.axhline(0, color='grey', linewidth=0.8, linestyle='--')
plt.axvline(0, color='grey', linewidth=0.8, linestyle='--')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Loadings Plot')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


