import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

diet = pd.read_csv("https://raw.githubusercontent.com/ELSTE-Master/Data-Science/main/Data/diet.csv")
# compute weight loss
diet["weight.loss"] = diet["initial.weight"] - diet["final.weight"]

# Variable of interest
dietA = diet["weight.loss"][diet["diet.type"]=="A"]
dietC = diet["weight.loss"][diet["diet.type"]=="C"]


stat, p_value = stats.mannwhitneyu(dietA+1, dietC, alternative="less")
print(f"Mann-Whitney U test statistic: {stat}")
print(f"P-value: {p_value}")














# ########################################



import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

diet = pd.read_csv("https://raw.githubusercontent.com/ELSTE-Master/Data-Science/main/Data/diet.csv")
# compute weight loss
diet["weight.loss"] = diet["initial.weight"] - diet["final.weight"]

# Variable of interest
dietA = diet["weight.loss"][diet["diet.type"]=="A"]
dietC = diet["weight.loss"][diet["diet.type"]=="C"]


stat, p_value = stats.mannwhitneyu(dietA+0.95, dietC, alternative="less")
print(f"Mann-Whitney U test statistic: {stat}")
print(f"P-value: {p_value}")




##############################


import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

diet = pd.read_csv("https://raw.githubusercontent.com/ELSTE-Master/Data-Science/main/Data/diet.csv")


import pandas as pd
import matplotlib.pyplot as plt

df_salamander = pd.read_csv("https://raw.githubusercontent.com/ELSTE-Master/Data-Science/main/Data/salamander_weights.csv")
df_salamander.head()

# Variable of interest
weight_CC = df_salamander["weight_g"][df_salamander["section"]=="CC"]
weight_OG = df_salamander["weight_g"][df_salamander["section"]=="OG"]

# Side-by-side boxplot
plt.figure(figsize=(6,4))
plt.boxplot([weight_CC, weight_OG],    #<1>
            labels=["CC", "OG"],
            patch_artist=True,  # color the boxes
            boxprops=dict(facecolor="lightblue", color="blue"),
            medianprops=dict(color="red"))

plt.ylabel("Weight (g)")
plt.show()

from scipy import stats
# Perform independent two-sample t-test (assuming unequal variances)
t_stat, p_val = stats.ttest_ind(weight_CC, weight_OG, equal_var=False)

print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_val:.3f}")





import pandas as pd
import matplotlib.pyplot as plt

df_salamander = pd.read_csv("https://raw.githubusercontent.com/ELSTE-Master/Data-Science/main/Data/salamander_weights.csv")
df_salamander.head()

# Variable of interest
weight_CC = df_salamander["weight_g"][df_salamander["section"]=="CC"]
weight_OG = df_salamander["weight_g"][df_salamander["section"]=="OG"]

stat, p_value = stats.mannwhitneyu(weight_CC, weight_OG)
print(f"Mann-Whitney U test statistic: {stat}")
print(f"P-value: {p_value}")
