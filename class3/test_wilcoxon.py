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
