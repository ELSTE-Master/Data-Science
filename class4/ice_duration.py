import pandas as pd
import matplotlib.pyplot as plt

# Example: your DataFrame `df`
df_icecover = pd.read_csv("https://raw.githubusercontent.com/ELSTE-Master/Data-Science/main/Data/df_icecover.csv")

# Filter lakes
df_lake_mendota = df_icecover[df_icecover['lakeid'] == "Lake Mendota"]
df_lake_monona  = df_icecover[df_icecover['lakeid'] == "Lake Monona"]

# Colors
mycol = ["#6a5acd", "#e64173"]

# Plot
plt.figure(figsize=(8,5))
plt.plot(df_lake_mendota['year'], df_lake_mendota['ice_duration'], 'o', color=mycol[0], label='Mendota')
plt.plot(df_lake_monona['year'], df_lake_monona['ice_duration'], 'o', color=mycol[1], label='Monona')

plt.xlabel("Year")
plt.ylabel("Ice duration (days)")
plt.title("Ice duration over years")
plt.legend(loc="upper right")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()















#



import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import numpy as np

# Load data
df_icecover = pd.read_csv("https://raw.githubusercontent.com/ELSTE-Master/Data-Science/main/Data/df_icecover.csv")

# Fit linear model with lake-specific intercepts
mod2 = smf.ols("ice_duration ~ year + lakeid", data=df_icecover).fit()
print(mod2.summary())

# Separate lakes
df_mendota = df_icecover[df_icecover['lakeid'] == "Lake Mendota"]
df_monona  = df_icecover[df_icecover['lakeid'] == "Lake Monona"]

# Colors
colors = ["#6a5acd", "#e64173"]

# Plot data points
plt.figure(figsize=(8,5))
plt.scatter(df_mendota['year'], df_mendota['ice_duration'], color=colors[0], label="Mendota")
plt.scatter(df_monona['year'], df_monona['ice_duration'], color=colors[1], label="Monona")

# Prepare years for predictions
years = np.linspace(df_icecover['year'].min(), df_icecover['year'].max(), 200)

# Predict for Mendota
pred_mendota = mod2.predict(pd.DataFrame({'year': years, 'lakeid': 'Lake Mendota'}))
plt.plot(years, pred_mendota, color=colors[0], linestyle='-', linewidth=2)

# Predict for Monona
pred_monona = mod2.predict(pd.DataFrame({'year': years, 'lakeid': 'Lake Monona'}))
plt.plot(years, pred_monona, color=colors[1], linestyle='-', linewidth=2)

# Labels, legend, title
plt.xlabel("Year")
plt.ylabel("Ice duration (days)")
plt.title("Ice duration over years with regression lines")
plt.legend()
plt.grid(True)
plt.show()









# predict using mod3 for years 2050
future_years = pd.DataFrame({'year': [2450], 'lakeid': ['Lake Mendota']})
pred_2050_mendota = mod2.predict(future_years)
print(f"Predicted ice duration for Lake Mendota in 2050: {pred_2050_mendota.values[0]:.2f} days")
