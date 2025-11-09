import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("cleaned_energy_data.csv")

# Energy over time
plt.figure(figsize=(10,5))
plt.plot(data['Date'], data['EnergyGenerated'])
plt.title("Energy Generation Over Time")
plt.xlabel("Date")
plt.ylabel("Energy Generated (kWh)")
plt.savefig("visuals/energy_trends_plot.png")

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig("visuals/correlations_heatmap.png")
