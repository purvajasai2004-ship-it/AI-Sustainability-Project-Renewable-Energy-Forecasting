import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("path_to_your_dataset.csv")

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Convert date column
data['Date'] = pd.to_datetime(data['Date'])

# Normalize numerical features
scaler = MinMaxScaler()
num_cols = ['Temperature', 'Humidity', 'WindSpeed', 'SolarRadiation', 'EnergyGenerated']
data[num_cols] = scaler.fit_transform(data[num_cols])

# Save cleaned data
data.to_csv("cleaned_energy_data.csv", index=False)
data.head()
