# Dataset Information

## 🧾 Overview
The dataset contains renewable energy generation statistics with weather parameters to predict solar/wind energy output.

| Column | Description | Type |
|--------|--------------|------|
| Date | Date of data collection | datetime |
| Temperature | Average temperature (°C) | numeric |
| Humidity | Relative humidity (%) | numeric |
| WindSpeed | Wind speed (m/s) | numeric |
| SolarRadiation | Solar radiation (W/m²) | numeric |
| EnergyGenerated | Actual energy produced (kWh) | numeric |

## ⚙️ Preprocessing Steps
1. Removed null and duplicate records  
2. Converted all column names to lowercase  
3. Converted date column to `datetime` format  
4. Normalized numeric values using MinMaxScaler  
5. Stored cleaned dataset as `cleaned_energy_data.csv`

## 📊 Data Snapshot
| Date | Temperature | Humidity | WindSpeed | EnergyGenerated |
|------|--------------|-----------|------------|----------------|
| 2023-01-01 | 32 | 55 | 3.5 | 1450 |
| 2023-01-02 | 33 | 52 | 4.1 | 1520 |
| ... | ... | ... | ... | ... |

## 📈 Insights
- Energy generation increases with solar radiation.
- High humidity slightly reduces solar energy output.
- Strong wind speeds positively impact wind energy.
- Seasonal variation observed in both solar and wind outputs.
