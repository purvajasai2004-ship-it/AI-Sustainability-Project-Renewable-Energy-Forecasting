#  AI-Based Renewable Energy Forecasting
# Week 2 – Data Preprocessing, EDA, and Model Training

# Import Libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Create Outputs Folder 
os.makedirs("outputs", exist_ok=True)

# Load Dataset 
# You can replace this with your actual dataset file name from Kaggle
# Example dataset: https://www.kaggle.com/datasets/anikannal/solar-power-generation-data
data = pd.read_csv("solar_generation.csv")

# Preview Dataset
print("✅ Dataset Loaded Successfully!\n")
print(data.head())
print("\nDataset Info:")
print(data.info())

# Handle Missing Values
print("\nMissing Values Before:")
print(data.isnull().sum())

data = data.dropna()

print("\nMissing Values After:")
print(data.isnull().sum())

# Select Relevant Columns
# Assume dataset has columns like ['DATE_TIME', 'PLANT_ID', 'DC_POWER', 'AC_POWER', 'DAILY_YIELD']
# We'll predict DAILY_YIELD based on other numeric features
num_cols = data.select_dtypes(include=[np.number]).columns
data = data[num_cols]

# Feature Correlation 
plt.figure(figsize=(8,6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig("outputs/correlation_heatmap.png")
plt.show()

# Basic Trend Visualization 
if 'DAILY_YIELD' in data.columns:
    plt.figure(figsize=(10,5))
    plt.plot(data['DAILY_YIELD'][:200])
    plt.title("Energy Generation Trend (Sample)")
    plt.xlabel("Time Index")
    plt.ylabel("Daily Energy Yield")
    plt.savefig("outputs/trend_plot.png")
    plt.show()

# Define Features and Target 
if 'DAILY_YIELD' in data.columns:
    X = data.drop('DAILY_YIELD', axis=1)
    y = data['DAILY_YIELD']
else:
    # fallback target variable
    y = data.iloc[:, -1]
    X = data.iloc[:, :-1]

# Split Data 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train AI Model (Linear Regression) 
model = LinearRegression()
model.fit(X_train, y_train)

# Predict 
y_pred = model.predict(X_test)

# Evaluate Model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Performance:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Save Metrics
with open("outputs/model_metrics.txt", "w") as f:
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"R2 Score: {r2:.4f}\n")

#  Visualization: Actual vs Predicted 
plt.figure(figsize=(10,5))
plt.plot(y_test.values[:100], label="Actual", color='blue')
plt.plot(y_pred[:100], label="Predicted", color='red')
plt.title("Actual vs Predicted Energy Generation (Sample)")
plt.xlabel("Sample Index")
plt.ylabel("Energy Generation")
plt.legend()
plt.savefig("outputs/actual_vs_predicted.png")
plt.show()

# Model Summary 
print("\n✅ Week 2 Progress Summary:")
print("✔ Data cleaning and preprocessing completed")
print("✔ EDA visualizations generated and saved")
print("✔ Baseline Linear Regression model trained")
print("✔ Model performance metrics saved in outputs folder")
print("\nNext Week (Week 3): Plan to use RandomForest/LSTM for better accuracy.")
