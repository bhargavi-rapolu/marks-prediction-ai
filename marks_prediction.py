import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("student_data.csv")

# Check for missing or invalid values
if data.isnull().values.any():
    print(" Warning: Dataset contains missing values. Please clean the data.")
    exit()

# Features and target
X = data[['study_hours', 'sleep_hours', 'attendance', 'previous_marks']]
y = data['current_marks']

# Ensure all values are positive for plotting
if (y <= 0).any():
    print(" Error: 'current_marks' contains zero or negative values. Cannot plot.")
    exit()

# Linear Regression
lr = LinearRegression()
lr.fit(X, y)
lr_preds = lr.predict(X)

# Random Forest Regressor
rf = RandomForestRegressor()
rf.fit(X, y)
rf_preds = rf.predict(X)

# Evaluation
print(" Linear Regression Results:")
print("R² Score:", round(r2_score(y, lr_preds), 3))
print("MAE:", round(mean_absolute_error(y, lr_preds), 2))

print("\n Random Forest Results:")
print("R² Score:", round(r2_score(y, rf_preds), 3))
print("MAE:", round(mean_absolute_error(y, rf_preds), 2))

# Create results folder if it doesn't exist
os.makedirs("results", exist_ok=True)

# Visualization
plt.figure(figsize=(8, 6))
plt.xscale('linear')
plt.yscale('linear')
plt.scatter(y, rf_preds, color='blue', label='Predicted vs Actual')
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Marks Prediction")
plt.legend()
plt.grid(True)
plt.savefig("results/prediction_plot.png")
plt.show()
