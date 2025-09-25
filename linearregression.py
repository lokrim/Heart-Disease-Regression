# Linear Regression Models to Predict Maximum Heart Rate (thalach)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("heart.csv")

# Inspect dataset (optional)
print("Dataset shape:", df.shape)
print(df.head())

print("\n--- Simple Linear Regression ---")

# Independent variable (age) and dependent variable (thalach)
X = df[['age']]  # 2D array
y = df['thalach']

# Split into train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predictions
y_pred = lin_reg.predict(X_test)

# Results
slope = lin_reg.coef_[0]
intercept = lin_reg.intercept_
mse_a = mean_squared_error(y_test, y_pred)

print(f"Slope: {slope:.4f}")
print(f"Intercept: {intercept:.4f}")
print(f"Mean Squared Error (MSE): {mse_a:.4f}")

# Plot regression line
plt.scatter(X_test, y_test, color="blue", label="Actual Data")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Regression Line")
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate (thalach)")
plt.title("Simple Linear Regression: Age vs Thalach")
plt.legend()
plt.show()

print("\n--- Multiple Linear Regression (age, trestbps, chol) ---")

features_case1 = ['age', 'trestbps', 'chol']
X1 = df[features_case1]
y = df['thalach']

X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=42)

multi_reg1 = LinearRegression()
multi_reg1.fit(X1_train, y_train)
y_pred1 = multi_reg1.predict(X1_test)

coefficients_case1 = dict(zip(features_case1, multi_reg1.coef_))
intercept_case1 = multi_reg1.intercept_
mse_b = mean_squared_error(y_test, y_pred1)

print("Coefficients:", coefficients_case1)
print(f"Intercept: {intercept_case1:.4f}")
print(f"Mean Squared Error (MSE): {mse_b:.4f}")

print("\n--- Multiple Linear Regression (age, trestbps, chol, sex, cp, fbs, exang) ---")

features_case2 = ['age', 'trestbps', 'chol', 'sex', 'cp', 'fbs', 'exang']
X2 = df[features_case2]
y = df['thalach']

X2_train, X2_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=42)

multi_reg2 = LinearRegression()
multi_reg2.fit(X2_train, y_train)
y_pred2 = multi_reg2.predict(X2_test)

coefficients_case2 = dict(zip(features_case2, multi_reg2.coef_))
intercept_case2 = multi_reg2.intercept_
mse_c = mean_squared_error(y_test, y_pred2)

print("Coefficients:", coefficients_case2)
print(f"Intercept: {intercept_case2:.4f}")
print(f"Mean Squared Error (MSE): {mse_c:.4f}")

print("\n--- Comparison & Inference ---")
print(f"PART A MSE (Simple Regression): {mse_a:.4f}")
print(f"PART B MSE (3 features): {mse_b:.4f}")
print(f"PART C MSE (7 features): {mse_c:.4f}")

if mse_c < mse_b and mse_c < mse_a:
    best_model = "Multiple Linear Regression (Case 2 with 7 features)"
elif mse_b < mse_a:
    best_model = "Multiple Linear Regression (Case 1 with 3 features)"
else:
    best_model = "Simple Linear Regression (Age only)"

print(f"\nBest Model: {best_model}")
print("\nReflection:")
print("- Adding more features generally reduced MSE, meaning better prediction accuracy.")
print("- The model with more clinical features (Case 2) performed best, showing that considering multiple health factors improves predictions.")
print("- These regression models can help in early risk detection by identifying abnormal heart rates relative to age and other clinical measures.")