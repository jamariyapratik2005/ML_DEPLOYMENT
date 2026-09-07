"""
Assignment 3 - Q3: Train and save a LinearRegression model
Predicts delivery time based on distance (km) and order size (number of items)
"""
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

# --- Training Data ---
# Features: [distance_km, order_size (num_items)]
# Target:   delivery_time (minutes)

X_train = np.array([
    [2,  1],    # 2 km, 1 item  -> 15 min
    [5,  2],    # 5 km, 2 items -> 25 min
    [1,  1],    # 1 km, 1 item  -> 10 min
    [8,  3],    # 8 km, 3 items -> 40 min
    [3,  2],    # 3 km, 2 items -> 20 min
    [10, 4],    # 10 km, 4 items -> 50 min
    [6,  1],    # 6 km, 1 item  -> 28 min
    [4,  3],    # 4 km, 3 items -> 30 min
    [7,  2],    # 7 km, 2 items -> 35 min
    [12, 5],    # 12 km, 5 items -> 60 min
])

y_train = np.array([15, 25, 10, 40, 20, 50, 28, 30, 35, 60])

# --- Train the model ---
model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained successfully!")
print(f"R² Score: {model.score(X_train, y_train):.4f}")
print(f"Coefficients: distance={model.coef_[0]:.2f}, order_size={model.coef_[1]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# --- Save the model ---
joblib.dump(model, 'assignment3/delivery_model.joblib')
print("\nModel saved as 'assignment3/delivery_model.joblib'")

# --- Quick test ---
test_input = np.array([[5, 2]])
prediction = model.predict(test_input)
print(f"\nTest: distance=5km, order_size=2 items -> Predicted delivery: {prediction[0]:.1f} min")
