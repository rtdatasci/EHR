# main.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes


print("Current Working Directory:", os.getcwd())


# Step 1: Load the data
print("Loading data...")
from src.data_loader import load_data
df = load_data()

# Step 2: Split data into features and target
X = df.drop(columns=['target'])
y = df['target']

# Step 3: Perform feature selection
print("Performing feature selection...")
from src.feature_selection import select_features
X_selected = select_features(X, y, k=5)

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Step 5: Train the model
print("Training model...")
from src.model_training import train_model
model = train_model(X_train, y_train)

# Step 6: Evaluate the model
from src.model_training import evaluate_model
mse, r2 = evaluate_model(model, X_test, y_test)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Step 7: Visualize results
from src.visualization import plot_actual_vs_predicted
y_pred = model.predict(X_test)
plot_actual_vs_predicted(y_test, y_pred)



# Save the model
os.makedirs("output/models", exist_ok=True)
model_path = "output/models/diabetes_model.pkl"
print(f"Saving model to {model_path}")
joblib.dump(model, model_path)
print("Model saved successfully.")



# Save metrics
os.makedirs("output/metrics", exist_ok=True)
metrics = {"Mean Squared Error": mse, "R² Score": r2}
with open("output/metrics/model_metrics.json", "w") as f:
    json.dump(metrics, f)

# Save visualization
os.makedirs("output/figures", exist_ok=True)
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
plt.savefig("output/figures/actual_vs_predicted.png")
plt.close()

print("Outputs saved successfully!")
