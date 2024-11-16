# main.py

from sklearn.model_selection import train_test_split
from src.data_loader import load_data
from src.feature_selection import select_features
from src.model_training import train_model, evaluate_model
from src.visualization import plot_actual_vs_predicted

# Step 1: Load the data
df = load_data()

# Step 2: Split data into features and target
X = df.drop(columns=['target'])
y = df['target']

# Step 3: Perform feature selection
X_selected = select_features(X, y, k=5)

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Step 5: Train the model
model = train_model(X_train, y_train)

# Step 6: Evaluate the model
mse, r2 = evaluate_model(model, X_test, y_test)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Step 7: Visualize results
y_pred = model.predict(X_test)
plot_actual_vs_predicted(y_test, y_pred)

