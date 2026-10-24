import pandas as pd
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load dataset
data = pd.read_csv("house_data.csv")

# Input features and target
X = data[["Area", "Bedrooms", "Bathrooms"]]
y = data["Price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print("Model Training Completed!")
print(f"Mean Absolute Error: ₹{mae:,.2f}")

# Save model
with open("house_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")