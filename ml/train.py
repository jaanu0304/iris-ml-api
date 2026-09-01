import pandas as pd
import joblib
import json

from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Model metadata
MODEL_VERSION = "1.0"

FEATURE_NAMES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


# 1. Load the dataset
data = pd.read_csv("ml/iris_dataset.csv")


# 2. Separate input features and target
X = data[FEATURE_NAMES]

y = data["species"]


# 3. Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 4. Create the ML model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# 5. Train the model
model.fit(X_train, y_train)


# 6. Make predictions on test data
y_pred = model.predict(X_test)


# 7. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("Model Training Completed!")
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 8. Save the trained model
joblib.dump(model, "ml/saved_model/model.joblib")

print("\nModel saved successfully!")
print("Location: ml/saved_model/model.joblib")


# 9. Save model metadata
metadata = {
    "model_type": type(model).__name__,
    "model_version": MODEL_VERSION,
    "training_date": datetime.now().strftime("%Y-%m-%d"),
    "expected_features": FEATURE_NAMES,
    "accuracy": round(float(accuracy), 4)
}

with open("ml/saved_model/model_metadata.json", "w") as file:
    json.dump(metadata, file, indent=4)

print("Model metadata saved successfully!")
print("Location: ml/saved_model/model_metadata.json")