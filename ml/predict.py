import joblib
import pandas as pd

# Load the saved model
model = joblib.load("ml/saved_model/model.joblib")

# Create one sample flower
sample = pd.DataFrame([
    {
        "sepal length (cm)": 5.1,
        "sepal width (cm)": 3.5,
        "petal length (cm)": 1.4,
        "petal width (cm)": 0.2
    }
])

# Make prediction
prediction = model.predict(sample)

print("Predicted species:", prediction[0])