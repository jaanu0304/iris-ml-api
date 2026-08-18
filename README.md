# ML Model Deployment as a Monitored REST API

## Iris ML API

## Project Overview

This project will build a simple REST API that uses a machine learning model to classify Iris flowers based on their physical measurements.

## Dataset

The project uses the Iris dataset provided by scikit-learn.

The dataset contains four input features:

* Sepal length
* Sepal width
* Petal length
* Petal width

The target variable is the Iris flower species.

## Machine Learning Problem

This is a classification problem.

The model will predict one of three Iris flower species:

* Setosa
* Versicolor
* Virginica

## Machine Learning Model

The project will use Logistic Regression for classification.

## API Contract

The `/predict` endpoint will accept four numerical input values: sepal length, sepal width, petal length, and petal width. The API will validate that the input values are provided and are valid numbers. The valid values will then be passed to the trained machine learning model. The API will return the predicted Iris flower species.

## Example Response


{
  "prediction": "setosa"
}


## Request Flow


```mermaid
flowchart LR
    A[Client Request] --> B[Input Validation]
    B --> C[Machine Learning Model]
    C --> D[Prediction]
    D --> E[API Response]
```

In our own words, the client first sends the flower measurements to the `/predict` endpoint. The API checks and validates the input values. If the values are valid, they are passed to the trained machine learning model. The model predicts the Iris flower species, and the API returns the prediction as the response.

## Project Scope

The goal of this project is to build a simple and reliable machine learning API. The focus is on understanding how a machine learning model can be served through a REST API rather than building a complex machine learning model.

## Future Work

The next task will set up the project folder structure and Python environment.
