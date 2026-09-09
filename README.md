# ML Model Deployment as a Monitored REST API

## Iris ML API

## Project Overview

This project will build a simple REST API that uses a machine learning model to classify Iris flowers based on their physical measurements.

## Dataset

The project uses the Iris dataset provided as a CSV file for this project.

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

Client Request → Input Validation → Machine Learning Model → Prediction → API Response

In our own words, the client first sends the flower measurements to the `/predict` endpoint. The API checks and validates the input values. If the values are valid, they are passed to the trained machine learning model. The model predicts the Iris flower species, and the API returns the prediction as the response.

## Project Scope

The goal of this project is to build a simple and reliable machine learning API. The focus is on understanding how a machine learning model can be served through a REST API rather than building a complex machine learning model.

## Future Work

The next task will set up the project folder structure and Python environment.


## How to Run This Project

### Using Docker Compose

Make sure Docker Desktop is running.

From the project root directory, run:

```bash
docker compose up --build
```


The API will be available at:

http://localhost:8000

Swagger API documentation:

http://localhost:8000/docs

### Run Without Rebuilding

After the image has already been built, the application can be started with:

```bash
docker compose up 
```

### Stop the Application

Press:

```text
Ctrl + C
```

to stop the running containers.

To remove the containers and network:

```bash
docker compose down
```

### Environment Variables

Environment variables are loaded from the `.env` file using Docker Compose.

The environment variables are not hardcoded in the `docker-compose.yml` file.

### Model Volume

The `ml/saved_model/` directory is mounted into the container so that the trained model can be replaced or updated without rebuilding the entire Docker image.