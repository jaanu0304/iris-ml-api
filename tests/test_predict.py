def test_predict_with_valid_input(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post(
        "/api/v1/predict",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "confidence" in data
    assert "model_version" in data
    assert "request_id" in data

    assert data["prediction"] in [
        "setosa",
        "versicolor",
        "virginica"
    ]

    assert 0 <= data["confidence"] <= 1


def test_predict_missing_field_returns_422(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4
    }

    response = client.post(
        "/api/v1/predict",
        json=payload
    )

    assert response.status_code == 422


def test_predict_invalid_value_returns_422(client):
    payload = {
        "sepal_length": -5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post(
        "/api/v1/predict",
        json=payload
    )

    assert response.status_code == 422


def test_v2_predict_with_valid_input(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post(
        "/api/v2/predict",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "probabilities" in data
    assert "model_version" in data
    assert "request_id" in data

    assert data["prediction"] in [
        "setosa",
        "versicolor",
        "virginica"
    ]

    assert isinstance(data["probabilities"], dict)

    assert set(data["probabilities"].keys()) == {
        "setosa",
        "versicolor",
        "virginica"
    }

    assert all(
        0 <= probability <= 1
        for probability in data["probabilities"].values()
    )

    assert abs(sum(data["probabilities"].values()) - 1.0) < 0.001


def test_v2_predict_missing_field_returns_422(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4
    }

    response = client.post(
        "/api/v2/predict",
        json=payload
    )

    assert response.status_code == 422


def test_v1_and_v2_have_different_response_shapes(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    v1_response = client.post(
        "/api/v1/predict",
        json=payload
    )

    v2_response = client.post(
        "/api/v2/predict",
        json=payload
    )

    assert v1_response.status_code == 200
    assert v2_response.status_code == 200

    v1_data = v1_response.json()
    v2_data = v2_response.json()

    # V1 response shape must remain unchanged
    assert set(v1_data.keys()) == {
        "prediction",
        "confidence",
        "model_version",
        "request_id"
    }

    # V2 has the new response shape
    assert set(v2_data.keys()) == {
        "prediction",
        "probabilities",
        "model_version",
        "request_id"
    }

    # Prove that V1 and V2 have different shapes
    assert set(v1_data.keys()) != set(v2_data.keys())

    # Verify V1 is individually correct
    assert v1_data["prediction"] in [
        "setosa",
        "versicolor",
        "virginica"
    ]

    assert 0 <= v1_data["confidence"] <= 1

    # Verify V2 is individually correct
    assert v2_data["prediction"] in [
        "setosa",
        "versicolor",
        "virginica"
    ]

    assert abs(sum(v2_data["probabilities"].values()) - 1.0) < 0.001