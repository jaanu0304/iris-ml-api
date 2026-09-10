def test_missing_api_key_returns_401(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post(
        "/api/v1/predict",
        json=payload,
        headers={"X-API-Key": ""}
    )

    assert response.status_code == 401

def test_invalid_api_key_returns_401(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post(
        "/api/v1/predict",
        json=payload,
        headers={"X-API-Key": "invalid-key"}
    )

    assert response.status_code == 401

def test_unexpected_extra_field_returns_422(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
        "unexpected_field": "not allowed"
    }

    response = client.post(
        "/api/v1/predict",
        json=payload
    )

    assert response.status_code == 422