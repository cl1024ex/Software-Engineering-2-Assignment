"""Integration tests for the Flask app and its HTML responses."""

import app


def test_home_page():
    client = app.app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert b"Car Price Predictor" in response.data
    assert b'name="brand"' in response.data
    assert b'name="year"' in response.data


def test_prediction_page():
    client = app.app.test_client()
    response = client.post(
        "/",
        data={
            "brand": "BMW",
            "model": "3 Series",
            "year": "2019",
            "transmission": "Automatic",
            "mileage": "45000",
            "fuel_type": "Diesel",
            "tax": "150",
            "mpg": "50",
            "engine_size": "2.0",
        },
    )

    assert response.status_code == 200
    assert b"Prediction Result" in response.data
    assert b"The predicted price is:" in response.data
    assert b"Predict Another Car" in response.data


def test_invalid_brand_error():
    client = app.app.test_client()

    response = client.post(
        "/",
        data={"brand": "not a car brand", "model": "3 Series", "year": "2019"},
    )

    assert response.status_code == 400
    assert b"Please select a brand from the available car brands" in response.data
    assert b"Car Price Predictor" in response.data
