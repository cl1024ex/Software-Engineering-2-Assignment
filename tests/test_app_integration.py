"""Verify Flask prediction and validation behavior through HTTP requests."""

import numpy as np

import app as car_app


class CapturingPreprocessor:
    """Record submitted data while delegating transformation to the real pipeline."""

    def __init__(self, preprocessing):
        """Keep the real preprocessing object for delegation."""

        self.preprocessing = preprocessing

    def transform(self, user_input):
        """Capture the input and return its transformed feature matrix."""

        self.user_input = user_input
        return self.preprocessing.transform(user_input)


class FixedPriceModel:
    """Return a fixed log-price prediction for route tests."""

    def predict(self, user_input):
        """Return the test price in the logarithmic scale used by the app."""

        return np.array([np.log1p(12500)])


def test_post_with_bad_car_values_still_returns_prediction(monkeypatch):
    """Check invalid optional values are cleaned and a price is rendered."""

    preprocessor = CapturingPreprocessor(car_app.preprocessing)
    monkeypatch.setattr(car_app, "preprocessing", preprocessor)
    monkeypatch.setattr(car_app, "model", FixedPriceModel())
    client = car_app.app.test_client()

    response = client.post(
        "/",
        data={
            "brand": "BMW",
            "model": "Unlisted Model",
            "year": "2011",
            "mileage": "infinity",
            "fuel_type": "Electric"
        }
    )

    assert response.status_code == 200
    assert b"12,500.00" in response.data
    assert preprocessor.user_input.loc[0, "brand"] == "BMW"
    assert preprocessor.user_input.loc[0, "year"] == 2011
    assert preprocessor.user_input.loc[0, "mileage"] == preprocessor.user_input[
        "mileage"
    ].median()


def test_post_with_unprocessable_input_shows_recoverable_error(monkeypatch):
    """Check transformation errors return a helpful form error."""

    class FailingPreprocessor:
        def transform(self, user_input):
            """Simulate a preprocessing failure for the submitted values."""

            raise ValueError("invalid input")

    monkeypatch.setattr(car_app, "preprocessing", FailingPreprocessor())
    client = car_app.app.test_client()

    response = client.post(
        "/",
        data={"brand": "BMW", "year": "2011"}
    )

    assert response.status_code == 400
    assert b"Some car details could not be processed" in response.data


def test_post_with_unknown_make_shows_validation_error():
    """Check an unsupported brand is rejected with a validation message."""

    response = car_app.app.test_client().post(
        "/",
        data={"brand": "randomletters", "year": "2011"}
    )

    assert response.status_code == 400
    assert b"Please select a brand from the available car brands" in response.data


def test_post_with_out_of_range_year_shows_validation_error():
    """Check years outside the supported range are rejected."""

    response = car_app.app.test_client().post(
        "/",
        data={"brand": "BMW", "year": "7894512"}
    )

    assert response.status_code == 400
    assert (
        f"Year must be a whole number between {car_app.YEAR_MIN} and {car_app.YEAR_MAX}."
        .encode()
        in response.data
    )


def test_post_accepts_current_year(monkeypatch):
    """Check the current year is accepted and passed to preprocessing."""

    preprocessor = CapturingPreprocessor(car_app.preprocessing)
    monkeypatch.setattr(car_app, "preprocessing", preprocessor)
    monkeypatch.setattr(car_app, "model", FixedPriceModel())

    response = car_app.app.test_client().post(
        "/",
        data={"brand": "BMW", "year": str(car_app.YEAR_MAX)}
    )

    assert response.status_code == 200
    assert preprocessor.user_input.loc[0, "year"] == car_app.YEAR_MAX