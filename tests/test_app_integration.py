import numpy as np

import app as car_app


class CapturingPreprocessor:
    def __init__(self, preprocessing):
        self.preprocessing = preprocessing

    def transform(self, user_input):
        self.user_input = user_input
        return self.preprocessing.transform(user_input)


class FixedPriceModel:
    def predict(self, user_input):
        return np.array([np.log1p(12500)])


def test_post_with_bad_car_values_still_returns_prediction(monkeypatch):
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
    class FailingPreprocessor:
        def transform(self, user_input):
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
    response = car_app.app.test_client().post(
        "/",
        data={"brand": "randomletters", "year": "2011"}
    )

    assert response.status_code == 400
    assert b"Please select a brand from the available car brands" in response.data


def test_post_with_out_of_range_year_shows_validation_error():
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
    preprocessor = CapturingPreprocessor(car_app.preprocessing)
    monkeypatch.setattr(car_app, "preprocessing", preprocessor)
    monkeypatch.setattr(car_app, "model", FixedPriceModel())

    response = car_app.app.test_client().post(
        "/",
        data={"brand": "BMW", "year": str(car_app.YEAR_MAX)}
    )

    assert response.status_code == 200
    assert preprocessor.user_input.loc[0, "year"] == car_app.YEAR_MAX