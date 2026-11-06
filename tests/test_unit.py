"""Unit tests for data loading and user input preparation."""

import pandas as pd
import pytest

from model import data_loader
from UserInputHandling.preprocessing_user_input import (
    UserInputValidationError,
    prepare_user_input,
    year_max,
)


valid_user_input = pd.DataFrame(
    [
        {
            "brand": "BMW",
            "model": "3 Series",
            "year": 2019,
            "transmission": "Automatic",
            "mileage": 45000,
            "fuel_type": "Diesel",
            "tax": 150,
            "mpg": 50,
            "engine_size": 2.0,
        }
    ]
)


def test_cleaning_a_car_value(tmp_path):
    source = tmp_path / "audi.csv"
    pd.DataFrame(
        [
            {
                "model": " A1 ",
                "year": "2017",
                "price": "£12,500",
                "transmission": " Manual ",
                "mileage": "15,735",
                "fuelType": " Petrol ",
                "tax": 150,
                "mpg": 55.4,
                "engineSize": 1.4,
            }
        ]
    ).to_csv(source, index=False)

    result = data_loader.load_source(source)

    assert result.loc[0, "brand"] == "Audi"
    assert result.loc[0, "model"] == "A1"
    assert result.loc[0, "price"] == 12500
    assert result.loc[0, "mileage"] == 15735
    assert result.loc[0, "fuel_type"] == "Petrol"
    assert result.columns.tolist() == data_loader.master_columns

def test_build_master_database_removes_duplicates(tmp_path, monkeypatch):
    source = tmp_path / "bmw.csv"

    car = {
        "model": "3 Series",
        "year": 2019,
        "price": 12000,
        "transmission": "Automatic",
        "mileage": 45000,
        "fuel_type": "Diesel",
        "tax": 150,
        "mpg": 50,
        "engine_size": 2.0,
    }

    pd.DataFrame([car, car]).to_csv(source, index=False)
    monkeypatch.setattr(data_loader, "data_directory", tmp_path)

    result = data_loader.build_master_database(output_path=None)

    assert len(result) == 1

def test_build_master_database_removes_invalid_price(tmp_path, monkeypatch):
    source = tmp_path / "bmw.csv"

    pd.DataFrame([
        {
            "model": "3 Series",
            "year": 2019,
            "price": 12000,
            "transmission": "Automatic",
            "mileage": 45000,
            "fuel_type": "Diesel",
            "tax": 150,
            "mpg": 50,
            "engine_size": 2.0,
        },
        {
            "model": "5 Series",
            "year": 2019,
            "price": 0,
            "transmission": "Automatic",
            "mileage": 30000,
            "fuel_type": "Petrol",
            "tax": 150,
            "mpg": 45,
            "engine_size": 2.0,
        },
    ]).to_csv(source, index=False)

    monkeypatch.setattr(data_loader, "data_directory", tmp_path)

    result = data_loader.build_master_database(output_path=None)

    assert len(result) == 1
    assert result.loc[0, "model"] == "3 Series"
    assert result.loc[0, "price"] == 12000

def test_converting_numeric_user_input_values():
    result = prepare_user_input(valid_user_input.copy())

    assert result.loc[0, "year"] == 2019
    assert result.loc[0, "mileage"] == 45000
    assert result.loc[0, "engine_size"] == 2.0


def test_prepare_user_input_missing_values():
    user_input = valid_user_input.copy()
    user_input["mileage"] = "unknown"
    user_input.loc[0, "transmission"] = ""

    expected_mileage = data_loader.load_master_database()["mileage"].median()

    result = prepare_user_input(user_input)

    assert result.loc[0, "mileage"] == expected_mileage
    assert result.loc[0, "transmission"] == "Unknown"
    assert user_input.loc[0, "mileage"] == "unknown"


def test_unknown_brand():
    user_input = valid_user_input.copy()
    user_input.loc[0, "brand"] = "Unknown brand"

    with pytest.raises(UserInputValidationError):
        prepare_user_input(user_input)


def test_future_year_invalid():
    user_input = valid_user_input.copy()
    user_input.loc[0, "year"] = year_max + 1

    with pytest.raises(UserInputValidationError):
        prepare_user_input(user_input)

def test_non_numeric_year():
    user_input = valid_user_input.copy()
    user_input["year"] = user_input["year"].astype("object")
    user_input.loc[0, "year"] = "not a year"

    with pytest.raises(UserInputValidationError):
        prepare_user_input(user_input)
