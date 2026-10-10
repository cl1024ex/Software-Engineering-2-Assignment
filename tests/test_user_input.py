import pandas as pd
import pytest

from UserInputHandling.preprocessing_user_input import (
    YEAR_MAX,
    UserInputValidationError,
    prepare_user_input
)


def make_user_input(**overrides):
    values = {
        "Make": "BMW",
        "Model": "1 Series M",
        "Year": "2011",
        "Engine Fuel Type": "premium unleaded (required)",
        "Engine HP": "335",
        "Engine Cylinders": "6",
        "Transmission Type": "MANUAL",
        "Driven_Wheels": "rear wheel drive",
        "Number of Doors": "2",
        "Market Category": "",
        "Vehicle Size": "Compact",
        "Vehicle Style": "Coupe",
        "highway MPG": "26",
        "city mpg": "19",
        "Popularity": "3916"
    }
    values.update(overrides)
    return pd.DataFrame([values])


def test_numeric_input_is_converted():
    result = prepare_user_input(make_user_input())

    assert pd.api.types.is_numeric_dtype(result["Engine HP"])
    assert result.loc[0, "Engine HP"] == 335


def test_invalid_numeric_values_are_replaced_with_medians():
    original = make_user_input(
        **{"Engine HP": "", "Engine Cylinders": "unknown"}
    )

    result = prepare_user_input(original)

    for column in ["Engine HP", "Engine Cylinders"]:
        assert result.loc[0, column] == result[column].median()
    assert original.loc[0, "Engine HP"] == ""


def test_infinite_numeric_values_are_replaced_with_medians():
    result = prepare_user_input(make_user_input(**{"Engine HP": "inf"}))

    assert result.loc[0, "Engine HP"] == result["Engine HP"].median()


def test_missing_columns_are_filled_with_safe_defaults():
    result = prepare_user_input(
        pd.DataFrame([{"Make": "BMW", "Year": "2011"}])
    )

    assert result.loc[0, "Make"] == "BMW"
    assert result.loc[0, "Model"] == "Unknown"
    assert result.loc[0, "Year"] == 2011
    assert result.loc[0, "Vehicle Size"] in ["Compact", "Midsize", "Large"]


def test_blank_categorical_values_are_replaced_with_unknown():
    result = prepare_user_input(
        make_user_input(**{"Model": "  ", "Market Category": None})
    )

    assert result.loc[0, "Model"] == "Unknown"
    assert result.loc[0, "Market Category"] == "Unknown"


def test_unseen_categorical_values_remain_safe_strings():
    result = prepare_user_input(
        make_user_input(**{"Model": "New Model"})
    )

    assert result.loc[0, "Make"] == "BMW"
    assert result.loc[0, "Model"] == "New Model"


def test_make_is_normalized_case_insensitively():
    result = prepare_user_input(make_user_input(**{"Make": "  bmw  "}))

    assert result.loc[0, "Make"] == "BMW"


def test_current_year_is_accepted():
    result = prepare_user_input(make_user_input(Year=str(YEAR_MAX)))

    assert result.loc[0, "Year"] == YEAR_MAX


def test_unknown_make_is_rejected():
    with pytest.raises(UserInputValidationError, match="available car makes"):
        prepare_user_input(make_user_input(**{"Make": "randomletters"}))


@pytest.mark.parametrize("year", ["", "not a year", "7894512", "1989", "2026.5"])
def test_invalid_year_is_rejected(year):
    with pytest.raises(UserInputValidationError, match="whole number"):
        prepare_user_input(make_user_input(Year=year))


def test_invalid_vehicle_size_uses_a_valid_default():
    result = prepare_user_input(
        make_user_input(**{"Vehicle Size": "aircraft"})
    )

    assert result.loc[0, "Vehicle Size"] in ["Compact", "Midsize", "Large"]


def test_vehicle_size_is_normalized_case_insensitively():
    result = prepare_user_input(
        make_user_input(**{"Vehicle Size": " compact "})
    )

    assert result.loc[0, "Vehicle Size"] == "Compact"