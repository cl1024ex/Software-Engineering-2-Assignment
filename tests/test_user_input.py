"""Test conversion, defaults, and validation for submitted vehicle details."""

import pandas as pd
import pytest

from UserInputHandling.preprocessing_user_input import (
    YEAR_MAX,
    UserInputValidationError,
    prepare_user_input
)


def make_user_input(**overrides):
    """Build one valid form row, applying any supplied field overrides."""

    values = {
        "brand": "BMW",
        "model": "1 Series",
        "year": "2011",
        "transmission": "Manual",
        "mileage": "25000",
        "fuel_type": "Petrol",
        "tax": "150",
        "mpg": "55.4",
        "engine_size": "2.0"
    }
    values.update(overrides)
    return pd.DataFrame([values])


def test_numeric_input_is_converted():
    """Check numeric form values are converted to numeric types."""

    result = prepare_user_input(make_user_input())

    assert pd.api.types.is_numeric_dtype(result["engine_size"])
    assert result.loc[0, "engine_size"] == 2.0


def test_invalid_numeric_values_are_replaced_with_medians():
    """Check malformed numeric values use dataset medians without mutating input."""

    original = make_user_input(
        **{"mileage": "", "engine_size": "unknown"}
    )

    result = prepare_user_input(original)

    for column in ["mileage", "engine_size"]:
        assert result.loc[0, column] == result[column].median()
    assert original.loc[0, "mileage"] == ""


def test_infinite_numeric_values_are_replaced_with_medians():
    """Check infinite numeric values use the corresponding dataset median."""

    result = prepare_user_input(make_user_input(**{"mileage": "inf"}))

    assert result.loc[0, "mileage"] == result["mileage"].median()


def test_missing_columns_are_filled_with_safe_defaults():
    """Check omitted fields receive defaults while valid values are retained."""

    result = prepare_user_input(
        pd.DataFrame([{"brand": "BMW", "year": "2011"}])
    )

    assert result.loc[0, "brand"] == "BMW"
    assert result.loc[0, "transmission"] == "Unknown"
    assert result.loc[0, "year"] == 2011


def test_blank_categorical_values_are_replaced_with_unknown():
    """Check blank categorical form values become the Unknown category."""

    result = prepare_user_input(
        make_user_input(**{"model": "  ", "fuel_type": None})
    )

    assert result.loc[0, "model"] == "Unknown"
    assert result.loc[0, "fuel_type"] == "Unknown"


def test_unseen_categorical_values_remain_safe_strings():
    """Check unseen categories are preserved for the encoder to handle."""

    result = prepare_user_input(
        make_user_input(**{"model": "New Model"})
    )

    assert result.loc[0, "brand"] == "BMW"
    assert result.loc[0, "model"] == "New Model"


def test_make_is_normalized_case_insensitively():
    """Check brand matching ignores case and surrounding whitespace."""

    result = prepare_user_input(make_user_input(**{"brand": "  bmw  "}))

    assert result.loc[0, "brand"] == "BMW"


def test_current_year_is_accepted():
    """Check the current calendar year passes year validation."""

    result = prepare_user_input(make_user_input(year=str(YEAR_MAX)))

    assert result.loc[0, "year"] == YEAR_MAX


def test_unknown_make_is_rejected():
    """Check brands absent from the dataset raise a validation error."""

    with pytest.raises(UserInputValidationError, match="available car brands"):
        prepare_user_input(make_user_input(**{"brand": "randomletters"}))


@pytest.mark.parametrize("year", ["", "not a year", "7894512", "1899", "2026.5"])
def test_invalid_year_is_rejected(year):
    """Check missing, non-integer, and out-of-range years are rejected."""

    with pytest.raises(UserInputValidationError, match="whole number"):
        prepare_user_input(make_user_input(year=year))


def test_brand_is_trimmed_and_normalized_case_insensitively():
    """Check the accepted brand value is normalized to its canonical form."""

    result = prepare_user_input(make_user_input(**{"brand": " bmw "}))

    assert result.loc[0, "brand"] == "BMW"