from datetime import date

import numpy as np
import pandas as pd

from model.data_loader import load_master_database

df = load_master_database()
YEAR_MIN = int(df["year"].min())
YEAR_MAX = date.today().year
BRAND_LOOKUP = {
    brand.strip().casefold(): brand
    for brand in df["brand"].dropna().unique()
}
BRANDS = sorted(BRAND_LOOKUP.values())


class UserInputValidationError(ValueError):
    pass


numerical_columns = ["year", "mileage", "tax", "mpg", "engine_size"]
categorical_columns = ["brand", "model", "transmission", "fuel_type"]


def prepare_user_input(user_input):

    # Create a copy so the original data is not changed
    user_input = user_input.copy()

    expected_columns = numerical_columns + categorical_columns
    for column in expected_columns:
        if column not in user_input:
            user_input[column] = pd.NA

    brand_values = user_input["brand"].map(
        lambda value: str(value).strip().casefold() if pd.notna(value) else ""
    )
    if not brand_values.isin(BRAND_LOOKUP).all():
        raise UserInputValidationError(
            "Please select a brand from the available car brands."
        )
    user_input["brand"] = brand_values.map(BRAND_LOOKUP)

    year_values = pd.to_numeric(user_input["year"], errors="coerce")
    valid_years = (
        year_values.notna()
        & np.isfinite(year_values)
        & year_values.mod(1).eq(0)
        & year_values.between(YEAR_MIN, YEAR_MAX)
    )
    if not valid_years.all():
        raise UserInputValidationError(
            f"Year must be a whole number between {YEAR_MIN} and {YEAR_MAX}."
        )
    user_input["year"] = year_values

    # Convert numerical values to numbers
    for column in numerical_columns:
        user_input[column] = pd.to_numeric(
            user_input[column],
            errors="coerce"
        )
        user_input[column] = user_input[column].replace(
            [np.inf, -np.inf], np.nan
        )

    for column in numerical_columns:
        user_input[column] = user_input[column].fillna(
            df[column].median()
        )

    for column in categorical_columns:
        if column == "brand":
            continue
        user_input[column] = user_input[column].map(
            lambda value: str(value).strip() if pd.notna(value) else ""
        )
        user_input[column] = user_input[column].replace("", "Unknown")

    return user_input