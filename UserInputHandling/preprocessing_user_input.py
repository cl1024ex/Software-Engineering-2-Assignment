from datetime import date

import numpy as np
import pandas as pd


# Load the original database
df = pd.read_csv("data/database.csv")
YEAR_MIN = int(df["Year"].min())
YEAR_MAX = date.today().year
MAKE_LOOKUP = {
    make.strip().casefold(): make
    for make in df["Make"].dropna().unique()
}


class UserInputValidationError(ValueError):
    pass


# Numerical columns
numerical_columns = [
    "Year",
    "Engine HP",
    "Engine Cylinders",
    "Number of Doors",
    "highway MPG",
    "city mpg",
    "Popularity"
]


# Categorical columns
categorical_columns = [
    "Make",
    "Model",
    "Engine Fuel Type",
    "Transmission Type",
    "Driven_Wheels",
    "Market Category",
    "Vehicle Style"
]


def prepare_user_input(user_input):

    # Create a copy so the original data is not changed
    user_input = user_input.copy()

    expected_columns = numerical_columns + categorical_columns + [
        "Vehicle Size"
    ]
    for column in expected_columns:
        if column not in user_input:
            user_input[column] = pd.NA

    make_values = user_input["Make"].map(
        lambda value: str(value).strip().casefold() if pd.notna(value) else ""
    )
    if not make_values.isin(MAKE_LOOKUP).all():
        raise UserInputValidationError(
            "Please enter a make from the available car makes."
        )
    user_input["Make"] = make_values.map(MAKE_LOOKUP)

    year_values = pd.to_numeric(user_input["Year"], errors="coerce")
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
    user_input["Year"] = year_values

    # Convert numerical values to numbers
    for column in numerical_columns:
        user_input[column] = pd.to_numeric(
            user_input[column],
            errors="coerce"
        )
        user_input[column] = user_input[column].replace(
            [np.inf, -np.inf], np.nan
        )

    # Fill missing numerical values
    for column in numerical_columns:
        median_value = df[column].median()
        user_input[column] = user_input[column].fillna(
            median_value
        )

    for column in categorical_columns:
        user_input[column] = user_input[column].map(
            lambda value: str(value).strip() if pd.notna(value) else ""
        )
        user_input[column] = user_input[column].replace("", "Unknown")

    valid_vehicle_sizes = {
        "compact": "Compact",
        "midsize": "Midsize",
        "large": "Large"
    }
    default_vehicle_size = df["Vehicle Size"].mode()[0]

    def normalize_vehicle_size(value):
        if pd.isna(value):
            return default_vehicle_size
        return valid_vehicle_sizes.get(
            str(value).strip().casefold(), default_vehicle_size
        )

    user_input["Vehicle Size"] = user_input["Vehicle Size"].map(
        normalize_vehicle_size
    )

    return user_input