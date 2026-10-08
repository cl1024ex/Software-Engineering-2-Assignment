import pandas as pd


# Load the original database
df = pd.read_csv("data/database.csv")


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

    # Convert numerical values to numbers
    for column in numerical_columns:
        user_input[column] = pd.to_numeric(
            user_input[column],
            errors="coerce"
        )

    # Fill missing numerical values
    for column in numerical_columns:
        median_value = df[column].median()
        user_input[column] = user_input[column].fillna(
            median_value
        )

    # Fill missing categorical values
    # Fill missing categorical values
    for column in categorical_columns:

        user_input[column] = user_input[column].replace(
            "", pd.NA)

        user_input[column] = user_input[column].fillna(
            "Unknown" )
        
    # Vehicle Size cannot be "Unknown" because the saved
    # OrdinalEncoder only knows Compact, Midsize and Large.
    # Vehicle Size cannot be "Unknown"
    user_input["Vehicle Size"] = user_input[
        "Vehicle Size"
    ].replace("", pd.NA)

    user_input["Vehicle Size"] = user_input[
        "Vehicle Size"
    ].fillna(
        df["Vehicle Size"].mode()[0]
    )

    return user_input