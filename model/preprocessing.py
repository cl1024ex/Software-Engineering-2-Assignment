"""Load training data and define the feature preprocessing pipeline."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(path):
    """Read a CSV dataset from the supplied path."""

    return pd.read_csv(path)


df = load_data("../data/database.csv")

X = df.drop(columns=["price"])
y = np.log1p(df["price"])  # Train on log prices; the app reverses this transform.

numerical_columns = [
    "year",
    "mileage",
    "tax",
    "mpg",
    "engine_size",
]

categorical_columns = [
    "brand",
    "model",
    "transmission",
    "fuel_type",
]

columnTransformer = ColumnTransformer(
    transformers=[
        (
            "numerical",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numerical_columns,
        ),
        (
            "categorical",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="constant",
                            fill_value="Unknown",
                        ),
                    ),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_columns,
        ),
    ]
)