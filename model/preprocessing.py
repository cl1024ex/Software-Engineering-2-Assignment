import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler


# first will clean the database 
def load_data(path):
    return pd.read_csv(path)


def clean_data(df):
    # cleaning the database
    df = df.drop(columns=["Car ID"])  # dropping unnecessary columns for machine learning model
    return df


df = load_data("../data/database.csv")
df = clean_data(df)

X = df.drop(columns=["Price"])  # dropping the target column for machine learning model
y = df["Price"]


# now encoding the data for machine learning model

# Encoding for fuel type, transmission and condition columns

columnTransformer = ColumnTransformer(
    transformers=[
        ("brand", OneHotEncoder(handle_unknown="ignore"), ["Brand"]),
        ("fuel_type", OneHotEncoder(handle_unknown="ignore"), ["Fuel Type"]),
        ("transmission", OneHotEncoder(handle_unknown="ignore"), ["Transmission"]),
        ("condition", OrdinalEncoder(categories=[["Used", "Like New", "New"]]), ["Condition"]),
        ("model", OneHotEncoder(handle_unknown="ignore"), ["Model"]),

        # data normalisation for numerical columns
        ("year", StandardScaler(), ["Year"]),
        ("engine_size", StandardScaler(), ["Engine Size"]),
        ("mileage", StandardScaler(), ["Mileage"]),
    ],
)

# Appling the preprocessing
X_processed = columnTransformer.fit_transform(X)

print("\nOriginal X columns:")
print(X.columns.tolist())

print("\nTarget column:")
print(y.name)

print("\nTransformed columns:")
print(columnTransformer.get_feature_names_out())

print("\nOriginal X shape:")
print(X.shape)

print("\nProcessed X shape:")
print(X_processed.shape)

# Show the first 5 processed rows
processed_df = pd.DataFrame(
    X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed,
    columns=columnTransformer.get_feature_names_out()
)

print("\nFirst 5 processed rows:")
print(processed_df.head())