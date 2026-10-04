import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


# first will clean the database 
def load_data(path):
    return pd.read_csv(path)

df = load_data("../data/database.csv")

# Check for missing values
# print("\n========== MISSING VALUES ==========")
# print(df.isnull().sum())
# print(df.isnull().values.any())

# Checking for duplicates
# print("\n========== DUPLICATES ==========")
# print("Number of duplicate rows:", df.duplicated().sum())
# print(df[df.duplicated()])  # Display duplicate rows

# Handling missing values
def handle_missing_values(df):
    # Fill missing numerical values with the median
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_columns:
        median_value = df[col].median()
        df[col].fillna(median_value, inplace=True)

    # Fill missing categorical values with "Unknown" 
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = df[col].fillna("Unknown")

    return df

def handle_duplicates(df):
    # Remove duplicate rows
    df = df.drop_duplicates()
    return df

df = handle_missing_values(df)
df = handle_duplicates(df)

X = df.drop(columns=["Price"])  # dropping the target column for machine learning model
y = df["Price"]

# Now we will preprocess the data using ColumnTransformer, OneHotEncoder, OrdinalEncoder, and StandardScaler
# Each column split based on preprocessing type
numerical_columns = [
    "Year",
    "Engine HP",
    "Engine Cylinders",
    "Number of Doors",
    "highway MPG",
    "city mpg",
    "Popularity"
]

onehotencoder_columns = [
    "Make",
    "Model",
    "Engine Fuel Type",
    "Transmission Type",
    "Driven_Wheels",
    "Market Category",
    "Vehicle Style"
]

ordinal_columns = [
    "Vehicle Size"
]

# Creating the ColumnTransformer for preprocessing and encoding each column type
columnTransformer = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_columns
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            onehotencoder_columns
        ),
        (
            "vehicle_size",
            OrdinalEncoder(
                categories=[["Compact", "Midsize", "Large"]]
            ),
            ["Vehicle Size"]
        )
    ]
)

# Appling the preprocessing
# X_processed = columnTransformer.fit_transform(X)

# print("\nOriginal X columns:")
# print(X.columns.tolist())

# print("\nTarget column:")
# print(y.name)

# print("\nTransformed columns:")
# print(columnTransformer.get_feature_names_out())

# print("\nOriginal X shape:")
# print(X.shape)

# print("\nProcessed X shape:")
# print(X_processed.shape)

# # Show the first 5 processed rows
# processed_df = pd.DataFrame(
#     X_processed.toarray() 
#     if hasattr(X_processed, "toarray") 
#     else X_processed,
#     columns=columnTransformer.get_feature_names_out()
# )

# print("\nFirst 5 processed rows:")
# print(processed_df.head())
