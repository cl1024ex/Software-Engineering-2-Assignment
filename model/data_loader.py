from datetime import date
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data"
MASTER_DATABASE_PATH = DATA_DIRECTORY / "database.csv"

BRAND_NAMES = {
    "audi": "Audi",
    "bmw": "BMW",
    "cclass": "Mercedes-Benz",
    "focus": "Ford",
    "ford": "Ford",
    "hyundi": "Hyundai",
    "merc": "Mercedes-Benz",
    "skoda": "Skoda",
    "toyota": "Toyota",
    "vauxhall": "Vauxhall",
    "vw": "Volkswagen",
}

MASTER_COLUMNS = [
    "brand",
    "model",
    "year",
    "price",
    "transmission",
    "mileage",
    "fuel_type",
    "tax",
    "mpg",
    "engine_size",
]
NUMERIC_COLUMNS = ["year", "price", "mileage", "tax", "mpg", "engine_size"]
CATEGORICAL_COLUMNS = ["brand", "model", "transmission", "fuel_type"]


def _load_source(path):
    is_unclean = path.name.startswith("unclean ")
    source_name = path.stem.removeprefix("unclean ")
    frame = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    frame = frame.replace(r"^\s*$", pd.NA, regex=True)

    if is_unclean:
        for primary, alternate in [
            ("mileage", "mileage2"),
            ("fuel type", "fuel type2"),
            ("engine size", "engine size2"),
        ]:
            frame[primary] = frame[primary].combine_first(frame[alternate])

    frame = frame.rename(
        columns={
            "fueltype": "fuel_type",
            "fuel type": "fuel_type",
            "enginesize": "engine_size",
            "engine size": "engine_size",
        }
    )
    tax_column = next(
        (column for column in frame.columns if column.startswith("tax")), None
    )
    if tax_column and tax_column != "tax":
        frame = frame.rename(columns={tax_column: "tax"})

    frame["brand"] = BRAND_NAMES[source_name]
    for column in MASTER_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    frame = frame[MASTER_COLUMNS].copy()

    for column in NUMERIC_COLUMNS:
        values = (
            frame[column]
            .astype("string")
            .str.replace(",", "", regex=False)
            .str.replace(r"[^0-9.\-]", "", regex=True)
        )
        frame[column] = pd.to_numeric(values, errors="coerce")

    for column in CATEGORICAL_COLUMNS:
        frame[column] = frame[column].astype("string").str.strip()
        frame[column] = frame[column].replace("", pd.NA)

    return frame


def build_master_database(output_path=MASTER_DATABASE_PATH):
    """Merge, clean, and optionally save all brand listings."""
    source_paths = sorted(
        path
        for path in DATA_DIRECTORY.glob("*.csv")
        if path.name != MASTER_DATABASE_PATH.name
    )
    frames = [
        _load_source(path).dropna(axis=1, how="all")
        for path in source_paths
    ]
    master = pd.concat(frames, ignore_index=True)

    master = master.dropna(subset=["brand", "model", "year", "price"])
    master = master[
        master["year"].between(1900, date.today().year)
        & master["price"].gt(0)
    ].copy()

    for column in ["mileage", "tax", "mpg", "engine_size"]:
        master.loc[master[column] < 0, column] = pd.NA
        master[column] = master[column].fillna(master[column].median())
    for column in ["transmission", "fuel_type"]:
        master[column] = master[column].fillna("Unknown")

    master["year"] = master["year"].astype(int)
    master = master.drop_duplicates(subset=MASTER_COLUMNS).reset_index(drop=True)
    master = master[MASTER_COLUMNS]

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        master.to_csv(output_path, index=False)

    return master


def load_master_database():
    """Load the saved master dataset, building it if it does not exist."""
    if not MASTER_DATABASE_PATH.exists():
        return build_master_database()
    return pd.read_csv(MASTER_DATABASE_PATH)


if __name__ == "__main__":
    master = build_master_database()
    print(f"Saved {len(master):,} cleaned listings to {MASTER_DATABASE_PATH}")