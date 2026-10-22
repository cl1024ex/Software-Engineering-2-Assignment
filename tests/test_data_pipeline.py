"""Check the generated master dataset is complete and internally consistent."""

from datetime import date

from model.data_loader import build_master_database, master_columns


def test_master_database_is_clean_and_contains_all_brands():
    """Verify the merged dataset has expected columns, brands, and clean values."""

    master = build_master_database(output_path=None)

    assert master.columns.tolist() == master_columns
    assert not master.isna().any().any()
    assert not master.duplicated().any()
    assert (master["price"] > 0).all()
    assert master["year"].between(1900, date.today().year).all()
    assert set(master["brand"]) == {
        "Audi",
        "BMW",
        "Ford",
        "Hyundai",
        "Mercedes-Benz",
        "Skoda",
        "Toyota",
        "Vauxhall",
        "Volkswagen",
    }