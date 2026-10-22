from datetime import date

from model.data_loader import MASTER_COLUMNS, build_master_database


def test_master_database_is_clean_and_contains_all_brands():
    master = build_master_database(output_path=None)

    assert master.columns.tolist() == MASTER_COLUMNS
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