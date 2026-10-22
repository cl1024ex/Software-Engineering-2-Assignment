"""Persist the trained regressor and fitted preprocessing pipeline."""

import joblib
from pathlib import Path

from train_model import model
from train_test_split import columnTransformer

saved_model_directory = Path(__file__).resolve().parents[1] / "Saved model"
saved_model_directory.mkdir(parents=True, exist_ok=True)
joblib.dump(model, saved_model_directory / "model.pkl")
joblib.dump(columnTransformer, saved_model_directory / "preprocessing.pkl")