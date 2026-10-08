#saving model and preprocessing so it can be used and called upon in flask app

import joblib
from train_model import model
from train_test_split import columnTransformer

joblib.dump(model, "model.pkl")
joblib.dump(columnTransformer, "preprocessing.pkl")