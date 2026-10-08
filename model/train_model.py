from sklearn.ensemble import RandomForestRegressor
from train_test_split import X_train_processed, y_train, X_test_processed, y_test

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_processed, y_train)
