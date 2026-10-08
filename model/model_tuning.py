#further tuning for Random forest metrics

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error,mean_absolute_percentage_error, r2_score
from train_test_split import X_train_processed, y_train, X_test_processed, y_test


# Random Forest models to compare
# min_samples_split and min_samples_leaf are fixed
# based on the previous GridSearchCV results

randomForestTuning = {
    "Depth 20, 100 Trees": RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    ),

    "Depth 20, 200 Trees": RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    ),

    "Depth 30, 100 Trees": RandomForestRegressor(
        n_estimators=100,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    ),

    "Depth 30, 200 Trees": RandomForestRegressor(
        n_estimators=200,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
}


# Train and evaluate each Random Forest

for name, model in randomForestTuning.items():

    print("\n==============================")
    print(name)
    print("==============================")


    model.fit(
        X_train_processed,
        y_train
    )

    predictions = model.predict(
        X_test_processed
    )

    # Calculate evaluation metrics
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    mape = mean_absolute_percentage_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print("MAE:", mae)
    print("RMSE:", rmse)
    print("MAPE:", mape)
    print("R²:", r2)