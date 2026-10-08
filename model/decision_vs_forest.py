import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_absolute_percentage_error
from train_test_split import X_train_processed, y_train, X_test_processed, y_test


# Scoring metrics for model evaluation
#MAE: Mean Absolute Error
#RMSE: Root Mean Squared Error
#MAPE: Mean Absolute Percentage Error
#R2: Coefficient of Determination

scoring = {
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "MAPE": make_scorer(
        mean_absolute_percentage_error,
        greater_is_better=False),
    "R2": "r2"
}

# Decision tree fine tuning using GridSearchCV

decision_tree = DecisionTreeRegressor(random_state=42)


decision_tree_parameters = {
    "max_depth": [10, 15, 20, 25, 30, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10]
}

decision_tree_grid = GridSearchCV(
    estimator=decision_tree,
    param_grid=decision_tree_parameters,
    scoring=scoring,
    cv=5,
    n_jobs=-1,
    refit=False
)

print("Tuning Decision Tree...")


decision_tree_grid.fit(
    X_train_processed,
    y_train
)

#Radom forest fine tuning using GridSearchCV

random_forest = RandomForestRegressor(random_state=42)

#parameter contain n_estimators, max_depth, min_samples_split, and min_samples_leaf for tuning the random forest model
#n_estimators: number of trees in the forest
#max_depth: maximum depth of the tree
#min_samples_split: minimum number of samples required to split an internal node
#min_samples_leaf: minimum number of samples required to be at a leaf node

random_forest_parameters = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, 30],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

random_forest_grid = GridSearchCV(
    estimator=random_forest,
    param_grid=random_forest_parameters,
    scoring=scoring,
    cv=5,
    n_jobs=-1,
    refit=False
)


print("\nTuning Random Forest...")


random_forest_grid.fit(
    X_train_processed,
    y_train
)

#Show best results for each model based on the scoring metrics

def show_best_results(grid, model_name):
    results = grid.cv_results_
    print("\n==============================")
    print(model_name)
    print("==============================")


#Show best results for each scoring metric
    index = results["mean_test_MAE"].argmax()

    print("\nBest MAE:")
    print("Parameters:", results["params"][index])
    print("MAE:", -results["mean_test_MAE"][index])


    index = results["mean_test_RMSE"].argmax()

    print("\nBest RMSE:")
    print("Parameters:", results["params"][index])
    print("RMSE:", -results["mean_test_RMSE"][index])

    index = results["mean_test_MAPE"].argmax()

    print("\nBest MAPE:")
    print("Parameters:", results["params"][index])
    print("MAPE:", -results["mean_test_MAPE"][index])

    index = results["mean_test_R2"].argmax()

    print("\nBest R²:")
    print("Parameters:", results["params"][index])
    print("R²:", results["mean_test_R2"][index])


show_best_results(
    decision_tree_grid,
    "DECISION TREE"
)

show_best_results(
    random_forest_grid,
    "RANDOM FOREST"
)