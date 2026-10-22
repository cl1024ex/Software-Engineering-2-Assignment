from sklearn.model_selection import train_test_split
from preprocessing import X, y, columnTransformer


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)


# Fit preprocessing on training data
X_train_processed = columnTransformer.fit_transform(X_train)

# Apply the same preprocessing to test data
X_test_processed = columnTransformer.transform(X_test)


# Displays results
print("\nTraining data:")
print(X_train.shape)

print("\nTesting data:")
print(X_test.shape)

print("\nProcessed training data:")
print(X_train_processed.shape)

print("\nProcessed testing data:")
print(X_test_processed.shape)