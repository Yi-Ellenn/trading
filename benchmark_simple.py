import polars as pl
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error

# Load data
train_data = pl.read_parquet("./train.parquet")
test_data = pl.read_parquet("./test.parquet")
sample_submission = pl.read_csv("./sample_submission.csv")

# Prepare features
# feature_cols = [f"X{i}" for i in range(1, 891)]

feature_cols = [
        "X863", "X856", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
        "X415", "X345", "X855", "X174", "X302", "X178", "X168", "X612",
         "X888", "X421", "X333", "X292",
    ]
# Convert to numpy arrays
X_train = train_data[feature_cols].to_numpy()
y_train = train_data["label"].to_numpy()

# Remove NaN values from y_train
valid_mask = ~np.isnan(y_train)
X_train = X_train[valid_mask]
y_train = y_train[valid_mask]


X_test = test_data[feature_cols].to_numpy()

# Create LightGBM datasets
train_dataset = lgb.Dataset(X_train, label=y_train)

# Set parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# Train model
print("Training LightGBM model...")
model = lgb.train(
    params,
    train_dataset,
    num_boost_round=1000
)

# Make predictions
print("Making predictions...")
predictions = model.predict(X_test)
n_predictions = len(X_test)
# Create submission file
submission = pl.DataFrame({
    "ID": range(1, n_predictions + 1),  # Get ID from test_data instead of sample_submission
    "prediction": predictions
})

# Save submission
submission.write_csv("submission_v0.csv")
print("Submission saved to submission_v0.csv")
