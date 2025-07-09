import polars as pl
import lightgbm as lgb
import numpy as np
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from joblib import Parallel, delayed

# Load data
train_data = pl.read_parquet("./train.parquet")
test_data = pl.read_parquet("./test.parquet")
sample_submission = pl.read_csv("./sample_submission.csv")

# Prepare features
import pickle
with open('/Users/puzhengheng/Documents/Course/drw-crypto-market-prediction/selected_features.pkl', 'rb') as f:
    feature_cols = pickle.load(f)

# Convert to numpy arrays
X_train = train_data[feature_cols].to_numpy()
y_train = train_data["label"].to_numpy()

# Remove NaN values from y_train
valid_mask = ~np.isnan(y_train)
X_train = X_train[valid_mask]
y_train = y_train[valid_mask]

# Split training data into train and validation sets
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.8, random_state=42
)

X_test = test_data[feature_cols].to_numpy()

# Create LightGBM datasets
train_dataset = lgb.Dataset(X_train_split, label=y_train_split)
val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)

# Base parameters that won't be tuned
base_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'verbose': -1
}

def ic_score(y_true, y_pred):
    """Calculate Information Coefficient (Correlation Coefficient)"""
    return float(np.corrcoef(y_true, y_pred)[0,1])

def train_fold(train_idx, val_idx, X_train_split, y_train_split, params):
    X_train_cv = X_train_split[train_idx]
    y_train_cv = y_train_split[train_idx]
    X_val_cv = X_train_split[val_idx]
    y_val_cv = y_train_split[val_idx]
    
    train_set = lgb.Dataset(X_train_cv, y_train_cv)
    val_set = lgb.Dataset(X_val_cv, y_val_cv, reference=train_set)
    
    model_cv = lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=1000,
        valid_sets=[val_set],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=0)
        ]
    )
    
    pred_cv = model_cv.predict(X_val_cv)
    score = abs(ic_score(y_val_cv, pred_cv))  # Use absolute IC value
    return score

def objective(trial):
    # Define the hyperparameters to optimize
    params = {
        **base_params,
        'num_leaves': trial.suggest_categorical('num_leaves', [15, 31, 63]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.001, 0.01, 0.1]),
        'max_depth': trial.suggest_categorical('max_depth', [3, 6, 9, 12]),
        'min_gain_to_split': trial.suggest_categorical('min_gain_to_split', [0.001, 0.01, 0.1])
    }

    # Use K-Fold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Parallel execution of cross validation folds
    cv_scores = Parallel(n_jobs=-1)(
        delayed(train_fold)(
            train_idx, val_idx, X_train_split, y_train_split, params
        )
        for train_idx, val_idx in kf.split(X_train_split)
    )

    return float(-np.mean(np.array(cv_scores)))  # Negative because we want to maximize absolute IC


study = optuna.create_study(direction='minimize')

print("Optimizing hyperparameters with Optuna...")
study.optimize(objective, n_trials=80, n_jobs=-1)


best_params = {**base_params, **study.best_params}
best_score = -study.best_value  

print(f"\nBest absolute IC: {best_score:.6f}")
print("Best parameters:", best_params)


print("\nTraining final model with best parameters...")
final_model = lgb.train(
    best_params,
    train_dataset,
    num_boost_round=1000,
    valid_sets=[val_dataset],
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)


print("Making predictions...")
predictions = final_model.predict(X_test)
n_predictions = len(X_test)


submission = pl.DataFrame({
    "ID": range(1, n_predictions + 1),
    "prediction": predictions
})


submission.write_csv("submission_v1.csv")
print("Submission saved to submission_v1.csv")
#  {'num_leaves': 15, 'learning_rate': 0.1, 'max_depth': 9, 'min_gain_to_split': 0.001}
# {'num_leaves': 63, 'learning_rate': 0.1, 'max_depth': 12, 'min_gain_to_split': 0.001}