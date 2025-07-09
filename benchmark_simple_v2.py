import polars as pl
import lightgbm as lgb
import numpy as np
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split, KFold
from joblib import Parallel, delayed
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load data
logger.info("Loading data...")
train_data = pl.read_parquet("./train.parquet")
test_data = pl.read_parquet("./test.parquet")
sample_submission = pl.read_csv("./sample_submission.csv")

# Prepare features
logger.info("Preparing features...")
import pickle

feature_cols = [
        "X863", "X856", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
        "X415", "X345", "X855", "X174", "X302", "X178", "X168", "X612",
        "buy_qty", "sell_qty", "volume", "X888", "X421", "X333", "X292",
    ]

# Convert to numpy arrays
X_train = train_data[feature_cols].to_numpy()
y_train = train_data["label"].to_numpy()

# Remove NaN values from y_train
logger.info("Preprocessing data...")
valid_mask = ~np.isnan(y_train)
X_train = X_train[valid_mask]
y_train = y_train[valid_mask]

# Split training data into train and validation sets
# Use the last 20% of data for validation
split_idx = int(len(X_train) * 0.8)
X_train_split = X_train[:split_idx]
X_val = X_train[split_idx:]
y_train_split = y_train[:split_idx]
y_val = y_train[split_idx:]

X_test = test_data[feature_cols].to_numpy()

logger.info(f"Data shapes - Train: {X_train_split.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Create LightGBM datasets
train_dataset = lgb.Dataset(X_train_split, label=y_train_split)
val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)

# Base parameters that won't be tuned
base_params = {
    'objective': 'regression',
    'metric': None,  # Remove default metric
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'verbose': -1,
    'feature_fraction': 0.8,  
    'bagging_fraction': 0.8,  
    'bagging_freq': 5,       
}

def ic_score(y_true, y_pred):
    """Calculate Information Coefficient (Correlation Coefficient)"""
    return float(np.corrcoef(y_true, y_pred)[0,1])

# Custom evaluation function for LightGBM
def ic_eval(preds, train_data):
    labels = train_data.get_label()
    score = abs(ic_score(labels, preds))
    return 'ic', score, True

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
        num_boost_round=500,
        valid_sets=[val_set],
        feval=ic_eval,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50)
        ]
    )
    
    pred_cv = model_cv.predict(X_val_cv)
    score = abs(ic_score(y_val_cv, pred_cv))  # Use absolute IC value
    return score

def objective(trial):
    params = {
        **base_params,
        'num_leaves': trial.suggest_categorical('num_leaves', [31, 63, 127]),
        'max_depth': trial.suggest_categorical('max_depth', [2, 3, 4]),
        'min_data_in_leaf': trial.suggest_categorical('min_data_in_leaf', [10, 20, 50, 100]), 
        'lambda_l1': trial.suggest_categorical('lambda_l1', [0.01, 0.1, 0.5, 1.0, 2.0]), 
        'lambda_l2': trial.suggest_categorical('lambda_l2', [0.01, 0.1, 0.5, 1.0, 2.0]), 
    }

    logger.info(f"Trial {trial.number} - Testing parameters: {params}")

    # 使用时间序列分割而不是随机K-Fold
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Parallel execution of cross validation folds
    cv_scores = Parallel(n_jobs=-1)(
        delayed(train_fold)(
            train_idx, val_idx, X_train_split, y_train_split, params
        )
        for train_idx, val_idx in tscv.split(X_train_split)
    )

    mean_score = float(-np.mean(np.array(cv_scores)))
    logger.info(f"Trial {trial.number} - Mean CV Score: {-mean_score:.6f}")
    return mean_score  # Negative because we want to maximize absolute IC

if __name__ == "__main__":
    logger.info("Starting hyperparameter optimization...")
    study = optuna.create_study(direction='minimize')

    # Run optimization
    logger.info("Optimizing hyperparameters with Optuna...")
    study.optimize(objective, n_trials=100, n_jobs=-1)

    # Get the best parameters
    best_params = {**base_params, **study.best_params}
    best_score = -study.best_value  # Convert back to positive absolute IC

    logger.info(f"\nBest absolute IC: {best_score:.6f}")
    logger.info(f"Best parameters: {best_params}")

    # Train final model with best parameters
    logger.info("\nTraining final model with best parameters...")
    final_model = lgb.train(
        best_params,
        train_dataset,
        num_boost_round=500,
        valid_sets=[val_dataset],
        feval=ic_eval,
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )

    # Make predictions
    logger.info("Making predictions...")
    predictions = final_model.predict(X_test)
    n_predictions = len(X_test)

    # Create submission file
    submission = pl.DataFrame({
        "ID": range(1, n_predictions + 1),
        "prediction": predictions
    })

    # Save submission
    submission.write_csv("submission_v2.csv")
    logger.info("Submission saved to submission_v2.csv")
