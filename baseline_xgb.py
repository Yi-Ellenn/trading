# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Configuration
class Config:
    # Automatically detect project path
    import os
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    TRAIN_PATH = os.path.join(PROJECT_DIR, "train.parquet")
    TEST_PATH = os.path.join(PROJECT_DIR, "test.parquet") 
    SUBMISSION_PATH = os.path.join(PROJECT_DIR, "sample_submission.csv")

    FEATURES = [
        "X863", "X856", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
        "X415", "X345", "X855", "X174", "X302", "X178", "X168", "X612", "X804", "X785", "X512", 
        "buy_qty", "sell_qty", "volume", "X888", "X421", "X333", "X292",
    ]

    LABEL_COLUMN = "label"
    N_FOLDS = 3
    RANDOM_STATE = 42
    USE_MODEL_SLICES = False  # Enable model slices

# XGBoost parameters
XGB_PARAMS = {
    'tree_method': 'hist',
    'device': 'gpu', 
    'n_jobs': -1,
    'colsample_bytree': 0.4111224922845363,
    'colsample_bynode': 0.28869302181383194,
    'gamma': 1.4665430311056709,
    'learning_rate': 0.014053505540364681,
    'max_depth': 7,
    'max_leaves': 40,
    'n_estimators': 500,
    'reg_alpha': 27.791606770656145,
    'reg_lambda': 84.90603428439086,
    'subsample': 0.06567,
    'verbosity': 0,
    'random_state': Config.RANDOM_STATE
}

LEARNERS = [
    {"name": "xgb", "Estimator": XGBRegressor, "params": XGB_PARAMS},
]

def feature_engineering(df):
    df['imbalance'] = df['ask_qty'] - df['bid_qty']
    df['imbalance_ratio'] = df['imbalance'] / (df['ask_qty'] + df['bid_qty'] + 1e-8)
    df['volume_weighted_sell'] = df['sell_qty'] * df['volume']
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-8)
    df['selling_pressure'] = df['sell_qty'] / (df['volume'] + 1e-8)
    df['effective_spread_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-8)
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    Config.FEATURES += ["imbalance", "imbalance_ratio", "volume_weighted_sell", "buy_sell_ratio", "selling_pressure", "effective_spread_proxy"]
    return df

def create_time_decay_weights(n: int, decay: float = 0.9, reverse: bool = False) -> np.ndarray:
    positions = np.arange(n)
    if reverse:
        normalized = 1.0 - (positions / (n - 1))
    else:
        normalized = positions / (n - 1)
    weights = decay ** (1.0 - normalized)
    return weights * n / weights.sum()

def load_data():
    train_df = pd.read_parquet(Config.TRAIN_PATH, columns=Config.FEATURES + [Config.LABEL_COLUMN])
    test_df = pd.read_parquet(Config.TEST_PATH, columns=Config.FEATURES)
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)

    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)
    
    print(f"Loaded data - Train: {train_df.shape}, Test: {test_df.shape}, Submission: {submission_df.shape}")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), submission_df

Config.FEATURES += ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"]
# Config.FEATURES += ["imbalance", "imbalance_ratio", "volume_weighted_sell", "buy_sell_ratio", "selling_pressure", "effective_spread_proxy"]
Config.FEATURES = list(set(Config.FEATURES))

EARLY_PERCENTAGE = 0.35

def get_model_slices(n_samples: int):
    return [
        {"name": "full_data", "type": "full", "cutoff": 0},
        {"name": "last_75pct", "type": "recent", "cutoff": int(0.25 * n_samples)},
        {"name": "last_50pct", "type": "recent", "cutoff": int(0.50 * n_samples)},
        {"name": f"first_{int(EARLY_PERCENTAGE*100)}pct", "type": "early", "cutoff": int(EARLY_PERCENTAGE * n_samples)},
    ]

def train_xgboost(train_df, test_df):
    n_samples = len(train_df)
    
    if Config.USE_MODEL_SLICES:
        model_slices = get_model_slices(n_samples)
    else:
        model_slices = [{"name": "full_data", "type": "full", "cutoff": 0}]

    # Initialize predictions dictionary for each learner
    oof_preds = {}
    test_preds = {}
    for learner in LEARNERS:
        oof_preds[learner["name"]] = {}
        test_preds[learner["name"]] = {}
        for s in model_slices:
            oof_preds[learner["name"]][s["name"]] = np.zeros(n_samples)
            test_preds[learner["name"]][s["name"]] = np.zeros(len(test_df))

    full_weights = create_time_decay_weights(n_samples)
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=False)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df), start=1):
        print(f"\n--- Fold {fold}/{Config.N_FOLDS} ---")
        X_valid = train_df.iloc[valid_idx][Config.FEATURES]
        y_valid = train_df.iloc[valid_idx][Config.LABEL_COLUMN]

        for s in model_slices:
            cutoff = s["cutoff"]
            slice_name = s["name"]
            slice_type = s["type"]
            
            if slice_type == "full":
                subset = train_df.reset_index(drop=True)
                rel_idx = train_idx
                sw = full_weights[train_idx]
            elif slice_type == "recent":
                subset = train_df.iloc[cutoff:].reset_index(drop=True)
                rel_idx = train_idx[train_idx >= cutoff] - cutoff
                if cutoff > 0:
                    sw = create_time_decay_weights(len(subset))[rel_idx]
                else:
                    sw = full_weights[train_idx]
            elif slice_type == "early":
                subset = train_df.iloc[:cutoff].reset_index(drop=True)
                rel_idx = train_idx[train_idx < cutoff]
                if len(rel_idx) > 0:
                    sw = create_time_decay_weights(len(subset))[rel_idx]
                else:
                    sw = np.array([])

            if len(rel_idx) == 0:
                print(f"  Skipping slice: {slice_name} (no training data in fold)")
                continue

            X_train = subset.iloc[rel_idx][Config.FEATURES]
            y_train = subset.iloc[rel_idx][Config.LABEL_COLUMN]
            
            X_train_np = X_train.values
            y_train_np = y_train.values
            X_valid_np = X_valid.values
            y_valid_np = y_valid.values
            
            print(f"  Training slice: {slice_name}, samples: {len(X_train)}")

            for learner in LEARNERS:
                model = learner["Estimator"](**learner["params"])
                model.fit(X_train_np, y_train_np, sample_weight=sw,
                         eval_set=[(X_valid_np, y_valid_np)], verbose=False)

                if slice_type == "early":
                    mask = valid_idx < cutoff
                    if mask.any():
                        idxs = valid_idx[mask]
                        oof_preds[learner["name"]][slice_name][idxs] = model.predict(train_df.iloc[idxs][Config.FEATURES].values)
                    if (~mask).any():
                        oof_preds[learner["name"]][slice_name][valid_idx[~mask]] = oof_preds[learner["name"]]["full_data"][valid_idx[~mask]]
                else:
                    mask = valid_idx >= cutoff if slice_type == "recent" else np.ones(len(valid_idx), dtype=bool)
                    if mask.any():
                        idxs = valid_idx[mask]
                        oof_preds[learner["name"]][slice_name][idxs] = model.predict(train_df.iloc[idxs][Config.FEATURES].values)
                    if slice_type == "recent" and cutoff > 0 and (~mask).any():
                        oof_preds[learner["name"]][slice_name][valid_idx[~mask]] = oof_preds[learner["name"]]["full_data"][valid_idx[~mask]]

                test_preds[learner["name"]][slice_name] += model.predict(test_df[Config.FEATURES].values) / Config.N_FOLDS

    # Evaluate and find best slice
    y_true = train_df[Config.LABEL_COLUMN].values
    xgb_scores = {}
    for slice_name in oof_preds["xgb"]:
        score = pearsonr(y_true, oof_preds["xgb"][slice_name])[0]
        xgb_scores[slice_name] = score
        print(f"\nOOF Score ({slice_name}): {score:.4f}")
    
    best_slice = max(xgb_scores.items(), key=lambda x: x[1])[0]
    print(f"\nBest slice: {best_slice} ({xgb_scores[best_slice]:.4f})")
    
    # Save submission for best slice
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    submission_df["prediction"] = test_preds["xgb"][best_slice]
    submission_df.to_csv(f"submission_xgb_{best_slice}.csv", index=False)
    print(f"Saved submission_xgb_{best_slice}.csv")
    
    return oof_preds, test_preds

def main():
    print("\n=== Training XGBoost Baseline ===")
    train_df, test_df, _ = load_data()
    oof_predictions, test_predictions = train_xgboost(train_df, test_df)
    print("\n✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main()
