# Imports
import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from scipy.stats import pearsonr
from rich.progress import Progress
warnings.filterwarnings('ignore')

# Configuration
class Config:
    # Automatically detect project path
    import os
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    TRAIN_PATH = os.path.join(PROJECT_DIR, "train.parquet")
    TEST_PATH = os.path.join(PROJECT_DIR, "test.parquet") 
    SUBMISSION_PATH = os.path.join(PROJECT_DIR, "sample_submission.csv")

    with open(os.path.join(PROJECT_DIR, "selected_features.pkl"), 'rb') as f:
        FEATURES = pickle.load(f)
    print(f"Loaded {len(FEATURES)} features")

    LABEL_COLUMN = "label"
    N_FOLDS = 3
    RANDOM_STATE = 42
    USE_MODEL_SLICES = False  # Enable model slices
    
    # Linear regression specific parameters
    PCA_COMPONENTS = 100
    TEST_SIZE = 0.2

# Linear regression models
LINEAR_MODELS = [
    {"name": "ridge", "model": RidgeCV(alphas=[0.01, 0.1, 1, 10], cv=5)},
    {"name": "lasso", "model": LassoCV(alphas=[0.01, 0.1, 1], cv=5, max_iter=10000)},
    {"name": "elastic", "model": ElasticNetCV(l1_ratio=0.5, alphas=[0.01, 0.1, 1], cv=5)},
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

def load_data():
    train_df = pd.read_parquet(Config.TRAIN_PATH, columns=Config.FEATURES + [Config.LABEL_COLUMN])
    test_df = pd.read_parquet(Config.TEST_PATH, columns=Config.FEATURES)
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)

    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)
    
    print(f"Loaded data - Train: {train_df.shape}, Test: {test_df.shape}, Submission: {submission_df.shape}")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), submission_df

Config.FEATURES += ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"]
Config.FEATURES = list(set(Config.FEATURES))

EARLY_PERCENTAGE = 0.35

def get_model_slices(n_samples: int):
    return [
        {"name": "full_data", "type": "full", "cutoff": 0},
        {"name": "last_75pct", "type": "recent", "cutoff": int(0.25 * n_samples)},
        {"name": "last_50pct", "type": "recent", "cutoff": int(0.50 * n_samples)},
        {"name": f"first_{int(EARLY_PERCENTAGE*100)}pct", "type": "early", "cutoff": int(EARLY_PERCENTAGE * n_samples)},
    ]

def train_linear_model(X_train, y_train, X_val, y_val, model_config):
    """Train a linear regression model with PCA preprocessing"""
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Apply PCA
    pca = PCA(n_components=Config.PCA_COMPONENTS, random_state=Config.RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    
    # Train model
    model = model_config["model"]
    model.fit(X_train_pca, y_train)
    
    # Validate
    val_pred = model.predict(X_val_pca)
    val_score = pearsonr(y_val, val_pred)[0]
    
    print(f"  {model_config['name']} - Best alpha: {getattr(model, 'alpha_', 'N/A')}, Validation IC: {val_score:.4f}")
    
    return model, scaler, pca, val_score

def train_linear_regression(train_df, test_df):
    """Main training function for linear regression models"""
    # Initialize result storage
    oof_preds = {model["name"]: {} for model in LINEAR_MODELS}
    test_preds = {model["name"]: {} for model in LINEAR_MODELS}
    
    # Get model slices if enabled
    if Config.USE_MODEL_SLICES:
        model_slices = get_model_slices(len(train_df))
    else:
        model_slices = [{"name": "full_data", "type": "full", "cutoff": 0}]
    
    # Initialize prediction arrays
    for model in LINEAR_MODELS:
        for slice_config in model_slices:
            slice_name = slice_config["name"]
            oof_preds[model["name"]][slice_name] = np.zeros(len(train_df))
            test_preds[model["name"]][slice_name] = np.zeros(len(test_df))
    
    # Cross-validation
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)
    
    with Progress() as progress:
        task = progress.add_task("Training Linear Models...", total=len(LINEAR_MODELS) * len(model_slices) * Config.N_FOLDS)
        
        for model_config in LINEAR_MODELS:
            print(f"\nTraining {model_config['name']}...")
            
            for slice_config in model_slices:
                slice_name = slice_config["name"]
                slice_type = slice_config["type"]
                cutoff = slice_config["cutoff"]
                
                print(f"  Slice: {slice_name}")
                
                for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df)):
                    progress.update(task, advance=1)
                    
                    # Prepare data based on slice type
                    if slice_type == "full":
                        train_data = train_df.iloc[train_idx]
                        X_train = train_data[Config.FEATURES].values
                        y_train = train_data[Config.LABEL_COLUMN].values
                    elif slice_type == "recent":
                        mask = train_idx >= cutoff
                        if mask.any():
                            recent_idx = train_idx[mask]
                            train_data = train_df.iloc[recent_idx]
                            X_train = train_data[Config.FEATURES].values
                            y_train = train_data[Config.LABEL_COLUMN].values
                        else:
                            continue
                    elif slice_type == "early":
                        mask = train_idx < cutoff
                        if mask.any():
                            early_idx = train_idx[mask]
                            train_data = train_df.iloc[early_idx]
                            X_train = train_data[Config.FEATURES].values
                            y_train = train_data[Config.LABEL_COLUMN].values
                        else:
                            continue
                    
                    # Remove NaN values
                    valid_mask = ~np.isnan(y_train)
                    X_train = X_train[valid_mask]
                    y_train = y_train[valid_mask]
                    
                    if len(y_train) == 0:
                        continue
                    
                    # Validation data
                    X_valid = train_df.iloc[valid_idx][Config.FEATURES].values
                    y_valid = train_df.iloc[valid_idx][Config.LABEL_COLUMN].values
                    
                    # Train model
                    model, scaler, pca, val_score = train_linear_model(
                        X_train, y_train, X_valid, y_valid, model_config
                    )
                    
                    # OOF predictions
                    if slice_type == "early":
                        mask = valid_idx < cutoff
                        if mask.any():
                            idxs = valid_idx[mask]
                            X_pred_scaled = scaler.transform(train_df.iloc[idxs][Config.FEATURES].values)
                            X_pred_pca = pca.transform(X_pred_scaled)
                            pred = model.predict(X_pred_pca)
                            oof_preds[model_config["name"]][slice_name][idxs] = pred
                        if (~mask).any():
                            oof_preds[model_config["name"]][slice_name][valid_idx[~mask]] = oof_preds[model_config["name"]]["full_data"][valid_idx[~mask]]
                    else:
                        mask = valid_idx >= cutoff if slice_type == "recent" else np.ones(len(valid_idx), dtype=bool)
                        if mask.any():
                            idxs = valid_idx[mask]
                            X_pred_scaled = scaler.transform(train_df.iloc[idxs][Config.FEATURES].values)
                            X_pred_pca = pca.transform(X_pred_scaled)
                            pred = model.predict(X_pred_pca)
                            oof_preds[model_config["name"]][slice_name][idxs] = pred
                        if slice_type == "recent" and cutoff > 0 and (~mask).any():
                            oof_preds[model_config["name"]][slice_name][valid_idx[~mask]] = oof_preds[model_config["name"]]["full_data"][valid_idx[~mask]]
                    
                    # Test predictions
                    X_test_scaled = scaler.transform(test_df[Config.FEATURES].values)
                    X_test_pca = pca.transform(X_test_scaled)
                    test_pred = model.predict(X_test_pca)
                    test_preds[model_config["name"]][slice_name] += test_pred / Config.N_FOLDS
    
    # Evaluate and find best model/slice combination
    y_true = train_df[Config.LABEL_COLUMN].values
    best_score = -1
    best_model = None
    best_slice = None
    
    for model_name in oof_preds:
        print(f"\n{model_name.upper()} Results:")
        for slice_name in oof_preds[model_name]:
            score = pearsonr(y_true, oof_preds[model_name][slice_name])[0]
            print(f"  OOF Score ({slice_name}): {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_model = model_name
                best_slice = slice_name
    
    print(f"\nBest combination: {best_model} - {best_slice} ({best_score:.4f})")
    
    # Save submission for best model/slice
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    submission_df["prediction"] = test_preds[best_model][best_slice]
    submission_df.to_csv(f"submission_lr_{best_model}_{best_slice}.csv", index=False)
    print(f"Saved submission_lr_{best_model}_{best_slice}.csv")
    
    return oof_preds, test_preds

def main():
    print("\n=== Training Linear Regression Baseline ===")
    
    # Set random seeds
    np.random.seed(Config.RANDOM_STATE)
    
    train_df, test_df, _ = load_data()
    oof_predictions, test_predictions = train_linear_regression(train_df, test_df)
    print("\n✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main()
