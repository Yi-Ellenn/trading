# Imports
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from rich.progress import Progress
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

    with open(os.path.join(PROJECT_DIR, "selected_features.pkl"), 'rb') as f:
        FEATURES = pickle.load(f)
    print(f"Loaded {len(FEATURES)} features")

    LABEL_COLUMN = "label"
    N_FOLDS = 3
    RANDOM_STATE = 42
    USE_MODEL_SLICES = False  # Enable model slices
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # MLP specific parameters
    BATCH_SIZE = 512
    LEARNING_RATE = 0.001
    EPOCHS = 100
    PATIENCE = 15

# MLP Model
class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout=0.3):
        super(MLPRegressor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
    
    def forward(self, x):
        return self.network(x)

# MLP parameters
MLP_PARAMS = {
    'hidden_dim': 512,
    'dropout': 0.2,
    'learning_rate': Config.LEARNING_RATE,
    'batch_size': Config.BATCH_SIZE,
    'epochs': Config.EPOCHS,
    'patience': Config.PATIENCE
}

LEARNERS = [
    {"name": "mlp", "Estimator": MLPRegressor, "params": MLP_PARAMS},
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

def ic_loss(y_pred, y_true):
    """
    Information Coefficient (IC) loss function for MLP training.
    IC is the correlation between predictions and true values.
    We minimize negative IC to maximize correlation.
    """
    # Flatten tensors to ensure they're 1D
    y_pred = y_pred.squeeze()
    y_true = y_true.squeeze()
    
    # Calculate means
    mean_pred = torch.mean(y_pred)
    mean_true = torch.mean(y_true)
    
    # Calculate centered values
    pred_centered = y_pred - mean_pred
    true_centered = y_true - mean_true
    
    # Calculate correlation (Pearson correlation coefficient)
    numerator = torch.sum(pred_centered * true_centered)
    
    # Calculate standard deviations
    pred_std = torch.sqrt(torch.sum(pred_centered ** 2))
    true_std = torch.sqrt(torch.sum(true_centered ** 2))
    
    # Avoid division by zero
    denominator = pred_std * true_std + 1e-8
    
    # Calculate correlation
    correlation = numerator / denominator
    
    # Return negative correlation as loss (we want to maximize correlation)
    return 1-correlation


class ICLoss(nn.Module):
    """IC Loss as a PyTorch module"""
    def __init__(self):
        super(ICLoss, self).__init__()
    
    def forward(self, y_pred, y_true):
        return ic_loss(y_pred, y_true)


def filter_features(feature_names):
    high_missing_features = [
        'X697', 'X708', 'X716', 'X715', 'X714', 'X713', 'X712', 'X711', 
        'X710', 'X709', 'X707', 'X698', 'X706', 'X705', 'X704', 'X703', 
        'X702', 'X701', 'X700', 'X699', 'X717'
    ]
    
    high_outlier_features = [
        'X62', 'X104', 'X146', 'X152', 'X110', 'X68', 'X393', 'X309', 
        'X351', 'X116'
    ]

    extreme_skew_features = [
        'X648', 'X874', 'X873', 'X636', 'X624', 'X618', 'X645', 'X231', 
        'X597', 'X630'
    ]

    # Filter out problematic features
    filtered_features = []
    features_to_remove = set(high_missing_features + high_outlier_features + extreme_skew_features)
    
    for feature in feature_names:
        if feature not in features_to_remove:
            filtered_features.append(feature)
    
    print(f"Filtered out {len(feature_names) - len(filtered_features)} problematic features")
    print(f"Remaining features: {len(filtered_features)}")
    
    return filtered_features

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

def train_mlp_model(X_train, y_train, X_val, y_val, model_params):
    """Train a single MLP model"""
    model = MLPRegressor(
        input_dim=X_train.shape[1],
        hidden_dim=model_params['hidden_dim'],
        dropout=model_params['dropout']
    ).to(Config.DEVICE)
    
    criterion = ICLoss()
    optimizer = optim.Adam(model.parameters(), lr=model_params['learning_rate'])
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(Config.DEVICE)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(Config.DEVICE)
    X_val_tensor = torch.FloatTensor(X_val).to(Config.DEVICE)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(Config.DEVICE)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TextColumn("Loss: {task.fields[val_loss]:.4f}"),
        refresh_per_second=1
    ) as progress:
        task = progress.add_task("Training MLP", total=model_params['epochs'], val_loss=0.0)
        
        for epoch in range(model_params['epochs']):
            # Training
            model.train()
            optimizer.zero_grad()
            pred = model(X_train_tensor)
            loss = criterion(pred, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_tensor)
                val_loss = criterion(val_pred, y_val_tensor).item()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                
            # Update progress bar
            progress.update(task, advance=1, val_loss=val_loss)
                
            if patience_counter >= model_params['patience']:
                progress.update(task, completed=model_params['epochs'])
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model

def train_mlp(train_df, test_df):
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

    Config.FEATURES = filter_features(Config.FEATURES)

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
            
            # Scale features for MLP
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train.values)
            X_valid_scaled = scaler.transform(X_valid.values)
            
            print(f"  Training slice: {slice_name}, samples: {len(X_train)}")

            for learner in LEARNERS:
                model = train_mlp_model(
                    X_train_scaled, y_train.values, 
                    X_valid_scaled, y_valid.values,
                    learner["params"]
                )

                # Make predictions
                model.eval()
                with torch.no_grad():
                    if slice_type == "early":
                        mask = valid_idx < cutoff
                        if mask.any():
                            idxs = valid_idx[mask]
                            X_pred_scaled = scaler.transform(train_df.iloc[idxs][Config.FEATURES].values)
                            X_pred_tensor = torch.FloatTensor(X_pred_scaled).to(Config.DEVICE)
                            pred = model(X_pred_tensor).cpu().numpy().flatten()
                            oof_preds[learner["name"]][slice_name][idxs] = pred
                        if (~mask).any():
                            oof_preds[learner["name"]][slice_name][valid_idx[~mask]] = oof_preds[learner["name"]]["full_data"][valid_idx[~mask]]
                    else:
                        mask = valid_idx >= cutoff if slice_type == "recent" else np.ones(len(valid_idx), dtype=bool)
                        if mask.any():
                            idxs = valid_idx[mask]
                            X_pred_scaled = scaler.transform(train_df.iloc[idxs][Config.FEATURES].values)
                            X_pred_tensor = torch.FloatTensor(X_pred_scaled).to(Config.DEVICE)
                            pred = model(X_pred_tensor).cpu().numpy().flatten()
                            oof_preds[learner["name"]][slice_name][idxs] = pred
                        if slice_type == "recent" and cutoff > 0 and (~mask).any():
                            oof_preds[learner["name"]][slice_name][valid_idx[~mask]] = oof_preds[learner["name"]]["full_data"][valid_idx[~mask]]

                    # Test predictions
                    X_test_scaled = scaler.transform(test_df[Config.FEATURES].values)
                    X_test_tensor = torch.FloatTensor(X_test_scaled).to(Config.DEVICE)
                    test_pred = model(X_test_tensor).cpu().numpy().flatten()
                    test_preds[learner["name"]][slice_name] += test_pred / Config.N_FOLDS

    # Evaluate and find best slice
    y_true = train_df[Config.LABEL_COLUMN].values
    mlp_scores = {}
    for slice_name in oof_preds["mlp"]:
        score = pearsonr(y_true, oof_preds["mlp"][slice_name])[0]
        mlp_scores[slice_name] = score
        print(f"\nOOF Score ({slice_name}): {score:.4f}")
    
    best_slice = max(mlp_scores.items(), key=lambda x: x[1])[0]
    print(f"\nBest slice: {best_slice} ({mlp_scores[best_slice]:.4f})")
    
    # Save submission for best slice
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    submission_df["prediction"] = test_preds["mlp"][best_slice]
    submission_df.to_csv(f"submission_mlp_{best_slice}.csv", index=False)
    print(f"Saved submission_mlp_{best_slice}.csv")
    
    return oof_preds, test_preds

def main():
    print("\n=== Training MLP Baseline ===")
    
    # Set random seeds
    np.random.seed(Config.RANDOM_STATE)
    torch.manual_seed(Config.RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.RANDOM_STATE)
    
    train_df, test_df, _ = load_data()
    oof_predictions, test_predictions = train_mlp(train_df, test_df)
    print("\n✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main()
