
EARLY_PERCENTAGE = 0.35  # Change this to 0.20, 0.25, 0.30, 0.35, 0.40, or 0.45

# Imports
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from xgboost import XGBRegressor
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set device and seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ====================================
# Original Feature Engineering (Keep Exactly)
# ====================================
def feature_engineering(df):
    df['imbalance'] = df['ask_qty'] - df['bid_qty']
    df['imbalance_ratio'] = df['imbalance'] / (df['ask_qty'] + df['bid_qty'] + 1e-8)
    df['volume_weighted_sell'] = df['sell_qty'] * df['volume']
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-8)
    df['selling_pressure'] = df['sell_qty'] / (df['volume'] + 1e-8)
    df['effective_spread_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-8)
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df 

# ====================================
# Configuration (Keep Original)
# ====================================
class Config:
    # Automatically detect project path
    import os
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    TRAIN_PATH = os.path.join(PROJECT_DIR, "train.parquet")
    TEST_PATH = os.path.join(PROJECT_DIR, "test.parquet")
    SUBMISSION_PATH = os.path.join(PROJECT_DIR, "sample_submission.csv")

    FEATURES = [
        "X863", "X856", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
        "X415", "X345", "X855", "X174", "X302", "X178", "X168", "X612",
        "buy_qty", "sell_qty", "volume", "X888", "X421", "X333", "X292",
    ]

    LABEL_COLUMN = "label"
    N_FOLDS = 3
    RANDOM_STATE = 42

# Original XGBoost parameters (DO NOT CHANGE)
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

# ====================================
# Neural Network Models
# ====================================

# 1. MLP with Heavy Regularization
class RegularizedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.6, noise_factor=0.1):
        super().__init__()
        self.noise_factor = noise_factor
        
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        # Add noise during training
        if self.training and self.noise_factor > 0:
            noise = torch.randn_like(x) * self.noise_factor
            x = x + noise
        
        return self.network(x)

# 2. SAINT (Self-Attention and Intersample Attention Transformer)
class SAINT(nn.Module):
    def __init__(self, input_dim, n_heads=8, n_layers=3, d_model=128, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            ) for _ in range(n_layers)
        ])
        
        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Project input
        x = self.input_projection(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :1, :]
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Global pooling
        x = x.mean(dim=1)
        
        # Output
        return self.output_head(x)

# 3. GANDALF (Simplified version)
class GANDALF(nn.Module):
    def __init__(self, input_dim, n_estimators=20, tree_dim=64, depth=3):
        super().__init__()
        self.n_estimators = n_estimators
        
        # Feature transformation
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, tree_dim),
            nn.BatchNorm1d(tree_dim)
        )
        
        # Soft decision trees
        self.trees = nn.ModuleList([
            nn.Sequential(
                nn.Linear(tree_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            ) for _ in range(n_estimators)
        ])
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_estimators),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # Transform features
        tree_input = self.feature_transform(x)
        
        # Get tree outputs
        tree_outputs = []
        for tree in self.trees:
            output = tree(tree_input)
            tree_outputs.append(output)
        
        tree_outputs = torch.cat(tree_outputs, dim=1)
        
        # Get gates
        gates = self.gate(x)
        
        # Weighted combination
        output = (tree_outputs * gates).sum(dim=1, keepdim=True)
        
        return output

# 4. Additive Model
class AdditiveModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        
        # Shape function for each feature
        self.shape_functions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(input_dim)
        ])
        
        # Global bias
        self.bias = nn.Parameter(torch.zeros(1))
        
        # Feature importance weights
        self.feature_weights = nn.Parameter(torch.ones(input_dim))
        
    def forward(self, x):
        outputs = []
        
        for i in range(x.shape[1]):
            feature = x[:, i:i+1]
            shape_output = self.shape_functions[i](feature)
            weighted_output = shape_output * torch.sigmoid(self.feature_weights[i])
            outputs.append(weighted_output)
        
        # Sum all shape functions
        output = sum(outputs) + self.bias
        
        return output

# ====================================
# Data Loading (Keep Original)
# ====================================
def create_time_decay_weights(n: int, decay: float = 0.9, reverse: bool = False) -> np.ndarray:
    """Create time decay weights. If reverse=True, older data gets higher weight."""
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
Config.FEATURES = list(set(Config.FEATURES))

# ====================================
# Original XGBoost Training (Keep Exactly)
# ====================================
def get_model_slices(n_samples: int):
    return [
        {"name": "full_data", "type": "full", "cutoff": 0},
        {"name": "last_75pct", "type": "recent", "cutoff": int(0.25 * n_samples)},
        {"name": "last_50pct", "type": "recent", "cutoff": int(0.50 * n_samples)},
        {"name": f"first_{int(EARLY_PERCENTAGE*100)}pct", "type": "early", "cutoff": int(EARLY_PERCENTAGE * n_samples)},
    ]

def train_xgboost(train_df, test_df):
    """Original XGBoost training function - DO NOT MODIFY"""
    n_samples = len(train_df)
    model_slices = get_model_slices(n_samples)

    oof_preds = {
        learner["name"]: {s["name"]: np.zeros(n_samples) for s in model_slices}
        for learner in LEARNERS
    }
    test_preds = {
        learner["name"]: {s["name"]: np.zeros(len(test_df)) for s in model_slices}
        for learner in LEARNERS
    }

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

                test_preds[learner["name"]][slice_name] += model.predict(test_df[Config.FEATURES].values)

    for learner_name in test_preds:
        for slice_name in test_preds[learner_name]:
            test_preds[learner_name][slice_name] /= Config.N_FOLDS

    return oof_preds, test_preds, model_slices

# ====================================
# Neural Network Training
# ====================================
def train_neural_model(model, train_loader, val_loader, epochs=30, lr=0.001, patience=5):
    """Generic training function for neural models"""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    criterion = nn.HuberLoss(delta=1.0)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                val_preds.extend(outputs.cpu().numpy().flatten())
                val_targets.extend(batch_y.cpu().numpy().flatten())
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model

def train_neural_models(train_df, test_df):
    """Train all neural network models"""
    print("\n=== Training Neural Network Models ===")
    
    X_train = train_df[Config.FEATURES].values
    y_train = train_df[Config.LABEL_COLUMN].values
    X_test = test_df[Config.FEATURES].values
    
    # Models to train
    models = {
        'mlp': RegularizedMLP(len(Config.FEATURES), hidden_dims=[256, 128, 64], dropout=0.6, noise_factor=0.1),
        'saint': SAINT(len(Config.FEATURES), n_heads=8, n_layers=3, d_model=128, dropout=0.3),
        'gandalf': GANDALF(len(Config.FEATURES), n_estimators=20, tree_dim=64),
        'additive': AdditiveModel(len(Config.FEATURES), hidden_dim=64)
    }
    
    # Store predictions
    nn_oof_preds = {name: np.zeros(len(train_df)) for name in models}
    nn_test_preds = {name: np.zeros(len(test_df)) for name in models}
    
    # K-fold training
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=False)
    
    for model_name, model_class in models.items():
        print(f"\nTraining {model_name.upper()}...")
        
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train), 1):
            print(f"  Fold {fold}/{Config.N_FOLDS}")
            
            # Prepare data
            X_tr, X_val = X_train[train_idx], X_train[valid_idx]
            y_tr, y_val = y_train[train_idx], y_train[valid_idx]
            
            # Scale data
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            # Create datasets
            train_dataset = TensorDataset(
                torch.tensor(X_tr_scaled, dtype=torch.float32),
                torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
            )
            val_dataset = TensorDataset(
                torch.tensor(X_val_scaled, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
            )
            
            # Adjust batch size based on model
            batch_size = 256 if model_name in ['mlp', 'additive'] else 128
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False)
            
            # Create new model instance
            if model_name == 'mlp':
                model = RegularizedMLP(len(Config.FEATURES), hidden_dims=[256, 128, 64], dropout=0.6, noise_factor=0.1)
            elif model_name == 'saint':
                model = SAINT(len(Config.FEATURES), n_heads=8, n_layers=3, d_model=128, dropout=0.3)
            elif model_name == 'gandalf':
                model = GANDALF(len(Config.FEATURES), n_estimators=20, tree_dim=64)
            else:  # additive
                model = AdditiveModel(len(Config.FEATURES), hidden_dim=64)
            
            model = model.to(device)
            
            # Train model
            model = train_neural_model(model, train_loader, val_loader, epochs=30, lr=0.001)
            
            # Make predictions
            model.eval()
            with torch.no_grad():
                # Validation predictions
                val_preds = []
                for batch_x, _ in val_loader:
                    batch_x = batch_x.to(device)
                    outputs = model(batch_x)
                    val_preds.extend(outputs.cpu().numpy().flatten())
                
                nn_oof_preds[model_name][valid_idx] = np.array(val_preds)
                
                # Test predictions
                test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32))
                test_loader = DataLoader(test_dataset, batch_size=batch_size*2, shuffle=False)
                
                test_preds = []
                for (batch_x,) in test_loader:
                    batch_x = batch_x.to(device)
                    outputs = model(batch_x)
                    test_preds.extend(outputs.cpu().numpy().flatten())
                
                nn_test_preds[model_name] += np.array(test_preds) / Config.N_FOLDS
        
        # Report OOF score
        oof_score = pearsonr(y_train, nn_oof_preds[model_name])[0]
        print(f"  {model_name.upper()} OOF Score: {oof_score:.4f}")
    
    return nn_oof_preds, nn_test_preds

# ====================================
# Final Ensemble
# ====================================
def create_final_ensemble(train_df, xgb_oof, xgb_test, nn_oof, nn_test, submission_df):
    """Create final ensemble with XGBoost getting 80% weight"""
    print("\n=== Creating Final Ensemble ===")
    
    y_true = train_df[Config.LABEL_COLUMN].values
    
    # Evaluate XGBoost slices
    xgb_scores = {}
    for slice_name in xgb_oof['xgb']:
        score = pearsonr(y_true, xgb_oof['xgb'][slice_name])[0]
        xgb_scores[slice_name] = score
        print(f"  XGB {slice_name}: {score:.4f}")
    
    # Find best XGBoost slice
    best_xgb_slice = max(xgb_scores.items(), key=lambda x: x[1])[0]
    best_xgb_oof = xgb_oof['xgb'][best_xgb_slice]
    best_xgb_test = xgb_test['xgb'][best_xgb_slice]
    print(f"\nBest XGBoost slice: {best_xgb_slice} ({xgb_scores[best_xgb_slice]:.4f})")
    
    # Evaluate neural models
    nn_scores = {}
    for model_name in nn_oof:
        score = pearsonr(y_true, nn_oof[model_name])[0]
        nn_scores[model_name] = score
        print(f"  {model_name.upper()}: {score:.4f}")
    
    # Create weighted ensemble
    # XGBoost gets 80% weight
    xgb_weight = 1.0
    remaining_weight = 1.0 - xgb_weight
    
    # Distribute remaining 20% among neural models based on performance
    total_nn_score = sum(nn_scores.values())
    nn_weights = {name: (score/total_nn_score) * remaining_weight for name, score in nn_scores.items()}
    
    print(f"\n=== Final Weights ===")
    print(f"XGBoost ({best_xgb_slice}): {xgb_weight:.1%}")
    for name, weight in nn_weights.items():
        print(f"{name.upper()}: {weight:.1%}")
    
    # Create final predictions
    final_oof = best_xgb_oof * xgb_weight
    final_test = best_xgb_test * xgb_weight
    
    for model_name, weight in nn_weights.items():
        final_oof += nn_oof[model_name] * weight
        final_test += nn_test[model_name] * weight
    
    final_score = pearsonr(y_true, final_oof)[0]
    print(f"\nFinal Ensemble OOF Score: {final_score:.4f}")
    
    # Alternative ensemble: Simple average of all models (for comparison)
    all_oof = [best_xgb_oof] + list(nn_oof.values())
    simple_oof = np.mean(all_oof, axis=0)
    simple_score = pearsonr(y_true, simple_oof)[0]
    print(f"Simple Average Score: {simple_score:.4f}")
    
    # Use the better ensemble  
    if abs(simple_score) > abs(final_score):
        print("\nUsing simple average ensemble")
        final_test = np.mean([best_xgb_test] + list(nn_test.values()), axis=0)
    
    # Save submission
    filename = f"submission_xgb80_neural20_early_{int(EARLY_PERCENTAGE*100)}pct.csv"
    submission_df["prediction"] = final_test
    submission_df.to_csv(filename, index=False)
    print(f"\nSaved: {filename}")
    
    return final_test

# ====================================
# Main Pipeline
# ====================================
def main():
    print(f"\n{'='*60}")
    print(f"XGBoost (100%) + Neural Models (0%) Ensemble")
    print(f"Running with EARLY_PERCENTAGE = {EARLY_PERCENTAGE} ({int(EARLY_PERCENTAGE*100)}%)")
    print(f"{'='*60}")
    
    # Load data
    train_df, test_df, submission_df = load_data()
    
    # Train original XGBoost (unchanged)
    print("\n=== Training XGBoost (Original) ===")
    xgb_oof, xgb_test, model_slices = train_xgboost(train_df, test_df)
    
    # Train neural models
    nn_oof, nn_test = train_neural_models(train_df, test_df)
    
    # Create final ensemble
    final_predictions = create_final_ensemble(
        train_df, xgb_oof, xgb_test, nn_oof, nn_test, submission_df
    )
    
    print("\n✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main()