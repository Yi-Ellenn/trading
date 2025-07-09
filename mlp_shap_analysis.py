import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Configuration
class Config:
    import os
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    TRAIN_PATH = os.path.join(PROJECT_DIR, "train.parquet")
    TEST_PATH = os.path.join(PROJECT_DIR, "test.parquet")
    SUBMISSION_PATH = os.path.join(PROJECT_DIR, "sample_submission.csv")
    
    LABEL_COLUMN = "label"
    N_FOLDS = 3
    RANDOM_STATE = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # MLP specific parameters
    BATCH_SIZE = 512
    LEARNING_RATE = 0.001
    EPOCHS = 100
    PATIENCE = 15
    
    # Feature selection parameters
    MAX_FEATURES = 50  # Maximum number of features for MLP
    IC_THRESHOLD = 0.01  # Minimum IC threshold for feature selection


def load_and_prepare_data():
    """Load data and get all available features"""
    print("Loading data...")
    
    # Load full data first to get all columns
    train_df = pd.read_parquet(Config.TRAIN_PATH)
    test_df = pd.read_parquet(Config.TEST_PATH)
    
    # Get all feature columns (excluding label and ID-like columns)
    all_features = [col for col in train_df.columns 
                   if col not in [Config.LABEL_COLUMN, 'ID'] and not col.startswith('id')]
    
    print(f"Total available features: {len(all_features)}")
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    
    return train_df, test_df, all_features


def calculate_feature_ic(train_df, features):
    """Calculate Information Coefficient for each feature"""
    print("\nCalculating feature ICs...")
    
    feature_ics = {}
    y = train_df[Config.LABEL_COLUMN].values
    
    for feature in tqdm(features):
        try:
            x = train_df[feature].values
            
            # Handle missing values
            valid_mask = ~np.isnan(x) & ~np.isnan(y) & np.isfinite(x) & np.isfinite(y)
            
            if np.sum(valid_mask) > 100:  # Need enough valid samples
                try:
                    ic = pearsonr(x[valid_mask], y[valid_mask])[0]
                    feature_ics[feature] = abs(ic) if not np.isnan(ic) else 0.0
                except:
                    feature_ics[feature] = 0.0
            else:
                feature_ics[feature] = 0.0
                
        except Exception as e:
            feature_ics[feature] = 0
    
    return feature_ics


def feature_selection_for_mlp(train_df, all_features):
    """Multi-stage feature selection optimized for MLP"""
    print("\n=== MLP Feature Selection ===")
    
    # Stage 1: IC-based filtering
    print("Stage 1: IC-based filtering...")
    feature_ics = calculate_feature_ic(train_df, all_features)
    
    # Filter by IC threshold
    ic_filtered = {k: v for k, v in feature_ics.items() if v > Config.IC_THRESHOLD}
    print(f"Features after IC filtering (>{Config.IC_THRESHOLD}): {len(ic_filtered)}")
    
    if len(ic_filtered) == 0:
        print("Warning: No features pass IC threshold, using top 50 by IC")
        ic_filtered = dict(sorted(feature_ics.items(), key=lambda x: x[1], reverse=True)[:50])
    
    # Stage 2: Statistical feature selection
    print("Stage 2: Statistical feature selection...")
    ic_features = list(ic_filtered.keys())
    
    X = train_df[ic_features].fillna(0).replace([np.inf, -np.inf], 0)
    y = train_df[Config.LABEL_COLUMN].fillna(0)
    
    # Remove constant features
    feature_variances = X.var()
    variable_features = feature_variances[feature_variances > 1e-8].index.tolist()
    print(f"Features after variance filtering: {len(variable_features)}")
    
    X_filtered = X[variable_features]
    
    # Apply SelectKBest with f_regression
    n_features_to_select = min(Config.MAX_FEATURES * 2, len(variable_features))
    selector = SelectKBest(score_func=f_regression, k=n_features_to_select)
    
    X_selected = selector.fit_transform(X_filtered, y)
    selected_features = X_filtered.columns[selector.get_support()].tolist()
    
    print(f"Features after statistical selection: {len(selected_features)}")
    
    # Stage 3: Correlation-based redundancy removal
    print("Stage 3: Removing redundant features...")
    
    # Calculate correlation matrix
    corr_matrix = X_filtered[selected_features].corr().abs()
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.95:  # High correlation threshold
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
    
    # Remove redundant features (keep the one with higher IC)
    features_to_remove = set()
    for feat1, feat2 in high_corr_pairs:
        if feat1 not in features_to_remove and feat2 not in features_to_remove:
            # Keep the feature with higher IC
            if feature_ics.get(feat1, 0) >= feature_ics.get(feat2, 0):
                features_to_remove.add(feat2)
            else:
                features_to_remove.add(feat1)
    
    final_features = [f for f in selected_features if f not in features_to_remove]
    
    # Ensure we don't exceed max features
    if len(final_features) > Config.MAX_FEATURES:
        # Sort by IC and take top features
        feature_ic_pairs = [(f, feature_ics.get(f, 0)) for f in final_features]
        feature_ic_pairs.sort(key=lambda x: x[1], reverse=True)
        final_features = [f for f, _ in feature_ic_pairs[:Config.MAX_FEATURES]]
    
    print(f"Final selected features for MLP: {len(final_features)}")
    
    # Print top features by IC
    print("\nTop 10 selected features by IC:")
    top_features = sorted([(f, feature_ics.get(f, 0)) for f in final_features], 
                         key=lambda x: x[1], reverse=True)[:10]
    for feat, ic in top_features:
        print(f"  {feat}: {ic:.4f}")
    
    return final_features, feature_ics


class MLPRegressor(nn.Module):
    """Simple MLP with dropout and batch normalization"""
    
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience=7, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0


def train_mlp_model(X_train, y_train, X_val, y_val, input_dim):
    """Train MLP model with early stopping"""
    
    # Create model
    model = MLPRegressor(
        input_dim=input_dim,
        hidden_dims=[256, 128, 64, 32],
        dropout=0.3
    ).to(Config.DEVICE)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.7)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=Config.PATIENCE)
    
    # Training loop
    model.train()
    train_losses = []
    val_losses = []
    
    for epoch in range(Config.EPOCHS):
        # Training
        train_loss = 0
        num_batches = 0
        for i in range(0, len(X_train), Config.BATCH_SIZE):
            batch_X = X_train[i:i+Config.BATCH_SIZE]
            batch_y = y_train[i:i+Config.BATCH_SIZE]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs.squeeze(), y_val).item()
            
            # Calculate IC for validation
            val_pred = val_outputs.squeeze().cpu().numpy()
            val_true = y_val.cpu().numpy()
            val_ic = pearsonr(val_pred, val_true)[0] if len(val_pred) > 1 else 0
        
        model.train()
        
        train_losses.append(train_loss / num_batches)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping check (use negative val_loss as score to maximize)
        early_stopping(-val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{Config.EPOCHS}, Train Loss: {train_losses[-1]:.6f}, "
                  f"Val Loss: {val_loss:.6f}, Val IC: {val_ic:.4f}")
    
    return model, train_losses, val_losses


def cross_validate_mlp(train_df, selected_features):
    """Cross-validate MLP model"""
    print("\n=== Cross-Validating MLP ===")
    
    X = train_df[selected_features].fillna(0).replace([np.inf, -np.inf], 0)
    y = train_df[Config.LABEL_COLUMN].fillna(0)
    
    # Remove samples with NaN targets
    valid_mask = ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]
    
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=False)
    fold_scores = []
    models = []
    scalers = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\nFold {fold}/{Config.N_FOLDS}")
        
        # Split data
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(Config.DEVICE)
        y_train_tensor = torch.FloatTensor(y_train_fold.values).to(Config.DEVICE)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(Config.DEVICE)
        y_val_tensor = torch.FloatTensor(y_val_fold.values).to(Config.DEVICE)
        
        # Train model
        model, train_losses, val_losses = train_mlp_model(
            X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, len(selected_features)
        )
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor).squeeze().cpu().numpy()
            val_true = y_val_fold.values
            
            # Calculate IC
            fold_ic = pearsonr(val_pred, val_true)[0]
            fold_scores.append(fold_ic)
            print(f"Fold {fold} IC: {fold_ic:.4f}")
        
        models.append(model)
        scalers.append(scaler)
    
    mean_ic = np.mean(fold_scores)
    std_ic = np.std(fold_scores)
    print(f"\nCross-validation results:")
    print(f"Mean IC: {mean_ic:.4f} ± {std_ic:.4f}")
    
    return models, scalers, fold_scores


def shap_analysis(models, scalers, train_df, selected_features):
    """Perform SHAP analysis on trained MLP models"""
    print("\n=== SHAP Analysis ===")
    
    X = train_df[selected_features].fillna(0).replace([np.inf, -np.inf], 0)
    y = train_df[Config.LABEL_COLUMN].fillna(0)
    
    # Remove samples with NaN targets
    valid_mask = ~np.isnan(y)
    X = X[valid_mask]
    
    # Use a subset for SHAP analysis (computational efficiency)
    n_samples = min(1000, len(X))
    sample_idx = np.random.choice(len(X), n_samples, replace=False)
    X_sample = X.iloc[sample_idx]
    
    # Use the first model for SHAP analysis
    model = models[0]
    scaler = scalers[0]
    
    # Scale the sample data
    X_sample_scaled = scaler.transform(X_sample)
    
    # Create a wrapper function for SHAP
    def model_predict(x):
        model.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(Config.DEVICE)
            return model(x_tensor).cpu().numpy().flatten()
    
    # SHAP analysis
    print("Computing SHAP values...")
    
    # Use a smaller background dataset for efficiency
    background_size = min(100, len(X_sample_scaled))
    background = X_sample_scaled[:background_size]
    
    # Create SHAP explainer
    explainer = shap.DeepExplainer(model, torch.FloatTensor(background).to(Config.DEVICE))
    
    # Calculate SHAP values for a subset
    shap_sample_size = min(200, len(X_sample_scaled))
    shap_values = explainer.shap_values(
        torch.FloatTensor(X_sample_scaled[:shap_sample_size]).to(Config.DEVICE)
    )
    
    # Handle different SHAP value formats
    if isinstance(shap_values, list):
        # For multi-output models, take the first output
        shap_values = shap_values[0]
    
    # Ensure shap_values is 2D (samples, features)
    if len(shap_values.shape) > 2:
        shap_values = shap_values.squeeze()
    
    print(f"SHAP values shape: {shap_values.shape}")
    print(f"Selected features count: {len(selected_features)}")
    
    # Create feature importance summary
    feature_importance = np.abs(shap_values).mean(axis=0)
    
    # Ensure feature_importance is 1D
    if len(feature_importance.shape) > 1:
        feature_importance = feature_importance.flatten()
    
    # Handle dimension mismatch
    if len(feature_importance) != len(selected_features):
        print(f"Warning: Feature importance length ({len(feature_importance)}) != selected features length ({len(selected_features)})")
        min_len = min(len(feature_importance), len(selected_features))
        feature_importance = feature_importance[:min_len]
        features_for_df = selected_features[:min_len]
    else:
        features_for_df = selected_features
    
    feature_importance_df = pd.DataFrame({
        'feature': features_for_df,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 features by SHAP importance:")
    print(feature_importance_df.head(20).to_string(index=False))
    
    # Create plots
    create_shap_plots(shap_values, X_sample.iloc[:shap_sample_size], selected_features, feature_importance_df)
    
    return feature_importance_df, shap_values


def create_shap_plots(shap_values, X_sample, feature_names, feature_importance_df):
    """Create SHAP visualization plots"""
    print("\nCreating SHAP plots...")
    
    # Set style
    plt.style.use('default')
    
    # 1. Feature importance plot
    plt.figure(figsize=(10, 8))
    top_features = feature_importance_df.head(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Mean |SHAP Value|')
    plt.title('Feature Importance (SHAP)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('shap_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Summary plot (if shap is available)
    try:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=20)
        plt.title('SHAP Summary Plot')
        plt.tight_layout()
        plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"Could not create SHAP summary plot: {e}")
    
    # 3. Feature correlation with SHAP values
    if len(feature_importance_df) >= 20:
        plt.figure(figsize=(12, 8))
        top_20_features = feature_importance_df.head(20)['feature'].tolist()
        
        # Calculate correlation between features and their SHAP importance
        feature_data = X_sample[top_20_features].values
        shap_top = shap_values[:, [feature_names.index(f) for f in top_20_features]]
        
        correlations = []
        for i, feat in enumerate(top_20_features):
            corr = np.corrcoef(feature_data[:, i], shap_top[:, i])[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0)
        
        plt.bar(range(len(top_20_features)), correlations)
        plt.xticks(range(len(top_20_features)), top_20_features, rotation=45, ha='right')
        plt.ylabel('Correlation (Feature Value vs SHAP Value)')
        plt.title('Feature-SHAP Correlation (Top 20)')
        plt.tight_layout()
        plt.savefig('feature_shap_correlation.png', dpi=300, bbox_inches='tight')
        plt.show()


def generate_predictions(models, scalers, test_df, selected_features):
    """Generate predictions using ensemble of trained models"""
    print("\n=== Generating Predictions ===")
    
    X_test = test_df[selected_features].fillna(0).replace([np.inf, -np.inf], 0)
    
    predictions = []
    for model, scaler in zip(models, scalers):
        X_test_scaled = scaler.transform(X_test)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(Config.DEVICE)
        
        model.eval()
        with torch.no_grad():
            pred = model(X_test_tensor).squeeze().cpu().numpy()
            predictions.append(pred)
    
    # Ensemble predictions (mean)
    final_predictions = np.mean(predictions, axis=0)
    
    return final_predictions


def main():
    """Main execution function"""
    print("=== MLP Feature Selection and SHAP Analysis ===")
    
    # Set random seeds
    np.random.seed(Config.RANDOM_STATE)
    torch.manual_seed(Config.RANDOM_STATE)
    
    # Load data
    train_df, test_df, all_features = load_and_prepare_data()
    
    # Feature selection
    selected_features, feature_ics = feature_selection_for_mlp(train_df, all_features)
    
    # Save selected features
    with open('mlp_selected_features.pkl', 'wb') as f:
        pickle.dump(selected_features, f)
    print(f"\nSaved {len(selected_features)} selected features to mlp_selected_features.pkl")
    
    # Cross-validate MLP
    models, scalers, fold_scores = cross_validate_mlp(train_df, selected_features)
    
    # SHAP analysis
    feature_importance_df, shap_values = shap_analysis(models, scalers, train_df, selected_features)
    
    # Save feature importance
    feature_importance_df.to_csv('mlp_feature_importance_shap.csv', index=False)
    print("Saved SHAP feature importance to mlp_feature_importance_shap.csv")
    
    # Save top 20 most important features
    top_50_features = feature_importance_df.head(50)['feature'].tolist()
    with open('mlp_top50_features.pkl', 'wb') as f:
        pickle.dump(top_50_features, f)
    print(f"Saved top 50 most important features to mlp_top50_features.pkl")
    
    # Generate predictions
    predictions = generate_predictions(models, scalers, test_df, selected_features)
    
    # Create submission
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    submission_df['prediction'] = predictions
    submission_df.to_csv('submission_mlp_shap.csv', index=False)
    print("Saved predictions to submission_mlp_shap.csv")
    
    print("\n=== Analysis Complete ===")
    print(f"Final model performance: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f} IC")
    print(f"Selected {len(selected_features)} features for MLP")
    print(f"Saved top 100 most important features")
    print("SHAP plots saved as PNG files")


if __name__ == "__main__":
    main() 