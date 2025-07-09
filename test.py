#%%
import pandas as pd

#%%
df = pd.read_parquet("train.parquet")

#%%
df.head()

#%%
# Read the two submission files
import pandas as pd
sub1 = pd.read_csv("/Users/puzhengheng/Documents/Course/drw-crypto-market-prediction/submission_xgb80_neural20_early_35pct.csv")
sub2 = pd.read_csv("/Users/puzhengheng/Documents/Course/drw-crypto-market-prediction/submission.csv")

# # Calculate correlation coefficient between the target columns
# corr = sub1['prediction'].corr(sub2['prediction'])
# print(f"Correlation coefficient between the two submissions: {corr:.4f}")

# Calculate weighted predictions
weighted_predictions = -0.12 * sub2['prediction'] + 0.99 * sub1['prediction']

# Create new submission dataframe
weighted_submission = pd.DataFrame({
    'ID': sub1['ID'],
    'prediction': weighted_predictions
})

# Save weighted submission
weighted_submission.to_csv('weighted_submission.csv', index=False)
print("Saved weighted submission to weighted_submission.csv")


# %%
# Get all anonymous features (starting with 'X')
import pandas as pd
from tqdm import tqdm

df = pd.read_parquet("train.parquet")
anon_features = [col for col in df.columns if col.startswith('X')]

# Initialize lists to store results
results = []
significant_expressions = []

# Calculate ICs for all pairs of features
import numpy as np
from scipy.stats import pearsonr

# Convert dataframe columns to numpy arrays for faster computation
label_array = df['label'].values
feature_arrays = {feat: df[feat].values for feat in anon_features}

for i, feat1 in list(enumerate(anon_features)):
    arr1 = feature_arrays[feat1]
    for feat2 in tqdm(anon_features[i+1:]):
        if feat1 == feat2:
            continue
            
        arr2 = feature_arrays[feat2]
        
        # Calculate operators using numpy operations
        minus = arr1 - arr2
        ratio = arr1 / (arr2 + 1e-8)
        imbalance = (arr1 - arr2) / (arr1 + arr2 + 1e-8)
        harmonic = (arr1 * arr2) / (arr1 + arr2 + 1e-8)
        
        # Replace inf/-inf with nan
        minus = np.where(np.isinf(minus), np.nan, minus)
        ratio = np.where(np.isinf(ratio), np.nan, ratio)
        imbalance = np.where(np.isinf(imbalance), np.nan, imbalance)
        harmonic = np.where(np.isinf(harmonic), np.nan, harmonic)
        
        # Calculate correlations using scipy.stats.pearsonr after removing nans
        minus_mask = ~np.isnan(minus)
        ratio_mask = ~np.isnan(ratio)
        imbalance_mask = ~np.isnan(imbalance)
        harmonic_mask = ~np.isnan(harmonic)
        
        # Check if we have enough non-nan values before calculating correlation
        try:
            minus_ic = pearsonr(minus[minus_mask], label_array[minus_mask])[0] if np.sum(minus_mask) >= 2 else np.nan
            ratio_ic = pearsonr(ratio[ratio_mask], label_array[ratio_mask])[0] if np.sum(ratio_mask) >= 2 else np.nan
            imbalance_ic = pearsonr(imbalance[imbalance_mask], label_array[imbalance_mask])[0] if np.sum(imbalance_mask) >= 2 else np.nan
            harmonic_ic = pearsonr(harmonic[harmonic_mask], label_array[harmonic_mask])[0] if np.sum(harmonic_mask) >= 2 else np.nan
        except:
            minus_ic = ratio_ic = imbalance_ic = harmonic_ic = np.nan
            
        # Store expressions with IC > 0.04
        if abs(minus_ic) > 0.06:
            significant_expressions.append({
                'expression': f'{feat1} - {feat2}',
                'ic': minus_ic
            })
            print(f"Found significant IC > 0.04: {feat1} - {feat2}, {minus_ic:.4f}")
        if abs(ratio_ic) > 0.06:
            significant_expressions.append({
                'expression': f'{feat1} / ({feat2} + 1e-8)',
                'ic': ratio_ic
            })
            print(f"Found significant IC > 0.04: {feat1} / ({feat2} + 1e-8), {ratio_ic:.4f}")
        if abs(imbalance_ic) > 0.06:
            significant_expressions.append({
                'expression': f'({feat1} - {feat2}) / ({feat1} + {feat2} + 1e-8)',
                'ic': imbalance_ic
            })
            print(f"Found significant IC > 0.04: ({feat1} - {feat2}) / ({feat1} + {feat2} + 1e-8), {imbalance_ic:.4f}")
        if abs(harmonic_ic) > 0.06:
            significant_expressions.append({
                'expression': f'({feat1} * {feat2}) / ({feat1} + {feat2} + 1e-8)',
                'ic': harmonic_ic
            })
            print(f"Found significant IC > 0.04: ({feat1} * {feat2}) / ({feat1} + {feat2} + 1e-8), {harmonic_ic:.4f}")
            
        results.append({
            'feature1': feat1,
            'feature2': feat2,
            'minus_ic': minus_ic,
            'ratio_ic': ratio_ic,
            'imbalance_ic': imbalance_ic, 
            'harmonic_ic': harmonic_ic
        })

# Convert results to dataframe
results_df = pd.DataFrame(results)
significant_df = pd.DataFrame(significant_expressions)

# Save significant expressions to CSV
significant_df.to_csv('significant_expressions.csv', index=False)
print("\nSaved significant expressions to significant_expressions.csv")

# Sort by absolute IC values
for col in ['minus_ic', 'ratio_ic', 'imbalance_ic', 'harmonic_ic']:
    print(f"\nTop 10 pairs by absolute {col}:")
    print(results_df.nlargest(10, col).to_string())


# %%
# Calculate median and 1% quantile of ICs for x-prefixed features
import pandas as pd
import numpy as np

# Load data
train = pd.read_parquet("train.parquet")
x_features = [col for col in train.columns if col.startswith('X')]
ics = []

print("\nIC Statistics for x-prefixed features:")
print("-" * 50)

# Calculate ICs with handling for infinite values
for feat in x_features:
    valid = np.isfinite(train[feat]) & np.isfinite(train['label'])
    ic = np.corrcoef(train[feat][valid], train['label'][valid])[0, 1]
    ics.append(abs(ic))

ics = np.array(ics)
# Get top 20 features by IC
top_features = pd.Series(ics, index=x_features).sort_values(ascending=False)[:20]

print("\nTop 10 features by absolute IC:")
print("-" * 50)
for feat, ic in top_features.items():
    print(f"{feat}: {ic:.4f}")

# %%
# Train XGBoost model to get feature importances
from xgboost import XGBRegressor

import matplotlib.pyplot as plt

print("\nCalculating XGBoost feature importance:")
print("-" * 50)

# Prepare data and handle infinite values
X = train[x_features].replace([np.inf, -np.inf], np.nan)
y = train['label']

# Initialize and train model with missing value handling
xgb = XGBRegressor(
    tree_method='hist',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    missing=np.nan  # Explicitly handle missing values
)

# Fit model on valid data only
valid_mask = np.isfinite(y) 
X_valid = X[valid_mask].copy()
y_valid = y[valid_mask].copy()

# Replace any remaining infinities with NaN

xgb.fit(X_valid, y_valid)

# Get feature importance
importance_dict = dict(zip(x_features, xgb.feature_importances_))
importance_series = pd.Series(importance_dict).sort_values(ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 8))
plt.bar(range(20), importance_series.head(20).values)
plt.xticks(range(20), importance_series.head(20).index, rotation=45, ha='right')
plt.title('Top 20 Features by XGBoost Importance')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.show()

print("\nTop 20 features by XGBoost importance:")
print("-" * 50)
for feat, imp in importance_series.head(20).items():
    print(f"{feat}: {imp:.4f}")

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_parquet("train.parquet")
# Check distribution of all X features for outliers and anomalies
print("\nAnalyzing X feature distributions for outliers:")
print("=" * 60)

# Get all X features
x_cols = [col for col in train.columns if col.startswith('X')]
print(f"Found {len(x_cols)} X features to analyze")

# Create summary statistics for all X features
summary_stats = []
for col in x_cols:
    data = train[col].replace([np.inf, -np.inf], np.nan)
    
    stats = {
        'feature': col,
        'count': data.count(),
        'missing': data.isna().sum(),
        'missing_pct': data.isna().sum() / len(data) * 100,
        'mean': data.mean(),
        'std': data.std(),
        'min': data.min(),
        'q1': data.quantile(0.25),
        'median': data.median(),
        'q3': data.quantile(0.75),
        'max': data.max(),
        'skewness': data.skew(),
        'kurtosis': data.kurtosis(),
        'outliers_iqr': ((data < (data.quantile(0.25) - 1.5 * (data.quantile(0.75) - data.quantile(0.25)))) | 
                        (data > (data.quantile(0.75) + 1.5 * (data.quantile(0.75) - data.quantile(0.25))))).sum(),
        'outliers_3std': (np.abs((data - data.mean()) / data.std()) > 3).sum()
    }
    summary_stats.append(stats)

# Convert to DataFrame for better analysis
stats_df = pd.DataFrame(summary_stats)

# Display features with high missing values
print("\nFeatures with high missing values (>5%):")
print("-" * 50)
high_missing = stats_df[stats_df['missing_pct'] > 5].sort_values('missing_pct', ascending=False)
if len(high_missing) > 0:
    print(high_missing[['feature', 'missing', 'missing_pct']].to_string(index=False))
else:
    print("No features with >5% missing values")

# Display features with extreme outliers
print("\nFeatures with high outlier counts (IQR method):")
print("-" * 50)
high_outliers = stats_df[stats_df['outliers_iqr'] > len(train) * 0.05].sort_values('outliers_iqr', ascending=False)
if len(high_outliers) > 0:
    print(high_outliers[['feature', 'outliers_iqr', 'outliers_3std']].head(10).to_string(index=False))
else:
    print("No features with excessive outliers (>5% of data)")

# Display features with extreme skewness
print("\nFeatures with extreme skewness (|skew| > 5):")
print("-" * 50)
extreme_skew = stats_df[np.abs(stats_df['skewness']) > 5].sort_values('skewness', key=abs, ascending=False)
if len(extreme_skew) > 0:
    print(extreme_skew[['feature', 'skewness', 'kurtosis']].head(10).to_string(index=False))
else:
    print("No features with extreme skewness")

# Create distribution plots for features with potential issues
problematic_features = set()
problematic_features.update(high_missing['feature'].tolist())
problematic_features.update(high_outliers['feature'].tolist())
problematic_features.update(extreme_skew['feature'].tolist())

if len(problematic_features) > 0:
    print(f"\nCreating distribution plots for {len(problematic_features)} problematic features...")
    
    # Limit to top 12 most problematic features for visualization
    top_problematic = list(problematic_features)[:12]
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, feat in enumerate(top_problematic):
        if i >= 12:
            break
            
        data = train[feat].replace([np.inf, -np.inf], np.nan).dropna()
        
        # Create histogram
        axes[i].hist(data, bins=50, alpha=0.7, density=True)
        axes[i].set_title(f'{feat}\nSkew: {data.skew():.2f}, Kurt: {data.kurtosis():.2f}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Density')
        
        # Add vertical lines for quartiles
        q1, q2, q3 = data.quantile([0.25, 0.5, 0.75])
        axes[i].axvline(q1, color='red', linestyle='--', alpha=0.5, label='Q1')
        axes[i].axvline(q2, color='green', linestyle='--', alpha=0.5, label='Median')
        axes[i].axvline(q3, color='red', linestyle='--', alpha=0.5, label='Q3')
    
    # Hide unused subplots
    for i in range(len(top_problematic), 12):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Summary of overall data quality
print("\nOverall X Features Data Quality Summary:")
print("=" * 60)
print(f"Total X features: {len(x_cols)}")
print(f"Features with missing values: {(stats_df['missing'] > 0).sum()}")
print(f"Features with >1% missing: {(stats_df['missing_pct'] > 1).sum()}")
print(f"Features with >5% missing: {(stats_df['missing_pct'] > 5).sum()}")
print(f"Features with high outliers (>5% of data): {(stats_df['outliers_iqr'] > len(train) * 0.05).sum()}")
print(f"Features with extreme skewness (|skew| > 5): {(np.abs(stats_df['skewness']) > 5).sum()}")
print(f"Features with extreme kurtosis (>20): {(stats_df['kurtosis'] > 20).sum()}")

# Save detailed statistics
stats_df.to_csv('x_features_distribution_analysis.csv', index=False)
print(f"\nDetailed statistics saved to 'x_features_distribution_analysis.csv'")

# %%
