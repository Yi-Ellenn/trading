#%%
import polars as pl
train_data = pl.read_parquet("./train.parquet")
train_data.head()

#%%
import matplotlib.pyplot as plt
import numpy as np

# Calculate correlations between X1-X890 and label
x_cols = [f"X{i}" for i in range(1, 891)]
correlations = []

for col in x_cols:
    # Convert to numpy arrays and use np.corrcoef for correlation
    x = train_data[col].to_numpy()
    y = train_data["label"].to_numpy()
    valid_mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[valid_mask]
    y = y[valid_mask]
    corr = np.corrcoef(x, y)[0,1]
    correlations.append(corr)


plt.figure(figsize=(10, 6))
plt.hist(correlations, bins=50, edgecolor='black')
plt.title('Distribution of Feature Correlations with Label')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)
plt.show()


print(f"Max correlation: {max(correlations):.3f}")
print(f"Min correlation: {min(correlations):.3f}")
print(f"Mean correlation: {np.nanmean(correlations):.3f}")
print(f"Median correlation: {np.nanmedian(correlations):.3f}")

# %%
test_data = pl.read_parquet("/Users/puzhengheng/Documents/Course/drw-crypto-market-prediction/test.parquet")
test_data.head()

# %%
plt.figure(figsize=(10, 6))
plt.hist(train_data["label"].to_numpy(), bins=50, edgecolor='black')
plt.title('Distribution of Label Values')
plt.xlabel('Label Value')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)
plt.show()

print(f"Label mean: {train_data['label'].mean():.3f}")
print(f"Label std: {train_data['label'].std():.3f}")
print(f"Label min: {train_data['label'].min():.3f}")
print(f"Label max: {train_data['label'].max():.3f}")

# %%
# Get absolute correlations and feature names
abs_correlations = np.abs(correlations)
feature_pairs = list(zip(x_cols, abs_correlations))

# Filter features with abs correlation > 0.02 and sort by correlation strength
strong_features = sorted([(col, corr) for col, corr in feature_pairs if corr > 0.02], 
                        key=lambda x: x[1], reverse=True)

# Initialize selected features list with the strongest feature
selected_features = [strong_features[0][0]]
selected_correlations = [strong_features[0][1]]

# For each remaining feature
for feature, corr in strong_features[1:]:
    # Get feature values
    feature_values = train_data[feature].to_numpy()
    
    # Check correlation with all already selected features
    is_independent = True
    for selected_feature in selected_features:
        selected_values = train_data[selected_feature].to_numpy()
        
        # Handle missing values
        valid_mask = ~np.isnan(feature_values) & ~np.isnan(selected_values)
        if sum(valid_mask) > 0:  # Only calculate if we have valid data points
            correlation = abs(np.corrcoef(
                feature_values[valid_mask], 
                selected_values[valid_mask]
            )[0,1])
            
            if correlation >= 0.8:
                is_independent = False
                break
                
    # If feature is independent enough, add it to selected features
    if is_independent:
        selected_features.append(feature)
        selected_correlations.append(corr)

# Print results
print(f"Selected {len(selected_features)} features with IC > 0.02 and low inter-correlation:")
for feature, corr in zip(selected_features, selected_correlations):
    print(f"{feature}: {corr:.4f}")

# %%
# Save selected features to pickle file
import pickle

with open('selected_features.pkl', 'wb') as f:
    pickle.dump(selected_features, f)

print("Selected features saved to selected_features.pkl")

# %%


# %%
# Load selected features from pickle file
import pickle

with open('selected_features.pkl', 'rb') as f:
    loaded_features = pickle.load(f)

print("Loaded features from selected_features.pkl:")
print(f"Number of features: {len(loaded_features)}")
print("Features:", loaded_features)

# %%

