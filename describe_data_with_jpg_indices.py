import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import plot_tree
import statsmodels.api as sm

# Main Analysis Function
def analyze_csv(file_path, title, is_pixel_data=True):
    df = pd.read_csv(file_path)
    
    # Basic Descriptive Analysis
    analyze_csv_basic(df, title)
    
    # Correlation Analysis
    analyze_nonlinear_correlations(df, title)
    
    # Regression: Predict NDVI from jpg indices
    regression_analysis(df, title)
    
    # Clustering & PCA
    clustering_analysis(df, title)
    dimensionality_reduction(df, title)
    
    if is_pixel_data:
        print(f"Skipping spatial analysis for {title} (coordinates missing).")


    # --- 🔗 Scatterplots: JPG Indices vs True NDVI ---
def plot_scatter_for_ndvi(df, indices, title):
    # Filter out 'jpg_NDVI' and 'NDVI' from the indices list
    filtered_indices = [idx for idx in indices if idx not in ['jpg_NDVI', 'NDVI']]
    
    # Calculate the number of subplots needed
    num_plots = len(filtered_indices)
    
    # Determine the number of rows and columns for the subplot grid
    n_cols = 3  # Number of columns (can be adjusted)
    n_rows = (num_plots // n_cols) + (1 if num_plots % n_cols != 0 else 0)
    
    # Create the subplot grid
    plt.figure(figsize=(16, 5 * n_rows))  # Adjust figure size based on rows
    for idx, jpg_idx in enumerate(filtered_indices, 1):
        plt.subplot(n_rows, n_cols, idx)
        sns.scatterplot(x=df['NDVI'], y=df[jpg_idx], alpha=0.5)
        plt.xlabel('True NDVI')
        plt.ylabel(jpg_idx)
        plt.title(f'{jpg_idx} vs True NDVI')
        plt.plot([df['NDVI'].min(), df['NDVI'].max()], 
                 [df['NDVI'].min(), df['NDVI'].max()], 'r--', alpha=0.5)  # Diagonal line

    plt.suptitle(f'Scatterplots: JPG Indices vs True NDVI ({title})', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
# Descriptive Stats, Correlation, Boxplot & Scatterplot
def analyze_csv_basic(df, title):
    print(f"\nDescriptive Stats for {title}:")
    print(df.describe())

    # Correlation Matrix for Multispectral & JPG Indices
    indices = ['NDVI', 'NGRDI', 'VARI', 'GLI', 'vNDVI', 'RGBVI','MGRVI']
    jpg_indices = [f'jpg_{idx}' for idx in indices]

    plt.figure(figsize=(12, 6))
    sns.heatmap(df[indices + jpg_indices].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f'Correlation Matrix: {title}')
    plt.show()

    # --- 📊 Boxplot for Multispectral and JPG Indices ---
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df[indices + jpg_indices])
    plt.xticks(rotation=45)
    plt.title(f'Boxplot of Vegetation Indices: {title}')
    plt.show()

 #   plot_scatter_for_ndvi(df,jpg_indices)
 #   plot_scatter_for_ndvi(df,indices)


# Non-linear Correlations
def analyze_nonlinear_correlations(df, title):
    indices = ['NDVI', 'NGRDI', 'VARI', 'GLI', 'vNDVI', 'RGBVI','MGRVI']
    jpg_indices = [f'jpg_{idx}' for idx in indices]

    spearman_results = pd.DataFrame(index=indices, columns=jpg_indices)
    mi_results = pd.DataFrame(index=indices, columns=jpg_indices)

    for i in indices:
        for j in jpg_indices:
            corr, _ = spearmanr(df[i], df[j], nan_policy='omit')
            spearman_results.loc[i, j] = corr

            # Mutual Information
            mi = mutual_info_score(df[i].rank(method='dense'), df[j].rank(method='dense'))
            mi_results.loc[i, j] = mi

    # Plot Spearman Correlation Heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(spearman_results.astype(float), annot=True, cmap='viridis', fmt='.2f', vmin=-1, vmax=1)
    plt.title(f'Spearman Correlation (Multispectral vs JPG): {title}')
    plt.show()

# Regression Analysis: Predict NDVI from jpg_* indices (RGB-Based)
def regression_analysis(df, title):
    jpg_indices = [f'jpg_{idx}' for idx in ['NGRDI', 'VARI', 'GLI', 'RGBVI','MGRVI']]
    y = df['NDVI']
    X = df[jpg_indices]

    # --- 🔗 Linear Regression ---
    lr = LinearRegression()
    lr = sm.OLS(y, X).fit()
    y_pred_lr = lr.predict(X)
    r2_lr = r2_score(y, y_pred_lr)
    mae_lr = mean_absolute_error(y, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y, y_pred_lr))
    print(f"\n📈 Linear Regression for {title}:")
    print(lr.summary())
    print(f"R²: {r2_lr:.3f}, MAE: {mae_lr:.3f}, RMSE: {rmse_lr:.3f}")

    # --- 🌳 Random Forest Regression ---
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    y_pred_rf = rf.predict(X)
    r2_rf = r2_score(y, y_pred_rf)
    mae_rf = mean_absolute_error(y, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y, y_pred_rf))
    print(f"\n🌳 Random Forest Regression for {title}:")
    print(f"R²: {r2_rf:.3f}, MAE: {mae_rf:.3f}, RMSE: {rmse_rf:.3f}")

    # --- 📊 Plot: True vs Predicted NDVI ---
    plt.figure(figsize=(12, 5))

    # Linear Regression Plot
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y, y=y_pred_lr, alpha=0.5, label='Predictions')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal Fit')
    plt.xlabel('True NDVI')
    plt.ylabel('Predicted NDVI')
    plt.title(f'Linear Regression: {title}')
    plt.legend()

    # Random Forest Plot
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=y, y=y_pred_rf, alpha=0.5, label='Predictions')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal Fit')
    plt.xlabel('True NDVI')
    plt.ylabel('Predicted NDVI')
    plt.title(f'Random Forest Regression: {title}')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # --- 🌳 Visualize One Tree from Random Forest using matplotlib ---
    tree = rf.estimators_[0]  # Take the first tree
    plt.figure(figsize=(20, 10))
    plot_tree(tree, feature_names=jpg_indices, filled=True, rounded=True, max_depth=3)
    plt.title(f"Decision Tree from Random Forest: {title}")
    plt.show()

def clustering_analysis(df, title, n_clusters=3):
    indices = ['NDVI', 'NGRDI', 'VARI', 'GLI', 'vNDVI', 'RGBVI','MGRVI']
    jpg_indices = [f'jpg_{idx}' for idx in indices]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[jpg_indices])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    sns.pairplot(df, vars=jpg_indices, hue='Cluster', palette='viridis')
    plt.suptitle(f'Clustering: {title}', y=1.02)
    plt.show()

# PCA for Dimensionality Reduction
def dimensionality_reduction(df, title):
    indices = ['NDVI', 'NGRDI', 'VARI', 'GLI', 'vNDVI', 'RGBVI','MGRVI']
    jpg_indices = [f'jpg_{idx}' for idx in indices]
    X = df[jpg_indices]

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1])
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)')
    plt.title(f'PCA: {title}')
    plt.show()

# Run Analysis on Pixel & Patch CSVs
# base_dir = r'D:/MRc/FIIT/DP_Model/Datasets/kazachstan_multispectral_UAV/filght_session_02/2022-06-09' 
# pixel_csv = os.path.join(base_dir, 'veg_indices_perpixel_factor_8.csv')
# patch_csv = os.path.join(base_dir, 'vegetation_indices_patched_factor8_patch_64.csv')

# analyze_csv(pixel_csv, 'Pixel-Level Data', is_pixel_data=True)
# analyze_csv(patch_csv, 'Patch-Level Data', is_pixel_data=False)
