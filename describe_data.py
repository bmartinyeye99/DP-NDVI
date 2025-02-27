import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import libpysal as ps
from esda.moran import Moran


def analyze_csv(file_path, title, is_pixel_data=True):
    # Load data
    df = pd.read_csv(file_path)
   
    # Descriptive stats and basic plots (from previous code)
    analyze_csv_basic(df, title)
    
    # Advanced analysis
    analyze_nonlinear_correlations(df, title)
    regression_analysis(df, title)
    clustering_analysis(df, title)
    dimensionality_reduction(df, title)
    
    if is_pixel_data:
        spatial_analysis(df, title)
    
    domain_adaptation_analysis(df, title, is_pixel_data)

# Basic analysis (from previous answer)
def analyze_csv_basic(df, title):
    print(f"\nDescriptive Stats for {title}:")
    print(df.describe())
    
    # Correlation matrix
    corr = df[['NDVI', 'NGRDI', 'VARI', 'GLI', 'vNDVI', 'RGBVI']].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f'Correlation Matrix: {title}')
    plt.show()
    
    # Boxplot
    df_melted = df.melt(value_vars=['NDVI', 'NGRDI', 'VARI', 'GLI', 'vNDVI', 'RGBVI'])
    sns.boxplot(x='variable', y='value', data=df_melted)
    plt.title(f'Boxplot: {title}')
    plt.xticks(rotation=45)
    plt.show()

# Non-linear correlations
def analyze_nonlinear_correlations(df, title):
    indices = ['NDVI', 'NGRDI', 'VARI', 'GLI', 'vNDVI', 'RGBVI']
    spearman_results = pd.DataFrame(index=indices, columns=indices)
    mi_results = pd.DataFrame(index=indices, columns=indices)
    
    for i in indices:
        for j in indices:
            # Calculate Spearman correlation
            corr, _ = spearmanr(df[i], df[j], nan_policy='omit')  # Handle NaNs
            spearman_results.loc[i, j] = corr
            
            # Calculate Mutual Information
            mi_results.loc[i, j] = mutual_info_score(df[i].rank(method='dense'), df[j].rank(method='dense'))
    
    # Convert Spearman results to float (in case of NaNs)
    spearman_results = spearman_results.astype(float)
    
    # Plot Spearman correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(spearman_results, annot=True, cmap='viridis', fmt='.2f', vmin=-1, vmax=1)
    plt.title(f'Spearman Correlation: {title}')
    plt.show()
    
    # Plot Mutual Information heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(mi_results.astype(float), annot=True, cmap='viridis', fmt='.2f')
    plt.title(f'Mutual Information: {title}')
    plt.show()

# Regression modeling
def regression_analysis(df, title):
    X = df[['vNDVI', 'NGRDI']]  # Predictors (visible indices)
    y = df['NDVI']   # Target
    
    # Linear regression
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, y)
    print(f"\nLinear Regression (NDVI ~ vNDVI + NGRDI) for {title}:")
    print(f"Coefficients: vNDVI={model.coef_[0]:.3f}, NGRDI={model.coef_[1]:.3f}")
    print(f"R²: {model.score(X, y):.3f}")
    
    # Non-linear regression (Random Forest)
    model_rf = RandomForestRegressor(n_estimators=100)
    model_rf.fit(X, y)
    print(f"Random Forest R²: {model_rf.score(X, y):.3f}")
    
    # Plot predictions
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y, y=model_rf.predict(X))
    plt.xlabel('True NDVI')
    plt.ylabel('Predicted NDVI')
    plt.title(f'Regression Fit: {title}')
    plt.show()

# Clustering analysis
def clustering_analysis(df, title, n_clusters=3):
    X = df[['NDVI', 'VARI', 'RGBVI']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    sns.pairplot(df, vars=['NDVI', 'VARI', 'RGBVI'], hue='Cluster', palette='viridis')
    plt.suptitle(f'Clustering Analysis: {title}', y=1.02)
    plt.show()

# Dimensionality reduction
def dimensionality_reduction(df, title):
    X = df[['NDVI', 'NGRDI', 'VARI', 'GLI', 'vNDVI', 'RGBVI']]
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1])
    plt.xlabel('PC1 (Variance: {:.1f}%)'.format(pca.explained_variance_ratio_[0]*100))
    plt.ylabel('PC2 (Variance: {:.1f}%)'.format(pca.explained_variance_ratio_[1]*100))
    plt.title(f'PCA: {title}')
    plt.show()

# Spatial analysis (for pixel-level data)
def spatial_analysis(df, title):
    # Moran's I for spatial autocorrelation
    w = ps.lib.weights.lat2W(df['Y'].max()+1, df['X'].max()+1)  # Grid weights
    moran = ps.Moran(df['NDVI'], w)
    print(f"\nSpatial Autocorrelation (Moran's I) for NDVI in {title}: {moran.I:.3f} (p={moran.p_norm:.3f})")

# Domain adaptation (compare pixel vs. patch)
def domain_adaptation_analysis(df, title, is_pixel_data):
    if is_pixel_data:
        # Train on pixels, test on patches (or vice versa)
        # Example: Use ReliefF to identify important indices
        X = df[['NDVI', 'NGRDI', 'VARI', 'GLI', 'vNDVI', 'RGBVI']].values
        y = df['NDVI'].values  # Example target
        
        fs = ReliefF()
        fs.fit(X, y)
        print(f"\nFeature Importance (ReliefF) for {title}:")
        print(dict(zip(['NDVI', 'NGRDI', 'VARI', 'GLI', 'vNDVI', 'RGBVI'], fs.feature_importances_)))

# Run analysis

# base_dir = r'D:/MRc/FIIT/DP_Model/Datasets/kazachstan_multispectral_UAV/filght_session_02/2022-06-09' 
# pixel_csv_name = os.path.join(base_dir,'veg_indices_perpixel_factor_8.csv')
# patch_csv_name = os.path.join(base_dir,'vegetation_indices_patched_factor8_patch_64.csv')

# analyze_csv(pixel_csv_name, 'Pixel-Level Data', is_pixel_data=True)
# analyze_csv(patch_csv_name, 'Patch-Level Data', is_pixel_data=False)



