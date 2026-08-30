from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


def normalize_retail_columns(df):
    """
    Standardize common Online Retail column names across workbook variants.
    This keeps the core functions intact while making both local files and
    online downloads compatible with the existing RFM pipeline.
    """
    column_map = {
        'Customer ID': 'CustomerID',
        'CustomerID': 'CustomerID',
        'Invoice': 'InvoiceNo',
        'InvoiceNo': 'InvoiceNo',
        'Price': 'UnitPrice',
        'UnitPrice': 'UnitPrice',
        'unit_price': 'UnitPrice',
    }
    return df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})


def find_dataset_file(candidates=None):
    """Look for a known Online Retail workbook in the current workspace."""
    candidate_list = candidates or ['online_retail_II.xlsx', 'Online Retail.xlsx', 'OnlineRetail.xlsx', 'online_retail.xlsx']
    for name in candidate_list:
        path = Path(name)
        if path.exists():
            return str(path)
    return None


def prepare_retail_dataset(file_path='Online Retail.xlsx'):
    """
    Prepare the Online Retail dataset for customer segmentation using clustering.
    """
    try:
        # Try loading as Excel file first (since it might be .xlsx)
        print("Loading dataset as Excel file...")
        df = pd.read_excel(file_path)
        print(f"✓ Dataset loaded successfully! Shape: {df.shape}")
        df = normalize_retail_columns(df)
        
    except Exception as e:
        print(f"⚠️  Error loading as Excel: {e}")
        
        # Try loading as CSV
        try:
            print("Attempting to load as CSV...")
            df = pd.read_csv(file_path, encoding='latin1')
            print(f"✓ Dataset loaded successfully! Shape: {df.shape}")
            
        except Exception as e2:
            print(f"⚠️  Error loading as CSV: {e2}")
            
            # If both fail, let's download the dataset
            print("Downloading dataset from UCI repository...")
            import requests
            import io
            
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
            response = requests.get(url)
            
            if response.status_code == 200:
                # Save the downloaded file
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"✓ Dataset downloaded and saved to {file_path}")
                
                # Try loading again
                df = pd.read_excel(file_path)
                print(f"✓ Dataset loaded successfully! Shape: {df.shape}")
            else:
                raise Exception("Could not download dataset from UCI repository")
    
    # Clean the dataset
    print("\n2. Cleaning dataset...")
    
    # Remove rows with missing CustomerID
    df_clean = df.dropna(subset=['CustomerID'])
    print(f"✓ Removed rows without CustomerID. New shape: {df_clean.shape}")
    
    # Remove negative quantities and zero unit prices
    df_clean = df_clean[df_clean['Quantity'] > 0]
    df_clean = df_clean[df_clean['UnitPrice'] > 0]
    print(f"✓ Removed negative/zero quantities and prices. New shape: {df_clean.shape}")
    
    # Convert CustomerID to int
    df_clean['CustomerID'] = df_clean['CustomerID'].astype(int)
    
    # Create TotalPrice column
    df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']
    
    # Convert InvoiceDate to datetime
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
    
    # Calculate Recency, Frequency, and Monetary (RFM) metrics
    print("\n3. Calculating RFM metrics...")
    
    current_date = df_clean['InvoiceDate'].max()
    
    # Recency: days since last purchase
    recency_df = df_clean.groupby('CustomerID')['InvoiceDate'].max().reset_index()
    recency_df['Recency'] = (current_date - recency_df['InvoiceDate']).dt.days
    
    # Frequency: number of transactions
    frequency_df = df_clean.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
    frequency_df.columns = ['CustomerID', 'Frequency']
    
    # Monetary: total amount spent
    monetary_df = df_clean.groupby('CustomerID')['TotalPrice'].sum().reset_index()
    monetary_df.columns = ['CustomerID', 'Monetary']
    
    # Merge all RFM metrics
    customer_features = recency_df.merge(frequency_df, on='CustomerID')
    customer_features = customer_features.merge(monetary_df, on='CustomerID')
    
    print(f"✓ RFM metrics calculated for {len(customer_features)} customers")
    print(f"Features: Recency, Frequency, Monetary")
    
    # Additional features: average order value
    customer_features['AvgOrderValue'] = customer_features['Monetary'] / customer_features['Frequency']
    
    # Log transform monetary and frequency for better distribution
    customer_features['LogMonetary'] = np.log1p(customer_features['Monetary'])
    customer_features['LogFrequency'] = np.log1p(customer_features['Frequency'])
    
    print("\n4. Feature Engineering complete. Final features:")
    print(customer_features.head())
    
    # Prepare features for clustering
    print("\n5. Preparing features for clustering...")
    
    # Select features for clustering
    X = customer_features[['Recency', 'Frequency', 'Monetary', 'AvgOrderValue']].copy()
    
    # Add log-transformed features
    X['LogMonetary'] = customer_features['LogMonetary']
    X['LogFrequency'] = customer_features['LogFrequency']
    
    print(f"✓ Features ready: {X.shape}")
    print(f"Features: {X.columns.tolist()}")
    
    # Scale the features
    print("\n6. Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"✓ Features scaled using StandardScaler")
    
    # Apply PCA for dimensionality reduction
    print("\n7. Applying PCA...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"✓ PCA applied. Explained variance ratio: {pca.explained_variance_ratio_.sum():.2%}")
    
    print("\n" + "="*50)
    print("✅ Dataset preparation complete!")
    print("="*50)
    
    return customer_features, X, X_scaled, X_pca

def find_optimal_clusters(X_scaled, max_k=10):
    """
    Find the optimal number of clusters using Elbow method and Silhouette score.
    """
    print("\n🔍 Finding optimal number of clusters...")
    
    inertias = []
    silhouette_scores = []
    K_range = range(2, max_k + 1)
    
    from sklearn.metrics import silhouette_score
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        
        # Silhouette score for k>1
        if k >= 2:
            score = silhouette_score(X_scaled, kmeans.labels_)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(0)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Elbow plot
    ax1.plot(K_range, inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.grid(True, alpha=0.3)
    
    # Silhouette plot
    ax2.plot(K_range, silhouette_scores, 'ro-')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score for Optimal k')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Recommend k
    best_k_silhouette = K_range[np.argmax(silhouette_scores)]
    print(f"\n📊 Recommended k based on Silhouette Score: {best_k_silhouette}")
    print(f"   (Highest Silhouette Score: {max(silhouette_scores):.3f})")
    
    return best_k_silhouette

# Run the preparation
if __name__ == "__main__":
    print("="*50)
    print("Preparing Online Retail Dataset for Clustering")
    print("="*50)
    
    # Try to load with automatic fallback
    try:
        # Try alternative file names, including the local workbook used in this workspace
        file_names = ['online_retail_II.xlsx', 'Online Retail.xlsx', 'OnlineRetail.xlsx', 'online_retail.xlsx']
        customer_features, X, X_scaled, X_pca = None, None, None, None
        
        for file in file_names:
            try:
                customer_features, X, X_scaled, X_pca = prepare_retail_dataset(file)
                break
            except Exception as e:
                print(f"Could not load {file}: {e}")
                continue

        if customer_features is None:
            discovered_file = find_dataset_file(file_names)
            if discovered_file:
                customer_features, X, X_scaled, X_pca = prepare_retail_dataset(discovered_file)
        
        if customer_features is None:
            # Download the dataset
            print("Downloading dataset...")
            import requests
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
            response = requests.get(url)
            with open('Online Retail.xlsx', 'wb') as f:
                f.write(response.content)
            
            customer_features, X, X_scaled, X_pca = prepare_retail_dataset('Online Retail.xlsx')
        
        # Check the data
        print(f"\n📊 Customer Features Summary:")
        print(customer_features.describe())
        
        # Find optimal clusters
        optimal_k = find_optimal_clusters(X_scaled)
        
        # Apply K-means with optimal k
        print(f"\n🎯 Applying K-means with k={optimal_k}...")
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        customer_features['Cluster'] = labels
        
        print(f"✅ Clustering complete! Added 'Cluster' column to customer_features")
        print(f"\n📊 Cluster Distribution:")
        print(customer_features['Cluster'].value_counts().sort_index())
        
        # Show cluster centers
        centers_scaled = kmeans.cluster_centers_
        scaler = StandardScaler()
        scaler.fit(X)
        centers_original = scaler.inverse_transform(centers_scaled)
        
        print(f"\n📊 Cluster Centers (Original Scale):")
        cluster_centers_df = pd.DataFrame(centers_original, columns=X.columns)
        print(cluster_centers_df)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have internet connection to download the dataset")
        print("2. Install required packages: pip install pandas openpyxl requests scikit-learn matplotlib seaborn")
        print("3. If using a local file, ensure it's in the correct directory")
        print("4. The file should be 'Online Retail.xlsx' from UCI Machine Learning Repository")