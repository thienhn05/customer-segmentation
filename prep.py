import os
import io
import zipfile
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Ensure writable matplotlib config in sandboxed environments
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl')


def get_canonical_dataset_path():
    """Return local path to the canonical Online Retail dataset file."""
    candidates = [
        'Online Retail.xlsx',
        'online_retail.xlsx',
        'data/Online Retail.xlsx',
        'data/Online_Retail.xlsx',
        'online_retail_II.xlsx',
    ]
    for c in candidates:
        if Path(c).exists() and Path(c).stat().st_size > 100000:
            return c
    return 'Online Retail.xlsx'


def download_dataset(target_path='Online Retail.xlsx'):
    """
    Download the canonical Online Retail dataset from verified sources
    with automatic mirror fallback and local caching.
    """
    urls = [
        'https://raw.githubusercontent.com/dipanjanS/practical-machine-learning-with-python/master/notebooks/Ch08_Customer_Segmentation_and_Effective_Cross_Selling/Online%20Retail.xlsx',
        'https://raw.githubusercontent.com/nelsoncardenas/Customer-segmentation-on-Online-Retail-Data-Set/master/Online%20Retail.xlsx',
        'https://archive.ics.uci.edu/static/public/352/online+retail.zip',
        'https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx',
    ]

    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Online Retail dataset to {target_path}...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

    for url in urls:
        try:
            print(f"  Attempting download from: {url}")
            response = requests.get(url, headers=headers, timeout=45)
            if response.status_code == 200 and len(response.content) > 100000:
                if url.endswith('.zip') or url.endswith('.zip?raw=true'):
                    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                        for name in z.namelist():
                            if name.endswith('.xlsx') or name.endswith('.csv'):
                                with open(target, 'wb') as f:
                                    f.write(z.read(name))
                                print(f"✓ Extracted and saved {name} to {target_path}")
                                return str(target)
                else:
                    with open(target, 'wb') as f:
                        f.write(response.content)
                    print(f"✓ Saved dataset directly to {target_path}")
                    return str(target)
        except Exception as e:
            print(f"  ⚠️ Warning: Mirror failed ({e}), trying next source...")

    raise RuntimeError(
        "Could not download Online Retail dataset from any mirror. "
        "Please check your internet connection or place 'Online Retail.xlsx' in the workspace."
    )


def normalize_retail_columns(df):
    """Standardize column names across workbook variants."""
    column_map = {
        'Customer ID': 'CustomerID',
        'CustomerID': 'CustomerID',
        'Invoice': 'InvoiceNo',
        'InvoiceNo': 'InvoiceNo',
        'Price': 'UnitPrice',
        'UnitPrice': 'UnitPrice',
        'unit_price': 'UnitPrice',
        'Stock Code': 'StockCode',
        'StockCode': 'StockCode',
        'Invoice Date': 'InvoiceDate',
        'InvoiceDate': 'InvoiceDate',
    }
    return df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})


def load_raw_dataset(file_path=None):
    """
    Load raw Online Retail transaction dataset with robust fallback
    and fast parquet caching.
    """
    cache_path = Path('data/online_retail_raw.parquet')
    if cache_path.exists() and file_path is None:
        try:
            df = pd.read_parquet(cache_path)
            return normalize_retail_columns(df)
        except Exception:
            pass

    if file_path is None or not Path(file_path).exists():
        file_path = get_canonical_dataset_path()
        if not Path(file_path).exists():
            file_path = download_dataset(file_path)

    path = Path(file_path)
    if not path.exists():
        file_path = download_dataset(str(path))
        path = Path(file_path)

    print(f"Loading raw dataset from {file_path}...")
    if path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    elif path.suffix.lower() == '.parquet':
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path, encoding='latin1')

    df = normalize_retail_columns(df)

    # Save to fast cache if not existing
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
    except Exception:
        pass

    return df


def clean_retail_transactions(df):
    """
    Clean transaction dataset:
    - Remove rows with missing CustomerID
    - Remove cancellations and invalid quantities (Quantity <= 0)
    - Remove zero or negative unit prices (UnitPrice <= 0)
    - Parse datetime and calculate TotalPrice
    """
    initial_count = len(df)

    # Missing CustomerID removal
    df_clean = df.dropna(subset=['CustomerID']).copy()
    missing_customer_count = initial_count - len(df_clean)

    # Convert CustomerID to integer
    df_clean['CustomerID'] = df_clean['CustomerID'].astype(int)

    # Filter invalid quantities and unit prices
    valid_mask = (df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)
    invalid_trx_count = len(df_clean) - valid_mask.sum()
    df_clean = df_clean[valid_mask].copy()

    # Convert InvoiceDate to datetime
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])

    # Calculate TotalPrice
    df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']

    # Remove exact duplicate transaction lines if any
    duplicate_count = df_clean.duplicated().sum()
    if duplicate_count > 0:
        df_clean = df_clean.drop_duplicates().copy()

    cleaning_stats = {
        'initial_transactions': initial_count,
        'missing_customer_rows': missing_customer_count,
        'invalid_transactions_removed': invalid_trx_count,
        'duplicates_removed': int(duplicate_count),
        'usable_transactions': len(df_clean),
        'min_date': str(df_clean['InvoiceDate'].min()),
        'max_date': str(df_clean['InvoiceDate'].max()),
    }

    return df_clean, cleaning_stats


def calculate_rfm_metrics(df_clean, reference_date=None):
    """
    Aggregate transaction data by CustomerID into RFM behavioral features:
    - Recency: Days since last transaction (relative to snapshot date)
    - Frequency: Number of unique purchase invoices
    - Monetary: Total spending amount
    - AvgOrderValue: Monetary / Frequency
    """
    if reference_date is None:
        reference_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)
    else:
        reference_date = pd.to_datetime(reference_date)

    rfm = df_clean.groupby('CustomerID').agg({
        'InvoiceDate': lambda d: (reference_date - d.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum',
    }).reset_index()

    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    rfm['AvgOrderValue'] = (rfm['Monetary'] / rfm['Frequency']).round(2)

    return rfm


def engineer_clustering_features(rfm_df):
    """
    Inspect skewness and apply log1p transformations to create
    a statistically sound clustering matrix without multicollinear duplication.
    """
    customer_features = rfm_df.copy()

    # Log1p transforms for skewed positive metrics
    customer_features['LogRecency'] = np.log1p(customer_features['Recency'])
    customer_features['LogFrequency'] = np.log1p(customer_features['Frequency'])
    customer_features['LogMonetary'] = np.log1p(customer_features['Monetary'])
    customer_features['LogAvgOrderValue'] = np.log1p(customer_features['AvgOrderValue'])

    # Skewness calculation
    skewness_raw = {
        'Recency': float(customer_features['Recency'].skew()),
        'Frequency': float(customer_features['Frequency'].skew()),
        'Monetary': float(customer_features['Monetary'].skew()),
        'AvgOrderValue': float(customer_features['AvgOrderValue'].skew()),
    }
    skewness_log = {
        'LogRecency': float(customer_features['LogRecency'].skew()),
        'LogFrequency': float(customer_features['LogFrequency'].skew()),
        'LogMonetary': float(customer_features['LogMonetary'].skew()),
        'LogAvgOrderValue': float(customer_features['LogAvgOrderValue'].skew()),
    }

    # Final feature set for clustering: Log-transformed RFM
    # Avoids collinear duplication of Monetary/Frequency and AvgOrderValue
    clustering_cols = ['LogRecency', 'LogFrequency', 'LogMonetary']
    X = customer_features[clustering_cols].copy()

    return customer_features, X, skewness_raw, skewness_log


def prepare_retail_dataset(file_path=None):
    """
    Main canonical pipeline function.
    Returns:
    - customer_features: DataFrame with CustomerID, RFM, AvgOrderValue, and Log features
    - X: DataFrame with the exact clustering features (LogRecency, LogFrequency, LogMonetary)
    - X_scaled: StandardScaler transformed numpy matrix
    - X_pca: 2D PCA transformed numpy matrix for visualization
    - metadata: Comprehensive dictionary of preprocessing metrics and dataset details
    """
    df_raw = load_raw_dataset(file_path)
    df_clean, cleaning_stats = clean_retail_transactions(df_raw)
    rfm_df = calculate_rfm_metrics(df_clean)
    customer_features, X, skew_raw, skew_log = engineer_clustering_features(rfm_df)

    # Standard scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    pca_variance_ratio = pca.explained_variance_ratio_.tolist()
    pca_total_variance = float(pca.explained_variance_ratio_.sum())

    metadata = {
        'dataset_name': 'Online Retail Dataset',
        'dataset_source': 'UCI Machine Learning Repository (Daqing Chen, 2015)',
        'doi': '10.24432/C5BW33',
        'time_period': f"{cleaning_stats['min_date'][:10]} to {cleaning_stats['max_date'][:10]}",
        'initial_transactions': cleaning_stats['initial_transactions'],
        'missing_customer_rows': cleaning_stats['missing_customer_rows'],
        'invalid_transactions_removed': cleaning_stats['invalid_transactions_removed'],
        'duplicates_removed': cleaning_stats['duplicates_removed'],
        'usable_transactions': cleaning_stats['usable_transactions'],
        'usable_customers': len(customer_features),
        'clustering_features': list(X.columns),
        'untransformed_features': ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue'],
        'skewness_raw': skew_raw,
        'skewness_log': skew_log,
        'scaler_type': 'StandardScaler (zero mean, unit variance)',
        'pca_n_components': 2,
        'pca_explained_variance_ratio': pca_variance_ratio,
        'pca_total_explained_variance': pca_total_variance,
    }

    return customer_features, X, X_scaled, X_pca, metadata


if __name__ == '__main__':
    print("=" * 60)
    print("Testing Canonical Preprocessing Pipeline")
    print("=" * 60)
    customer_features, X, X_scaled, X_pca, metadata = prepare_retail_dataset()
    print(f"✓ Usable customers: {len(customer_features):,}")
    print(f"✓ Clustering features: {list(X.columns)}")
    print(f"✓ PCA Total Explained Variance: {metadata['pca_total_explained_variance']:.2%}")
    print(f"✓ Sample customer features:\n{customer_features.head(3)}")