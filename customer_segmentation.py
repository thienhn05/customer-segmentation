import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score

from prep import prepare_retail_dataset


def evaluate_k_range(X_scaled, max_k=12):
    """Evaluate K values from 2 to max_k and return inertia and silhouette scores."""
    ks = list(range(2, max_k + 1))
    inertias = []
    silhouette_scores = []

    for k in ks:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        inertias.append(model.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))

    return ks, inertias, silhouette_scores


def choose_optimal_k(ks, inertias, silhouette_scores):
    """Select K using elbow and silhouette heuristics."""
    inertia_array = np.array(inertias)
    if len(inertia_array) >= 3:
        elbow_deltas = np.abs(np.diff(inertia_array, n=2))
        elbow_k = ks[int(np.argmax(elbow_deltas))]
    else:
        elbow_k = ks[0]

    silhouette_k = ks[int(np.argmax(silhouette_scores))]
    return elbow_k, silhouette_k


def plot_optimal_k_selection(ks, inertias, silhouette_scores, elbow_k, silhouette_k):
    """Plot the elbow curve and silhouette scores and save the result."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(ks, inertias, 'bo-')
    ax1.axvline(elbow_k, linestyle='--', color='tab:red', linewidth=1.5)
    ax1.set_title('Elbow Method')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(ks, silhouette_scores, 'ro-')
    ax2.axvline(silhouette_k, linestyle='--', color='tab:green', linewidth=1.5)
    ax2.set_title('Silhouette Scores')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('optimal_k_selection.png', dpi=200)
    plt.close(fig)


def train_kmeans_model(X_scaled, k):
    """Train and time the K-Means model."""
    start = time.time()
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    elapsed = time.time() - start
    return model, labels, elapsed


def compute_clustering_metrics(X_scaled, labels, model):
    """Compute evaluation metrics for cluster quality."""
    return {
        'n_clusters': len(np.unique(labels)),
        'silhouette_score': silhouette_score(X_scaled, labels),
        'davies_bouldin_index': davies_bouldin_score(X_scaled, labels),
        'inertia': model.inertia_,
    }


def plot_pca_clusters(X_pca, labels):
    """Create a PCA scatter plot showing cluster assignments."""
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=35, alpha=0.8)
    ax.set_title('PCA Visualization of Customer Clusters')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Cluster ID')
    plt.tight_layout()
    plt.savefig('pca_cluster_visualization.png', dpi=200)
    plt.close(fig)


def plot_cluster_profiles(customer_features, labels, clustering_features):
    """Plot mean feature values for each cluster."""
    df = customer_features.copy()
    df['Cluster'] = labels
    cluster_means = df.groupby('Cluster')[clustering_features].mean().T

    fig, ax = plt.subplots(figsize=(12, 6))
    cluster_means.plot(kind='bar', ax=ax)
    ax.set_title('Cluster Profiles by Feature Mean')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Average Value')
    ax.legend(title='Cluster')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('cluster_profiles.png', dpi=200)
    plt.close(fig)
    return df.groupby('Cluster')[clustering_features].mean().round(2)


def plot_cluster_sizes(labels):
    """Bar chart of cluster sizes."""
    counts = pd.Series(labels).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind='bar', color='steelblue', ax=ax)
    ax.set_title('Cluster Size Distribution')
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Number of Customers')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('cluster_sizes.png', dpi=200)
    plt.close(fig)
    return counts


def label_cluster(row, recency_median, frequency_median, monetary_median, avg_order_median):
    """Assign a business-friendly label to a cluster."""
    recency = row['Recency']
    frequency = row['Frequency']
    monetary = row['Monetary']
    avg_order = row['AvgOrderValue']

    if frequency >= frequency_median and monetary >= monetary_median:
        return 'High-value frequent customers'
    if recency >= recency_median and frequency < frequency_median:
        return 'Low-frequency inactive customers'
    if avg_order >= avg_order_median and monetary >= monetary_median:
        return 'Premium high-ticket buyers'
    if recency < recency_median and frequency >= frequency_median:
        return 'Recent loyal buyers'
    return 'Moderate spenders'


def summarize_clusters(customer_features, labels):
    """Build a summary table showing cluster interpretation and customer counts."""
    summary_df = customer_features.copy()
    summary_df['Cluster'] = labels

    feature_cols = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue']
    cluster_summary = summary_df.groupby('Cluster')[feature_cols].mean().round(2)
    cluster_summary['Customers'] = summary_df['Cluster'].value_counts().sort_index().values

    medians = summary_df[feature_cols].median()
    cluster_summary['Interpretation'] = cluster_summary.apply(
        lambda row: label_cluster(
            row,
            medians['Recency'],
            medians['Frequency'],
            medians['Monetary'],
            medians['AvgOrderValue'],
        ),
        axis=1,
    )

    cluster_summary = cluster_summary.reset_index().rename(columns={'Cluster': 'ClusterID'})
    cluster_summary['ClusterID'] = cluster_summary['ClusterID'].astype(int)
    return cluster_summary


def run_customer_segmentation(file_path='online_retail_II.xlsx', max_k=12):
    """Run the full customer segmentation flow using the prepared retail dataset."""
    try:
        print('Loading prepared retail dataset...')
        customer_features, X, X_scaled, X_pca = prepare_retail_dataset(file_path)

        clustering_features = list(X.columns)
        ks, inertias, silhouette_scores = evaluate_k_range(X_scaled, max_k=max_k)
        elbow_k, silhouette_k = choose_optimal_k(ks, inertias, silhouette_scores)
        recommended_k = silhouette_k

        print('\nOptimal K values:')
        print(f' - Elbow method: {elbow_k}')
        print(f' - Silhouette method: {silhouette_k}')
        plot_optimal_k_selection(ks, inertias, silhouette_scores, elbow_k, silhouette_k)

        print(f'\nTraining K-Means with k={recommended_k}...')
        model, labels, training_time = train_kmeans_model(X_scaled, recommended_k)

        metrics = compute_clustering_metrics(X_scaled, labels, model)
        print('\nClustering metrics:')
        print(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
        print(f"  Davies-Bouldin Index: {metrics['davies_bouldin_index']:.4f}")
        print(f"  Inertia: {metrics['inertia']:.2f}")
        print(f"  Training time: {training_time:.4f} seconds")

        result_df = customer_features.copy()
        result_df['Cluster'] = labels
        result_df.to_csv('customer_segments.csv', index=False)

        cluster_summary = summarize_clusters(result_df, labels)
        cluster_summary.to_csv('cluster_summary.csv', index=False)

        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv('clustering_metrics.csv', index=False)

        plot_pca_clusters(X_pca, labels)
        plot_cluster_profiles(result_df, labels, clustering_features)
        plot_cluster_sizes(labels)

        print('\nCluster summary:')
        print(cluster_summary.to_string(index=False))

        return {
            'customer_features': result_df,
            'cluster_summary': cluster_summary,
            'metrics': metrics,
            'model': model,
            'labels': labels,
            'recommended_k': recommended_k,
            'elbow_k': elbow_k,
            'silhouette_k': silhouette_k,
            'training_time': training_time,
        }

    except Exception as exc:
        print(f'ERROR: {exc}')
        raise


if __name__ == '__main__':
    print('=' * 60)
    print('Customer Segmentation with K-Means')
    print('=' * 60)
    run_customer_segmentation(file_path='online_retail_II.xlsx', max_k=12)
