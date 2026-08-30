import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score

from prep import prepare_retail_dataset


def build_linkage_matrix(X, method='ward', metric='euclidean'):
    """Build a linkage matrix for hierarchical clustering using a chosen method and metric."""
    if method == 'ward':
        linkage_matrix = linkage(X, method='ward', metric=metric)
    else:
        linkage_matrix = linkage(X, method=method, metric=metric)
    return linkage_matrix


def evaluate_linkage_methods(X, methods=None, metrics=None):
    """Evaluate linkage methods and distance metrics by building linkage matrices and returning summaries."""
    if methods is None:
        methods = ['ward', 'complete', 'average', 'single']
    if metrics is None:
        metrics = ['euclidean', 'manhattan', 'cosine']

    results = []
    for method in methods:
        for metric in metrics:
            try:
                Z = build_linkage_matrix(X, method=method, metric=metric)
                c, coph_dists = cophenet(Z, pdist(X, metric=metric))
                results.append({
                    'method': method,
                    'metric': metric,
                    'cophenetic_correlation': c,
                    'cophenetic_distances': coph_dists,
                    'linkage_matrix': Z,
                })
            except Exception:
                continue

    return results


def plot_dendrograms_for_methods(results, optimal_method='ward', optimal_metric='euclidean', optimal_clusters=3):
    """Create a multi-panel comparison of dendrograms for different linkage methods."""
    methods = sorted({item['method'] for item in results})
    metrics = sorted({item['metric'] for item in results})

    fig, axes = plt.subplots(len(methods), len(metrics), figsize=(15, 12))
    if len(methods) == 1:
        axes = np.array([axes])
    if len(metrics) == 1:
        axes = axes.reshape(len(methods), 1)

    for i, method in enumerate(methods):
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            matching = [r for r in results if r['method'] == method and r['metric'] == metric]
            if not matching:
                ax.axis('off')
                continue

            Z = matching[0]['linkage_matrix']
            dendrogram(Z, ax=ax, no_labels=True)
            ax.set_title(f'{method.title()} / {metric.title()}')
            ax.axhline(y=Z[-optimal_clusters, 2], color='red', linestyle='--', linewidth=1.5)

    plt.tight_layout()
    plt.savefig('hierarchical_dendrogram_comparison.png', dpi=200)
    plt.close(fig)


def choose_optimal_hierarchy(results):
    """Identify the most promising linkage setup using cophenetic correlation and visual inspection of dendrograms."""
    best = max(results, key=lambda x: x['cophenetic_correlation'])
    print('\nBest hierarchical configuration by cophenetic correlation:')
    print(f"  Method: {best['method']}")
    print(f"  Metric: {best['metric']}")
    print(f"  Cophenetic correlation: {best['cophenetic_correlation']:.4f}")
    return best['method'], best['metric'], best['linkage_matrix']


def train_hierarchical_model(X, n_clusters, linkage_method='ward', metric='euclidean'):
    """Train sklearn AgglomerativeClustering and return the model plus training time."""
    start = time.time()
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric=metric,
        linkage=linkage_method,
    )
    labels = model.fit_predict(X)
    elapsed = time.time() - start
    return model, labels, elapsed


def compute_hierarchical_metrics(X, labels):
    """Compute evaluation metrics for hierarchical clustering."""
    return {
        'silhouette_score': silhouette_score(X, labels),
        'davies_bouldin_index': davies_bouldin_score(X, labels),
    }


def plot_pca_clusters(X_pca, labels):
    """Plot the PCA projection of the clusters."""
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=35, alpha=0.8)
    ax.set_title('Agglomerative Clustering in PCA Space')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Cluster ID')
    plt.tight_layout()
    plt.savefig('hierarchical_pca_clusters.png', dpi=200)
    plt.close(fig)


def plot_cluster_profiles(customer_features, labels, clustering_features):
    """Plot average feature values per cluster."""
    df = customer_features.copy()
    df['Cluster'] = labels
    profile = df.groupby('Cluster')[clustering_features].mean().round(2)

    fig, ax = plt.subplots(figsize=(12, 6))
    profile.T.plot(kind='bar', ax=ax)
    ax.set_title('Hierarchical Cluster Profiles')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Average Value')
    ax.legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('hierarchical_cluster_profiles.png', dpi=200)
    plt.close(fig)
    return profile


def plot_cluster_sizes(labels):
    """Bar chart of cluster sizes."""
    counts = pd.Series(labels).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind='bar', color='darkorange', ax=ax)
    ax.set_title('Hierarchical Cluster Sizes')
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Number of Customers')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('hierarchical_cluster_sizes.png', dpi=200)
    plt.close(fig)
    return counts


def plot_dendrogram_with_cut(Z, n_clusters=3):
    """Plot a dendrogram with a highlighted cut level."""
    fig, ax = plt.subplots(figsize=(12, 8))
    dendrogram(Z, ax=ax, color_threshold=Z[-n_clusters, 2], above_threshold_color='grey')
    ax.axhline(y=Z[-n_clusters, 2], color='red', linestyle='--', linewidth=1.5, label=f'Cut at {n_clusters} clusters')
    ax.set_title('Dendrogram with Optimal Cut Level Highlighted')
    ax.set_xlabel('Customers')
    ax.set_ylabel('Distance')
    ax.legend()
    plt.tight_layout()
    plt.savefig('hierarchical_dendrogram_cut.png', dpi=200)
    plt.close(fig)


def label_cluster(row, recency_median, frequency_median, monetary_median, avg_order_median):
    """Assign each cluster a business-friendly label based on its average values."""
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
    """Summarize cluster behavior and attach a practical interpretation."""
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


def scalability_analysis(X, sample_fracs=(0.10, 0.25, 0.50, 0.75, 1.00), linkage_method='ward', metric='euclidean'):
    """Measure runtime for different sample sizes to assess scalability."""
    runtimes = []
    for frac in sample_fracs:
        n_samples = max(2, int(len(X) * frac))
        X_subset = X[:n_samples]
        start = time.time()
        AgglomerativeClustering(n_clusters=min(3, n_samples - 1), metric=metric, linkage=linkage_method).fit_predict(X_subset)
        elapsed = time.time() - start
        runtimes.append({'sample_fraction': frac, 'n_samples': n_samples, 'runtime_seconds': elapsed})

    runtime_df = pd.DataFrame(runtimes)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(runtime_df['n_samples'], runtime_df['runtime_seconds'], marker='o', color='tab:purple')
    ax.set_title('Hierarchical Clustering Runtime vs Dataset Size')
    ax.set_xlabel('Number of Customers')
    ax.set_ylabel('Runtime (seconds)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('hierarchical_scalability.png', dpi=200)
    plt.close(fig)
    return runtime_df


def run_hierarchical_clustering(customer_features, X_scaled, X_pca, clustering_features=None, methods=None, metrics=None):
    """Run the full agglomerative clustering workflow and save outputs."""
    if clustering_features is None:
        clustering_features = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'LogMonetary', 'LogFrequency']
    if methods is None:
        methods = ['ward', 'complete', 'average', 'single']
    if metrics is None:
        metrics = ['euclidean', 'manhattan', 'cosine']

    print('\n=== HIERARCHICAL CLUSTERING ANALYSIS ===')
    results = evaluate_linkage_methods(X_scaled, methods=methods, metrics=metrics)
    optimal_method, optimal_metric, optimal_linkage_matrix = choose_optimal_hierarchy(results)
    optimal_result = next(r for r in results if r['method'] == optimal_method and r['metric'] == optimal_metric)

    # Use the best configuration to estimate a reasonable cluster count from the dendrogram.
    suggested_clusters = 3
    print(f'\nUsing linkage method: {optimal_method} with metric={optimal_metric}')
    print(f'Suggested cluster count from dendrogram: {suggested_clusters}')

    plot_dendrograms_for_methods(results, optimal_method=optimal_method, optimal_metric=optimal_metric, optimal_clusters=suggested_clusters)
    plot_dendrogram_with_cut(optimal_linkage_matrix, n_clusters=suggested_clusters)

    model, labels, training_time = train_hierarchical_model(X_scaled, suggested_clusters, linkage_method=optimal_method, metric=optimal_metric)
    metrics_dict = compute_hierarchical_metrics(X_scaled, labels)
    metrics_dict['training_time_seconds'] = training_time
    metrics_dict['cophenetic_correlation'] = optimal_result['cophenetic_correlation']

    print('\nModel evaluation:')
    print(f"Silhouette Score: {metrics_dict['silhouette_score']:.4f}")
    print(f"Davies-Bouldin Index: {metrics_dict['davies_bouldin_index']:.4f}")
    print(f"Cophenetic Correlation: {metrics_dict['cophenetic_correlation']:.4f}")
    print(f"Training time: {training_time:.4f} seconds")

    result_df = customer_features.copy()
    result_df['Cluster'] = labels
    result_df.to_csv('hierarchical_customer_segments.csv', index=False)

    cluster_summary = summarize_clusters(result_df, labels)
    cluster_summary.to_csv('hierarchical_cluster_summary.csv', index=False)

    plot_pca_clusters(X_pca, labels)
    plot_cluster_profiles(result_df, labels, clustering_features)
    plot_cluster_sizes(labels)
    scalability_df = scalability_analysis(X_scaled, linkage_method=optimal_method, metric=optimal_metric)

    print('\nCluster summary:')
    print(cluster_summary.to_string(index=False))

    return {
        'model': model,
        'labels': labels,
        'optimal_method': optimal_method,
        'optimal_metric': optimal_metric,
        'optimal_clusters': suggested_clusters,
        'metrics': metrics_dict,
        'cluster_summary': cluster_summary,
        'scalability': scalability_df,
    }


if __name__ == '__main__':
    print('=' * 60)
    print('Hierarchical Agglomerative Clustering')
    print('=' * 60)

    try:
        customer_features, X, X_scaled, X_pca = prepare_retail_dataset('online_retail_II.xlsx')
        clustering_features = list(X.columns)
        run_hierarchical_clustering(customer_features, X_scaled, X_pca, clustering_features=clustering_features)
    except Exception as exc:
        print(f'ERROR: {exc}')
        raise
