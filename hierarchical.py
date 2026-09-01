import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

from prep import prepare_retail_dataset
from segmentation_profiles import create_final_segment_profiles


def evaluate_hierarchical_candidates(X_scaled, k_range=None, methods=None, metrics=None):
    """
    Evaluate Hierarchical Agglomerative Clustering across cluster counts (2 to 12)
    and valid linkage / distance metric combinations.
    """
    if k_range is None:
        k_range = list(range(2, 13))
    if methods is None:
        methods = ['ward', 'complete', 'average', 'single']
    if metrics is None:
        metrics = ['euclidean', 'manhattan', 'cosine']

    results = []

    for method in methods:
        valid_metrics = ['euclidean'] if method == 'ward' else metrics
        for metric in valid_metrics:
            for k in k_range:
                try:
                    model = AgglomerativeClustering(
                        n_clusters=k,
                        metric=metric,
                        linkage=method,
                    )
                    labels = model.fit_predict(X_scaled)

                    if len(np.unique(labels)) < 2:
                        continue

                    sil = float(silhouette_score(X_scaled, labels))
                    dbi = float(davies_bouldin_score(X_scaled, labels))

                    results.append({
                        'method': method,
                        'metric': metric,
                        'n_clusters': k,
                        'silhouette': sil,
                        'davies_bouldin': dbi,
                        'labels': labels,
                        'model': model,
                    })
                except Exception as e:
                    print(f"  ⚠️ Warning: Hierarchical failed for {method}/{metric} with k={k}: {e}")
                    continue

    return results


def select_best_hierarchical(results, prefer_balanced=True):
    """
    Select best hierarchical configuration.
    If prefer_balanced is True, filters out degenerate clusterings where
    a single outlier cluster has < 5% of the data (e.g. single/average linkage artifacts)
    to select the best actionable business clustering (Ward / Euclidean).
    """
    if not results:
        return None

    candidate_pool = results
    if prefer_balanced:
        balanced_pool = []
        for r in results:
            counts = pd.Series(r['labels']).value_counts()
            # Require smallest cluster to have at least 5% of total dataset
            if counts.min() >= len(r['labels']) * 0.05:
                balanced_pool.append(r)
        if balanced_pool:
            candidate_pool = balanced_pool

    return max(
        candidate_pool,
        key=lambda x: (
            x['silhouette'],
            -x['davies_bouldin']
        )
    )


def build_linkage_matrix(X, method='ward', metric='euclidean'):
    """Compute scipy linkage matrix with metric translation."""
    scipy_metric = 'cityblock' if metric == 'manhattan' else metric
    if method == 'ward':
        return linkage(X, method='ward', metric='euclidean')
    else:
        return linkage(X, method=method, metric=scipy_metric)


def calculate_cophenetic_correlation(X, methods=None, metrics=None):
    """
    Calculate Cophenetic Correlation Coefficient for linkage/metric setups.
    Measures how faithfully the tree preserves pairwise distances.
    """
    if methods is None:
        methods = ['ward', 'complete', 'average', 'single']
    if metrics is None:
        metrics = ['euclidean', 'manhattan', 'cosine']

    coph_results = []
    for method in methods:
        valid_metrics = ['euclidean'] if method == 'ward' else metrics
        for metric in valid_metrics:
            try:
                scipy_metric = 'cityblock' if metric == 'manhattan' else metric
                Z = build_linkage_matrix(X, method=method, metric=metric)
                dists = pdist(X, metric=scipy_metric)
                c, _ = cophenet(Z, dists)
                coph_results.append({
                    'method': method,
                    'metric': metric,
                    'cophenetic_correlation': round(float(c), 4),
                    'linkage_matrix': Z,
                })
            except Exception as e:
                print(f"  ⚠️ Cophenetic calculation failed for {method}/{metric}: {e}")
                continue

    return coph_results


def plot_dendrogram(linkage_matrix=None, X_scaled=None, n_clusters=2, sample_size=150, output_path='final_artifacts/dendrogram.png'):
    """
    Generate a clear, presentation-quality dendrogram using a stratified representative
    sample of customers for visual clarity while the model clusters the full dataset.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if linkage_matrix is None and X_scaled is not None:
        # Take reproducible stratified sample for clean visualization
        np.random.seed(42)
        idx_sample = np.random.choice(len(X_scaled), size=min(sample_size, len(X_scaled)), replace=False)
        X_sample = X_scaled[idx_sample]
        linkage_matrix = build_linkage_matrix(X_sample, method='ward', metric='euclidean')

    fig, ax = plt.subplots(figsize=(14, 7))
    cut_threshold = linkage_matrix[-n_clusters + 1, 2] if n_clusters > 1 else linkage_matrix[-1, 2]

    dendrogram(
        linkage_matrix,
        ax=ax,
        color_threshold=cut_threshold,
        above_threshold_color='navy',
        no_labels=True,
    )
    ax.axhline(
        y=cut_threshold,
        color='crimson',
        linestyle='--',
        linewidth=2,
        label=f'Optimal Cut Level (K = {n_clusters}, Distance = {cut_threshold:.2f})',
    )
    ax.set_title('Hierarchical Agglomerative Dendrogram (Ward Linkage, Euclidean Distance)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Customer Subtree Clusters', fontsize=11)
    ax.set_ylabel('Euclidean Merge Distance (Ward Criterion)', fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"✓ Saved Hierarchical dendrogram plot to {output_path}")


def plot_hierarchical_selection(results, best_config, output_path='final_artifacts/hierarchical_selection.png'):
    """Plot Silhouette and DBI comparison across hierarchical configurations."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_res = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Ward Linkage Curve across K
    ward_df = df_res[df_res['method'] == 'ward'].sort_values('n_clusters')
    if not ward_df.empty:
        axes[0].plot(ward_df['n_clusters'], ward_df['silhouette'], 'bo-', linewidth=2, label='Ward (Euclidean)')
        axes[0].axvline(best_config['n_clusters'], linestyle='--', color='red', label=f"Best K = {best_config['n_clusters']}")
        axes[0].set_title('Hierarchical Silhouette Score by Cluster Count (K)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Number of Clusters (K)', fontsize=10)
        axes[0].set_ylabel('Silhouette Score', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(ward_df['n_clusters'], ward_df['davies_bouldin'], 'go-', linewidth=2, label='Ward (Euclidean)')
        axes[1].axvline(best_config['n_clusters'], linestyle='--', color='red', label=f"Best K = {best_config['n_clusters']}")
        axes[1].set_title('Hierarchical Davies-Bouldin Index by K', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Number of Clusters (K)', fontsize=10)
        axes[1].set_ylabel('Davies-Bouldin Index', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"✓ Saved Hierarchical selection plot to {output_path}")


def run_hierarchical_clustering(customer_features=None, X_scaled=None, X_pca=None):
    """Run full standalone Hierarchical clustering analysis."""
    if customer_features is None or X_scaled is None or X_pca is None:
        customer_features, X, X_scaled, X_pca, metadata = prepare_retail_dataset()

    print("\n=== RUNNING HIERARCHICAL AGGLOMERATIVE CLUSTERING ANALYSIS ===")
    results = evaluate_hierarchical_candidates(X_scaled)
    best_result = select_best_hierarchical(results, prefer_balanced=True)

    print(f"\nOptimal Hierarchical Configuration:")
    print(f"  Linkage Method:       {best_result['method']}")
    print(f"  Distance Metric:      {best_result['metric']}")
    print(f"  Number of Clusters:   {best_result['n_clusters']}")
    print(f"  Silhouette Score:     {best_result['silhouette']:.4f}")
    print(f"  Davies-Bouldin Index: {best_result['davies_bouldin']:.4f}")

    # Cophenetic correlations
    coph_results = calculate_cophenetic_correlation(X_scaled)
    print("\nCophenetic Correlation Summary:")
    for r in coph_results:
        print(f"  {r['method'].title():10s} / {r['metric'].title():10s}: {r['cophenetic_correlation']:.4f}")

    # Plots
    plot_dendrogram(X_scaled=X_scaled, n_clusters=best_result['n_clusters'])
    plot_hierarchical_selection(results, best_result)

    profiles = create_final_segment_profiles(customer_features, best_result['labels'])
    print("\nCustomer Segment Profiles:")
    print(profiles[['ClusterID', 'SegmentName', 'CustomerCount', 'Percentage',
                    'Recency_Median', 'Frequency_Median', 'Monetary_Median']].to_string(index=False))

    return {
        'best_result': best_result,
        'all_results': results,
        'cophenetic_results': coph_results,
        'profiles': profiles,
    }


if __name__ == '__main__':
    run_hierarchical_clustering()
