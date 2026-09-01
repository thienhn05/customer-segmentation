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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

from prep import prepare_retail_dataset
from segmentation_profiles import create_final_segment_profiles


def evaluate_kmeans_range(X_scaled, min_k=2, max_k=12):
    """
    Evaluate K-Means across cluster counts from min_k to max_k.
    Computes Silhouette Score, Davies-Bouldin Index, and Inertia.
    """
    results = []
    ks = list(range(min_k, max_k + 1))

    for k in ks:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        sil = float(silhouette_score(X_scaled, labels))
        dbi = float(davies_bouldin_score(X_scaled, labels))
        inertia = float(model.inertia_)

        results.append({
            'k': k,
            'silhouette': sil,
            'davies_bouldin': dbi,
            'inertia': inertia,
            'labels': labels,
            'model': model,
        })

    return results


def select_best_kmeans(results):
    """
    Select best K using Silhouette Score as primary criterion,
    with Davies-Bouldin Index as supporting tie-breaker.
    """
    if not results:
        return None
    return max(results, key=lambda x: (x['silhouette'], -x['davies_bouldin']))


def train_kmeans_model(X_scaled, k):
    """Train single K-Means model with timed execution."""
    start_time = time.time()
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    elapsed = time.time() - start_time
    return model, labels, elapsed


def plot_kmeans_selection(results, best_k, output_path='final_artifacts/kmeans_selection.png'):
    """Plot Elbow (Inertia) and Silhouette curves for model selection."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ks = [r['k'] for r in results]
    inertias = [r['inertia'] for r in results]
    silhouettes = [r['silhouette'] for r in results]
    dbis = [r['davies_bouldin'] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Elbow curve
    axes[0].plot(ks, inertias, 'bo-', linewidth=2, markersize=7)
    axes[0].axvline(best_k, linestyle='--', color='tab:red', linewidth=1.8, label=f'Best K = {best_k}')
    axes[0].set_title('K-Means Elbow Method (Inertia vs K)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Number of Clusters (K)', fontsize=10)
    axes[0].set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 2. Silhouette Score
    axes[1].plot(ks, silhouettes, 'ro-', linewidth=2, markersize=7)
    axes[1].axvline(best_k, linestyle='--', color='tab:green', linewidth=1.8, label=f'Best K = {best_k}')
    axes[1].set_title('Silhouette Score by K (Higher is Better)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Number of Clusters (K)', fontsize=10)
    axes[1].set_ylabel('Silhouette Score', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # 3. Davies-Bouldin Index
    axes[2].plot(ks, dbis, 'go-', linewidth=2, markersize=7)
    axes[2].axvline(best_k, linestyle='--', color='tab:blue', linewidth=1.8, label=f'Best K = {best_k}')
    axes[2].set_title('Davies-Bouldin Index by K (Lower is Better)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Number of Clusters (K)', fontsize=10)
    axes[2].set_ylabel('Davies-Bouldin Index', fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"✓ Saved K-Means selection plot to {output_path}")


def run_kmeans_analysis(customer_features=None, X_scaled=None, X_pca=None):
    """Run full standalone K-Means analysis."""
    if customer_features is None or X_scaled is None or X_pca is None:
        customer_features, X, X_scaled, X_pca, metadata = prepare_retail_dataset()

    print("\n=== RUNNING K-MEANS CLUSTERING ANALYSIS ===")
    results = evaluate_kmeans_range(X_scaled, min_k=2, max_k=12)
    best_result = select_best_kmeans(results)
    best_k = best_result['k']

    print(f"\nOptimal K-Means Selection: K = {best_k}")
    print(f"  Silhouette Score:     {best_result['silhouette']:.4f}")
    print(f"  Davies-Bouldin Index: {best_result['davies_bouldin']:.4f}")
    print(f"  Inertia:              {best_result['inertia']:.2f}")

    plot_kmeans_selection(results, best_k)

    # Final profiles
    profiles = create_final_segment_profiles(customer_features, best_result['labels'])
    print("\nCustomer Segment Profiles:")
    print(profiles[['ClusterID', 'SegmentName', 'CustomerCount', 'Percentage',
                    'Recency_Median', 'Frequency_Median', 'Monetary_Median']].to_string(index=False))

    return {
        'best_k': best_k,
        'best_result': best_result,
        'all_results': results,
        'profiles': profiles,
    }


if __name__ == '__main__':
    run_kmeans_analysis()
