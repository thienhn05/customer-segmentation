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
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score

from prep import prepare_retail_dataset
from segmentation_profiles import create_final_segment_profiles


def evaluate_gmm_candidates(X_scaled, component_range=None, covariance_types=None):
    """
    Evaluate Gaussian Mixture Models across component counts (2 to 12)
    and all four covariance types (full, tied, diag, spherical).
    """
    if component_range is None:
        component_range = list(range(2, 13))
    if covariance_types is None:
        covariance_types = ['full', 'tied', 'diag', 'spherical']

    results = []

    for cov_type in covariance_types:
        for n_comp in component_range:
            try:
                gmm = GaussianMixture(
                    n_components=n_comp,
                    covariance_type=cov_type,
                    random_state=42,
                    n_init=5,
                    reg_covar=1e-6,
                )
                labels = gmm.fit_predict(X_scaled)

                if len(np.unique(labels)) < 2:
                    continue

                sil = float(silhouette_score(X_scaled, labels))
                dbi = float(davies_bouldin_score(X_scaled, labels))
                aic = float(gmm.aic(X_scaled))
                bic = float(gmm.bic(X_scaled))

                results.append({
                    'covariance_type': cov_type,
                    'n_components': n_comp,
                    'silhouette': sil,
                    'davies_bouldin': dbi,
                    'aic': aic,
                    'bic': bic,
                    'labels': labels,
                    'model': gmm,
                })
            except Exception as e:
                print(f"  ⚠️ Warning: GMM failed for {cov_type} with {n_comp} components: {e}")
                continue

    return results


def select_best_gmm(results):
    """
    Unified GMM model selection policy:
    Primary metric: Silhouette Score (for fair cross-model comparison)
    Supporting metrics: Davies-Bouldin Index (lower is better), BIC, AIC.
    """
    if not results:
        return None
    return max(
        results,
        key=lambda x: (
            x['silhouette'],
            -x['davies_bouldin'],
            -x['bic'],
            -x['aic']
        )
    )


def train_gmm_model(X_scaled, n_components, covariance_type='spherical'):
    """Train a single GMM model with timing and convergence check."""
    start_time = time.time()
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=42,
        n_init=5,
        reg_covar=1e-6,
    )
    labels = gmm.fit_predict(X_scaled)
    elapsed = time.time() - start_time
    return gmm, labels, elapsed


def analyze_gmm_uncertainty(gmm_model, X_scaled, threshold=0.60):
    """
    Analyze soft clustering probabilities:
    - Probability matrix for each customer across clusters
    - Maximum assignment confidence
    - Ambiguous customers with max probability below threshold
    """
    probs = gmm_model.predict_proba(X_scaled)
    max_probs = probs.max(axis=1)
    ambiguous_idx = np.where(max_probs < threshold)[0]
    ambiguous_count = len(ambiguous_idx)
    ambiguous_pct = (ambiguous_count / len(X_scaled)) * 100

    return {
        'probabilities': probs,
        'max_probabilities': max_probs,
        'ambiguous_indices': ambiguous_idx,
        'ambiguous_count': ambiguous_count,
        'ambiguous_percentage': round(ambiguous_pct, 2),
        'mean_max_probability': round(float(np.mean(max_probs)), 4),
        'median_max_probability': round(float(np.median(max_probs)), 4),
        'threshold': threshold,
    }


def plot_gmm_selection(results, output_path='final_artifacts/gmm_selection.png'):
    """Plot AIC, BIC, Silhouette, and DBI across covariance types."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_res = pd.DataFrame(results)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Silhouette Score by Covariance & Components
    for cov in df_res['covariance_type'].unique():
        sub = df_res[df_res['covariance_type'] == cov].sort_values('n_components')
        axes[0, 0].plot(sub['n_components'], sub['silhouette'], marker='o', label=cov.title(), linewidth=2)
    axes[0, 0].set_title('GMM Silhouette Score by Components (Higher is Better)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Number of Components', fontsize=10)
    axes[0, 0].set_ylabel('Silhouette Score', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # 2. Davies-Bouldin Index
    for cov in df_res['covariance_type'].unique():
        sub = df_res[df_res['covariance_type'] == cov].sort_values('n_components')
        axes[0, 1].plot(sub['n_components'], sub['davies_bouldin'], marker='o', label=cov.title(), linewidth=2)
    axes[0, 1].set_title('GMM Davies-Bouldin Index (Lower is Better)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Number of Components', fontsize=10)
    axes[0, 1].set_ylabel('Davies-Bouldin Index', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # 3. AIC Curve
    for cov in df_res['covariance_type'].unique():
        sub = df_res[df_res['covariance_type'] == cov].sort_values('n_components')
        axes[1, 0].plot(sub['n_components'], sub['aic'], marker='o', label=cov.title(), linewidth=2)
    axes[1, 0].set_title('GMM Akaike Information Criterion (AIC)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Number of Components', fontsize=10)
    axes[1, 0].set_ylabel('AIC Score', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # 4. BIC Curve
    for cov in df_res['covariance_type'].unique():
        sub = df_res[df_res['covariance_type'] == cov].sort_values('n_components')
        axes[1, 1].plot(sub['n_components'], sub['bic'], marker='o', label=cov.title(), linewidth=2)
    axes[1, 1].set_title('GMM Bayesian Information Criterion (BIC)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Number of Components', fontsize=10)
    axes[1, 1].set_ylabel('BIC Score', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"✓ Saved GMM selection plot to {output_path}")


def plot_gmm_probability_histogram(max_probs, threshold=0.60, output_path='final_artifacts/gmm_probability_histogram.png'):
    """Plot distribution of maximum cluster membership probabilities."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(max_probs, bins=35, color='royalblue', edgecolor='black', alpha=0.85)
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Ambiguity Threshold ({threshold:.2f})')
    ax.set_title('GMM Soft Clustering: Maximum Membership Probability Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Maximum Assignment Probability', fontsize=10)
    ax.set_ylabel('Number of Customers', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"✓ Saved GMM probability histogram to {output_path}")


def run_gmm_segmentation(customer_features=None, X_scaled=None, X_pca=None):
    """Run full standalone GMM analysis."""
    if customer_features is None or X_scaled is None or X_pca is None:
        customer_features, X, X_scaled, X_pca, metadata = prepare_retail_dataset()

    print("\n=== RUNNING GAUSSIAN MIXTURE MODEL (GMM) ANALYSIS ===")
    results = evaluate_gmm_candidates(X_scaled)
    best_result = select_best_gmm(results)

    print(f"\nOptimal GMM Configuration:")
    print(f"  Covariance Type:      {best_result['covariance_type']}")
    print(f"  Number of Components: {best_result['n_components']}")
    print(f"  Silhouette Score:     {best_result['silhouette']:.4f}")
    print(f"  Davies-Bouldin Index: {best_result['davies_bouldin']:.4f}")
    print(f"  AIC:                  {best_result['aic']:.1f}")
    print(f"  BIC:                  {best_result['bic']:.1f}")

    uncertainty = analyze_gmm_uncertainty(best_result['model'], X_scaled, threshold=0.60)
    print(f"\nSoft Clustering Diagnostics:")
    print(f"  Mean Confidence:      {uncertainty['mean_max_probability']:.2%}")
    print(f"  Ambiguous Customers:  {uncertainty['ambiguous_count']:,} ({uncertainty['ambiguous_percentage']:.2f}%)")

    plot_gmm_selection(results)
    plot_gmm_probability_histogram(uncertainty['max_probabilities'])

    profiles = create_final_segment_profiles(customer_features, best_result['labels'])
    print("\nCustomer Segment Profiles:")
    print(profiles[['ClusterID', 'SegmentName', 'CustomerCount', 'Percentage',
                    'Recency_Median', 'Frequency_Median', 'Monetary_Median']].to_string(index=False))

    return {
        'best_result': best_result,
        'all_results': results,
        'uncertainty': uncertainty,
        'profiles': profiles,
    }


if __name__ == '__main__':
    run_gmm_segmentation()
