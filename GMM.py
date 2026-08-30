import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import davies_bouldin_score, silhouette_score

from prep import prepare_retail_dataset


def evaluate_gmm_candidates(X_scaled, max_components=12, covariance_types=None):
    """Evaluate Gaussian Mixture models across component counts and covariance types."""
    if covariance_types is None:
        covariance_types = ['full', 'tied', 'diag', 'spherical']

    aic_results = {}
    bic_results = {}
    best_models = {}

    for cov_type in covariance_types:
        aic_scores = []
        bic_scores = []
        fitted_models = []

        for n_components in range(2, max_components + 1):
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=cov_type,
                random_state=42,
                n_init=5,
                reg_covar=1e-6,
            )
            gmm.fit(X_scaled)
            aic_scores.append(gmm.aic(X_scaled))
            bic_scores.append(gmm.bic(X_scaled))
            fitted_models.append(gmm)

        aic_results[cov_type] = aic_scores
        bic_results[cov_type] = bic_scores
        best_models[cov_type] = fitted_models

    return aic_results, bic_results, best_models, list(range(2, max_components + 1)), covariance_types


def select_best_gmm(aic_results, bic_results, component_range, covariance_types):
    """Select the best covariance type and number of components using AIC/BIC minima."""
    best = {'covariance_type': None, 'n_components': None, 'aic': np.inf, 'bic': np.inf}

    for cov_type in covariance_types:
        for n_idx, n_components in enumerate(component_range):
            aic_score = aic_results[cov_type][n_idx]
            bic_score = bic_results[cov_type][n_idx]

            if aic_score < best['aic']:
                best['aic'] = aic_score
                best['covariance_type'] = cov_type
                best['n_components'] = n_components

            if bic_score < best['bic']:
                best['bic'] = bic_score
                best['bic_covariance_type'] = cov_type
                best['bic_n_components'] = n_components

    return best


def plot_aic_bic_grid(aic_results, bic_results, component_range, covariance_types):
    """Create a grid of AIC/BIC plots for each covariance type."""
    fig, axes = plt.subplots(len(covariance_types), 2, figsize=(15, 12))

    for idx, cov_type in enumerate(covariance_types):
        ax_aic = axes[idx, 0]
        ax_bic = axes[idx, 1]

        ax_aic.plot(component_range, aic_results[cov_type], marker='o', linewidth=2)
        ax_aic.set_title(f'{cov_type.title()} Covariance - AIC')
        ax_aic.set_xlabel('Number of Components')
        ax_aic.set_ylabel('AIC')
        ax_aic.grid(True, alpha=0.3)

        ax_bic.plot(component_range, bic_results[cov_type], marker='o', linewidth=2, color='tab:orange')
        ax_bic.set_title(f'{cov_type.title()} Covariance - BIC')
        ax_bic.set_xlabel('Number of Components')
        ax_bic.set_ylabel('BIC')
        ax_bic.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gmm_aic_bic_grid.png', dpi=200)
    plt.close(fig)


def train_gmm_model(X_scaled, n_components, covariance_type='full'):
    """Train a Gaussian Mixture Model and time the training process."""
    start_time = time.time()
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=42,
        n_init=5,
        reg_covar=1e-6,
    )
    labels = gmm.fit_predict(X_scaled)
    elapsed_time = time.time() - start_time
    return gmm, labels, elapsed_time


def calculate_hard_metrics(X_scaled, labels):
    """Calculate clustering evaluation metrics from hard assignments."""
    metrics = {
        'silhouette_score': silhouette_score(X_scaled, labels),
        'davies_bouldin_index': davies_bouldin_score(X_scaled, labels),
    }
    return metrics


def analyze_uncertainty(gmm_model, X_scaled):
    """Calculate cluster membership probabilities and identify ambiguous customers."""
    probabilities = gmm_model.predict_proba(X_scaled)
    max_prob = probabilities.max(axis=1)
    ambiguous_idx = np.where(max_prob < 0.60)[0]

    uncertainty_summary = {
        'probability_matrix': probabilities,
        'max_probability': max_prob,
        'ambiguous_customers': ambiguous_idx,
        'ambiguous_count': len(ambiguous_idx),
        'threshold': 0.60,
    }
    return uncertainty_summary


def plot_probability_histogram(max_probability):
    """Plot the distribution of maximum cluster membership probabilities."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(max_probability, bins=30, color='steelblue', edgecolor='black')
    ax.axvline(0.60, color='red', linestyle='--', linewidth=1.5, label='Ambiguity threshold = 0.60')
    ax.set_title('Distribution of Maximum Membership Probability')
    ax.set_xlabel('Maximum Probability')
    ax.set_ylabel('Number of Customers')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('gmm_max_probability_histogram.png', dpi=200)
    plt.close(fig)


def plot_pca_clusters(X_pca, labels):
    """Create a PCA scatter plot with cluster assignments."""
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=35, alpha=0.8)
    ax.set_title('GMM Cluster Visualization in PCA Space')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Cluster ID')
    plt.tight_layout()
    plt.savefig('gmm_pca_cluster_visualization.png', dpi=200)
    plt.close(fig)


def plot_cluster_profiles(customer_features, labels, clustering_features):
    """Show average values of each feature by cluster."""
    df = customer_features.copy()
    df['Cluster'] = labels
    cluster_profile = df.groupby('Cluster')[clustering_features].mean().round(2)

    fig, ax = plt.subplots(figsize=(12, 6))
    cluster_profile.T.plot(kind='bar', ax=ax)
    ax.set_title('GMM Cluster Profiles')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Average Value')
    ax.legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('gmm_cluster_profiles.png', dpi=200)
    plt.close(fig)
    return cluster_profile


def label_cluster(row, recency_median, frequency_median, monetary_median, avg_order_median):
    """Assign a business-friendly label to each cluster based on feature patterns."""
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
    """Create a summary table of cluster statistics and interpretation."""
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


def compare_with_kmeans(customer_features, gmm_labels, kmeans_labels=None):
    """Compare GMM results with K-Means if labels are available."""
    if kmeans_labels is None:
        return None

    comparison = pd.DataFrame({
        'CustomerID': customer_features['CustomerID'],
        'GMM_Cluster': gmm_labels,
        'KMeans_Cluster': kmeans_labels,
    })
    return comparison


def run_gmm_segmentation(customer_features, X_scaled, X_pca, clustering_features=None, max_components=12, covariance_types=None):
    """Run the full GMM workflow and save outputs."""
    if clustering_features is None:
        clustering_features = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'LogMonetary', 'LogFrequency']

    if covariance_types is None:
        covariance_types = ['full', 'tied', 'diag', 'spherical']

    print('\n=== GAUSSIAN MIXTURE MODEL ANALYSIS ===')
    aic_results, bic_results, _, component_range, covariance_types = evaluate_gmm_candidates(
        X_scaled,
        max_components=max_components,
        covariance_types=covariance_types,
    )

    plot_aic_bic_grid(aic_results, bic_results, component_range, covariance_types)

    best = select_best_gmm(aic_results, bic_results, component_range, covariance_types)
    print('\nOptimal model selection:')
    print(f"  Best covariance type by AIC: {best['covariance_type']} with {best['n_components']} components")
    print(f"  Best covariance type by BIC: {best.get('bic_covariance_type')} with {best.get('bic_n_components')} components")

    selected_cov_type = best['covariance_type']
    selected_components = best['n_components']

    print(f'\nTraining GMM with {selected_components} components and covariance type={selected_cov_type}...')
    start = time.time()
    gmm_model, labels, training_time = train_gmm_model(X_scaled, selected_components, selected_cov_type)
    elapsed = time.time() - start

    metrics = calculate_hard_metrics(X_scaled, labels)
    print('\nModel evaluation:')
    print(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
    print(f"  Davies-Bouldin Index: {metrics['davies_bouldin_index']:.4f}")
    print(f"  Training time: {training_time:.4f} seconds")
    print(f"  Converged: {gmm_model.converged_}")

    uncertainty = analyze_uncertainty(gmm_model, X_scaled)
    probability_df = pd.DataFrame(
        uncertainty['probability_matrix'],
        columns=[f'Cluster_{i}' for i in range(uncertainty['probability_matrix'].shape[1])],
    )
    probability_df['CustomerID'] = customer_features['CustomerID'].values
    probability_df['MaxProbability'] = uncertainty['max_probability']
    probability_df.to_csv('gmm_probability_matrix.csv', index=False)

    plot_probability_histogram(uncertainty['max_probability'])
    plot_pca_clusters(X_pca, labels)
    cluster_profile = plot_cluster_profiles(customer_features, labels, clustering_features)
    cluster_summary = summarize_clusters(customer_features, labels)

    cluster_summary.to_csv('gmm_cluster_summary.csv', index=False)
    print('\nCluster summary:')
    print(cluster_summary.to_string(index=False))

    ambiguous_customers = probability_df.loc[probability_df['MaxProbability'] < 0.60, ['CustomerID', 'MaxProbability']]
    print(f'\nAmbiguous customers (MaxProbability < 0.60): {len(ambiguous_customers)}')
    if not ambiguous_customers.empty:
        print(ambiguous_customers.head().to_string(index=False))

    return {
        'gmm_model': gmm_model,
        'labels': labels,
        'metrics': metrics,
        'training_time': training_time,
        'cluster_summary': cluster_summary,
        'probability_matrix': probability_df,
        'ambiguous_customers': ambiguous_customers,
        'optimal_components': selected_components,
        'optimal_covariance_type': selected_cov_type,
        'aic_results': aic_results,
        'bic_results': bic_results,
    }


if __name__ == '__main__':
    print('=' * 60)
    print('Gaussian Mixture Model Customer Segmentation')
    print('=' * 60)

    try:
        customer_features, X, X_scaled, X_pca = prepare_retail_dataset('online_retail_II.xlsx')
        clustering_features = list(X.columns)
        run_gmm_segmentation(customer_features, X_scaled, X_pca, clustering_features=clustering_features)
    except Exception as exc:
        print(f'ERROR: {exc}')
        raise
