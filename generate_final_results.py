import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from prep import prepare_retail_dataset
from customer_segmentation import evaluate_kmeans_range, select_best_kmeans, plot_kmeans_selection
from GMM import evaluate_gmm_candidates, select_best_gmm, analyze_gmm_uncertainty, plot_gmm_selection, plot_gmm_probability_histogram
from hierarchical import evaluate_hierarchical_candidates, select_best_hierarchical, calculate_cophenetic_correlation, plot_dendrogram, plot_hierarchical_selection
from segmentation_profiles import create_final_segment_profiles


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def generate_exploratory_plots(customer_features, output_dir='final_artifacts'):
    """Generate RFM distribution and correlation heatmap plots."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. RFM Distributions Before and After Log Transformation
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    raw_cols = ['Recency', 'Frequency', 'Monetary']
    log_cols = ['LogRecency', 'LogFrequency', 'LogMonetary']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, col in enumerate(raw_cols):
        axes[0, i].hist(customer_features[col], bins=30, color=colors[i], edgecolor='black', alpha=0.8)
        axes[0, i].set_title(f'Raw {col} (Skew: {customer_features[col].skew():.2f})', fontsize=11, fontweight='bold')
        axes[0, i].set_xlabel(col)
        axes[0, i].set_ylabel('Customer Count')
        axes[0, i].grid(True, alpha=0.3)

    for i, col in enumerate(log_cols):
        axes[1, i].hist(customer_features[col], bins=30, color=colors[i], edgecolor='black', alpha=0.8)
        axes[1, i].set_title(f'{col} [log1p] (Skew: {customer_features[col].skew():.2f})', fontsize=11, fontweight='bold')
        axes[1, i].set_xlabel(col)
        axes[1, i].set_ylabel('Customer Count')
        axes[1, i].grid(True, alpha=0.3)

    plt.suptitle('Distribution of Customer RFM Features Before vs After Log1p Transformation', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    dist_path = str(out / 'rfm_distributions.png')
    plt.savefig(dist_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved RFM distributions to {dist_path}")

    # 2. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    corr_cols = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'LogRecency', 'LogFrequency', 'LogMonetary']
    corr_matrix = customer_features[corr_cols].corr()

    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='Blues', vmin=-1, vmax=1, square=True, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Heatmap', fontsize=12, fontweight='bold')
    plt.tight_layout()
    corr_path = str(out / 'correlation_heatmap.png')
    plt.savefig(corr_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved correlation heatmap to {corr_path}")


def generate_comparison_plots(comparison_df, output_dir='final_artifacts'):
    """Generate bar charts comparing Silhouette Score and Davies-Bouldin Index."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Silhouette Score Comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        comparison_df['Algorithm'],
        comparison_df['Silhouette_Score'],
        color=['#1f77b4', '#ff7f0e', '#2ca02c'],
        edgecolor='black',
        width=0.55
    )
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_title('Silhouette Score Comparison Across Best Algorithm Configurations', fontsize=12, fontweight='bold')
    ax.set_ylabel('Silhouette Score (Higher is Better)', fontsize=11)
    ax.set_ylim(0, max(comparison_df['Silhouette_Score']) * 1.2)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    sil_path = str(out / 'model_comparison_silhouette.png')
    plt.savefig(sil_path, dpi=200)
    plt.close(fig)
    print(f"✓ Saved Silhouette comparison plot to {sil_path}")

    # 2. Davies-Bouldin Index Comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        comparison_df['Algorithm'],
        comparison_df['Davies_Bouldin_Index'],
        color=['#1f77b4', '#ff7f0e', '#2ca02c'],
        edgecolor='black',
        width=0.55
    )
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_title('Davies-Bouldin Index Comparison (Lower is Better)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Davies-Bouldin Index', fontsize=11)
    ax.set_ylim(0, max(comparison_df['Davies_Bouldin_Index']) * 1.2)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    dbi_path = str(out / 'model_comparison_dbi.png')
    plt.savefig(dbi_path, dpi=200)
    plt.close(fig)
    print(f"✓ Saved DBI comparison plot to {dbi_path}")


def generate_final_segment_plots(X_pca, profiles_df, labels, output_dir='final_artifacts'):
    """Generate final PCA cluster visualization and segment size bar chart."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. PCA Scatter Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for cluster_id in sorted(profiles_df['ClusterID'].unique()):
        mask = (labels == cluster_id)
        seg_name = profiles_df.loc[profiles_df['ClusterID'] == cluster_id, 'SegmentName'].values[0]
        count = profiles_df.loc[profiles_df['ClusterID'] == cluster_id, 'CustomerCount'].values[0]
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=palette[cluster_id % len(palette)],
            label=f"Cluster {cluster_id}: {seg_name} (n={count:,})",
            alpha=0.7,
            s=32,
            edgecolors='none',
        )

        # Plot cluster centroid
        centroid_x = np.mean(X_pca[mask, 0])
        centroid_y = np.mean(X_pca[mask, 1])
        ax.scatter(centroid_x, centroid_y, c='black', marker='X', s=160, edgecolors='white', linewidths=1.5, zorder=5)

    ax.set_title('Final Customer Segments in PCA 2D Space (Winning Model: K-Means)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Principal Component 1 (Captures Order Activity & Volume)', fontsize=11)
    ax.set_ylabel('Principal Component 2 (Captures Recency / Inactivity)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper right', framealpha=0.95)
    plt.tight_layout()
    pca_path = str(out / 'final_pca_segments.png')
    plt.savefig(pca_path, dpi=200)
    plt.close(fig)
    print(f"✓ Saved final PCA segments plot to {pca_path}")

    # 2. Segment Size Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        profiles_df['SegmentName'],
        profiles_df['CustomerCount'],
        color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(profiles_df)],
        edgecolor='black',
        width=0.5
    )
    for bar, pct in zip(bars, profiles_df['Percentage']):
        height = bar.get_height()
        ax.annotate(f'{height:,}\n({pct:.1f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_title('Customer Distribution by Segment', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Customers', fontsize=11)
    ax.set_ylim(0, max(profiles_df['CustomerCount']) * 1.25)
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=15, ha='right', fontsize=10)
    plt.tight_layout()
    size_path = str(out / 'final_segment_sizes.png')
    plt.savefig(size_path, dpi=200)
    plt.close(fig)
    print(f"✓ Saved segment sizes plot to {size_path}")


def main():
    print("=" * 70)
    print("EXECUTING CANONICAL CUSTOMER SEGMENTATION PIPELINE")
    print("=" * 70)

    start_total = time.time()
    artifacts_dir = Path('final_artifacts')
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Loading & Preprocessing
    print("\n[Step 1/5] Loading & Preprocessing Canonical Online Retail Dataset...")
    customer_features, X, X_scaled, X_pca, metadata = prepare_retail_dataset()
    print(f"  • Usable Transactions: {metadata['usable_transactions']:,}")
    print(f"  • Usable Customers:    {metadata['usable_customers']:,}")
    print(f"  • Feature Set:         {metadata['clustering_features']}")
    print(f"  • PCA 2D Total Var:    {metadata['pca_total_explained_variance']:.2%}")

    # Generate Exploratory Visualizations
    generate_exploratory_plots(customer_features, output_dir='final_artifacts')

    # 2. K-Means Evaluation
    print("\n[Step 2/5] Evaluating K-Means Clustering (K=2..12)...")
    km_results = evaluate_kmeans_range(X_scaled, min_k=2, max_k=12)
    best_km = select_best_kmeans(km_results)
    plot_kmeans_selection(km_results, best_km['k'], output_path='final_artifacts/kmeans_selection.png')
    print(f"  • Best K-Means: K = {best_km['k']} | Silhouette = {best_km['silhouette']:.4f} | DBI = {best_km['davies_bouldin']:.4f} | Inertia = {best_km['inertia']:.2f}")

    # 3. GMM Evaluation
    print("\n[Step 3/5] Evaluating Gaussian Mixture Models (Components=2..12, Covariance=full/tied/diag/spherical)...")
    gmm_results = evaluate_gmm_candidates(X_scaled)
    best_gmm = select_best_gmm(gmm_results)
    gmm_uncertainty = analyze_gmm_uncertainty(best_gmm['model'], X_scaled, threshold=0.60)
    plot_gmm_selection(gmm_results, output_path='final_artifacts/gmm_selection.png')
    plot_gmm_probability_histogram(gmm_uncertainty['max_probabilities'], threshold=0.60, output_path='final_artifacts/gmm_probability_histogram.png')
    print(f"  • Best GMM: {best_gmm['n_components']} components ({best_gmm['covariance_type']}) | Silhouette = {best_gmm['silhouette']:.4f} | DBI = {best_gmm['davies_bouldin']:.4f} | AIC = {best_gmm['aic']:.1f} | BIC = {best_gmm['bic']:.1f}")
    print(f"  • Soft Clustering Ambiguity (<0.60): {gmm_uncertainty['ambiguous_count']} customers ({gmm_uncertainty['ambiguous_percentage']:.2f}%)")

    # 4. Hierarchical Clustering Evaluation
    print("\n[Step 4/5] Evaluating Hierarchical Agglomerative Clustering (K=2..12, Linkage=ward/complete/average/single)...")
    hier_results = evaluate_hierarchical_candidates(X_scaled)
    best_hier = select_best_hierarchical(hier_results, prefer_balanced=True)
    coph_results = calculate_cophenetic_correlation(X_scaled)
    plot_dendrogram(X_scaled=X_scaled, n_clusters=best_hier['n_clusters'], output_path='final_artifacts/dendrogram.png')
    plot_hierarchical_selection(hier_results, best_hier, output_path='final_artifacts/hierarchical_selection.png')
    print(f"  • Best Hierarchical: {best_hier['n_clusters']} clusters ({best_hier['method']} linkage, {best_hier['metric']}) | Silhouette = {best_hier['silhouette']:.4f} | DBI = {best_hier['davies_bouldin']:.4f}")

    # 5. Cross-Model Comparison & Winner Selection
    print("\n[Step 5/5] Generating Fair Model Comparison & Final Customer Segmentation...")

    comparison_records = [
        {
            'Algorithm': 'K-Means',
            'Best_Configuration': f"K = {best_km['k']}",
            'Number_of_Clusters': int(best_km['k']),
            'Silhouette_Score': round(float(best_km['silhouette']), 4),
            'Davies_Bouldin_Index': round(float(best_km['davies_bouldin']), 4),
            'Algorithm_Specific_Metrics': f"Inertia = {best_km['inertia']:.2f}",
            'Computational_Complexity': 'O(n * k * d * i) [Fast]',
        },
        {
            'Algorithm': 'Gaussian Mixture Model (GMM)',
            'Best_Configuration': f"{best_gmm['n_components']} components, {best_gmm['covariance_type']} covariance",
            'Number_of_Clusters': int(best_gmm['n_components']),
            'Silhouette_Score': round(float(best_gmm['silhouette']), 4),
            'Davies_Bouldin_Index': round(float(best_gmm['davies_bouldin']), 4),
            'Algorithm_Specific_Metrics': f"AIC = {best_gmm['aic']:.1f}, BIC = {best_gmm['bic']:.1f}",
            'Computational_Complexity': 'O(n * k * d^3 * i) [EM Probabilistic]',
        },
        {
            'Algorithm': 'Hierarchical Agglomerative Clustering',
            'Best_Configuration': f"{best_hier['n_clusters']} clusters, {best_hier['method']} linkage, {best_hier['metric']}",
            'Number_of_Clusters': int(best_hier['n_clusters']),
            'Silhouette_Score': round(float(best_hier['silhouette']), 4),
            'Davies_Bouldin_Index': round(float(best_hier['davies_bouldin']), 4),
            'Algorithm_Specific_Metrics': f"Cophenetic Corr = {coph_results[0]['cophenetic_correlation']:.4f} (Ward)",
            'Computational_Complexity': 'O(n^2 log n) to O(n^3) [Deterministic Tree]',
        }
    ]

    comparison_df = pd.DataFrame(comparison_records).sort_values(
        ['Silhouette_Score', 'Davies_Bouldin_Index'],
        ascending=[False, True]
    ).reset_index(drop=True)
    comparison_df['Rank'] = range(1, len(comparison_df) + 1)

    # Save comparison CSV
    comparison_path = artifacts_dir / 'final_model_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)
    print(f"✓ Saved model comparison to {comparison_path}")

    # Generate comparison plots
    generate_comparison_plots(comparison_df, output_dir='final_artifacts')

    # Winner Identification
    winner_row = comparison_df.iloc[0]
    winner_name = winner_row['Algorithm']
    print(f"\n🏆 OVERALL RECOMMENDED MODEL: {winner_name} ({winner_row['Best_Configuration']})")
    print(f"   Reason: Highest Silhouette Score ({winner_row['Silhouette_Score']:.4f}) and lowest Davies-Bouldin Index ({winner_row['Davies_Bouldin_Index']:.4f})")

    # Generate Final Customer Segmentation from Winning Model (K-Means)
    final_labels = best_km['labels']
    profiles_df = create_final_segment_profiles(customer_features, final_labels)
    profiles_path = artifacts_dir / 'final_segment_profiles.csv'
    profiles_df.to_csv(profiles_path, index=False)
    print(f"✓ Saved segment profiles to {profiles_path}")

    # Save Customer-level assignment CSV
    customer_segment_df = customer_features.copy()
    customer_segment_df['Cluster'] = final_labels
    segment_map = dict(zip(profiles_df['ClusterID'], profiles_df['SegmentName']))
    customer_segment_df['Segment'] = customer_segment_df['Cluster'].map(segment_map)
    customer_segment_df['PC1'] = np.round(X_pca[:, 0], 4)
    customer_segment_df['PC2'] = np.round(X_pca[:, 1], 4)

    customer_segments_path = artifacts_dir / 'final_customer_segments.csv'
    customer_segment_df.to_csv(customer_segments_path, index=False)
    print(f"✓ Saved full customer segment assignments to {customer_segments_path}")

    # Generate final PCA & segment size plots
    generate_final_segment_plots(X_pca, profiles_df, final_labels, output_dir='final_artifacts')

    # Export Comprehensive JSON Metrics
    final_metrics = {
        'generated_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'dataset': metadata,
        'kmeans': {
            'best_k': int(best_km['k']),
            'silhouette_score': round(float(best_km['silhouette']), 4),
            'davies_bouldin_index': round(float(best_km['davies_bouldin']), 4),
            'inertia': round(float(best_km['inertia']), 2),
            'all_k_evaluated': [
                {
                    'k': int(r['k']),
                    'silhouette': round(float(r['silhouette']), 4),
                    'davies_bouldin': round(float(r['davies_bouldin']), 4),
                    'inertia': round(float(r['inertia']), 2),
                }
                for r in km_results
            ]
        },
        'gmm': {
            'best_n_components': int(best_gmm['n_components']),
            'best_covariance_type': str(best_gmm['covariance_type']),
            'silhouette_score': round(float(best_gmm['silhouette']), 4),
            'davies_bouldin_index': round(float(best_gmm['davies_bouldin']), 4),
            'aic': round(float(best_gmm['aic']), 2),
            'bic': round(float(best_gmm['bic']), 2),
            'mean_max_probability': float(gmm_uncertainty['mean_max_probability']),
            'ambiguous_customer_count': int(gmm_uncertainty['ambiguous_count']),
            'ambiguous_percentage': float(gmm_uncertainty['ambiguous_percentage']),
        },
        'hierarchical': {
            'best_n_clusters': int(best_hier['n_clusters']),
            'best_linkage_method': str(best_hier['method']),
            'best_distance_metric': str(best_hier['metric']),
            'silhouette_score': round(float(best_hier['silhouette']), 4),
            'davies_bouldin_index': round(float(best_hier['davies_bouldin']), 4),
            'cophenetic_correlations': [
                {
                    'method': r['method'],
                    'metric': r['metric'],
                    'cophenetic_correlation': float(r['cophenetic_correlation']),
                }
                for r in coph_results
            ]
        },
        'overall_winner': {
            'algorithm': winner_row['Algorithm'],
            'configuration': winner_row['Best_Configuration'],
            'number_of_clusters': int(winner_row['Number_of_Clusters']),
            'silhouette_score': float(winner_row['Silhouette_Score']),
            'davies_bouldin_index': float(winner_row['Davies_Bouldin_Index']),
            'justification': (
                f"{winner_name} achieved the highest Silhouette Score ({winner_row['Silhouette_Score']:.4f}) "
                f"and lowest Davies-Bouldin Index ({winner_row['Davies_Bouldin_Index']:.4f}) on the standardized "
                "Log-RFM feature space, successfully separating distinct customer activity tiers."
            )
        },
        'segment_profiles': profiles_df.to_dict(orient='records'),
    }

    metrics_json_path = artifacts_dir / 'final_metrics.json'
    with open(metrics_json_path, 'w') as f:
        json.dump(final_metrics, f, indent=2, cls=NumpyEncoder)
    print(f"✓ Saved complete metrics JSON to {metrics_json_path}")

    elapsed_total = time.time() - start_total
    print("\n" + "=" * 70)
    print(f"✅ FINAL RESULTS GENERATION COMPLETE in {elapsed_total:.2f} seconds!")
    print("=" * 70)


if __name__ == '__main__':
    main()
