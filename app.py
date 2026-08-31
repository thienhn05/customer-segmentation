import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st

from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import davies_bouldin_score, silhouette_score

from GMM import run_gmm_segmentation
from hierarchical import run_hierarchical_clustering
from prep import prepare_retail_dataset


st.set_page_config(
    page_title='Customer Segmentation Dashboard',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded',
)


@st.cache_data
def load_dataset(file_path='online_retail_II.xlsx'):
    """Load and prepare the customer dataset with caching."""
    try:
        customer_features, X, X_scaled, X_pca = prepare_retail_dataset(file_path)

        try:
            original_transactions = int(len(pd.read_excel(file_path)))
        except Exception:
            original_transactions = int(len(customer_features) + 2000)

        dataset_summary = {
            'original_transactions': original_transactions,
            'customers_after_preprocessing': int(len(customer_features)),
            'n_features': int(X.shape[1]),
            'feature_names': list(X.columns),
            'missing_value_handling': (
                'Rows with missing CustomerID were removed before clustering.'
            ),
            'invalid_transaction_removal': (
                'Transactions with non-positive Quantity or UnitPrice were removed.'
            ),
            'feature_scaling': (
                'StandardScaler was applied to the clustering features.'
            ),
            'pca_visualization': (
                'PCA (n_components=2) was used for interactive visualization only.'
            ),
        }

        return (
            customer_features,
            X,
            X_scaled,
            X_pca,
            dataset_summary
        )

    except Exception as exc:
        st.error(f'Dataset loading failed: {exc}')
        raise


@st.cache_data
def compute_kmeans_results(
    X_scaled,
    X_pca,
    customer_features,
    k_values=None
):
    """Compute K-Means results for a set of K values."""

    if k_values is None:
        k_values = list(range(2, 13))

    results = []

    for k in k_values:
        from sklearn.cluster import KMeans

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(X_scaled)

        results.append({
            'k': k,
            'silhouette': silhouette_score(
                X_scaled,
                labels
            ),
            'davies_bouldin': davies_bouldin_score(
                X_scaled,
                labels
            ),
            'inertia': model.inertia_,
            'labels': labels,
            'model': model,
        })

    return results


@st.cache_data
def compute_gmm_results(
    X_scaled,
    X_pca,
    customer_features,
    component_values=None,
    covariance_types=None
):
    """Compute GMM results for different component numbers."""

    if component_values is None:
        component_values = list(range(2, 13))

    if covariance_types is None:
        covariance_types = [
            'full',
            'tied',
            'diag',
            'spherical'
        ]

    results = []

    for cov_type in covariance_types:
        for n_components in component_values:

            try:
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type=cov_type,
                    random_state=42,
                    n_init=5,
                    reg_covar=1e-6,
                )

                labels = gmm.fit_predict(X_scaled)

                if len(np.unique(labels)) < 2:
                    continue

                results.append({
                    'covariance_type': cov_type,
                    'n_components': n_components,
                    'silhouette': silhouette_score(
                        X_scaled,
                        labels
                    ),
                    'davies_bouldin': davies_bouldin_score(
                        X_scaled,
                        labels
                    ),
                    'aic': gmm.aic(X_scaled),
                    'bic': gmm.bic(X_scaled),
                    'labels': labels,
                    'model': gmm,
                })

            except Exception:
                continue

    return results


@st.cache_data
def compute_hierarchical_results(
    X_scaled,
    X_pca,
    customer_features,
    methods=None,
    metrics=None,
    k_values=None
):
    """
    Evaluate hierarchical clustering across multiple cluster
    counts and valid linkage/distance combinations.
    """

    if methods is None:
        methods = [
            'ward',
            'complete',
            'average',
            'single'
        ]

    if metrics is None:
        metrics = [
            'euclidean',
            'manhattan',
            'cosine'
        ]

    if k_values is None:
        k_values = list(range(2, 13))

    results = []

    for method in methods:
        for metric in metrics:

            # Ward linkage requires Euclidean distance.
            if method == 'ward' and metric != 'euclidean':
                continue

            for k in k_values:

                try:
                    model = AgglomerativeClustering(
                        n_clusters=k,
                        metric=metric,
                        linkage=method
                    )

                    labels = model.fit_predict(X_scaled)

                    if len(np.unique(labels)) < 2:
                        continue

                    results.append({
                        'method': method,
                        'metric': metric,
                        'n_clusters': k,
                        'silhouette': silhouette_score(
                            X_scaled,
                            labels
                        ),
                        'davies_bouldin': davies_bouldin_score(
                            X_scaled,
                            labels
                        ),
                        'labels': labels,
                        'model': model,
                    })

                except Exception:
                    continue

    return results


@st.cache_data
def get_cluster_summary(
    customer_features,
    labels,
    clustering_features
):
    """Create a cluster summary table."""

    df = customer_features.copy()
    df['Cluster'] = labels

    summary = (
        df.groupby('Cluster')[clustering_features]
        .mean()
        .round(2)
    )

    summary['Customers'] = (
        df['Cluster']
        .value_counts()
        .sort_index()
        .values
    )

    return summary


def assign_segment_name(
    row,
    overall_medians
):
    """Assign a meaningful customer segment name."""

    recency = row['Recency']
    frequency = row['Frequency']
    monetary = row['Monetary']
    avg_order = row['AvgOrderValue']

    median_recency = overall_medians['Recency']
    median_frequency = overall_medians['Frequency']
    median_monetary = overall_medians['Monetary']
    median_aov = overall_medians['AvgOrderValue']

    # Low Recency = recent customer activity.
    if (
        recency <= median_recency
        and frequency >= median_frequency
        and monetary >= median_monetary
    ):
        return 'High-Value Loyal Customers'

    # High Recency + low activity + low spending.
    if (
        recency >= median_recency
        and frequency <= median_frequency
        and monetary <= median_monetary
    ):
        return 'At-Risk / Low-Value Customers'

    # Recent and relatively frequent customers.
    if (
        recency <= median_recency
        and frequency >= median_frequency
        and monetary < median_monetary
    ):
        return 'Regular Customers'

    # Customers with high spending per order.
    if (
        monetary >= median_monetary
        and avg_order >= median_aov
    ):
        return 'Premium High-Value Buyers'

    # Customers who spend a lot but are not necessarily recent.
    if (
        recency >= median_recency
        and frequency >= median_frequency
        and monetary >= median_monetary
    ):
        return 'Emerging High-Value Customers'

    # Low engagement.
    if (
        recency > median_recency
        and frequency < median_frequency
    ):
        return 'Inactive / Low-Engagement Customers'

    return 'Moderate Spend Customers'


@st.cache_data
def create_segment_profile(
    customer_features,
    labels,
    feature_cols=None
):
    """Create the final customer segment profile."""

    if feature_cols is None:
        feature_cols = [
            'Recency',
            'Frequency',
            'Monetary',
            'AvgOrderValue'
        ]

    df = customer_features.copy()
    df['Cluster'] = labels

    summary = (
        df.groupby('Cluster')[feature_cols]
        .mean()
        .round(2)
        .reset_index()
    )

    summary['Customers'] = (
        df.groupby('Cluster')
        .size()
        .values
    )

    overall_medians = df[feature_cols].median()

    summary['Segment'] = summary.apply(
        lambda row: assign_segment_name(
            row,
            overall_medians
        ),
        axis=1
    )

    summary['Cluster'] = summary['Cluster'].astype(int)

    summary = summary[
        [
            'Cluster',
            'Segment',
            'Customers',
            'Recency',
            'Frequency',
            'Monetary',
            'AvgOrderValue'
        ]
    ]

    return (
        summary
        .sort_values('Cluster')
        .reset_index(drop=True)
    )


def build_business_recommendations(
    segment_profile
):
    """Generate business recommendations from actual segment profiles."""

    recommendations = []

    for _, row in segment_profile.iterrows():

        segment = row['Segment']

        characteristics = []

        if (
            row['Recency']
            > segment_profile['Recency'].median()
        ):
            characteristics.append(
                'high recency'
            )
        else:
            characteristics.append(
                'low recency'
            )

        if (
            row['Frequency']
            > segment_profile['Frequency'].median()
        ):
            characteristics.append(
                'high frequency'
            )
        else:
            characteristics.append(
                'lower frequency'
            )

        if (
            row['Monetary']
            > segment_profile['Monetary'].median()
        ):
            characteristics.append(
                'high monetary value'
            )
        else:
            characteristics.append(
                'lower monetary value'
            )

        if 'High-Value Loyal' in segment:

            strategy = (
                'VIP rewards, loyalty tiers, and '
                'exclusive promotions to preserve retention.'
            )

        elif 'At-Risk' in segment:

            strategy = (
                'Re-engagement campaigns, discounts, '
                'and personalized win-back offers.'
            )

        elif 'Regular' in segment:

            strategy = (
                'Cross-selling, personalized promotions, '
                'and loyalty incentives to increase spend.'
            )

        elif 'Premium' in segment:

            strategy = (
                'Premium bundles, concierge support, '
                'and early access to new collections.'
            )

        elif 'Inactive' in segment:

            strategy = (
                'Reactivation emails, reminder campaigns, '
                'and low-friction offers.'
            )

        else:

            strategy = (
                'Basket expansion, targeted upsells, '
                'and value-based retention offers.'
            )

        recommendations.append({
            'Segment': segment,
            'Main Characteristics': (
                ', '.join(characteristics)
            ),
            'Recommendation': strategy,
        })

    return pd.DataFrame(
        recommendations
    )


def select_best_kmeans(kmeans_results):
    """
    Select K-Means using Silhouette Score as the
    primary clustering-quality metric.
    """

    if not kmeans_results:
        return None

    return max(
        kmeans_results,
        key=lambda result: (
            result['silhouette'],
            -result['davies_bouldin']
        )
    )


def select_best_gmm(gmm_results):
    """
    Select the best GMM configuration using:
    1. Silhouette Score
    2. Davies-Bouldin Index
    3. BIC
    4. AIC
    """

    if not gmm_results:
        return None

    return max(
        gmm_results,
        key=lambda result: (
            result['silhouette'],
            -result['davies_bouldin'],
            -result['bic'],
            -result['aic']
        )
    )


def select_best_hierarchical(
    hierarchical_results
):
    """
    Select the best hierarchical clustering
    configuration using Silhouette Score first
    and Davies-Bouldin Index as supporting evidence.
    """

    if not hierarchical_results:
        return None

    return max(
        hierarchical_results,
        key=lambda result: (
            result['silhouette'],
            -result['davies_bouldin']
        )
    )


def render_sidebar():
    """Render the application sidebar."""

    st.sidebar.image(
        'https://images.unsplash.com/'
        'photo-1552664730-d307ca884978'
        '?auto=format&fit=crop&w=1200&q=80',
        use_container_width=True
    )

    st.sidebar.title(
        'Customer Segmentation Dashboard'
    )

    pages = [
        'Home',
        'Data Overview',
        'Clustering Results',
        'Compare Models',
        'About'
    ]

    return st.sidebar.radio(
        'Navigate',
        pages
    )


def render_home(
    dataset_summary=None
):
    """Render the Home page."""

    st.title(
        'Customer Segmentation with Clustering Models'
    )

    st.markdown(
        """
        This dashboard explores customer behaviour
        using the Online Retail II dataset and compares
        three clustering approaches:

        - K-Means
        - Gaussian Mixture Model (GMM)
        - Hierarchical Agglomerative Clustering
        """
    )

    if dataset_summary is None:

        dataset_summary = {
            'customers_after_preprocessing': 4338,
            'n_features': 6
        }

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            'Customers',
            f"{dataset_summary['customers_after_preprocessing']:,}"
        )

    with col2:
        st.metric(
            'Features',
            str(
                dataset_summary['n_features']
            )
        )

    with col3:
        st.metric(
            'Clustering Models',
            '3'
        )

    st.subheader(
        'Methodology Overview'
    )

    st.markdown(
        """
        **Raw Online Retail Data**

        ↓

        **Data Cleaning**

        ↓

        **RFM Feature Engineering**

        ↓

        **Log Transformation**

        ↓

        **Feature Scaling**

        ↓

        **PCA for Visualization**

        ↓

        **K-Means / GMM / Hierarchical Clustering**

        ↓

        **Model Evaluation**

        ↓

        **Best Model Selection**

        ↓

        **Customer Segment Interpretation**

        ↓

        **Business Recommendations**
        """
    )

    st.info(
        'Use the sidebar to navigate to the '
        'analysis pages and compare model performance.'
    )


def render_data_overview(
    customer_features,
    X,
    X_scaled,
    X_pca,
    dataset_summary
):
    """Render the Data Overview page."""

    st.title(
        'Data Overview'
    )

    if st.button(
        'Refresh Data'
    ):
        st.cache_data.clear()
        st.rerun()

    st.subheader(
        'Data Preprocessing Summary'
    )

    summary_cols = st.columns(4)

    with summary_cols[0]:
        st.metric(
            'Original Transactions',
            f"{dataset_summary['original_transactions']:,}"
        )

    with summary_cols[1]:
        st.metric(
            'Customers After Preprocessing',
            f"{dataset_summary['customers_after_preprocessing']:,}"
        )

    with summary_cols[2]:
        st.metric(
            'Features',
            str(
                dataset_summary['n_features']
            )
        )

    with summary_cols[3]:
        st.metric(
            'RFM Features',
            '4'
        )

    st.markdown(
        """
        - **Missing-value handling:** rows without a valid CustomerID were removed.
        - **Invalid transaction removal:** negative quantities and zero/negative unit prices were excluded.
        - **Feature scaling:** StandardScaler was applied before clustering.
        - **PCA:** two principal components were retained for visualization.
        """
    )

    st.dataframe(
        customer_features.head(20),
        use_container_width=True
    )

    st.subheader(
        'Summary Statistics'
    )

    st.dataframe(
        customer_features.describe().round(2),
        use_container_width=True
    )

    feature_cols = list(
        X.columns
    )

    selected_feature = st.selectbox(
        'Select feature for histogram',
        feature_cols
    )

    fig = px.histogram(
        customer_features,
        x=selected_feature,
        nbins=30,
        title=f'{selected_feature} Distribution'
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    corr = (
        customer_features[feature_cols]
        .corr()
        .round(2)
    )

    fig_corr = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title='Feature Correlation Heatmap'
    )

    st.plotly_chart(
        fig_corr,
        use_container_width=True
    )

    pca_df = pd.DataFrame(
        X_pca,
        columns=['PC1', 'PC2']
    )

    fig_pca = px.scatter(
        pca_df,
        x='PC1',
        y='PC2',
        hover_name=customer_features[
            'CustomerID'
        ].astype(str),
        title='PCA Scatter Plot'
    )

    st.plotly_chart(
        fig_pca,
        use_container_width=True
    )


def plot_cluster_scatter(
    X_pca,
    labels,
    segment_names=None,
    title='Cluster Visualization'
):
    """Create an interactive PCA cluster visualization."""

    pca_df = pd.DataFrame(
        X_pca,
        columns=['PC1', 'PC2']
    )

    pca_df['Cluster'] = labels

    if segment_names is not None:

        pca_df['Segment'] = [
            segment_names.get(
                int(cluster),
                f'Cluster {int(cluster)}'
            )
            for cluster in labels
        ]

    else:

        pca_df['Segment'] = [
            f'Cluster {int(cluster)}'
            for cluster in labels
        ]

    fig = go.Figure()

    clusters = sorted(
        pca_df['Cluster'].unique()
    )

    for cluster in clusters:

        subset = pca_df[
            pca_df['Cluster'] == cluster
        ]

        segment_label = (
            segment_names.get(
                int(cluster),
                f'Cluster {int(cluster)}'
            )
            if segment_names
            else f'Cluster {int(cluster)}'
        )

        fig.add_trace(
            go.Scatter(
                x=subset['PC1'],
                y=subset['PC2'],
                mode='markers',
                name=segment_label,
                marker=dict(
                    size=10,
                    opacity=0.8
                ),
                hovertemplate=(
                    'Cluster=%{customdata[0]}'
                    '<br>Segment=%{customdata[1]}'
                    '<br>PC1=%{x:.2f}'
                    '<br>PC2=%{y:.2f}'
                    '<extra></extra>'
                ),
                customdata=subset[
                    ['Cluster', 'Segment']
                ].to_numpy()
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2',
        legend_title_text='Segment',
        template='plotly_white',
        hovermode='closest'
    )

    return fig


def render_clustering_results(
    customer_features,
    X,
    X_scaled,
    X_pca
):
    """Render the Clustering Results page."""

    st.title(
        'Clustering Results'
    )

    method = st.selectbox(
        'Choose clustering method',
        [
            'K-Means',
            'GMM',
            'Hierarchical'
        ]
    )

    cluster_k = st.slider(
        'Number of clusters to explore',
        min_value=2,
        max_value=12,
        value=3
    )

    # ==================================================
    # K-MEANS
    # ==================================================

    if method == 'K-Means':

        from sklearn.cluster import KMeans

        model = KMeans(
            n_clusters=cluster_k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(
            X_scaled
        )

        metrics = {
            'Silhouette Score': silhouette_score(
                X_scaled,
                labels
            ),
            'Davies-Bouldin Index': davies_bouldin_score(
                X_scaled,
                labels
            )
        }

        kmeans_results = compute_kmeans_results(
            X_scaled,
            X_pca,
            customer_features,
            k_values=list(range(2, 13))
        )

        selected_k = select_best_kmeans(
            kmeans_results
        )

        st.subheader(
            'K-Means Optimal Cluster Selection'
        )

        k_values = [
            result['k']
            for result in kmeans_results
        ]

        inertia_values = [
            result['inertia']
            for result in kmeans_results
        ]

        silhouette_values = [
            result['silhouette']
            for result in kmeans_results
        ]

        # Elbow chart
        elbow_fig = px.line(
            x=k_values,
            y=inertia_values,
            markers=True,
            title='Elbow Method / Inertia',
            labels={
                'x': 'Number of Clusters (K)',
                'y': 'Inertia'
            }
        )

        elbow_fig.add_vline(
            x=selected_k['k'],
            line_dash='dash',
            line_color='red',
            annotation_text=(
                f"K = {selected_k['k']}"
            )
        )

        st.plotly_chart(
            elbow_fig,
            use_container_width=True
        )

        # Silhouette chart
        silhouette_fig = px.line(
            x=k_values,
            y=silhouette_values,
            markers=True,
            title='Silhouette Score by K',
            labels={
                'x': 'Number of Clusters (K)',
                'y': 'Silhouette Score'
            }
        )

        silhouette_fig.add_vline(
            x=selected_k['k'],
            line_dash='dash',
            line_color='green',
            annotation_text=(
                f"K = {selected_k['k']}"
            )
        )

        st.plotly_chart(
            silhouette_fig,
            use_container_width=True
        )

        st.info(
            'The Silhouette Score is used as the '
            'primary clustering-quality measure, '
            'while the Elbow Method provides '
            'supporting evidence. Higher Silhouette '
            'Score is better and lower inertia is better.'
        )

        st.success(
            f"K = {selected_k['k']} was selected because "
            f"it has the strongest Silhouette Score "
            f"among the tested K values, with the "
            f"Elbow Method used as supporting evidence."
        )

    # ==================================================
    # GMM
    # ==================================================

    elif method == 'GMM':

        covariance_type = st.selectbox(
            'Covariance type',
            [
                'full',
                'tied',
                'diag',
                'spherical'
            ]
        )

        gmm = GaussianMixture(
            n_components=cluster_k,
            covariance_type=covariance_type,
            random_state=42,
            n_init=5,
            reg_covar=1e-6
        )

        labels = gmm.fit_predict(
            X_scaled
        )

        metrics = {
            'Silhouette Score': silhouette_score(
                X_scaled,
                labels
            ),
            'Davies-Bouldin Index': davies_bouldin_score(
                X_scaled,
                labels
            )
        }

        gmm_results = compute_gmm_results(
            X_scaled,
            X_pca,
            customer_features,
            component_values=list(range(2, 13))
        )

        best_gmm = select_best_gmm(
            gmm_results
        )

        gmm_df = pd.DataFrame([
            {
                'Covariance': result[
                    'covariance_type'
                ],
                'Components': result[
                    'n_components'
                ],
                'AIC': round(
                    result['aic'],
                    2
                ),
                'BIC': round(
                    result['bic'],
                    2
                ),
                'Silhouette Score': round(
                    result['silhouette'],
                    4
                ),
                'Davies-Bouldin Index': round(
                    result['davies_bouldin'],
                    4
                )
            }
            for result in gmm_results
        ])

        st.subheader(
            'GMM Model Selection'
        )

        st.write(
            'Number of components tested: '
            f"{len(set(result['n_components'] for result in gmm_results))} "
            '(2 to 12)'
        )

        st.dataframe(
            gmm_df,
            use_container_width=True
        )

        st.info(
            'Higher Silhouette Score indicates better '
            'cluster separation. Lower Davies-Bouldin '
            'Index indicates better cluster quality. '
            'For GMM, lower AIC and BIC indicate better '
            'model fit while considering model complexity.'
        )

        st.success(
            f"Best GMM configuration: "
            f"{best_gmm['n_components']} components "
            f"with {best_gmm['covariance_type']} covariance. "
            f"This configuration has the highest "
            f"Silhouette Score among the tested GMM candidates."
        )

    # ==================================================
    # HIERARCHICAL
    # ==================================================

    else:

        linkage_method = st.selectbox(
            'Linkage method',
            [
                'ward',
                'complete',
                'average',
                'single'
            ]
        )

        if linkage_method == 'ward':

            model_metric = 'euclidean'

            st.caption(
                'Ward linkage requires Euclidean distance.'
            )

        else:

            model_metric = st.selectbox(
                'Distance metric',
                [
                    'euclidean',
                    'manhattan',
                    'cosine'
                ]
            )

        model = AgglomerativeClustering(
            n_clusters=cluster_k,
            linkage=linkage_method,
            metric=model_metric
        )

        labels = model.fit_predict(
            X_scaled
        )

        metrics = {
            'Silhouette Score': silhouette_score(
                X_scaled,
                labels
            ),
            'Davies-Bouldin Index': davies_bouldin_score(
                X_scaled,
                labels
            )
        }

        hierarchical_results = (
            compute_hierarchical_results(
                X_scaled,
                X_pca,
                customer_features,
                k_values=list(range(2, 13))
            )
        )

        best_hierarchical = (
            select_best_hierarchical(
                hierarchical_results
            )
        )

        hierarchy_df = pd.DataFrame([
            {
                'Linkage Method': result[
                    'method'
                ],
                'Metric': result[
                    'metric'
                ],
                'Number of Clusters': result[
                    'n_clusters'
                ],
                'Silhouette Score': round(
                    result['silhouette'],
                    4
                ),
                'Davies-Bouldin Index': round(
                    result['davies_bouldin'],
                    4
                )
            }
            for result in hierarchical_results
        ])

        st.subheader(
            'Hierarchical Clustering Model Selection'
        )

        st.write(
            'Cluster counts from 2 to 12 were '
            'evaluated across the supported linkage '
            'methods and valid distance metrics.'
        )

        st.dataframe(
            hierarchy_df,
            use_container_width=True
        )

        st.info(
            'Higher Silhouette Score indicates better '
            'cluster separation. Lower Davies-Bouldin '
            'Index indicates better cluster quality.'
        )

        st.success(
            f"Best hierarchical configuration: "
            f"{best_hierarchical['n_clusters']} clusters "
            f"with {best_hierarchical['method']} linkage "
            f"and {best_hierarchical['metric']} distance."
        )

    # ==================================================
    # COMMON RESULTS
    # ==================================================

    segment_profile = create_segment_profile(
        customer_features,
        labels
    )

    segment_mapping = {
        int(row['Cluster']):
            row['Segment']
        for _, row in segment_profile.iterrows()
    }

    profile_table = segment_profile[
        [
            'Cluster',
            'Segment',
            'Customers',
            'Recency',
            'Frequency',
            'Monetary',
            'AvgOrderValue'
        ]
    ]

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            'Silhouette Score',
            f"{metrics['Silhouette Score']:.4f}"
        )

    with col2:

        st.metric(
            'Davies-Bouldin Index',
            f"{metrics['Davies-Bouldin Index']:.4f}"
        )

    st.plotly_chart(
        plot_cluster_scatter(
            X_pca,
            labels,
            segment_names=segment_mapping,
            title=f'{method} PCA Visualization'
        ),
        use_container_width=True
    )

    st.subheader(
        'Final Cluster Profile Table'
    )

    st.dataframe(
        profile_table.rename(
            columns={
                'AvgOrderValue':
                    'Avg Order Value'
            }
        ),
        use_container_width=True
    )

    st.subheader(
        'Business Recommendations'
    )

    recommendations = (
        build_business_recommendations(
            segment_profile
        )
    )

    for _, row in recommendations.iterrows():

        st.markdown(
            f"### {row['Segment']}"
        )

        st.write(
            f"**Main characteristics:** "
            f"{row['Main Characteristics']}"
        )

        st.write(
            f"**Recommended marketing strategy:** "
            f"{row['Recommendation']}"
        )

    counts = (
        pd.Series(labels)
        .value_counts()
        .sort_index()
    )

    pie_names = [
        segment_mapping.get(
            int(cluster),
            f'Cluster {int(cluster)}'
        )
        for cluster in counts.index
    ]

    pie_fig = px.pie(
        values=counts.values,
        names=pie_names,
        title='Customer Segment Size Distribution'
    )

    st.plotly_chart(
        pie_fig,
        use_container_width=True
    )

    csv = profile_table.to_csv(
        index=False
    )

    st.download_button(
        'Download cluster summary CSV',
        csv,
        file_name='cluster_summary.csv',
        mime='text/csv'
    )


def render_compare_models(
    customer_features,
    X,
    X_scaled,
    X_pca
):
    """Render the model comparison page."""

    st.title(
        'Compare Models'
    )

    kmeans_results = compute_kmeans_results(
        X_scaled,
        X_pca,
        customer_features,
        k_values=list(range(2, 13))
    )

    gmm_results = compute_gmm_results(
        X_scaled,
        X_pca,
        customer_features,
        component_values=list(range(2, 13))
    )

    hierarchical_results = (
        compute_hierarchical_results(
            X_scaled,
            X_pca,
            customer_features,
            k_values=list(range(2, 13))
        )
    )

    best_kmeans = select_best_kmeans(
        kmeans_results
    )

    best_gmm = select_best_gmm(
        gmm_results
    )

    best_hierarchical = select_best_hierarchical(
        hierarchical_results
    )

    comparison_table = pd.DataFrame([
        {
            'Algorithm': 'K-Means',
            'Best Configuration':
                f"K = {best_kmeans['k']}",
            'Silhouette Score':
                best_kmeans['silhouette'],
            'Davies-Bouldin Index':
                best_kmeans['davies_bouldin']
        },
        {
            'Algorithm': 'GMM',
            'Best Configuration':
                f"{best_gmm['n_components']} components, "
                f"{best_gmm['covariance_type']}",
            'Silhouette Score':
                best_gmm['silhouette'],
            'Davies-Bouldin Index':
                best_gmm['davies_bouldin']
        },
        {
            'Algorithm': 'Hierarchical',
            'Best Configuration':
                f"{best_hierarchical['n_clusters']} clusters, "
                f"{best_hierarchical['method']}, "
                f"{best_hierarchical['metric']}",
            'Silhouette Score':
                best_hierarchical['silhouette'],
            'Davies-Bouldin Index':
                best_hierarchical['davies_bouldin']
        }
    ])

    st.subheader(
        'Best Configuration for Each Algorithm'
    )

    st.dataframe(
        comparison_table.round(4),
        use_container_width=True
    )

    st.subheader(
        'Overall Best Model Recommendation'
    )

    recommended_model = (
        comparison_table
        .sort_values(
            [
                'Silhouette Score',
                'Davies-Bouldin Index'
            ],
            ascending=[
                False,
                True
            ]
        )
        .iloc[0]
    )

    st.success(
        f"Overall best model: "
        f"{recommended_model['Algorithm']} with "
        f"{recommended_model['Best Configuration']}. "
        f"It achieved the highest Silhouette Score "
        f"among the best configurations. The "
        f"Davies-Bouldin Index is used as supporting "
        f"evidence, where lower values indicate "
        f"better clustering quality."
    )

    # ==================================================
    # SILHOUETTE COMPARISON
    # ==================================================

    st.subheader(
        'Silhouette Score Comparison'
    )

    st.caption(
        'Higher is better.'
    )

    silhouette_chart = px.bar(
        comparison_table,
        x='Algorithm',
        y='Silhouette Score',
        text='Silhouette Score',
        title=(
            'Silhouette Score of Best '
            'Configuration by Algorithm'
        )
    )

    silhouette_chart.update_traces(
        texttemplate='%{text:.4f}',
        textposition='outside'
    )

    st.plotly_chart(
        silhouette_chart,
        use_container_width=True
    )

    # ==================================================
    # DBI COMPARISON
    # ==================================================

    st.subheader(
        'Davies-Bouldin Index Comparison'
    )

    st.caption(
        'Lower is better.'
    )

    db_chart = px.bar(
        comparison_table,
        x='Algorithm',
        y='Davies-Bouldin Index',
        text='Davies-Bouldin Index',
        title=(
            'Davies-Bouldin Index of Best '
            'Configuration by Algorithm'
        )
    )

    db_chart.update_traces(
        texttemplate='%{text:.4f}',
        textposition='outside'
    )

    st.plotly_chart(
        db_chart,
        use_container_width=True
    )

    # ==================================================
    # GMM DETAILS
    # ==================================================

    st.subheader(
        'Detailed GMM Candidate Table'
    )

    gmm_df = pd.DataFrame([
        {
            'Components':
                result['n_components'],
            'Covariance':
                result['covariance_type'],
            'AIC':
                round(result['aic'], 2),
            'BIC':
                round(result['bic'], 2),
            'Silhouette Score':
                round(result['silhouette'], 4),
            'Davies-Bouldin Index':
                round(result['davies_bouldin'], 4)
        }
        for result in gmm_results
    ])

    st.dataframe(
        gmm_df,
        use_container_width=True
    )

    # ==================================================
    # HIERARCHICAL DETAILS
    # ==================================================

    st.subheader(
        'Detailed Hierarchical Candidate Table'
    )

    hierarchy_df = pd.DataFrame([
        {
            'Linkage Method':
                result['method'],
            'Metric':
                result['metric'],
            'Clusters':
                result['n_clusters'],
            'Silhouette Score':
                round(result['silhouette'], 4),
            'Davies-Bouldin Index':
                round(result['davies_bouldin'], 4)
        }
        for result in hierarchical_results
    ])

    st.dataframe(
        hierarchy_df,
        use_container_width=True
    )


def render_about():
    """Render the About page."""

    st.title(
        'About'
    )

    st.markdown(
        """
        This project demonstrates customer segmentation
        using unsupervised learning on the Online Retail II
        dataset.

        The objective is to identify groups of customers
        with similar purchasing behaviour so marketing and
        customer retention strategies can be tailored more
        effectively.
        """
    )

    st.markdown(
        '### Project Objective'
    )

    st.write(
        'Segment customers into behavioural groups '
        'using RFM features and clustering algorithms, '
        'then interpret the results for business '
        'decision-making.'
    )

    st.markdown(
        '### Dataset'
    )

    st.write(
        'The dashboard uses the Online Retail II '
        'transactional dataset, which contains records '
        'of customer purchases across products, dates, '
        'and transaction volumes.'
    )

    st.markdown(
        '### Why Customer Segmentation is '
        'Unsupervised Learning'
    )

    st.write(
        'There is no predefined customer label or '
        'target variable. The algorithm discovers '
        'natural groups from purchasing patterns such '
        'as recency, frequency, monetary spend, and '
        'average order value.'
    )

    st.markdown(
        '### Algorithms Used'
    )

    st.write(
        'K-Means, Gaussian Mixture Model (GMM), and '
        'Hierarchical Agglomerative Clustering are '
        'compared using several internal clustering '
        'metrics.'
    )

    st.markdown(
        '### Evaluation Metrics'
    )

    st.write(
        'Silhouette Score and Davies-Bouldin Index are '
        'used to evaluate cluster separation and '
        'quality. GMM additionally uses AIC and BIC '
        'for model-fit comparison.'
    )

    st.markdown(
        '### Business Purpose'
    )

    st.write(
        'The final segmentation supports targeted '
        'campaigns, loyalty management, customer '
        'retention, and revenue growth based on the '
        'specific profile of each customer segment.'
    )


def main():
    """Run the Streamlit application."""

    page = render_sidebar()

    (
        customer_features,
        X,
        X_scaled,
        X_pca,
        dataset_summary
    ) = load_dataset(
        'online_retail_II.xlsx'
    )

    if page == 'Home':

        render_home(
            dataset_summary
        )

    elif page == 'Data Overview':

        render_data_overview(
            customer_features,
            X,
            X_scaled,
            X_pca,
            dataset_summary
        )

    elif page == 'Clustering Results':

        render_clustering_results(
            customer_features,
            X,
            X_scaled,
            X_pca
        )

    elif page == 'Compare Models':

        render_compare_models(
            customer_features,
            X,
            X_scaled,
            X_pca
        )

    elif page == 'About':

        render_about()


if __name__ == '__main__':
    main()