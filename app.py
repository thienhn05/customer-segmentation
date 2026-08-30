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
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

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
        return customer_features, X, X_scaled, X_pca
    except Exception as exc:
        st.error(f'Dataset loading failed: {exc}')
        raise


@st.cache_data
def compute_kmeans_results(X_scaled, X_pca, customer_features, k_values=None):
    """Compute K-Means results for a set of k values."""
    if k_values is None:
        k_values = list(range(2, 11))

    results = []
    for k in k_values:
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        results.append({
            'k': k,
            'silhouette': silhouette_score(X_scaled, labels),
            'davies_bouldin': davies_bouldin_score(X_scaled, labels),
            'inertia': model.inertia_,
            'labels': labels,
            'model': model,
        })
    return results


@st.cache_data
def compute_gmm_results(X_scaled, X_pca, customer_features, component_values=None, covariance_types=None):
    """Compute GMM results for a range of component counts and covariance types."""
    if component_values is None:
        component_values = list(range(2, 11))
    if covariance_types is None:
        covariance_types = ['full', 'tied', 'diag', 'spherical']

    results = []
    for cov_type in covariance_types:
        for n_components in component_values:
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=cov_type,
                random_state=42,
                n_init=5,
                reg_covar=1e-6,
            )
            labels = gmm.fit_predict(X_scaled)
            results.append({
                'covariance_type': cov_type,
                'n_components': n_components,
                'silhouette': silhouette_score(X_scaled, labels),
                'davies_bouldin': davies_bouldin_score(X_scaled, labels),
                'aic': gmm.aic(X_scaled),
                'bic': gmm.bic(X_scaled),
                'labels': labels,
                'model': gmm,
            })
    return results


@st.cache_data
def compute_hierarchical_results(X_scaled, X_pca, customer_features, methods=None, metrics=None):
    """Compute hierarchical clustering results for several methods and metrics."""
    if methods is None:
        methods = ['ward', 'complete', 'average', 'single']
    if metrics is None:
        metrics = ['euclidean', 'manhattan', 'cosine']

    results = []
    for method in methods:
        for metric in metrics:
            try:
                model = AgglomerativeClustering(n_clusters=3, metric=metric, linkage=method)
                labels = model.fit_predict(X_scaled)
                results.append({
                    'method': method,
                    'metric': metric,
                    'silhouette': silhouette_score(X_scaled, labels),
                    'davies_bouldin': davies_bouldin_score(X_scaled, labels),
                    'labels': labels,
                    'model': model,
                })
            except Exception:
                continue
    return results


@st.cache_data
def get_cluster_summary(customer_features, labels, clustering_features):
    """Create a cluster summary table from the results."""
    df = customer_features.copy()
    df['Cluster'] = labels
    summary = df.groupby('Cluster')[clustering_features].mean().round(2)
    summary['Customers'] = df['Cluster'].value_counts().sort_index().values
    return summary


def render_sidebar():
    """Render the app navigation sidebar."""
    st.sidebar.image('https://images.unsplash.com/photo-1552664730-d307ca884978?auto=format&fit=crop&w=1200&q=80', use_container_width=True)
    st.sidebar.title('Customer Segmentation Dashboard')
    pages = ['Home', 'Data Overview', 'Clustering Results', 'Compare Models', 'About']
    return st.sidebar.radio('Navigate', pages)


def render_home():
    """Render the home page."""
    st.title('Customer Segmentation with Clustering Models')
    st.markdown(
        """
        This dashboard explores customer behavior using the Online Retail II dataset and compares multiple clustering approaches:
        - K-Means
        - Gaussian Mixture Models
        - Hierarchical Agglomerative Clustering
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('Customers', '4,338')
    with col2:
        st.metric('Features', '6')
    with col3:
        st.metric('Clusters', '3')

    st.subheader('Methodology Overview')
    st.markdown(
        """
        1. Load and clean the retail transaction data.
        2. Create RFM features such as Recency, Frequency, Monetary, and Avg Order Value.
        3. Scale the clustering features and reduce dimensionality with PCA.
        4. Train and compare multiple clustering models.
        5. Interpret the resulting customer segments and export the final results.
        """
    )

    st.info('Use the sidebar to navigate to the analysis pages and compare model performance.')


def render_data_overview(customer_features, X, X_scaled, X_pca):
    """Render the data overview page."""
    st.title('Data Overview')

    if st.button('Refresh Data'):
        st.cache_data.clear()
        st.rerun()

    with st.spinner('Loading data overview...'):
        st.dataframe(customer_features.head(20), use_container_width=True)

    st.subheader('Summary Statistics')
    st.dataframe(customer_features.describe().round(2), use_container_width=True)

    feature_cols = list(X.columns)
    selected_feature = st.selectbox('Select feature for histogram', feature_cols)
    fig = px.histogram(customer_features, x=selected_feature, nbins=30, title=f'{selected_feature} Distribution')
    st.plotly_chart(fig, use_container_width=True)

    corr = customer_features[feature_cols].corr().round(2)
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title='Feature Correlation Heatmap')
    st.plotly_chart(fig_corr, use_container_width=True)

    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    fig_pca = px.scatter(
        pca_df,
        x='PC1',
        y='PC2',
        hover_name=customer_features['CustomerID'].astype(str),
        title='PCA Scatter Plot',
    )
    st.plotly_chart(fig_pca, use_container_width=True)


def plot_cluster_scatter(X_pca, labels, title='Cluster Visualization'):
    """Create a PCA scatter chart with cluster labels."""
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = labels
    fig = px.scatter(
        pca_df,
        x='PC1',
        y='PC2',
        color='Cluster',
        title=title,
        color_continuous_scale='viridis',
        hover_data={'PC1': ':.2f', 'PC2': ':.2f', 'Cluster': True},
    )
    return fig


def render_clustering_results(customer_features, X, X_scaled, X_pca):
    """Render the clustering results page."""
    st.title('Clustering Results')

    method = st.selectbox('Choose clustering method', ['K-Means', 'GMM', 'Hierarchical'])
    cluster_k = st.slider('Number of clusters', min_value=2, max_value=12, value=3)

    if method == 'K-Means':
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=cluster_k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        metrics = {
            'Silhouette Score': silhouette_score(X_scaled, labels),
            'Davies-Bouldin Index': davies_bouldin_score(X_scaled, labels),
        }
        cluster_summary = get_cluster_summary(customer_features, labels, list(X.columns))

    elif method == 'GMM':
        covariance_type = st.selectbox('Covariance type', ['full', 'tied', 'diag', 'spherical'])
        gmm = GaussianMixture(n_components=cluster_k, covariance_type=covariance_type, random_state=42, n_init=5)
        labels = gmm.fit_predict(X_scaled)
        metrics = {
            'Silhouette Score': silhouette_score(X_scaled, labels),
            'Davies-Bouldin Index': davies_bouldin_score(X_scaled, labels),
        }
        cluster_summary = get_cluster_summary(customer_features, labels, list(X.columns))

    else:
        linkage_method = st.selectbox('Linkage method', ['ward', 'complete', 'average', 'single'])
        model = AgglomerativeClustering(n_clusters=cluster_k, linkage=linkage_method)
        labels = model.fit_predict(X_scaled)
        metrics = {
            'Silhouette Score': silhouette_score(X_scaled, labels),
            'Davies-Bouldin Index': davies_bouldin_score(X_scaled, labels),
        }
        cluster_summary = get_cluster_summary(customer_features, labels, list(X.columns))

    col1, col2 = st.columns(2)
    with col1:
        st.metric('Silhouette Score', f"{metrics['Silhouette Score']:.4f}")
    with col2:
        st.metric('Davies-Bouldin Index', f"{metrics['Davies-Bouldin Index']:.4f}")

    st.plotly_chart(plot_cluster_scatter(X_pca, labels, title=f'{method} PCA Visualization'), use_container_width=True)

    st.subheader('Cluster Profiles')
    st.dataframe(cluster_summary.reset_index().rename(columns={'index': 'ClusterID'}), use_container_width=True)

    counts = pd.Series(labels).value_counts().sort_index()
    pie_fig = px.pie(values=counts.values, names=[f'Cluster {i}' for i in counts.index], title='Cluster Size Distribution')
    st.plotly_chart(pie_fig, use_container_width=True)

    csv = cluster_summary.reset_index().rename(columns={'index': 'ClusterID'}).to_csv(index=False)
    st.download_button('Download cluster summary CSV', csv, file_name='cluster_summary.csv', mime='text/csv')


def render_compare_models(customer_features, X, X_scaled, X_pca):
    """Render the model comparison page."""
    st.title('Compare Models')

    kmeans_results = compute_kmeans_results(X_scaled, X_pca, customer_features, k_values=list(range(2, 11)))
    gmm_results = compute_gmm_results(X_scaled, X_pca, customer_features, component_values=list(range(2, 11)))
    hierarchical_results = compute_hierarchical_results(X_scaled, X_pca, customer_features)

    model_metrics = []
    for res in kmeans_results:
        model_metrics.append({'Model': 'K-Means', 'k': res['k'], 'Silhouette': res['silhouette'], 'Davies-Bouldin': res['davies_bouldin']})
    for res in gmm_results:
        model_metrics.append({'Model': 'GMM', 'k': res['n_components'], 'Silhouette': res['silhouette'], 'Davies-Bouldin': res['davies_bouldin']})
    for res in hierarchical_results:
        model_metrics.append({'Model': 'Hierarchical', 'k': 3, 'Silhouette': res['silhouette'], 'Davies-Bouldin': res['davies_bouldin']})

    metrics_df = pd.DataFrame(model_metrics)
    st.dataframe(metrics_df, use_container_width=True)

    fig_bar = px.bar(
        metrics_df.groupby('Model')[['Silhouette', 'Davies-Bouldin']].mean().reset_index(),
        x='Model',
        y=['Silhouette', 'Davies-Bouldin'],
        barmode='group',
        title='Average Model Performance',
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    best_recommendation = metrics_df.sort_values('Silhouette', ascending=False).iloc[0]
    st.success(f"Best model recommendation: {best_recommendation['Model']} (Silhouette = {best_recommendation['Silhouette']:.4f})")


def render_about():
    """Render the about page."""
    st.title('About')
    st.markdown(
        """
        This project demonstrates customer segmentation using several unsupervised learning techniques.
        It is designed for business analytics and marketing segmentation use cases.
        """
    )
    st.markdown('### Team')
    st.write('Customer Analytics Team')
    st.markdown('### License')
    st.write('This project is for educational and demonstration purposes.')


def main():
    """Run the Streamlit application."""
    page = render_sidebar()

    customer_features, X, X_scaled, X_pca = load_dataset('online_retail_II.xlsx')

    if page == 'Home':
        render_home()
    elif page == 'Data Overview':
        render_data_overview(customer_features, X, X_scaled, X_pca)
    elif page == 'Clustering Results':
        render_clustering_results(customer_features, X, X_scaled, X_pca)
    elif page == 'Compare Models':
        render_compare_models(customer_features, X, X_scaled, X_pca)
    elif page == 'About':
        render_about()


if __name__ == '__main__':
    main()
