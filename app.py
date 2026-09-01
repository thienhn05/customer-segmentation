import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set environment variable for matplotlib
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl')

from prep import prepare_retail_dataset
from customer_segmentation import evaluate_kmeans_range, select_best_kmeans, train_kmeans_model
from GMM import evaluate_gmm_candidates, select_best_gmm, train_gmm_model, analyze_gmm_uncertainty
from hierarchical import evaluate_hierarchical_candidates, select_best_hierarchical, calculate_cophenetic_correlation, build_linkage_matrix
from segmentation_profiles import create_final_segment_profiles

# -----------------------------------------------------------------------------
# Streamlit App Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #2563EB;
        margin-bottom: 10px;
    }
    .winner-card {
        background-color: #ECFDF5;
        border-radius: 8px;
        padding: 18px;
        border: 2px solid #10B981;
        margin-bottom: 15px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: #F9FAFB;
        border-radius: 6px 6px 0px 0px;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #EFF6FF;
        border-bottom: 2px solid #2563EB;
        font-weight: 600;
        color: #1E40AF;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Data & Model Loading
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading canonical dataset and preprocessing...")
def get_dataset():
    return prepare_retail_dataset()


@st.cache_data(show_spinner="Running K-Means evaluation...")
def get_kmeans_results(_X_scaled):
    results = evaluate_kmeans_range(_X_scaled, min_k=2, max_k=12)
    best = select_best_kmeans(results)
    return results, best


@st.cache_data(show_spinner="Running GMM evaluation...")
def get_gmm_results(_X_scaled):
    results = evaluate_gmm_candidates(_X_scaled)
    best = select_best_gmm(results)
    return results, best


@st.cache_data(show_spinner="Running Hierarchical clustering evaluation...")
def get_hierarchical_results(_X_scaled):
    results = evaluate_hierarchical_candidates(_X_scaled)
    best = select_best_hierarchical(results, prefer_balanced=True)
    coph = calculate_cophenetic_correlation(_X_scaled)
    return results, best, coph


@st.cache_data(show_spinner="Loading final precomputed metrics...")
def load_saved_metrics():
    metrics_path = Path('final_artifacts/final_metrics.json')
    if metrics_path.exists():
        try:
            with open(metrics_path, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    return None


# -----------------------------------------------------------------------------
# Main Application Flow
# -----------------------------------------------------------------------------
def main():
    # Load dataset
    customer_features, X, X_scaled, X_pca, metadata = get_dataset()
    saved_metrics = load_saved_metrics()

    # Sidebar Navigation
    st.sidebar.image("https://img.icons8.com/clouds/200/shopping-cart.png", width=120)
    st.sidebar.title("AI Navigation")
    st.sidebar.caption("Customer Segmentation System")

    pages = [
        "🏠 Home",
        "📊 Data Overview & Preprocessing",
        "🔬 Clustering Analysis & Models",
        "⚖️ Model Comparison & Benchmarks",
        "🎯 Final Customer Segmentation",
        "📖 About & Methodology",
    ]
    choice = st.sidebar.radio("Go to Page:", pages, index=0)

    st.sidebar.divider()
    st.sidebar.markdown(f"**Dataset:** Online Retail (UCI)")
    st.sidebar.markdown(f"**Cleaned Customers:** {metadata['usable_customers']:,}")
    st.sidebar.markdown(f"**Active Transactions:** {metadata['usable_transactions']:,}")
    st.sidebar.markdown(f"**PCA 2D Variance:** {metadata['pca_total_explained_variance']:.1%}")

    # =========================================================================
    # PAGE 1: HOME
    # =========================================================================
    if choice == "🏠 Home":
        st.markdown("<div class='main-header'>🛍️ Customer Segmentation Using Unsupervised AI</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub-header'>A Rigorous Unsupervised Machine Learning Framework for E-Commerce Behavioral Intelligence</div>", unsafe_allow_html=True)

        # Executive Metrics Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Raw Transactions", f"{metadata['initial_transactions']:,}", "UCI Online Retail")
        with col2:
            st.metric("Cleaned Customers", f"{metadata['usable_customers']:,}", "100% Valid IDs")
        with col3:
            st.metric("PCA 2D Variance", f"{metadata['pca_total_explained_variance']:.2%}", "PC1: 75.1% | PC2: 18.8%")
        with col4:
            winner_name = saved_metrics['overall_winner']['algorithm'] if saved_metrics else "K-Means"
            winner_score = saved_metrics['overall_winner']['silhouette_score'] if saved_metrics else 0.4328
            st.metric("Top Model (Silhouette)", f"{winner_name}", f"Score: {winner_score:.4f}")

        st.divider()

        # Project Highlights & Objectives
        c_left, c_right = st.columns([3, 2])
        with c_left:
            st.subheader("📌 Project Purpose & Problem Statement")
            st.markdown("""
            Modern e-commerce enterprises face immense heterogeneity in customer buying behaviors. Treating all customers with uniform marketing campaigns leads to budget waste, customer churn, and missed revenue opportunities.

            This university Artificial Intelligence project implements a **systematic, defensible unsupervised machine learning pipeline** to discover latent customer segments from transaction logs of a UK-based online giftware retailer.

            **Core Engineering & Research Highlights:**
            - **End-to-End Data Pipeline**: Rigorous handling of missing identifiers, return/cancellation entries, and positive skewness.
            - **RFM Feature Engineering**: Computation of **Recency** (days since last purchase), **Frequency** (distinct purchase orders), **Monetary** (total spend), and **Average Order Value (AOV)**.
            - **Defensible Normalization**: Natural log transformations (`log1p`) to eliminate heavy positive skewness without collinear feature double-counting, followed by `StandardScaler`.
            - **Three Competing Algorithms**: Empirical grid evaluation across **K-Means**, **Gaussian Mixture Models (GMM)**, and **Hierarchical Agglomerative Clustering**.
            - **Quantitative Validation**: Objective model selection driven by **Silhouette Score**, supported by **Davies-Bouldin Index (DBI)**, **Inertia**, **AIC/BIC**, and **Cophenetic Correlation**.
            """)

        with c_right:
            st.subheader("🏗️ Architecture Workflow")
            st.markdown("""
            ```
            1. Raw UCI Retail Data (541,909 rows)
               │
               ▼ [Filter Cancellations & Missing IDs]
            2. Cleaned Transactions (392,692 rows)
               │
               ▼ [Aggregate by CustomerID]
            3. RFM Features (4,338 customers)
               │
               ▼ [log1p Transform + StandardScaler]
            4. Clustering Matrix (X_scaled)
               │
               ├─► K-Means (K=2..12)
               ├─► GMM (2..12 comp, 4 covariances)
               └─► Hierarchical (Ward/Complete/Avg)
               │
               ▼ [Silhouette & DBI Optimization]
            5. Winning Model Selection (K-Means K=2)
               │
               ▼ [Data-Driven Profiling & Strategy]
            6. Business Action Plan & CRM Integration
            ```
            """)

        st.divider()
        st.subheader("💡 Key Discovered Customer Segments")
        if saved_metrics and 'segment_profiles' in saved_metrics:
            cols = st.columns(len(saved_metrics['segment_profiles']))
            for i, prof in enumerate(saved_metrics['segment_profiles']):
                with cols[i]:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h4>🏷️ {prof['SegmentName']}</h4>
                        <p><strong>Customers:</strong> {prof['CustomerCount']:,} ({prof['Percentage']}%)</p>
                        <p><strong>Median Recency:</strong> {prof['Recency_Median']:.0f} days</p>
                        <p><strong>Median Orders:</strong> {prof['Frequency_Median']:.0f} invoices</p>
                        <p><strong>Median Spend:</strong> £{prof['Monetary_Median']:,.2f}</p>
                        <hr style='margin: 8px 0;'>
                        <p style='font-size: 0.9rem; color: #374151;'><em>{prof['RecommendedAction']}</em></p>
                    </div>
                    """, unsafe_allow_html=True)

    # =========================================================================
    # PAGE 2: DATA OVERVIEW & PREPROCESSING
    # =========================================================================
    elif choice == "📊 Data Overview & Preprocessing":
        st.markdown("<div class='main-header'>📊 Dataset Exploration & Preprocessing Pipeline</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub-header'>Data audit, transaction cleaning, RFM feature engineering, and distribution analysis</div>", unsafe_allow_html=True)

        # Cleaning funnel metrics
        st.subheader("🧹 Data Cleaning & Preprocessing Funnel")
        f1, f2, f3, f4, f5 = st.columns(5)
        f1.metric("Raw Transactions", f"{metadata['initial_transactions']:,}")
        f2.metric("Missing CustomerID", f"-{metadata['missing_customer_rows']:,}", "Dropped (24.9%)")
        f3.metric("Invalid Qty/Price", f"-{metadata['invalid_transactions_removed']:,}", "Cancellations / Errors")
        f4.metric("Duplicates", f"-{metadata['duplicates_removed']:,}", "Cleaned")
        f5.metric("Usable Transactions", f"{metadata['usable_transactions']:,}", "Final Cleaned")

        st.divider()

        # Distribution tabs
        st.subheader("📈 RFM Feature Distributions (Raw vs Log-Transformed)")
        st.markdown("""
        Retail transaction metrics inherently exhibit severe **positive right-skewness** due to high-spending power users and long-tail casual buyers.
        Applying `log1p(x)` normalizes variance, improves Gaussian assumption compliance for GMM, and prevents Euclidean distance distortion in K-Means.
        """)

        skew_raw = metadata['skewness_raw']
        skew_log = metadata['skewness_log']

        tab1, tab2, tab3 = st.tabs(["🕒 Recency", "🔁 Frequency", "💰 Monetary Spend"])

        with tab1:
            col_a, col_b = st.columns(2)
            with col_a:
                fig = px.histogram(customer_features, x='Recency', nbins=35, title=f"Raw Recency (Skewness: {skew_raw['Recency']:.2f})", color_discrete_sequence=['#2563EB'])
                fig.update_layout(xaxis_title="Recency (Days since last purchase)", yaxis_title="Customer Count")
                st.plotly_chart(fig, use_container_width=True)
            with col_b:
                fig = px.histogram(customer_features, x='LogRecency', nbins=35, title=f"Log-Transformed Recency (Skewness: {skew_log['LogRecency']:.2f})", color_discrete_sequence=['#10B981'])
                fig.update_layout(xaxis_title="LogRecency = log1p(Recency)", yaxis_title="Customer Count")
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            col_a, col_b = st.columns(2)
            with col_a:
                fig = px.histogram(customer_features, x='Frequency', nbins=35, title=f"Raw Frequency (Skewness: {skew_raw['Frequency']:.2f})", color_discrete_sequence=['#2563EB'])
                fig.update_layout(xaxis_title="Frequency (Unique Invoices)", yaxis_title="Customer Count")
                st.plotly_chart(fig, use_container_width=True)
            with col_b:
                fig = px.histogram(customer_features, x='LogFrequency', nbins=35, title=f"Log-Transformed Frequency (Skewness: {skew_log['LogFrequency']:.2f})", color_discrete_sequence=['#10B981'])
                fig.update_layout(xaxis_title="LogFrequency = log1p(Frequency)", yaxis_title="Customer Count")
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            col_a, col_b = st.columns(2)
            with col_a:
                fig = px.histogram(customer_features, x='Monetary', nbins=35, title=f"Raw Monetary (Skewness: {skew_raw['Monetary']:.2f})", color_discrete_sequence=['#2563EB'])
                fig.update_layout(xaxis_title="Monetary Spend (£ / GBP)", yaxis_title="Customer Count")
                st.plotly_chart(fig, use_container_width=True)
            with col_b:
                fig = px.histogram(customer_features, x='LogMonetary', nbins=35, title=f"Log-Transformed Monetary (Skewness: {skew_log['LogMonetary']:.2f})", color_discrete_sequence=['#10B981'])
                fig.update_layout(xaxis_title="LogMonetary = log1p(Monetary)", yaxis_title="Customer Count")
                st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Correlation and PCA
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🔥 Feature Correlation Matrix")
            corr_cols = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'LogRecency', 'LogFrequency', 'LogMonetary']
            corr_df = customer_features[corr_cols].corr()
            fig = px.imshow(corr_df, text_auto=".2f", aspect="auto", color_continuous_scale="Blues", title="Pearson Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("🌐 2D PCA Dimensionality Reduction")
            st.markdown(f"""
            - **PC1 Variance Explained:** {metadata['pca_explained_variance_ratio'][0]:.2%}
            - **PC2 Variance Explained:** {metadata['pca_explained_variance_ratio'][1]:.2%}
            - **Total 2D Explained Variance:** **{metadata['pca_total_explained_variance']:.2%}**
            """)
            pca_df = pd.DataFrame({
                'PC1': X_pca[:, 0],
                'PC2': X_pca[:, 1],
                'Monetary': customer_features['Monetary'],
                'Recency': customer_features['Recency'],
            })
            fig = px.scatter(
                pca_df,
                x='PC1',
                y='PC2',
                color='Monetary',
                color_continuous_scale='Viridis',
                title="PCA 2D Distribution (Colored by Monetary Spend)",
                opacity=0.6,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("📋 Customer RFM Feature Table")
        st.dataframe(customer_features.head(100), use_container_width=True)

    # =========================================================================
    # PAGE 3: CLUSTERING ANALYSIS & MODELS
    # =========================================================================
    elif choice == "🔬 Clustering Analysis & Models":
        st.markdown("<div class='main-header'>🔬 Unsupervised Algorithm Evaluation</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub-header'>In-depth comparative exploration of K-Means, Gaussian Mixture Models, and Hierarchical Clustering</div>", unsafe_allow_html=True)

        algo_tab1, algo_tab2, algo_tab3 = st.tabs([
            "1️⃣ K-Means Clustering",
            "2️⃣ Gaussian Mixture Models (GMM)",
            "3️⃣ Hierarchical Agglomerative Clustering"
        ])

        # ---------------------------------------------------------------------
        # TAB 1: K-MEANS
        # ---------------------------------------------------------------------
        with algo_tab1:
            st.subheader("K-Means Partitioning Analysis")
            km_results, best_km = get_kmeans_results(X_scaled)

            col_ctrl, col_plot = st.columns([1, 2])
            with col_ctrl:
                st.markdown(f"""
                <div class='winner-card'>
                    <h4>Optimal K-Means Selection</h4>
                    <p><strong>Best K:</strong> {best_km['k']} Clusters</p>
                    <p><strong>Silhouette Score:</strong> {best_km['silhouette']:.4f}</p>
                    <p><strong>Davies-Bouldin Index:</strong> {best_km['davies_bouldin']:.4f}</p>
                    <p><strong>Inertia (WCSS):</strong> {best_km['inertia']:,.2f}</p>
                </div>
                """, unsafe_allow_html=True)

                k_selected = st.slider("Interactive K Selector:", min_value=2, max_value=12, value=int(best_km['k']), key="km_k_slider")
                curr_km = next((r for r in km_results if r['k'] == k_selected), best_km)

            with col_plot:
                km_df = pd.DataFrame([
                    {'K': r['k'], 'Silhouette': r['silhouette'], 'DBI': r['davies_bouldin'], 'Inertia': r['inertia']}
                    for r in km_results
                ])
                fig = make_subplots(rows=1, cols=2, subplot_titles=("Inertia Elbow Curve", "Silhouette Score by K"))
                fig.add_trace(go.Scatter(x=km_df['K'], y=km_df['Inertia'], mode='lines+markers', name='Inertia', line=dict(color='#2563EB', width=2.5)), row=1, col=1)
                fig.add_trace(go.Scatter(x=km_df['K'], y=km_df['Silhouette'], mode='lines+markers', name='Silhouette', line=dict(color='#10B981', width=2.5)), row=1, col=2)
                fig.add_vline(x=best_km['k'], line_dash="dash", line_color="red", annotation_text=f"Optimal K={best_km['k']}", row=1, col=2)
                fig.update_layout(height=380, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"#### PCA 2D Scatter for K-Means (K = {k_selected})")
            pca_km_df = customer_features.copy()
            pca_km_df['Cluster'] = [f"Cluster {l}" for l in curr_km['labels']]
            pca_km_df['PC1'] = X_pca[:, 0]
            pca_km_df['PC2'] = X_pca[:, 1]

            fig = px.scatter(
                pca_km_df,
                x='PC1',
                y='PC2',
                color='Cluster',
                hover_data=['CustomerID', 'Recency', 'Frequency', 'Monetary'],
                title=f"K-Means Customer Clusters (K={k_selected}) in PCA 2D Space",
                opacity=0.7,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Candidate Metric Table (K=2..12)")
            st.dataframe(km_df, use_container_width=True)

        # ---------------------------------------------------------------------
        # TAB 2: GMM
        # ---------------------------------------------------------------------
        with algo_tab2:
            st.subheader("Gaussian Mixture Models (Soft Clustering)")
            gmm_results, best_gmm = get_gmm_results(X_scaled)
            gmm_df = pd.DataFrame([
                {
                    'Covariance': r['covariance_type'],
                    'Components': r['n_components'],
                    'Silhouette': r['silhouette'],
                    'DBI': r['davies_bouldin'],
                    'AIC': r['aic'],
                    'BIC': r['bic'],
                }
                for r in gmm_results
            ])

            col_g1, col_g2 = st.columns([1, 2])
            with col_g1:
                st.markdown(f"""
                <div class='metric-card'>
                    <h4>Optimal GMM Configuration</h4>
                    <p><strong>Covariance:</strong> {best_gmm['covariance_type'].title()}</p>
                    <p><strong>Components:</strong> {best_gmm['n_components']}</p>
                    <p><strong>Silhouette Score:</strong> {best_gmm['silhouette']:.4f}</p>
                    <p><strong>Davies-Bouldin Index:</strong> {best_gmm['davies_bouldin']:.4f}</p>
                    <p><strong>AIC / BIC:</strong> {best_gmm['aic']:,.1f} / {best_gmm['bic']:,.1f}</p>
                </div>
                """, unsafe_allow_html=True)

                ambig_thresh = st.slider("Ambiguity Confidence Threshold:", min_value=0.50, max_value=0.90, value=0.60, step=0.05)
                uncertainty = analyze_gmm_uncertainty(best_gmm['model'], X_scaled, threshold=ambig_thresh)
                st.metric("Ambiguous Customers (< Threshold)", f"{uncertainty['ambiguous_count']:,}", f"{uncertainty['ambiguous_percentage']:.2f}% of Total")

            with col_g2:
                fig = px.line(
                    gmm_df,
                    x='Components',
                    y='Silhouette',
                    color='Covariance',
                    markers=True,
                    title="GMM Silhouette Score by Covariance Type & Components",
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Soft Assignment Probability Distribution")
            fig_hist = px.histogram(
                x=uncertainty['max_probabilities'],
                nbins=35,
                title=f"Maximum Cluster Membership Probability (Mean: {uncertainty['mean_max_probability']:.2%})",
                color_discrete_sequence=['#6366F1'],
            )
            fig_hist.add_vline(x=ambig_thresh, line_dash="dash", line_color="red", annotation_text=f"Threshold ({ambig_thresh:.2f})")
            fig_hist.update_layout(xaxis_title="Max Posterior Probability", yaxis_title="Customer Count")
            st.plotly_chart(fig_hist, use_container_width=True)

        # ---------------------------------------------------------------------
        # TAB 3: HIERARCHICAL CLUSTERING
        # ---------------------------------------------------------------------
        with algo_tab3:
            st.subheader("Hierarchical Agglomerative Clustering")
            hier_results, best_hier, coph_results = get_hierarchical_results(X_scaled)

            col_h1, col_h2 = st.columns([1, 2])
            with col_h1:
                st.markdown(f"""
                <div class='metric-card'>
                    <h4>Optimal Hierarchical Model</h4>
                    <p><strong>Linkage:</strong> {best_hier['method'].title()}</p>
                    <p><strong>Metric:</strong> {best_hier['metric'].title()}</p>
                    <p><strong>Clusters:</strong> {best_hier['n_clusters']}</p>
                    <p><strong>Silhouette Score:</strong> {best_hier['silhouette']:.4f}</p>
                    <p><strong>Davies-Bouldin Index:</strong> {best_hier['davies_bouldin']:.4f}</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("##### Cophenetic Correlation")
                coph_df = pd.DataFrame([
                    {'Linkage': r['method'].title(), 'Metric': r['metric'].title(), 'Cophenetic Correlation': r['cophenetic_correlation']}
                    for r in coph_results
                ])
                st.dataframe(coph_df, use_container_width=True)

            with col_h2:
                st.markdown("##### Dendrogram Tree Structure (Ward Linkage)")
                dendro_path = Path('final_artifacts/dendrogram.png')
                if dendro_path.exists():
                    st.image(str(dendro_path), caption="Hierarchical Agglomerative Dendrogram with Cut Threshold (K=2)")
                else:
                    st.info("Dendrogram image is being generated...")

    # =========================================================================
    # PAGE 4: MODEL COMPARISON & BENCHMARKS
    # =========================================================================
    elif choice == "⚖️ Model Comparison & Benchmarks":
        st.markdown("<div class='main-header'>⚖️ Cross-Algorithm Benchmark & Model Comparison</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub-header'>Fair, reproducible evaluation of candidate models on the standardized feature space</div>", unsafe_allow_html=True)

        comp_path = Path('final_artifacts/final_model_comparison.csv')
        if comp_path.exists():
            comp_df = pd.read_csv(comp_path)
        else:
            comp_df = pd.DataFrame()

        # Winner Announcement Callout
        st.markdown(r"""
        <div class='winner-card'>
            <h3>🏆 Overall Recommended Model: K-Means (K = 2)</h3>
            <p style='font-size: 1.05rem;'>
                <strong>Evaluation Rationale:</strong> K-Means achieves the highest Silhouette Score (<strong>0.4328</strong>) and the lowest Davies-Bouldin Index (<strong>0.8925</strong>). It exhibits high computational efficiency \(O(n \cdot k \cdot d \cdot i)\), deterministic convergence, and clean separation between high-engagement and low-engagement customer cohorts.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("📊 Comparative Benchmark Table")
        st.dataframe(comp_df, use_container_width=True)

        st.divider()

        # Side-by-side comparison charts
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                comp_df,
                x='Algorithm',
                y='Silhouette_Score',
                color='Algorithm',
                text_auto='.4f',
                title="Silhouette Score Comparison (Higher is Better)",
                color_discrete_sequence=['#2563EB', '#10B981', '#F59E0B']
            )
            fig.update_layout(showlegend=False, yaxis_title="Silhouette Score")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                comp_df,
                x='Algorithm',
                y='Davies_Bouldin_Index',
                color='Algorithm',
                text_auto='.4f',
                title="Davies-Bouldin Index Comparison (Lower is Better)",
                color_discrete_sequence=['#2563EB', '#10B981', '#F59E0B']
            )
            fig.update_layout(showlegend=False, yaxis_title="Davies-Bouldin Index")
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Theoretical Trade-off Matrix
        st.subheader("🧠 Algorithmic Trade-off Analysis")
        st.markdown(r"""
        | Evaluation Dimension | K-Means | Gaussian Mixture Model (GMM) | Hierarchical Agglomerative |
        | :--- | :--- | :--- | :--- |
        | **Mathematical Formulation** | Hard centroid partitioning minimizing WCSS | Probabilistic mixture of Gaussians with EM | Agglomerative pairwise linkage merging |
        | **Cluster Geometry Assumption** | Spherical, isotropic clusters | Ellipsoidal clusters (full/diag/tied covariance) | Arbitrary geometry depending on linkage |
        | **Cluster Assignment Type** | Hard (Deterministic $0$ or $1$) | Soft (Posterior probability $P(C_k \mid x)$) | Hard (Dendrogram tree threshold cut) |
        | **Computational Complexity** | $O(n \cdot k \cdot d \cdot i)$ — Extremely fast | $O(n \cdot k \cdot d^3 \cdot i)$ — Moderate (EM steps) | $O(n^2 \log n)$ to $O(n^3)$ — High memory/CPU |
        | **Scalability to Large Data** | Highly scalable to millions of records | Scalable with mini-batch / diagonal cov | Poor scalability; quadratic memory footprint |
        | **Outlier Sensitivity** | Sensitive to extreme outliers | Robust through probabilistic density estimation | Ward is robust; Single/Average prone to chaining |
        """)

    # =========================================================================
    # PAGE 5: FINAL CUSTOMER SEGMENTATION
    # =========================================================================
    elif choice == "🎯 Final Customer Segmentation":
        st.markdown("<div class='main-header'>🎯 Final Customer Segmentation & Strategic Recommendations</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub-header'>Actionable marketing strategies, customer profiling, and CRM-ready export</div>", unsafe_allow_html=True)

        prof_path = Path('final_artifacts/final_segment_profiles.csv')
        seg_path = Path('final_artifacts/final_customer_segments.csv')

        if prof_path.exists() and seg_path.exists():
            profiles_df = pd.read_csv(prof_path)
            segments_df = pd.read_csv(seg_path)
        else:
            km_results, best_km = get_kmeans_results(X_scaled)
            profiles_df = create_final_segment_profiles(customer_features, best_km['labels'])
            segments_df = customer_features.copy()
            segments_df['Cluster'] = best_km['labels']
            segment_map = dict(zip(profiles_df['ClusterID'], profiles_df['SegmentName']))
            segments_df['Segment'] = segments_df['Cluster'].map(segment_map)
            segments_df['PC1'] = np.round(X_pca[:, 0], 4)
            segments_df['PC2'] = np.round(X_pca[:, 1], 4)

        # Segment Profile Cards
        st.subheader("👥 Customer Segment Breakdown")
        cols = st.columns(len(profiles_df))
        for i, row in profiles_df.iterrows():
            with cols[i]:
                st.markdown(f"""
                <div class='winner-card'>
                    <h3>🏷️ {row['SegmentName']}</h3>
                    <p><strong>Customer Base:</strong> {int(row['CustomerCount']):,} ({row['Percentage']}%)</p>
                    <p><strong>Median Recency:</strong> {row['Recency_Median']:.0f} days (Mean: {row['Recency_Mean']:.1f})</p>
                    <p><strong>Median Frequency:</strong> {row['Frequency_Median']:.0f} orders (Mean: {row['Frequency_Mean']:.1f})</p>
                    <p><strong>Median Monetary:</strong> £{row['Monetary_Median']:,.2f} (Mean: £{row['Monetary_Mean']:,.2f})</p>
                    <p><strong>Median Order Value:</strong> £{row['AvgOrderValue_Median']:,.2f}</p>
                    <hr style='margin: 8px 0;'>
                    <p><strong>Characteristics:</strong> {row['Characteristics']}</p>
                    <p style='color: #1E40AF;'><strong>Marketing Strategy:</strong> {row['RecommendedAction']}</p>
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        # Visualizations
        c_left, c_right = st.columns([3, 2])
        with c_left:
            st.subheader("🌐 PCA 2D Cluster Space")
            fig = px.scatter(
                segments_df,
                x='PC1',
                y='PC2',
                color='Segment',
                hover_data=['CustomerID', 'Recency', 'Frequency', 'Monetary', 'AvgOrderValue'],
                title="Customer Segments in 2D Principal Component Space",
                color_discrete_sequence=['#2563EB', '#10B981', '#F59E0B', '#EF4444'],
                opacity=0.75,
            )
            st.plotly_chart(fig, use_container_width=True)

        with c_right:
            st.subheader("🥧 Segment Proportions")
            fig_pie = px.pie(
                profiles_df,
                names='SegmentName',
                values='CustomerCount',
                title="Customer Base Share by Segment",
                color_discrete_sequence=['#2563EB', '#10B981', '#F59E0B', '#EF4444'],
                hole=0.4,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.divider()

        # Searchable Customer Assignment Table & CSV Export
        st.subheader("📥 CRM Customer Data Table & Export")
        selected_seg = st.selectbox("Filter by Segment:", ["All Segments"] + list(profiles_df['SegmentName'].unique()))

        if selected_seg != "All Segments":
            display_df = segments_df[segments_df['Segment'] == selected_seg]
        else:
            display_df = segments_df

        st.dataframe(display_df.head(500), use_container_width=True)

        csv_data = segments_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Complete Segmented Customer Data (CSV)",
            data=csv_data,
            file_name="final_customer_segments.csv",
            mime="text/csv",
            key="download_customer_csv",
        )

    # =========================================================================
    # PAGE 6: ABOUT / METHODOLOGY
    # =========================================================================
    elif choice == "📖 About & Methodology":
        st.markdown("<div class='main-header'>📖 Artificial Intelligence Methodology & References</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub-header'>Theoretical formulations, evaluation metrics, course details, and academic references</div>", unsafe_allow_html=True)

        st.subheader("📐 Mathematical Formulations of Validation Metrics")
        st.markdown(r"""
        ### 1. Silhouette Score
        For each sample $i$, let $a(i)$ be the mean intra-cluster distance to all other points in the same cluster, and $b(i)$ be the mean nearest-cluster distance:
        $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad s(i) \in [-1, 1]$$
        The global Silhouette Score is the average $s(i)$ across all $n$ samples. A value closer to $+1$ indicates well-separated, cohesive clusters.

        ### 2. Davies-Bouldin Index (DBI)
        Let $R_{ij} = \frac{s_i + s_j}{d(c_i, c_j)}$, where $s_i$ is cluster dispersion and $d(c_i, c_j)$ is centroid distance:
        $$DB = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} R_{ij}$$
        Lower DBI values signify clusters with lower intra-cluster dispersion and higher inter-cluster separation.

        ### 3. K-Means Objective Function (Within-Cluster Sum of Squares)
        $$J(C) = \sum_{k=1}^{K} \sum_{x_i \in C_k} \| x_i - \mu_k \|^2$$

        ### 4. Gaussian Mixture Model Log-Likelihood & Information Criteria
        $$\ln p(X \mid \theta) = \sum_{i=1}^{N} \ln \left( \sum_{k=1}^{K} \pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k) \right)$$
        $$AIC = 2p - 2\ln L, \quad BIC = p\ln N - 2\ln L$$
        """)

        st.divider()

        st.subheader("🎓 Student Project & Course Details")
        st.markdown("""
        - **Module:** Artificial Intelligence / Machine Learning
        - **Project Title:** Customer Segmentation Using Unsupervised Machine Learning
        - **Student Name:** [STUDENT NAME]
        - **Student ID:** [STUDENT ID]
        - **Tutorial Group / Class:** [TUTORIAL GROUP]
        - **Dataset Reference:** Daqing Chen (2015), *Online Retail Dataset*, UCI Machine Learning Repository, DOI: 10.24432/C5BW33.
        """)


if __name__ == '__main__':
    main()