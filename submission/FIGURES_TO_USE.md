# Figure Placement & Usage Guide

This guide maps each artifact generated in `final_artifacts/` to its corresponding section in the academic report (`submission/FINAL_REPORT.md`), presentation slides, and evaluation demo.

---

## Artifact Index & Placement Map

| Figure Filename | Artifact Path | Recommended Section in Report | Description & Key Insights to Highlight |
| :--- | :--- | :--- | :--- |
| **Figure 1** | `final_artifacts/rfm_distributions.png` | **Section 3.2: Description and Analysis of Dataset** | Multi-panel histogram showing severe positive skewness in raw RFM metrics (Monetary skew +19.34) and the resulting near-normal symmetry achieved by `log1p` transformation (LogMonetary skew +0.40). |
| **Figure 2** | `final_artifacts/correlation_heatmap.png` | **Section 3.2: Description and Analysis of Dataset** | Pearson correlation matrix displaying inter-feature relationships and justifying why only log-transformed features are passed to clustering to avoid multicollinear weighting. |
| **Figure 3** | `final_artifacts/kmeans_selection.png` | **Section 3.3.1 & 4.1: K-Means Evaluation** | Three-panel plot displaying Inertia Elbow curve, Silhouette Score peak at $K=2$ ($0.4328$), and Davies-Bouldin Index minimum at $K=2$ ($0.8925$). |
| **Figure 4** | `final_artifacts/gmm_selection.png` | **Section 3.3.2 & 4.1: GMM Evaluation** | Four-panel plot showing Silhouette Score, Davies-Bouldin Index, AIC, and BIC across full, tied, diagonal, and spherical covariance structures. Spherical covariance at 2 components optimizes silhouette ($0.4307$). |
| **Figure 5** | `final_artifacts/gmm_probability_histogram.png` | **Section 3.3.2: GMM Uncertainty Analysis** | Posterior membership confidence distribution showing 93.5% mean confidence and identifying that only 4.10% (178 customers) fall into the ambiguous boundary zone ($P < 0.60$). |
| **Figure 6** | `final_artifacts/hierarchical_selection.png` | **Section 3.3.3 & 4.1: Hierarchical Evaluation** | Silhouette and DBI curves for Ward linkage across $K=2 \dots 12$, showing peak performance at $K=2$ ($0.4040$). |
| **Figure 7** | `final_artifacts/dendrogram.png` | **Section 3.3.3: Hierarchical Tree Analysis** | Stratified tree dendrogram displaying cluster merge distances and the optimal horizontal cut line corresponding to $K=2$. |
| **Figure 8** | `final_artifacts/model_comparison_silhouette.png` | **Section 4.1: Results (Model Comparison)** | Comparative bar chart illustrating K-Means ($0.4328$) outperforming GMM ($0.4307$) and Hierarchical ($0.4040$). |
| **Figure 9** | `final_artifacts/model_comparison_dbi.png` | **Section 4.1: Results (Model Comparison)** | Comparative bar chart demonstrating K-Means achieving the lowest Davies-Bouldin Index ($0.8925$). |
| **Figure 10** | `final_artifacts/final_pca_segments.png` | **Section 4.2: Discussion / Interpretation** | 2D PCA scatter plot showing distinct spatial separation between High-Value Active Customers and Low-Engagement Lapsed Spenders with marked centroids. |
| **Figure 11** | `final_artifacts/final_segment_sizes.png` | **Section 4.2: Discussion / Interpretation** | Customer distribution chart showing the 61.6% vs 38.4% partition of the customer base. |

---

## Embedding Syntax for Markdown & LaTeX

### In Markdown Documents
```markdown
![Figure 1: RFM Distributions Before and After Log Transformation](final_artifacts/rfm_distributions.png)
![Figure 3: K-Means Selection Curves](final_artifacts/kmeans_selection.png)
![Figure 7: Hierarchical Dendrogram](final_artifacts/dendrogram.png)
![Figure 10: Final PCA Customer Segments](final_artifacts/final_pca_segments.png)
```

### In LaTeX Reports
```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.85\textwidth]{final_artifacts/final_pca_segments.png}
    \caption{Final Customer Segments Projected in 2D PCA Latent Space ($K=2$ K-Means).}
    \label{fig:pca_segments}
\end{figure}
```
