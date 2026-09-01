# Online Retail Customer Segmentation: Empirical Reference Data

This document contains the exact numerical results, empirical metrics, and statistical benchmarks generated from the canonical execution of the customer segmentation pipeline. All values in the academic report, presentation slides, and evaluation rubrics correspond directly to these numbers.

---

## 1. Dataset & Preprocessing Pipeline Statistics

| Metric | Empirical Value | Description / Rationale |
| :--- | :--- | :--- |
| **Dataset Name** | Online Retail Dataset | Canonical giftware transaction dataset from UCI Machine Learning Repository |
| **Citation / DOI** | Daqing Chen (2015) / `10.24432/C5BW33` | Transactions occurring from non-store online retail |
| **Observation Period** | `2010-12-01` to `2011-12-09` | ~1.02 years of longitudinal transactional records |
| **Currency** | Pound Sterling (£ / GBP) | Standard currency of transaction logs |
| **Initial Raw Records** | **541,909** transactions | Full raw dataset shape: `(541909, 8)` |
| **Missing CustomerID Rows** | **135,080** rows (24.93%) | Dropped; cannot attribute customer-level behavior without identifiers |
| **Invalid Records Removed** | **8,945** rows | Cancellations (`Quantity <= 0`) and zero/negative unit prices (`UnitPrice <= 0`) |
| **Duplicate Lines Removed** | **5,192** rows | Exact duplicate transaction rows removed |
| **Final Usable Transactions** | **392,692** transactions | High-quality cleaned transactional records |
| **Final Usable Customers** | **4,338** unique customers | Aggregate customer cohort for unsupervised learning |

---

## 2. RFM Feature Distribution & Normalization Statistics

| Feature Dimension | Raw Mean | Raw Median | Raw Std Dev | Raw Skewness | Transformed Feature | Log1p Skewness | Skewness Reduction |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Recency (Days)** | 92.54 | 51.00 | 100.01 | **+1.25** | `LogRecency` | **-0.38** | Moderate normal symmetry |
| **Frequency (Invoices)** | 4.27 | 2.00 | 7.70 | **+12.07** | `LogFrequency` | **+1.21** | 90.0% reduction in skewness |
| **Monetary Spend (£)** | £2,054.27 | £674.49 | £8,989.23 | **+19.34** | `LogMonetary` | **+0.40** | 97.9% reduction in skewness |
| **Avg Order Value (£)** | £419.17 | £293.90 | £1,796.54 | **+41.69** | `LogAvgOrderValue` | **+0.24** | 99.4% reduction in skewness |

### Principal Component Analysis (PCA) on Standardized Log-RFM Space
- **PC1 Explained Variance:** **75.08%**
- **PC2 Explained Variance:** **18.79%**
- **Cumulative 2D Explained Variance:** **93.87%**
- *Note:* PCA was fitted strictly for 2D visualization without reducing the 3D feature space used during model training.

---

## 3. Algorithm Grid Search & Optimization Results

### A. K-Means Clustering Evaluation Grid ($K = 2 \dots 12$)

| Clusters ($K$) | Silhouette Score ($\uparrow$) | Davies-Bouldin Index ($\downarrow$) | Inertia (WCSS) ($\downarrow$) | Evaluation Status |
| :---: | :---: | :---: | :---: | :--- |
| **2** | **0.4328** | **0.8925** | **6,483.59** | **Optimal Configuration (Global Silhouette Peak)** |
| 3 | 0.3365 | 1.0483 | 4,869.49 | Secondary Local Optimum |
| 4 | 0.3375 | 1.0086 | 3,939.05 | Granular Business Partition |
| 5 | 0.3162 | 0.9878 | 3,296.71 | Over-partitioning inflection point |
| 6 | 0.3124 | 1.0210 | 2,855.76 | Sub-optimal |
| 7 | 0.3092 | 0.9823 | 2,548.82 | Sub-optimal |
| 8 | 0.3033 | 0.9892 | 2,336.34 | Sub-optimal |
| 9 | 0.2811 | 1.0193 | 2,156.01 | Sub-optimal |
| 10 | 0.2767 | 1.0268 | 2,005.75 | Sub-optimal |
| 11 | 0.2748 | 1.0280 | 1,872.68 | Sub-optimal |
| 12 | 0.2750 | 1.0279 | 1,767.40 | Sub-optimal |

---

### B. Gaussian Mixture Model (GMM) Evaluation Summary

- **Best Candidate Configuration:** 2 components, `spherical` covariance
- **Silhouette Score ($\uparrow$):** **0.4307**
- **Davies-Bouldin Index ($\downarrow$):** **0.9023**
- **Akaike Information Criterion (AIC):** **32,563.45**
- **Bayesian Information Criterion (BIC):** **32,620.83**
- **Mean Posterior Probability (Confidence):** **93.50%**
- **Ambiguous Customers ($P_{\max} < 0.60$):** **178 customers** (4.10% of cohort)
- *Note:* The 2-component spherical GMM was selected primarily for cluster separation under the common Silhouette-based comparison policy. AIC and BIC are reported as model-fit diagnostics.

---

### C. Hierarchical Agglomerative Clustering Summary

- **Best Actionable Configuration:** 2 clusters, `ward` linkage, `euclidean` metric
- **Silhouette Score ($\uparrow$):** **0.4040**
- **Davies-Bouldin Index ($\downarrow$):** **0.9405**
- **Cophenetic Correlation (Ward / Euclidean):** **0.6096** (moderate diagnostic value confirming tree distance preservation)

---

## 4. Fair Cross-Model Comparison Table

| Algorithm | Best Configuration | Number of Clusters ($K$) | Silhouette Score ($\uparrow$) | Davies-Bouldin Index ($\downarrow$) | Algorithm-Specific Diagnostic | Computational Complexity | Rank |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- | :---: |
| **K-Means** | **$K = 2$** | **2** | **0.4328** | **0.8925** | **Inertia = 6,483.59** | **$O(n \cdot k \cdot d \cdot i)$ [Fast]** | **1 (Winner)** |
| **Gaussian Mixture Model (GMM)** | 2 components, `spherical` cov | 2 | **0.4307** | **0.9023** | AIC = 32,563.45, BIC = 32,620.83 | $O(n \cdot k \cdot d^3 \cdot i)$ [EM Probabilistic] | 2 |
| **Hierarchical Agglomerative** | 2 clusters, `ward` linkage, Euclidean | 2 | **0.4040** | **0.9405** | Cophenetic Corr = 0.6096 | $O(n^2 \log n)$ to $O(n^3)$ [Tree] | 3 |

---

## 5. Final Customer Segment Profiles (Recommended Model: K-Means)

| Cluster ID | Segment Name | Customer Count | Share (%) | Recency Median (Mean) | Frequency Median (Mean) | Monetary Median (Mean) | Avg Order Value Median (Mean) | Core Behavioral Trait |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Cluster 0** | **Low-Engagement / Lapsed Spenders** | **2,672** | **61.60%** | **96.0 days** (134.09) | **1.0 order** (1.67) | **£363.08** (£495.59) | **£239.40** (£320.20) | Infrequent, long absence, low cumulative spend |
| **Cluster 1** | **High-Value Active Customers** | **1,666** | **38.40%** | **16.0 days** (25.89) | **6.0 orders** (8.44) | **£2,061.08** (£4,539.60) | **£346.26** (£573.93) | Highly recent, recurring purchases, high monetary value |
