# Viva & Lecturer Q&A Preparation Guide

This document contains 30 comprehensive, technically rigorous questions and model answers designed to prepare students for live examination, viva voce defense, and Q&A sessions.

---

## Category 1: Problem Formulation & Preprocessing

### Q1: Why did you drop records with missing `CustomerID` instead of imputing them?
**Answer:**
> *"In customer segmentation, `CustomerID` is the fundamental entity key around which longitudinal transactional behaviors are aggregated. A missing `CustomerID` (which accounted for 135,080 rows or ~24.9% of the dataset) represents unregistered guest checkouts. Imputing a synthetic customer ID or aggregating them into a single dummy ID would introduce catastrophic bias, artificially fabricating a 'mega-customer' with over 130,000 orders and distorting the entire RFM distribution. Removing unauthenticated records is standard scientific practice in customer analytics literature (Chen, Sain & Guo, 2012)."*

### Q2: How did you handle negative quantities and zero unit prices?
**Answer:**
> *"Negative quantities in the UCI Online Retail dataset denote transaction cancellations, merchandise returns (prefixed with 'C' in `InvoiceNo`), or inventory write-offs. Zero unit prices represent promotional giveaways or administrative adjustments. Because our objective is to measure baseline purchasing propensity and monetary value, we filtered out records where `Quantity <= 0` or `UnitPrice <= 0` (removing 8,945 invalid records), ensuring only genuine completed purchases were aggregated into RFM metrics."*

### Q3: Why did you apply `log1p` transformation to RFM features?
**Answer:**
> *"E-commerce retail metrics exhibit extreme positive (right) skewness due to the Pareto distribution, where a small fraction of power buyers generate the majority of spend. For instance, raw Monetary spend had a skewness coefficient of **+19.34**, and Frequency had **+12.07**. Unsupervised algorithms relying on Euclidean distances (like K-Means and Ward linkage) are sensitive to extreme outliers, which pull centroids away from dense data regions. Applying `log1p(x) = \ln(1 + x)` reduced Monetary skewness to **+0.40** and Frequency skewness to **+1.21**, restoring near-normal symmetry while preserving monotonicity and non-negativity."*

### Q4: Why did you not include both raw features and log features in clustering?
**Answer:**
> *"Including both an original feature and its log-transformed counterpart (e.g., `Monetary` and `LogMonetary`) doubles the geometric dimensionality and introduces severe multicollinearity. In Euclidean space, this artificially doubles the relative weight of that business dimension in distance calculations. We preserved raw RFM features in our database for human-interpretable business profiling, but fed strictly the log-transformed normalized features (`LogRecency`, `LogFrequency`, `LogMonetary`) into `StandardScaler` for algorithm training."*

### Q5: Why did you not include `AvgOrderValue` in the final clustering feature set?
**Answer:**
> *"By definition, Average Order Value is a deterministic mathematical ratio of Monetary divided by Frequency: $\text{AOV} = \frac{\text{Monetary}}{\text{Frequency}}$. Including AOV alongside Frequency and Monetary introduces collinear dependency. In empirical testing, clustering on the 3-feature log-RFM space yielded higher PCA 2D variance explanation (**93.87%** vs 88.03%) and higher Silhouette Scores (**0.4328** vs 0.2788) than the 4-feature setup."*

### Q6: Why is `StandardScaler` necessary after log transformation?
**Answer:**
> *"Even after log transformation, features have different empirical variances and scales. Without standardization, a feature with a standard deviation of 2.5 would dominate Euclidean distance calculations over a feature with a standard deviation of 0.8. `StandardScaler` standardizes each feature to have $\mu = 0$ and $\sigma = 1$, ensuring all three behavioral axes contribute equally to cluster formation."*

---

## Category 2: Algorithm Mechanics & Hyperparameter Selection

### Q7: How does K-Means optimize cluster assignments?
**Answer:**
> *"K-Means is a centroid-based partitioning algorithm that minimizes the Within-Cluster Sum of Squares (WCSS), also known as Inertia:
> $$J = \sum_{k=1}^{K} \sum_{x_i \in C_k} \| x_i - \mu_k \|^2$$
> It operates iteratively via Lloyd's algorithm:
> 1. **Assignment step**: Each observation $x_i$ is assigned to the nearest centroid $\mu_k$ based on Euclidean distance.
> 2. **Update step**: Centroids $\mu_k$ are recomputed as the arithmetic mean of all points assigned to cluster $k$.
> The algorithm terminates when centroid shifts fall below a convergence threshold ($10^{-4}$) or maximum iterations are reached. We used `n_init=10` with K-Means++ initialization and `random_state=42` for deterministic reproducibility."*

### Q8: Why did K-Means select $K=2$ as the optimal number of clusters?
**Answer:**
> *"We conducted a systematic grid search over $K = 2 \dots 12$. The Silhouette Score peaked at $K=2$ with **0.4328** and the Davies-Bouldin Index reached its minimum at **0.8925**. At $K=2$, the data exhibits a clear separation between highly active, frequent shoppers and dormant, low-spend occasional buyers. While $K=4$ offers granular sub-tiers (Silhouette 0.3375), $K=2$ represents the mathematical global optimum in terms of geometric cluster compactness and separation."*

### Q9: How does Gaussian Mixture Model (GMM) differ from K-Means?
**Answer:**
> *"While K-Means performs hard, deterministic spherical partitioning, GMM is a generative probabilistic model that assumes the dataset is generated from a mixture of $K$ multivariate Gaussian distributions:
> $$p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)$$
> Key differences:
> 1. **Soft Clustering**: GMM provides posterior probabilities $P(C_k \mid x_i)$ for each cluster, quantifying assignment uncertainty.
> 2. **Flexible Geometry**: Through the covariance matrix $\Sigma_k$ (`full`, `tied`, `diag`, `spherical`), GMM can model ellipsoidal clusters with varying orientations, unlike K-Means which enforces spherical clusters."*

### Q10: How does the Expectation-Maximization (EM) algorithm work in GMM?
**Answer:**
> *"GMM is optimized using the EM algorithm:
> - **E-step (Expectation)**: Compute the responsibilities (posterior probabilities) $\gamma_{ik}$ that component $k$ generated data point $x_i$:
>   $$\gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}$$
> - **M-step (Maximization)**: Update mixture weights $\pi_k$, means $\mu_k$, and covariance matrices $\Sigma_k$ using the computed responsibilities to maximize the expected log-likelihood.
> Iteration continues until the log-likelihood gain falls below $10^{-3}$."*

### Q11: What covariance structures did you test in GMM and which performed best?
**Answer:**
> *"We evaluated all four covariance types across components $2 \dots 12$:
> 1. `full`: Each component has its own general covariance matrix.
> 2. `tied`: All components share the same general covariance matrix.
> 3. `diag`: Each component has a diagonal covariance matrix (axes-aligned).
> 4. `spherical`: Each component has a single variance parameter ($\sigma_k^2 I$).
> 
> The **spherical covariance** with 2 components achieved the highest GMM Silhouette Score (**0.4307**), indicating that after our log-transformation and standardization, the feature space is isotropic, making complex full covariance structures prone to slight parameter overfitting."*

### Q12: How did you diagnose uncertainty in GMM clustering?
**Answer:**
> *"We analyzed the distribution of maximum assignment probabilities $\max_k P(C_k \mid x_i)$. The mean assignment confidence across all 4,338 customers was **93.50%**. By defining an ambiguity threshold of $P_{\max} < 0.60$, we identified that only **178 customers (4.10%)** occupy the boundary region between clusters. These ambiguous customers represent transitioning buyers who can be targeted with targeted marketing tests."*

### Q13: How does Hierarchical Agglomerative Clustering work?
**Answer:**
> *"Hierarchical Agglomerative Clustering is a bottom-up clustering technique. It initializes with each customer as a single-element cluster ($N$ clusters) and iteratively merges the closest pair of clusters until only a single cluster remains, producing a dendrogram tree.
> The distance between clusters depends on the linkage criterion:
> - **Ward**: Minimizes total within-cluster variance increase (uses Euclidean distance).
> - **Complete**: Maximum pairwise distance between points in clusters.
> - **Average**: Average pairwise distance between points in clusters.
> - **Single**: Minimum pairwise distance between points in clusters."*

### Q14: Why is Ward linkage preferred over Single and Average linkage in retail segmentation?
**Answer:**
> *"Single linkage suffers from the **chaining phenomenon**, where intermediate points connect distant clusters into a single giant chain, isolating outliers into 1-element clusters (e.g., 4,337 vs 1). Average linkage also produced degenerate splits (4,334 vs 4).
> In contrast, **Ward linkage** minimizes the increase in within-cluster sum of squares upon merging, directly encouraging cohesive, balanced clusters. Ward linkage produced balanced clusters of 2,672 and 1,666 customers with a Silhouette Score of **0.4040** and a Cophenetic Correlation of **0.6096**."*

### Q15: What is the Cophenetic Correlation Coefficient?
**Answer:**
> *"The Cophenetic Correlation Coefficient ($c$) measures how faithfully the hierarchical dendrogram preserves the original pairwise distances between data points in the input space:
> $$c = \frac{\sum_{i < j} (d_{ij} - \bar{d})(t_{ij} - \bar{t})}{\sqrt{\sum_{i < j} (d_{ij} - \bar{d})^2 \sum_{i < j} (t_{ij} - \bar{t})^2}}$$
> where $d_{ij}$ is the Euclidean distance between points $i$ and $j$, and $t_{ij}$ is the dendrogram height at which they are first merged. Ward linkage achieved $c = 0.6096$, providing moderate supporting diagnostic evidence of structural consistency."*

---

## Category 3: Validation Metrics & Model Comparison

### Q16: How is the Silhouette Score calculated and what does it measure?
**Answer:**
> *"The Silhouette Score measures how similar an object is to its own cluster compared to other clusters. For sample $i$:
> $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
> where $a(i)$ is the mean distance from $i$ to all other points in the same cluster (compactness), and $b(i)$ is the mean distance from $i$ to all points in the nearest neighboring cluster (separation).
> The global score is the average across all samples, bounded between $-1$ (incorrect clustering) and $+1$ (dense, well-separated clusters). K-Means achieved $s = 0.4328$."*

### Q17: What does the Davies-Bouldin Index (DBI) represent?
**Answer:**
> *"The Davies-Bouldin Index evaluates clustering quality based on the ratio of within-cluster scatter to between-cluster separation:
> $$R_{ij} = \frac{s_i + s_j}{d(c_i, c_j)}, \quad DB = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} R_{ij}$$
> where $s_i$ is average distance of points to their centroid, and $d(c_i, c_j)$ is distance between centroids. A lower DBI signifies superior clustering (tight clusters that are far apart). K-Means achieved the lowest DBI of **0.8925**."*

### Q18: What are AIC and BIC and why are they used for GMM?
**Answer:**
> *"Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC) are penalized likelihood criteria used for model selection:
> $$\text{AIC} = 2p - 2\ln L, \quad \text{BIC} = p\ln N - 2\ln L$$
> where $p$ is number of free parameters, $N$ is sample size, and $L$ is maximized likelihood.
> They evaluate model fit while penalizing parameter complexity. In GMM, they serve as model-fit diagnostics alongside internal cluster validation metrics."*

### Q19: Why was K-Means selected as the overall winner over GMM and Hierarchical?
**Answer:**
> *"K-Means was selected because:
> 1. **Empirical Performance**: Highest Silhouette Score (**0.4328**) and lowest Davies-Bouldin Index (**0.8925**).
> 2. **Computational Efficiency**: Runs in $O(n \cdot k \cdot d \cdot i)$ time (fast execution on 4,338 rows), whereas Hierarchical requires $O(n^2)$ memory for pairwise distances.
> 3. **Deployment Feasibility**: Readily assigns new incoming streaming transactions via nearest-centroid lookup without retraining."*

---

## Category 4: Dimensionality Reduction & Visualization

### Q20: Why did you use PCA for visualization rather than t-SNE or UMAP?
**Answer:**
> *"PCA is a linear orthogonal transformation that preserves global geometric variance. Fitting 2 PCA components captured **93.87%** of the total variance (PC1: 75.08%, PC2: 18.79%).
> In contrast, t-SNE and UMAP are non-linear stochastic embeddings that do not preserve global distances and can create misleading visual density artifacts. PCA provided an exact, geometrically faithful 2D projection."*

---

## Category 5: Business Impact & Strategic Recommendations

### Q21: Describe the two final customer segments and their business profiles.
**Answer:**
> *"Our winning model identified two distinct customer segments:
> 1. **High-Value Active Customers (Cluster 1 / 38.4% / 1,666 customers)**: Highly recent (median 16 days vs pop median 51), frequent buyers (median 6 orders), high spend (median **£2,061.08** vs pop median £674.49, mean £4,539.60).
> 2. **Low-Engagement / Lapsed Spenders (Cluster 0 / 61.6% / 2,672 customers)**: Infrequent (median 1 order), high latency (median 96 days since last purchase), low spend (median **£363.08**, mean £495.59)."*

### Q22: What actionable strategies do you propose for High-Value Active Customers?
**Answer:**
> *"For High-Value Active Customers (Cluster 1):
> - **VIP Loyalty Program**: Tiered rewards and exclusive cashback points.
> - **Early Access**: Pre-launch previews for new giftware lines.
> - **Dedicated Support**: Priority customer service to maintain engagement.
> - **Upsell / Cross-Sell Bundles**: Recommendations based on item affinity (`StockCode`) to increase basket size."*

### Q23: What actionable strategies do you propose for Low-Engagement / Lapsed Spenders?
**Answer:**
> *"For Low-Engagement / Lapsed Spenders (Cluster 0):
> - **Automated Win-Back Workflows**: Triggered email campaigns with time-limited discounts (e.g., '10% off your next order').
> - **Churn Surveys**: Micro-surveys to understand friction points in user experience or pricing.
> - **Re-engagement Catalog**: Highlighting best-sellers under £20 to trigger second purchases."*

---

## Category 6: Engineering Rigor & Code Architecture

### Q24: How did you ensure reproducibility in your codebase?
**Answer:**
> *"We enforced reproducibility through:
> 1. **Fixed Random Seeds**: `random_state=42` across K-Means, GMM, and PCA.
> 2. **Deterministic Algorithm Parameters**: `n_init=10` for K-Means, `reg_covar=1e-6` for GMM.
> 3. **Headless Generation Script**: `generate_final_results.py` exports identical CSVs, PNGs, and JSON metrics on any machine without human intervention.
> 4. **Automated Data Retrieval**: `prep.py` loads the canonical UCI dataset with automated path detection."*

### Q25: How does your Streamlit application ensure high performance?
**Answer:**
> *"We leveraged Streamlit's `@st.cache_data` decorators across the dataset loader, K-Means evaluation, GMM fitting, and Hierarchical clustering. Heavy computations run once and remain cached in memory, ensuring responsive UI page transitions and slider updates."*

### Q26: What are the main limitations of the current segmentation model?
**Answer:**
> *"Key limitations:
> 1. **Feature Scope**: RFM captures behavioral transaction history but lacks demographic, psychographic, and web clickstream data.
> 2. **Temporal Stationarity**: The dataset spans December 2010 to December 2011; seasonality affects recency. A production pipeline could implement rolling-window RFM.
> 3. **Static Snapshots**: Customer segments shift over time; dynamic clustering or hidden Markov models would capture transition probabilities."*

### Q27: What future improvements would you implement?
**Answer:**
> *"Future extensions:
> 1. **Customer Lifetime Value (CLV) Prediction**: Integrating Gamma-Gamma and BG/NBD probabilistic models to forecast future purchasing frequency and monetary spend.
> 2. **Product-Level Embeddings**: Applying NLP text embeddings on transaction item descriptions (`Description`) to segment by product category affinity.
> 3. **Real-Time Streaming Ingestion**: Deploying the K-Means centroid model as a lightweight microservice to assign segments to new customers dynamically."*
