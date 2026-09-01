# Live Demonstration Script: Customer Segmentation Using Unsupervised AI

**Project Title:** Customer Segmentation Using Unsupervised Machine Learning  
**Target Duration:** 5 – 7 Minutes  
**Audience:** Course Lecturer, AI Teaching Assistants, and Academic Examiners  
**Presenters:** [STUDENT NAME] ([STUDENT ID]), Tutorial Group: [TUTORIAL GROUP]

---

## ⏱️ Timeline & Presentation Flow Overview

| Stage | Dashboard Page | Time Allocation | Key Objective |
| :--- | :--- | :---: | :--- |
| **Phase 1** | `🏠 Home` | 1:00 min | Set problem context, e-commerce challenges, and pipeline architecture |
| **Phase 2** | `📊 Data Overview & Preprocessing` | 1:30 min | Explain cleaning decisions, RFM feature engineering, and log normalization |
| **Phase 3** | `🔬 Clustering Analysis & Models` | 2:00 min | Demonstrate K-Means, GMM uncertainty, and Hierarchical Dendrogram |
| **Phase 4** | `⚖️ Model Comparison & Benchmarks` | 1:00 min | Present empirical comparison table, metrics, and algorithmic trade-offs |
| **Phase 5** | `🎯 Final Customer Segmentation` | 1:00 min | Showcase business segment profiles, marketing strategies, and CRM data export |

---

## 🎙️ Step-by-Step Script & Click-by-Click Guide

### Phase 1: Introduction & Architecture (0:00 – 1:00)
**Action:** Launch the dashboard (`streamlit run app.py`) and remain on the **🏠 Home** page.

**Spoken Script:**
> *"Good morning/afternoon, Professor and Examiners. Today, we present our Artificial Intelligence project on **Customer Segmentation Using Unsupervised Machine Learning**.*
> 
> *In digital commerce, customer behavior is highly heterogeneous. Treating all users identically results in inefficient marketing expenditure and customer churn. Our objective was to engineer a defensible, mathematically grounded unsupervised learning pipeline on real-world transactional data from the UCI Online Retail dataset.*
> 
> *As shown on our Home page overview, we preprocessed over 540,000 raw transaction logs into 4,338 unique customer profiles. We developed and benchmarked three distinct unsupervised paradigms: **K-Means**, **Gaussian Mixture Models**, and **Hierarchical Agglomerative Clustering** to establish an optimal, data-driven customer segmentation strategy."*

---

### Phase 2: Data Preprocessing & Feature Engineering (1:00 – 2:30)
**Action:** Click on the sidebar: **📊 Data Overview & Preprocessing**. Scroll down through the Cleaning Funnel and the RFM Distributions.

**Spoken Script:**
> *"Moving to our data pipeline: Real-world retail data is inherently noisy and severely skewed. We enforced strict cleaning rules: dropping 135,080 records lacking Customer IDs, removing cancellations with negative quantities, and eliminating duplicates, leaving 392,692 valid transactions.*
> 
> *Next, we aggregated transactions into the classic **RFM framework**: **Recency** (days since last transaction), **Frequency** (unique order count), **Monetary** (total spend), and **Average Order Value**.*
> 
> *Crucially, as seen in these distribution charts, raw retail metrics exhibit extreme positive skewness—Monetary spend had a skewness of +19.34. Feeding raw features directly into distance-based algorithms heavily distorts Euclidean distances due to high-spending outliers. We resolved this by applying a `log1p` transformation, reducing Monetary skewness to +0.40 and Frequency skewness from +12.07 to +1.21.*
> 
> *To prevent multicollinearity, we passed only the log-transformed RFM features into `StandardScaler`. Our 2D PCA retains **93.87%** of the total multidimensional variance, guaranteeing high-fidelity geometric representations."*

---

### Phase 3: Algorithm Deep Dive & Interactive Analysis (2:30 – 4:30)
**Action:** Click on **🔬 Clustering Analysis & Models**. Walk through all three tabs.

#### Tab 1: K-Means
- **Action:** Select Tab 1. Move the interactive slider to $K=2$, then briefly to $K=4$.
- **Spoken Script:**
  > *"On Tab 1, we evaluated K-Means across $K=2$ to $12$. The Silhouette score peaks decisively at $K=2$ with **0.4328** and a minimum Davies-Bouldin Index of **0.8925**. The 2D PCA projection clearly separates the high-activity customer cluster from the low-engagement cohort."*

#### Tab 2: Gaussian Mixture Models (GMM)
- **Action:** Select Tab 2. Point to the Covariance lines, then adjust the Ambiguity slider.
- **Spoken Script:**
  > *"On Tab 2, we implemented GMM to explore soft probabilistic clustering. We evaluated full, tied, diagonal, and spherical covariance matrices across 2 to 12 components. Spherical covariance with 2 components proved optimal with a Silhouette score of **0.4307** and an average assignment confidence of **93.50%**. By analyzing posterior probabilities, we identified that only 4.10% of customers are ambiguous ($P < 0.60$), residing along the decision boundary."*

#### Tab 3: Hierarchical Agglomerative Clustering
- **Action:** Select Tab 3. Point to the Cophenetic correlation table and the generated dendrogram.
- **Spoken Script:**
  > *"On Tab 3, we evaluated Hierarchical Clustering across Ward, Complete, Average, and Single linkages. Ward linkage with Euclidean distance produced cohesive, balanced clusters with a Silhouette score of **0.4040** and a Cophenetic correlation of **0.6096**, avoiding the chaining artifacts that plague single linkage."*

---

### Phase 4: Cross-Model Comparison & Winner Selection (4:30 – 5:30)
**Action:** Click on **⚖️ Model Comparison & Benchmarks**. Point to the Comparative Benchmark Table and Bar Charts.

**Spoken Script:**
> *"Here on the Model Comparison page, we present our empirical benchmark. Under identical standardized feature spaces:
> - **K-Means ($K=2$)** ranks #1 with the highest Silhouette score (**0.4328**) and lowest DBI (**0.8925**).
> - **GMM (2 components, spherical)** ranks #2 with a Silhouette score of **0.4307**.
> - **Hierarchical (Ward, $K=2$)** ranks #3 with a Silhouette score of **0.4040**.
> 
> *Beyond metric performance, K-Means offers superior computational complexity of $O(n \cdot k \cdot d \cdot i)$, making it the ideal production-grade model for dynamic retail segmentation."*

---

### Phase 5: Business Segments & CRM Action Plan (5:30 – 6:30)
**Action:** Click on **🎯 Final Customer Segmentation**. Highlight the two segment profile cards, the PCA scatter plot, and click the Download CSV button.

**Spoken Script:**
> *"Finally, we translated our unsupervised findings into actionable business strategies:
> 
> 1. **High-Value Active Customers (38.4% of base / 1,666 customers)**: Median spend of **$2,061.08**, median of 6 orders, and highly recent activity (median 16 days). We recommend VIP loyalty tiers, dedicated account managers, and exclusive product previews to maximize Customer Lifetime Value (CLV).
> 2. **Low-Engagement / Lapsed Spenders (61.6% of base / 2,672 customers)**: Median spend of only **$363.08**, single orders, and long inactivity (median 96 days). We recommend automated re-engagement email sequences, discount triggers, and low-friction catalog suggestions.
> 
> *The entire segmented dataset is exportable via our CSV download button for immediate ingestion into enterprise CRM platforms like Salesforce or HubSpot.*
> 
> *Thank you. We welcome your questions."*

---

## 🎯 Quick Emergency Troubleshooting Tips for Live Demo

- **If port 8501 is busy:** Run `.venv/bin/streamlit run app.py --server.port 8502`.
- **If charts take time to render:** The caching `@st.cache_data` ensures fast subsequent interactions once initialized.
- **If asked for raw statistics:** Point directly to `submission/REPORT_DATA.md` or the Data Overview page table.
