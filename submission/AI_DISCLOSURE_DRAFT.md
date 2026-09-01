# Academic AI Tool Usage & Integrity Declaration

**Module / Course:** Artificial Intelligence / Machine Learning  
**Assignment Title:** Customer Segmentation Using Unsupervised Machine Learning  
**Student Name(s):** [STUDENT NAME 1], [STUDENT NAME 2], [STUDENT NAME 3]  
**Student ID(s):** [STUDENT ID 1], [STUDENT ID 2], [STUDENT ID 3]  
**Tutorial Group / Class:** [TUTORIAL GROUP]  
**Tutor / Lecturer:** [TUTOR / LECTURER NAME]  
**Submission Date:** [SUBMISSION DATE]

---

## 1. Declaration of Generative AI Tool Usage

In accordance with university academic integrity policies and guidelines, the authors disclose that Generative Artificial Intelligence (GenAI) tools were utilized in the development, code refactoring, verification, and documentation drafting of this assignment.

The submitting students have actively reviewed, executed, debugged, validated, and verified all code, mathematical formulations, empirical metrics, and written analyses contained within this submission.

---

## 2. Generative AI Tools Employed

| AI Tool / Platform | Developer / Provider | Primary Purpose & Scope of Assistance |
| :--- | :--- | :--- |
| **ChatGPT (GPT-4o / Reasoning Models)** | OpenAI | Assignment rubric interpretation, architectural planning, generation and refinement of repository completion prompts, and review guidance. |
| **[CODING AGENT / MODEL USED, e.g. Gemini 2.5 Pro / Antigravity Assistant]** | [DEVELOPER, e.g. Google DeepMind] | Codebase auditing, script refactoring, bug fixing, test execution, generation of empirical artifacts, Streamlit UI enhancement, and initial documentation drafting. |

---

## 3. Example Prompts Used During Project Development

Below are representative examples of prompts utilized during the project lifecycle:

### Prompt 1: Repository Audit & Pipeline Standardization
> *"Audit the customer-segmentation repository against the university AI assignment rubric. Standardize the data preprocessing pipeline on the canonical Online Retail dataset, implement and evaluate K-Means, Gaussian Mixture Models, and Hierarchical Agglomerative Clustering across K=2..12, generate reproducible final artifacts, complete the Streamlit prototype, and draft submission documentation."*

### Prompt 2: Controlled Empirical Benchmarking & Feature Engineering
> *"Ensure the feature engineering pipeline uses the Log-RFM representation (`LogRecency`, `LogFrequency`, `LogMonetary`) with `StandardScaler` to correct severe right-skewness without multicollinear feature double-counting. Ensure PCA is used strictly for 2D visualization without altering the 3D training space. Export all evaluation metrics to CSV, JSON, and PNG plots."*

### Prompt 3: Rubric Alignment & Final Submission Correction
> *"Perform a final submission-correction pass focusing on rubric alignment, factual correctness, currency units (ensure all dataset values use GBP £ instead of $), AI disclosure, and ensure all empirical numbers in the final report match the actual output files with zero fabricated data."*

---

## 4. Human Oversight & Verification Protocol

To ensure academic rigor, empirical validity, and compliance with course learning outcomes, the following verification steps were conducted:

1. **Local Code Execution**: All Python scripts (`prep.py`, `customer_segmentation.py`, `GMM.py`, `hierarchical.py`, `generate_final_results.py`, `app.py`) were executed locally within the project virtual environment.
2. **Deterministic Artifact Verification**: The master script `generate_final_results.py` was executed end-to-end (runtime ~134s) to produce all 15 final artifacts in `final_artifacts/`.
3. **Cross-Validation of Numerical Data**: All numerical statistics in `submission/FINAL_REPORT.md` and `submission/REPORT_DATA.md` were directly cross-referenced against `final_artifacts/final_metrics.json` and `final_artifacts/final_model_comparison.csv` to ensure zero fabricated or hallucinated values.
4. **Interactive Dashboard Testing**: The Streamlit application was imported and tested across all 6 pages to ensure zero runtime exceptions and clean visualization rendering.
5. **Human Review & Defense Readiness**: All group members have reviewed the code implementation, understand the underlying mathematical theory, and are prepared to defend their work during live examination and Q&A.

---

## 5. Software & Open-Source Libraries Used

*(Note: The following open-source software libraries were utilized for computational modeling and are distinct from generative AI tools)*

- **Python (v3.10+)**: Core programming runtime.
- **Scikit-Learn (v1.6+)**: Implementations of K-Means, Gaussian Mixture Models, Agglomerative Clustering, StandardScaler, PCA, and evaluation metrics (Silhouette Score, Davies-Bouldin Index).
- **SciPy (v1.15+)**: Distance computations, hierarchical linkage matrices, and cophenetic correlation analysis.
- **Pandas (v2.2+) & NumPy (v2.1+)**: Transactional data manipulation, aggregation, and vectorised mathematical operations.
- **Streamlit (v1.42+) & Plotly (v6.0+)**: Interactive web dashboard development and data visualization.

---

## 6. Student Signatures & Integrity Confirmation

I/We confirm that this submission is our own collective work, that the disclosures above are accurate, and that we take full academic responsibility for the contents of this project.

| Student Name | Student ID | Signature | Date |
| :--- | :--- | :--- | :--- |
| [STUDENT NAME 1] | [STUDENT ID 1] | ________________________ | ______________ |
| [STUDENT NAME 2] | [STUDENT ID 2] | ________________________ | ______________ |
| [STUDENT NAME 3] | [STUDENT ID 3] | ________________________ | ______________ |
