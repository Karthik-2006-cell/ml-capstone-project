# 23CSE301 Machine Learning — Capstone Project (Review 2)

**Course:** B.Tech Computer Science & Engineering — 23CSE301 Machine Learning  
**Academic Year:** 2026-27  
**Evaluation Scope:** Review 2 (Classification Track Part B + Full Clustering Track + Bonus Interactive Deployment)

---

##  1. Project Overview & Problem Statements

This project implements an end-to-end Machine Learning pipeline across supervised and unsupervised domains under formal course capstone guidelines.

###  Track 1: Bank Marketing Term Deposit Classification (Supervised)
* **Dataset:** Portuguese Bank Direct Marketing Campaign (`bank.csv` / 41,188 instances, 20 features)
* **Problem Statement:** Predict whether a retail banking client will subscribe to a long-term deposit product (`y`: yes/no) based on client attributes, past campaign interactions, and macro-economic conditions.
* **Review 1 Scope (Part A):** Logistic Regression, K-Nearest Neighbors, Naive Bayes (Gaussian), Decision Tree Classifier, Support Vector Machine (SVC).
* **Review 2 Scope (Part B):** Random Forest Classifier, AdaBoost Classifier, Gradient Boosting Classifier (GBM), Bagging Classifier, Multi-Layer Perceptron (MLP Neural Network).
* **Optimization & Selection:** Identification of the top candidate model, hyperparameter tuning via `GridSearchCV`, and leak-free 5-fold cross-validation.

###  Track 2: Credit Card Customer Segmentation (Unsupervised)
* **Dataset:** Credit Card Usage & Balance Behavior (`CC GENERAL.csv` / 8,950 accounts, 18 features)
* **Problem Statement:** Discover natural customer segments and spending personas using unsupervised clustering without ground-truth labels during training.
* **Algorithms Implemented:** K-Means Clustering and Agglomerative Hierarchical Clustering.
* **Evaluation Metrics:** Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index.
* **Visualizations:** Elbow curve, Dendrogram with cut-off threshold, PCA 2D projections, t-SNE 2D manifold embeddings, and cluster persona profiling.

---

##  2. Consolidated Benchmark Results Summary Tables

### Classification Track (10-Algorithm Consolidated Comparison Table — Rubric A2)

| # | Classification Algorithm | Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score | ROC-AUC |
|---|--------------------------|:--------:|:------------------:|:---------------:|:-----------------:|:-------:|
| 1 | **8. Gradient Boosting Classifier (Tuned)** | **0.9022** | **0.8879** | **0.9022** | **0.8841** | **0.7985** |
| 2 | 8. Gradient Boosting Classifier (Baseline) | 0.9016 | 0.8872 | 0.9016 | 0.8834 | 0.7964 |
| 3 | 6. Random Forest Classifier | 0.9031 | 0.8864 | 0.9031 | 0.8828 | 0.7958 |
| 4 | 9. Bagging Classifier (Decision Tree Base) | 0.9020 | 0.8875 | 0.9020 | 0.8831 | 0.7925 |
| 5 | 4. Decision Tree Classifier | 0.9019 | 0.8870 | 0.9019 | 0.8829 | 0.7865 |
| 6 | 1. Logistic Regression | 0.9004 | 0.8852 | 0.9004 | 0.8787 | 0.7954 |
| 7 | 10. MLP Classifier (Neural Net) | 0.9016 | 0.8861 | 0.9016 | 0.8815 | 0.7842 |
| 8 | 5. Support Vector Machine (SVC) | 0.9003 | 0.8842 | 0.9003 | 0.8812 | 0.7051 |
| 9 | 2. K-Nearest Neighbors (KNN) | 0.8921 | 0.8736 | 0.8921 | 0.8774 | 0.7418 |
| 10 | 7. AdaBoost Classifier | 0.9003 | 0.8835 | 0.9003 | 0.8795 | 0.7812 |
| 11 | 3. Naive Bayes (Gaussian) | 0.8139 | 0.8673 | 0.8139 | 0.8344 | 0.7685 |

* **Final Model Selection:** **Gradient Boosting Classifier (GBM)** achieves the best balance between discriminatory capacity (**ROC-AUC: 0.7985**) and classification performance (**Weighted F1: 0.8841**).

---

### Clustering Track (Mandatory Evaluation Metrics Table — Rubric B2)

| # | Clustering Algorithm | Silhouette Score (Higher is Better) | Davies-Bouldin Index (Lower is Better) | Calinski-Harabasz Index (Higher is Better) |
|---|----------------------|:-----------------------------------:|:--------------------------------------:|:------------------------------------------:|
| 1 | **K-Means Clustering ($k=4$)** | **0.2185** | **1.4820** | **1648.50** |
| 2 | Agglomerative Hierarchical (Ward, $k=4$) | 0.1782 | 1.5640 | 1320.15 |

* **Optimal Partition:** Both Elbow curve and Dendrogram threshold confirm that **$k=4$** represents the optimal balance of cohesion and separation.
* **Customer Personas Discovered:**
  1. **Cluster 0 — Active Installment Shoppers:** Moderate balances, consistent installment purchasing patterns.
  2. **Cluster 1 — Cash Advance Borrowers:** Elevated balance, frequent ATM cash advances, minimal purchase frequency.
  3. **Cluster 2 — VIP High Spenders & Transactors:** High transaction volume, large credit limits, substantial payments.
  4. **Cluster 3 — Conservative / Low Activity Users:** Low balance, minimal utilization, dormant cardholders.

---

##  3. Environment Setup & Installation

### Requirements
* Python 3.10 or higher
* Recommended virtual environment: `venv` or `conda`

### Installation Steps
```bash
# 1. Clone repository
git clone <repository-url>
cd notebooks

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

##  4. How to Run

### Running Jupyter Notebooks
All notebooks are configured to run top-to-bottom without manual modifications:
```bash
# Launch Jupyter Lab or Notebook
jupyter notebook

# Open and execute:
# - Classification.ipynb  (Review 2 Full Classification Track)
# - Clustering.ipynb      (Review 2 Full Clustering Track)
# - Regression.ipynb      (Review 1 Regression Track)
```

### Running the Interactive Web App (Bonus GUI — +2 Marks)
An interactive Streamlit dashboard is included to demonstrate live inference:
```bash
streamlit run app.py
```
* **Local URL:** `http://localhost:8501`
* **Features:**
  * Real-time Bank Deposit Subscription Predictor with client profiles and probability gauge.
  * Interactive Credit Card Customer Segment Profiler with persona cards and financial loading charts.
  * Consolidated benchmark comparison tables and diagnostic visualizations.

---

##  5. 1-Click Public Cloud Deployment (Streamlit Community Cloud)

To claim the full **+2 bonus marks** in Review 2:
1. Push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub.
3. Click **"New App"**, choose repository, branch `main`, and enter `app.py` as main file path.
4. Click **"Deploy"** — the live public link is ready within 2 minutes!

---

##  6. Repository Structure
```
/ (root)
├── README.md                              # Project documentation and summary
├── requirements.txt                       # Version-pinned dependencies
├── bank.csv                               # Classification dataset
├── CC GENERAL.csv                         # Clustering dataset
├── kc_house_data.csv                      # Regression dataset
├── Classification.ipynb                   # Review 2 Classification Notebook (10 models)
├── Clustering.ipynb                       # Review 2 Clustering Notebook (K-Means & Hierarchical)
├── Regression.ipynb                       # Review 1 Regression Notebook (10 models)
├── app.py                                 # Streamlit interactive application
├── app/
│   └── app.py                             # Deployment application copy
└── models/
    ├── bank_marketing_best_model.joblib   # Serialized classification inference pipeline
    └── credit_card_clustering_pipeline.joblib # Serialized clustering scaler & model
```

---

##  7. Academic Integrity & Tool Citation
* All model pipelines, feature engineering transformations, and diagnostic visualizations were constructed strictly in adherence to the 23CSE301 Capstone Guidelines.
* Standard libraries utilized: `scikit-learn`, `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `streamlit`, and `joblib`.
