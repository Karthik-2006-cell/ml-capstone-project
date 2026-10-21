# 23CSE301 Machine Learning -- Capstone Project

**Course:** B.Tech Computer Science & Engineering -- 23CSE301 Machine Learning  
**Academic Year:** 2026-27  
**Evaluation Scope:** Review 1 & Review 2 (Regression Track + Classification Track + Clustering Track + Interactive Deployment)  
**Live Cloud Application:** [https://ml-capstone-project-sigma.streamlit.app/](https://ml-capstone-project-sigma.streamlit.app/)

---

## 1. Project Overview & Problem Statements

This project implements an end-to-end Machine Learning pipeline across supervised regression, supervised classification, and unsupervised clustering domains under formal course capstone guidelines.

### Track 1: King County Housing Price Prediction (Supervised Regression)
* **Dataset:** King County House Sales Dataset (`kc_house_data.csv` / 21,613 properties, 21 attributes)
* **Problem Statement:** Accurately estimate the fair market sale price (`price` in USD) of residential properties in King County, Washington (including Seattle) based on square footage, bedroom/bathroom configurations, structural grading, view ratings, waterfront status, renovation history, and geographic coordinates (latitude and longitude).
* **Algorithms Implemented (All 10 Models):**
  1. Linear Regression (Ordinary Least Squares Baseline)
  2. Ridge Regression (L2 Tikhonov Regularization)
  3. Lasso Regression (L1 Feature-Shrinkage Regularization)
  4. ElasticNet Regression (Convex Combination of L1 and L2)
  5. Polynomial Regression (Degree 2 Quadratic Expansion on Top Drivers)
  6. Decision Tree Regressor (CART Recursive Binary Splitting)
  7. Random Forest Regressor (Bootstrap Aggregation of 100 Trees)
  8. Gradient Boosting Regressor (Sequential Gradient Boosting with Learning Rate Shrinkage)
  9. Support Vector Regressor (SVR with Radial Basis Function Kernel)
  10. K-Nearest Neighbors Regressor (Instance-Based Distance Weighting)
* **Feature Engineering:** Logarithmic price transformation (`log1p`), temporal property age and years since renovation derivation, total living area interaction metrics, and geographic target encoding across 70 ZIP codes.

---

### Track 2: Bank Marketing Term Deposit Prediction (Supervised Classification)
* **Dataset:** Portuguese Bank Direct Marketing Campaign (`bank.csv` / 41,188 interactions, 20 attributes)
* **Problem Statement:** Predict whether a retail banking client will subscribe to a long-term deposit certificate (`y`: yes / no) following direct telemarketing contacts, while effectively managing extreme class imbalance (88.7% negative vs. 11.3% positive).
* **Algorithms Implemented (All 10 Models):**
  1. Logistic Regression (Calibrated Log-Odds with Balanced Class Weights)
  2. K-Nearest Neighbors (KNN Distance-Weighted Voting)
  3. Naive Bayes (Gaussian Prior Probability Smoothing)
  4. Decision Tree Classifier (Cost-Complexity Pruning and Balanced Weights)
  5. Support Vector Machine (SVC with Radial Basis Function Margin)
  6. Random Forest Classifier (Ensemble of 100 De-correlated Trees)
  7. AdaBoost Classifier (Sequential Weight Adaptation over Decision Stumps)
  8. Gradient Boosting Classifier (GBM with Cross-Entropy Loss Minimization)
  9. Bagging Classifier (Bootstrap Aggregation of 50 Decision Trees)
  10. Multi-Layer Perceptron (MLP Neural Network with (64, 32) Hidden Architecture)
* **Optimization & Selection:** Threshold calibration, ROC-AUC maximization, and leak-free 5-fold cross-validation.

---

### Track 3: Credit Card Customer Segmentation (Unsupervised Clustering)
* **Dataset:** Credit Card Customer Usage & Balance Portfolio (`CC GENERAL.csv` / 8,950 accounts, 18 features)
* **Problem Statement:** Discover natural behavioral segments and financial personas across credit card holders without relying on ground-truth training labels.
* **Algorithms Implemented:** K-Means Clustering and Agglomerative Hierarchical Clustering (Ward Linkage).
* **Evaluation Metrics:** Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index.
* **Visualizations & Diagnostics:** Elbow Inertia Curve, Dendrogram Tree with Linkage Threshold, PCA 2D Dimension Projections, and Customer Persona Loading Profiles.

---

## 2. Consolidated Benchmark Results Summary Tables

### Regression Track: 10-Algorithm Benchmark Comparison

Evaluated on King County Holdout Test Partition (4,323 test samples):

| # | Regression Algorithm | Test R-Squared (R2) | Test RMSE (USD) | Test MAE (USD) | Model Category |
|---|----------------------|:-------------------:|:---------------:|:--------------:|:---------------|
| 1 | **8. Gradient Boosting Regressor (Tuned)** | **0.8709** | **$139,696** | **$71,432** | **Ensemble Gradient Champion** |
| 2 | 7. Random Forest Regressor | 0.8554 | $147,847 | $72,943 | Ensemble Bagging Trees |
| 3 | 10. K-Nearest Neighbors (KNN) | 0.8102 | $169,391 | $85,422 | Spatial Instance-Based |
| 4 | 2. Ridge Regression (L2) | 0.8068 | $170,903 | $99,219 | L2 Regularized Parametric |
| 5 | 1. Linear Regression (OLS) | 0.8068 | $170,897 | $98,737 | Parametric Linear Baseline |
| 6 | 3. Lasso Regression (L1) | 0.8058 | $171,358 | $98,787 | L1 Sparse Regularized |
| 7 | 6. Decision Tree Regressor | 0.7890 | $178,584 | $94,776 | Tree-Based Partitioning |
| 8 | 5. Polynomial Regression (Degree 2) | 0.7495 | $194,607 | $113,038 | Non-Linear Quadratic Expansion |
| 9 | 4. ElasticNet Regression | 0.7256 | $203,664 | $117,694 | Hybrid L1/L2 Regularization |
| 10 | 9. Support Vector Regressor (SVR) | 0.5227 | $268,622 | $105,578 | Non-Linear Kernel Margin |

* **Regression Champion:** **Gradient Boosting Regressor** delivers the highest variance explanation (**R2 = 0.8709**) and lowest prediction error (**MAE = $71,432**).

---

### Classification Track: 10-Algorithm Benchmark Comparison

Evaluated on Portuguese Bank Holdout Test Partition (8,238 test samples):

| # | Classification Algorithm | Test ROC-AUC | Test Accuracy | Weighted F1-Score | Model Category |
|---|--------------------------|:------------:|:-------------:|:-----------------:|:---------------|
| 1 | **8. Gradient Boosting Classifier (Tuned)** | **0.7994** | **89.98%** | **0.8802** | **Ensemble Gradient Champion** |
| 2 | 6. Random Forest Classifier | 0.7989 | 90.00% | 0.8802 | Ensemble Bagged Forest |
| 3 | 10. MLP Classifier (Neural Net) | 0.7965 | 89.90% | 0.8818 | Deep Multi-Layer Perceptron |
| 4 | 1. Logistic Regression | 0.7954 | 90.04% | 0.8787 | Calibrated Log-Odds |
| 5 | 9. Bagging Classifier (Decision Tree Base) | 0.7944 | 90.10% | 0.8826 | Bootstrap Aggregated Trees |
| 6 | 4. Decision Tree Classifier | 0.7865 | 90.19% | 0.8829 | Interpretable Tree Rules |
| 7 | 7. AdaBoost Classifier | 0.7838 | 89.87% | 0.8722 | Adaptive Weight Boosting |
| 8 | 3. Naive Bayes (Gaussian NB) | 0.7685 | 81.39% | 0.8344 | Probabilistic Bayes |
| 9 | 2. K-Nearest Neighbors (KNN) | 0.7418 | 89.21% | 0.8774 | Instance-Based Voting |
| 10 | 5. Support Vector Machine (SVC) | 0.7051 | 90.03% | 0.8812 | Maximum Margin Hyperplane |

* **Classification Champion:** **Gradient Boosting Classifier** achieves the highest class discriminability (**ROC-AUC = 0.7994**) while maintaining high overall precision and recall on imbalanced subscription outcomes.

---

### Clustering Track: Unsupervised Validation Metrics

Evaluated on Credit Card Dataset (8,950 accounts, standardized 17 continuous indicators):

| # | Clustering Algorithm | Silhouette Score (Higher is Better) | Davies-Bouldin Index (Lower is Better) | Calinski-Harabasz Index (Higher is Better) |
|---|----------------------|:-----------------------------------:|:--------------------------------------:|:------------------------------------------:|
| 1 | **K-Means Clustering (k=4)** | **0.2185** | **1.4820** | **1648.50** |
| 2 | Agglomerative Hierarchical (Ward, k=4) | 0.1782 | 1.5640 | 1320.15 |

* **Identified Customer Segments:**
  1. **Cluster 0 -- Active Installment Shoppers:** Moderate balances, consistent installment purchasing patterns.
  2. **Cluster 1 -- Cash Advance Borrowers:** Elevated balance, frequent ATM cash advances, minimal purchase frequency.
  3. **Cluster 2 -- VIP High Spenders & Transactors:** High transaction volume, large credit limits, substantial payments.
  4. **Cluster 3 -- Conservative / Low Activity Users:** Low balance, minimal utilization, dormant cardholders.

---

## 3. Interactive Web Application & Cloud Deployment

* **Live Streamlit Cloud Deployment:** [https://ml-capstone-project-sigma.streamlit.app/](https://ml-capstone-project-sigma.streamlit.app/)
* **Local Run Command:** `streamlit run app.py`

### Interactive Application Features:
1. **Multi-Model Selection Engine:**
   * **Regression Track:** Select and evaluate any of the 10 trained regression models on demand.
   * **Classification Track:** Select and evaluate any of the 10 trained classification models on demand.
   * **Clustering Track:** Switch between K-Means and Agglomerative Hierarchical Clustering models.
2. **Instant Dynamic KPIs:** Header metric cards ($R^2$, RMSE, MAE or ROC-AUC, F1, Accuracy) dynamically update to reflect the exact test benchmark of the selected algorithm.
3. **Comparative Evaluation Tables:** Live expandable comparison view that benchmarks all 10 algorithms side-by-side with active model highlighting.
4. **1-Click Quick Preset Scenarios:**
   * *Urban Starter Home (Affordable):* Realistically priced entry-level home (~$182,934).
   * *Luxury Waterfront Estate:* High-tier King County estate (~$1.4M+).
   * *Rural Acreage Property:* Mid-to-high acreage home (~$720k+).
   * *Receptive Prime Customer:* Validated bank marketing profile yielding a positive subscription recommendation (**WILL SUBSCRIBE / YES**).
   * *Skeptical Cold Prospect:* Validated profile yielding a negative subscription recommendation (**WILL NOT SUBSCRIBE / NO**).
5. **Clean Interface Design:** Built strictly with high-contrast, pure ASCII, and professional zero-emoji formatting.

---

## 4. Environment Setup & Installation

### System Requirements
* Python 3.10, 3.11, or 3.12
* Recommended virtual environment: `venv` or `conda`

### Installation Steps
```bash
# 1. Clone the repository
git clone <repository-url>
cd notebooks

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Linux / macOS:
source venv/bin/activate

# 4. Install all dependencies
pip install -r requirements.txt
```

---

## 5. How to Run

### Option A: Open Live Cloud Web App (No Installation Needed)
Access the live deployed application instantly in your browser:  
[https://ml-capstone-project-sigma.streamlit.app/](https://ml-capstone-project-sigma.streamlit.app/)

### Option B: Run Web App Locally
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### Option C: Run Jupyter Notebooks
All notebooks can be executed sequentially from top to bottom:
```bash
jupyter notebook
```
* [Regression.ipynb](file:///c:/Users/kr978/Downloads/notebooks/Regression.ipynb) -- King County Housing Price Prediction (All 10 Regression Algorithms)
* [Classification.ipynb](file:///c:/Users/kr978/Downloads/notebooks/Classification.ipynb) -- Bank Marketing Term Deposit Prediction (All 10 Classification Algorithms)
* [Clustering.ipynb](file:///c:/Users/kr978/Downloads/notebooks/Clustering.ipynb) -- Credit Card Customer Segmentation (K-Means and Hierarchical Clustering)

---

## 6. Repository Structure

```
notebooks/
|-- README.md                              # Comprehensive project documentation
|-- requirements.txt                       # Version-pinned Python dependencies
|-- app.py                                 # Streamlit multi-model interactive dashboard
|-- app/
|   \-- app.py                             # Deployment application copy
|-- bank.csv                               # Portuguese bank marketing dataset
|-- CC GENERAL.csv                         # Credit card usage dataset
|-- kc_house_data.csv                      # King County property sales dataset
|-- Regression.ipynb                       # Supervised regression track notebook (10 algorithms)
|-- Classification.ipynb                   # Supervised classification track notebook (10 algorithms)
|-- Clustering.ipynb                       # Unsupervised clustering track notebook (k=4)
\-- models/
    |-- regression_models_bundle.joblib    # Bundle containing preprocessor + all 10 regression models
    |-- classification_models_bundle.joblib# Bundle containing preprocessor + all 10 classification models
    |-- credit_card_clustering_pipeline.joblib # Serialized clustering scaler and cluster centroids
    |-- kc_house_price_model.joblib        # Standalone best regression model pipeline (GBM)
    \-- bank_marketing_best_model.joblib   # Standalone best classification model pipeline (GBM)
```

---

## 7. Academic Integrity & Course Rubric Alignment

* **Supervised Regression:** All 10 algorithms implemented, evaluated via cross-validation, and ranked with $R^2$, RMSE, and MAE.
* **Supervised Classification:** All 10 algorithms implemented, evaluated across accuracy, precision, recall, weighted F1, and ROC-AUC.
* **Unsupervised Clustering:** Partitions validated with Silhouette Scores, Davies-Bouldin Index, Calinski-Harabasz Index, Elbow method, and Dendrogram cuts.
* **Bonus Mark Deployment:** Fully functional interactive web application with real-time inference and live cloud URL at [https://ml-capstone-project-sigma.streamlit.app/](https://ml-capstone-project-sigma.streamlit.app/).
