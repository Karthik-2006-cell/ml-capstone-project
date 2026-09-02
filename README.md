# Machine Learning Capstone: Regression & Classification Pipelines(Review 1)

## 1. Executive Summary & Problem Statements

This project implements robust, leak-free, modular machine learning pipelines across distinct operational paradigms:

### Track 1: Regression — King County Real Estate Valuation
* **Problem Statement:** Real estate price dynamics are shaped by multi-collinear physical attributes, architectural conditions, and temporal construction variances. The objective is to construct an end-to-end regression engine to predict residential housing market valuations (`price`) with minimal error variance and high explanatory power ($R^2$).
* **Dataset:** House Sales in King County, USA (`kc_house_data.csv`)
* **Key Challenge:** Addressing heavy positive skewness in transaction prices, handling multi-collinear dimension spaces (e.g., `sqft_living`, `sqft_above`, `grade`), and preventing data leakage during scaling and polynomial transformation.

### Track 2: Classification (Part A) — Bank Telemarketing Campaign Prediction
* **Problem Statement:** Financial institutions expend substantial capital on unsolicited client outreach for fixed-term deposit acquisition. The goal is to build a predictive classification system that identifies high-propensity subscribers (`y` $\in \{0, 1\}$) to maximize conversion efficiency.
* **Dataset:** Bank Marketing Data Set (`bank-full.csv` / `bank.csv`)
* **Key Challenge:** Managing severe class imbalance (~88% non-subscribers vs. ~12% subscribers) and building an isolated encoding pipeline for heterogeneous categorical features (e.g., job titles, marital status, education levels).





## 2. Exploratory Data Analysis & Preprocessing Protocols

### Regression Track

1. **Target Analysis:** Evaluated the distribution of `price`. The feature displays right-skewness, validated via Seaborn KDE plots.
2. **Feature Engineering:**
* `house_age`: Derived via yr_sold - yr_built to capture structural depreciation.
* `is_renovated`: Binary flag  (1 if yr_renovated > 0, else 0) mitigating sparse zero-inflated year entries.
* Removed non-predictive identifiers (`id`, `date`, `yr_built`, `yr_renovated`) to eliminate redundant multi-collinearity.


3. **Leakage-Free Splitting & Scaling:**
* Split: $80\%$ Training set, $20\%$ Test split using fixed seed `random_state=42`.
* Scaler: `StandardScaler` fitted strictly on the training partition and transformed across the test set to avoid distribution leakage.



### Classification Track (Part A)

1. **Target Encoding:** Mapped deposit acquisition flags (`no` / `yes`) directly to binary indicators ($0$ and $1$).
2. **Stratified Partitioning:** Implemented stratified train-test splitting ($80:20$) to maintain exact class proportions across validation boundaries.
3. **Pipeline Construction:**
* Utilized `ColumnTransformer` to isolate transformations.
* Numerical features: Standardized with `StandardScaler`.
* Categorical features: Encoded via `OneHotEncoder(drop='first', handle_unknown='ignore')` to avoid dummy variable traps.





## 3. Benchmark Results & Comparative Analysis

### 3.1. Regression Track — 10 Algorithm Benchmark

Evaluated on the exact same 20% held-out test set. Ranked by explanatory power ($R^2$ Score).

| Rank | Algorithm Category | Algorithm Name | $R^2$ Score | RMSE ($) | MAE ($) | Evaluation Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Ensemble Boosting | Gradient Boosting Regressor (Tuned) | 0.8924 | 118,452.12 | 68,340.40 | Captures complex non-linear spatial interactions. |
| 2 | Ensemble Bagging | Random Forest Regressor | 0.8812 | 124,475.89 | 71,890.15 | Robust against variance; baseline ensemble. |
| 3 | Non-Parametric | Decision Tree Regressor | 0.7845 | 167,310.22 | 96,412.30 | Pruned at `max_depth=10` to limit overfitting. |
| 4 | Polynomial / Non-linear | Polynomial Regression (deg=2) | 0.7610 | 176,140.05 | 114,890.75 | Degree 2 with Ridge regularizer. |
| 5 | Instance-Based | K-Nearest Neighbors Regressor | 0.7431 | 182,650.40 | 111,230.10 | Distance-weighted; highly dependent on scaling. |
| 6 | $L_2$ Regularized | Ridge Regression | 0.7011 | 197,320.18 | 125,840.60 | $\alpha=10.0$; penalizes extreme regression weights. |
| 7 | Linear Baseline | Linear Regression | 0.7010 | 197,330.45 | 125,870.12 | Standard ordinary least squares baseline. |
| 8 | $L_1$ Regularized | Lasso Regression | 0.7008 | 197,410.80 | 125,790.35 | $\alpha=100.0$; drives minor coefficients to zero. |
| 9 | Elastic Regularized | ElasticNet Regression | 0.6945 | 199,490.65 | 127,110.45 | $L_1/L_2$ hybrid penalty balance. |
| 10 | Kernel Method | Support Vector Regressor (SVR) | 0.5512 | 241,890.30 | 151,620.80 | RBF kernel; computationally bounded scaling. |

#### 5-Fold Cross-Validation (Top 2 Performers)

* **Gradient Boosting Regressor (GridSearchCV Tuned):** Mean CV $R^2 = 0.8876 \pm 0.0084$
* **Random Forest Regressor:** Mean CV $R^2 = 0.8794 \pm 0.0091$



### 3.2. Classification Track (Part A) — 5 Algorithm Benchmark

Evaluated on the exact same 20% stratified test set using weighted class metrics.

| Rank | Algorithm Name | Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score | Evaluation Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Decision Tree Classifier | 0.9015 | 0.8920 | 0.9015 | 0.8942 | Interpretable tree structure (`max_depth=5`). |
| 2 | Support Vector Machine (SVC) | 0.8998 | 0.8845 | 0.8998 | 0.8860 | RBF kernel with balanced margin separation. |
| 3 | Logistic Regression | 0.8925 | 0.8760 | 0.8925 | 0.8810 | Calibrated log-odds baseline. |
| 4 | K-Nearest Neighbors | 0.8910 | 0.8715 | 0.8910 | 0.8785 | $k=7$ Euclidean distance neighborhood. |
| 5 | Naive Bayes (Gaussian) | 0.8420 | 0.8640 | 0.8420 | 0.8510 | Lower recall due to feature independence assumptions. |



## 4. Setup & Run Instructions

1. **Clone the repository:**
   * git clone https://github.com/Karthik-2006-cell/ml-capstone-project.git
   *  cd ml-capstone-project



2. **Install dependencies:**
   * pip install -r requirements.txt



4. **Launch notebooks:**
   * jupyter notebook


Open and run all cells sequentially in `notebooks/regression.ipynb` and `notebooks/classification.ipynb`.


## 5. Model Interpretation & Key Findings

* **Regression Track:** Tree-based ensembles (Gradient Boosting, Random Forest) systematically outperformed linear baselines, demonstrating that property values exhibit non-linear interactions between geographical grade and square footage that polynomial terms alone cannot capture.
* **Classification Track:** Decision tree thresholds on duration and outcome of previous marketing campaigns proved to be the most decisive split points for isolating positive term-deposit conversions.





