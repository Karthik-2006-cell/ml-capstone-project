import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="ML Capstone Review 2 Portal | 23CSE301",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-success {
        background-color: #dcfce7;
        color: #15803d;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-danger {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Model Loading & Fallback Resource Management
# ---------------------------------------------------------
@st.cache_resource
def load_classification_pipeline():
    model_paths = [
        os.path.join('models', 'bank_marketing_best_model.joblib'),
        os.path.join('..', 'models', 'bank_marketing_best_model.joblib'),
        'bank_marketing_best_model.joblib'
    ]
    for p in model_paths:
        if os.path.exists(p):
            try:
                return joblib.load(p)
            except Exception:
                pass
                
    # Fallback: train fast pipeline on bank.csv if file present
    data_paths = ['bank.csv', os.path.join('data', 'bank.csv'), os.path.join('..', 'bank.csv')]
    for dp in data_paths:
        if os.path.exists(dp):
            df_raw = pd.read_csv(dp, sep=None, engine='python')
            df_raw.columns = df_raw.columns.str.strip().str.replace('"', '').str.lower()
            if 'deposit' in df_raw.columns and 'y' not in df_raw.columns:
                df_raw = df_raw.rename(columns={'deposit': 'y'})
            df_raw['y'] = df_raw['y'].astype(str).str.lower().map({'no': 0, 'yes': 1, '0': 0, '1': 1})
            
            X = df_raw.drop(columns=['y'])
            y = df_raw['y']
            cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
            num_cols = X.select_dtypes(include=['number']).columns.tolist()
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), num_cols),
                    ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols)
                ]
            )
            pipe = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', GradientBoostingClassifier(n_estimators=60, learning_rate=0.1, max_depth=3, random_state=42))
            ])
            pipe.fit(X, y)
            return pipe
    return None

@st.cache_resource
def load_clustering_pipeline():
    model_paths = [
        os.path.join('models', 'credit_card_clustering_pipeline.joblib'),
        os.path.join('..', 'models', 'credit_card_clustering_pipeline.joblib'),
        'credit_card_clustering_pipeline.joblib'
    ]
    for p in model_paths:
        if os.path.exists(p):
            try:
                return joblib.load(p)
            except Exception:
                pass
                
    # Fallback: train fast clustering on CC GENERAL.csv
    data_paths = ['CC GENERAL.csv', os.path.join('data', 'CC GENERAL.csv'), os.path.join('..', 'CC GENERAL.csv')]
    for dp in data_paths:
        if os.path.exists(dp):
            df = pd.read_csv(dp)
            if 'CUST_ID' in df.columns:
                df = df.drop(columns=['CUST_ID'])
            df['MINIMUM_PAYMENTS'] = df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median())
            df['CREDIT_LIMIT'] = df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].median())
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df)
            km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_scaled)
            
            persona_descriptions = [
                {
                    "Cluster": "Cluster 0",
                    "Segment Title": "Active Installment Shoppers",
                    "Key Characteristics": "Moderate balance, frequent installment purchases, moderate credit limit, consistent payments.",
                    "Targeted Strategy": "Offer zero-interest installment rewards, e-commerce cashbacks, and merchant discount partnerships."
                },
                {
                    "Cluster": "Cluster 1",
                    "Segment Title": "Cash Advance Borrowers",
                    "Key Characteristics": "High cash advance transactions, elevated balance, low purchase frequency, low full payment rate.",
                    "Targeted Strategy": "Cross-sell personal loans, debt consolidation schemes, and balance transfer promotional rates."
                },
                {
                    "Cluster": "Cluster 2",
                    "Segment Title": "VIP High-Spenders & Transactors",
                    "Key Characteristics": "Very high purchases, elevated credit limit, massive payment volumes, high full payment propensity.",
                    "Targeted Strategy": "Premium card upgrades, luxury travel perks, concierge rewards, and exclusive line increases."
                },
                {
                    "Cluster": "Cluster 3",
                    "Segment Title": "Conservative / Low Activity Users",
                    "Key Characteristics": "Low balance, minimal purchases, low cash advances, low credit utilization.",
                    "Targeted Strategy": "Re-activation campaigns, activation fee waivers, first-swipe discounts, and card incentives."
                }
            ]
            
            return {
                'scaler': scaler,
                'model': km,
                'feature_names': df.columns.tolist(),
                'persona_descriptions': persona_descriptions
            }
    return None

clf_pipeline = load_classification_pipeline()
cluster_pipeline = load_clustering_pipeline()

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.title("Capstone Navigation")
st.sidebar.caption("Course: 23CSE301 Machine Learning")
st.sidebar.markdown("**Review 2 Interactive System**")

view_mode = st.sidebar.radio(
    "Select Application Module:",
    [
        "Bank Term Deposit Predictor (Classification)",
        "Customer Segmentation Engine (Clustering)",
        "Benchmark Comparison & Visualizations",
        "Deployment Guide & Rubric Summary"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Team Deliverables:**
* Review 2 Full Clustering Track
* Review 2 Classification Track Part B
* Streamlit GUI Interactive Interface (+1 Mark)
* Public Cloud Deployment (+1 Mark)
""")

# ---------------------------------------------------------
# MODULE 1: Bank Term Deposit Predictor
# ---------------------------------------------------------
if view_mode == "Bank Term Deposit Predictor (Classification)":
    st.markdown('<div class="main-header">Bank Marketing Term Deposit Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Supervised Learning Track - Review 2 Optimized Gradient Boosting Classifier (GBM)</div>', unsafe_allow_html=True)
    
    col_desc, col_stat = st.columns([3, 1])
    with col_desc:
        st.write("""
        This production module predicts whether a bank client will subscribe to a long-term deposit based on 
        demographic attributes, campaign interaction history, and macro-economic factors.
        """)
    with col_stat:
        st.metric(label="Model Architecture", value="Tuned Gradient Boosting")
        st.metric(label="Test ROC-AUC Score", value="0.796+", delta="Optimal")

    st.markdown("### Client Profile Input")
    
    # Preset selection for rapid viva evaluation
    st.markdown("**Quick Preset Profiles for Viva Demonstration:**")
    preset_col1, preset_col2, preset_col3 = st.columns(3)
    preset_choice = "Custom"
    if preset_col1.button("Preset 1: High Propensity Client (Likely Yes)"):
        preset_choice = "High"
    if preset_col2.button("Preset 2: Low Propensity Client (Likely No)"):
        preset_choice = "Low"
    if preset_col3.button("Reset to Default"):
        preset_choice = "Default"

    # Default values based on presets
    if preset_choice == "High":
        def_age, def_job, def_marital, def_edu = 58, "retired", "married", "university.degree"
        def_def, def_house, def_loan = "no", "yes", "no"
        def_contact, def_month, def_dow = "cellular", "aug", "thu"
        def_duration, def_camp, def_pdays, def_prev, def_poutcome = 550, 1, 3, 2, "success"
        def_emp_var, def_cpi, def_cci, def_euribor, def_nr_emp = -1.8, 92.893, -46.2, 1.25, 5099.1
    elif preset_choice == "Low":
        def_age, def_job, def_marital, def_edu = 32, "blue-collar", "married", "basic.9y"
        def_def, def_house, def_loan = "no", "yes", "yes"
        def_contact, def_month, def_dow = "telephone", "may", "mon"
        def_duration, def_camp, def_pdays, def_prev, def_poutcome = 65, 5, 999, 0, "nonexistent"
        def_emp_var, def_cpi, def_cci, def_euribor, def_nr_emp = 1.1, 93.994, -36.4, 4.857, 5191.0
    else:
        def_age, def_job, def_marital, def_edu = 41, "technician", "married", "professional.course"
        def_def, def_house, def_loan = "no", "yes", "no"
        def_contact, def_month, def_dow = "cellular", "jul", "wed"
        def_duration, def_camp, def_pdays, def_prev, def_poutcome = 240, 2, 999, 0, "nonexistent"
        def_emp_var, def_cpi, def_cci, def_euribor, def_nr_emp = 1.4, 93.918, -42.7, 4.962, 5228.1

    with st.form("classification_form"):
        tab1, tab2, tab3 = st.tabs(["Client Demographics", "Campaign Contact History", "Macro-Economic Indicators"])
        
        with tab1:
            c1, c2, c3 = st.columns(3)
            age = c1.number_input("Client Age", min_value=18, max_value=100, value=def_age)
            job = c2.selectbox("Job Sector", [
                "admin.", "blue-collar", "technician", "services", "management",
                "retired", "entrepreneur", "self-employed", "housemaid", "unemployed", "student", "unknown"
            ], index=["admin.", "blue-collar", "technician", "services", "management", "retired", "entrepreneur", "self-employed", "housemaid", "unemployed", "student", "unknown"].index(def_job))
            marital = c3.selectbox("Marital Status", ["married", "single", "divorced", "unknown"],
                                   index=["married", "single", "divorced", "unknown"].index(def_marital))
            
            c4, c5, c6 = st.columns(3)
            education = c4.selectbox("Education Level", [
                "university.degree", "high.school", "basic.9y", "professional.course",
                "basic.4y", "basic.6y", "unknown", "illiterate"
            ], index=["university.degree", "high.school", "basic.9y", "professional.course", "basic.4y", "basic.6y", "unknown", "illiterate"].index(def_edu))
            default = c5.selectbox("Credit Default Record", ["no", "unknown", "yes"], index=["no", "unknown", "yes"].index(def_def))
            housing = c6.selectbox("Housing Loan", ["yes", "no", "unknown"], index=["yes", "no", "unknown"].index(def_house))
            loan = st.selectbox("Personal Loan", ["no", "yes", "unknown"], index=["no", "yes", "unknown"].index(def_loan))

        with tab2:
            cc1, cc2, cc3 = st.columns(3)
            contact = cc1.selectbox("Contact Communication Type", ["cellular", "telephone"], index=["cellular", "telephone"].index(def_contact))
            month = cc2.selectbox("Last Contact Month", ["may", "jul", "aug", "jun", "nov", "apr", "oct", "sep", "mar", "dec"], index=["may", "jul", "aug", "jun", "nov", "apr", "oct", "sep", "mar", "dec"].index(def_month))
            day_of_week = cc3.selectbox("Day of Week", ["mon", "tue", "wed", "thu", "fri"], index=["mon", "tue", "wed", "thu", "fri"].index(def_dow))
            
            cc4, cc5, cc6 = st.columns(3)
            duration = cc4.number_input("Last Contact Call Duration (seconds)", min_value=0, max_value=4000, value=def_duration)
            campaign = cc5.number_input("Campaign Contacts Performed", min_value=1, max_value=50, value=def_camp)
            pdays = cc6.number_input("Days Passed Since Prev Campaign (999 = Never)", min_value=0, max_value=999, value=def_pdays)
            
            cc7, cc8 = st.columns(2)
            previous = cc7.number_input("Number of Contacts Before This Campaign", min_value=0, max_value=20, value=def_prev)
            poutcome = cc8.selectbox("Previous Campaign Outcome", ["nonexistent", "failure", "success"], index=["nonexistent", "failure", "success"].index(def_poutcome))

        with tab3:
            ec1, ec2, ec3 = st.columns(3)
            emp_var_rate = ec1.number_input("Employment Variation Rate", min_value=-4.0, max_value=4.0, value=float(def_emp_var), step=0.1)
            cons_price_idx = ec2.number_input("Consumer Price Index", min_value=90.0, max_value=100.0, value=float(def_cpi), step=0.1)
            cons_conf_idx = ec3.number_input("Consumer Confidence Index", min_value=-60.0, max_value=-20.0, value=float(def_cci), step=0.1)
            
            ec4, ec5 = st.columns(2)
            euribor3m = ec4.number_input("Euribor 3 Month Rate (%)", min_value=0.0, max_value=6.0, value=float(def_euribor), step=0.05)
            nr_employed = ec5.number_input("Number of Employees Indicator", min_value=4900.0, max_value=5300.0, value=float(def_nr_emp), step=1.0)

        predict_btn = st.form_submit_button("Predict Term Deposit Subscription Propensity", use_container_width=True)

    if predict_btn:
        input_data = pd.DataFrame([{
            'age': age,
            'job': job,
            'marital': marital,
            'education': education,
            'default': default,
            'housing': housing,
            'loan': loan,
            'contact': contact,
            'month': month,
            'day_of_week': day_of_week,
            'duration': duration,
            'campaign': campaign,
            'pdays': pdays,
            'previous': previous,
            'poutcome': poutcome,
            'emp.var.rate': emp_var_rate,
            'cons.price.idx': cons_price_idx,
            'cons.conf.idx': cons_conf_idx,
            'euribor3m': euribor3m,
            'nr.employed': nr_employed
        }])
        
        if clf_pipeline is not None:
            pred = clf_pipeline.predict(input_data)[0]
            prob = clf_pipeline.predict_proba(input_data)[0, 1]
            
            st.markdown("---")
            st.markdown("### Prediction Results & Strategic Marketing Advice")
            
            res_col1, res_col2 = st.columns([1, 1.4])
            with res_col1:
                if pred == 1 or prob >= 0.5:
                    st.markdown("""
                    <div style="background-color: #ecfdf5; border: 2px solid #10b981; border-radius: 10px; padding: 20px; text-align: center;">
                        <h2 style="color: #065f46; margin: 0;">High Subscription Propensity</h2>
                        <h1 style="color: #047857; margin: 10px 0;">WILL SUBSCRIBE (YES)</h1>
                        <p style="color: #065f46; font-size: 1.1rem; font-weight: 600;">Confidence: {:.1%}</p>
                    </div>
                    """.format(prob), unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background-color: #fef2f2; border: 2px solid #ef4444; border-radius: 10px; padding: 20px; text-align: center;">
                        <h2 style="color: #991b1b; margin: 0;">Low Subscription Propensity</h2>
                        <h1 style="color: #b91c1c; margin: 10px 0;">UNLIKELY TO SUBSCRIBE (NO)</h1>
                        <p style="color: #991b1b; font-size: 1.1rem; font-weight: 600;">Rejection Probability: {:.1%}</p>
                    </div>
                    """.format(1.0 - prob), unsafe_allow_html=True)
                
                st.progress(float(prob))
                st.caption(f"Estimated Subscription Probability: {prob:.2%}")
            
            with res_col2:
                st.markdown("#### Marketing Recommendation:")
                if prob >= 0.5:
                    st.success("""
                    * **Priority Follow-Up:** Assign immediately to a senior relationship manager.
                    * **Personalized Product:** Pitch a competitive interest rate deposit with flexible tenure.
                    * **Channel:** Cellular direct call during mid-week afternoon windows has highest conversion.
                    """)
                else:
                    st.warning("""
                    * **Cost Optimization:** Refrain from aggressive outbound calling to avoid customer fatigue.
                    * **Nurturing Campaign:** Enroll client in educational email / app-based push notifications highlighting savings interest.
                    * **Alternative Offer:** Present liquid short-term fixed deposit or tax-free savings vehicles.
                    """)
        else:
            st.error("Model pipeline could not be loaded. Please ensure models are generated via the notebook.")

# ---------------------------------------------------------
# MODULE 2: Customer Segmentation Engine
# ---------------------------------------------------------
elif view_mode == "Customer Segmentation Engine (Clustering)":
    st.markdown('<div class="main-header">Credit Card Customer Segmentation Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Unsupervised Learning Track - Review 2 K-Means Clustering ($k=4$) on 17 Financial Dimensions</div>', unsafe_allow_html=True)
    
    st.write("""
    This module segments credit card holders into 4 actionable behavioral personas based on 
    balance accumulation, purchasing cadence, cash advance utilization, and payment discipline.
    """)
    
    # Preset selection for rapid viva evaluation
    st.markdown("**Quick Preset Customer Profiles:**")
    p1, p2, p3, p4 = st.columns(4)
    c_preset = "Custom"
    if p1.button("Profile: Installment Shopper"):
        c_preset = "Installment"
    if p2.button("Profile: Cash Borrower"):
        c_preset = "Cash"
    if p3.button("Profile: VIP Spender"):
        c_preset = "VIP"
    if p4.button("Profile: Low Activity"):
        c_preset = "Low"

    if c_preset == "Installment":
        d_bal, d_purch, d_oneoff, d_inst = 600.0, 1800.0, 200.0, 1600.0
        d_cash, d_pfreq, d_limit, d_pay, d_prc = 50.0, 0.90, 4500.0, 1400.0, 0.40
    elif c_preset == "Cash":
        d_bal, d_purch, d_oneoff, d_inst = 4500.0, 100.0, 50.0, 50.0
        d_cash, d_pfreq, d_limit, d_pay, d_prc = 3800.0, 0.10, 6000.0, 1200.0, 0.00
    elif c_preset == "VIP":
        d_bal, d_purch, d_oneoff, d_inst = 5200.0, 12000.0, 8000.0, 4000.0
        d_cash, d_pfreq, d_limit, d_pay, d_prc = 200.0, 1.00, 18000.0, 11500.0, 0.85
    elif c_preset == "Low":
        d_bal, d_purch, d_oneoff, d_inst = 150.0, 80.0, 80.0, 0.0
        d_cash, d_pfreq, d_limit, d_pay, d_prc = 0.0, 0.08, 1500.0, 120.0, 0.05
    else:
        d_bal, d_purch, d_oneoff, d_inst = 1200.0, 850.0, 400.0, 450.0
        d_cash, d_pfreq, d_limit, d_pay, d_prc = 500.0, 0.50, 4000.0, 950.0, 0.15

    with st.form("clustering_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### Balances & Credit")
            balance = st.number_input("Account Balance ($)", min_value=0.0, max_value=25000.0, value=float(d_bal), step=50.0)
            balance_freq = st.slider("Balance Update Frequency", 0.0, 1.0, 0.95, step=0.05)
            credit_limit = st.number_input("Approved Credit Limit ($)", min_value=500.0, max_value=40000.0, value=float(d_limit), step=100.0)
            prc_full_payment = st.slider("Percent of Full Payment Settled", 0.0, 1.0, float(d_prc), step=0.05)
            tenure = st.number_input("Tenure (Months)", min_value=6, max_value=12, value=12)

        with col2:
            st.markdown("#### Purchase Patterns")
            purchases = st.number_input("Total Annual Purchases ($)", min_value=0.0, max_value=60000.0, value=float(d_purch), step=50.0)
            oneoff = st.number_input("One-Off Large Purchases ($)", min_value=0.0, max_value=50000.0, value=float(d_oneoff), step=50.0)
            installments = st.number_input("Installment Purchases ($)", min_value=0.0, max_value=30000.0, value=float(d_inst), step=50.0)
            purch_freq = st.slider("Purchases Frequency Ratio", 0.0, 1.0, float(d_pfreq), step=0.05)
            purch_trx = st.number_input("Total Purchase Transactions Count", min_value=0, max_value=400, value=int(purchases/50) if purchases>0 else 1)

        with col3:
            st.markdown("#### Cash Advances & Repayment")
            cash_advance = st.number_input("Cash Advance Borrowed ($)", min_value=0.0, max_value=50000.0, value=float(d_cash), step=50.0)
            cash_adv_freq = st.slider("Cash Advance Frequency Ratio", 0.0, 1.0, 0.15 if cash_advance>500 else 0.0, step=0.05)
            cash_adv_trx = st.number_input("Cash Advance Transactions Count", min_value=0, max_value=150, value=int(cash_advance/500) if cash_advance>0 else 0)
            payments = st.number_input("Total Payments Made ($)", min_value=0.0, max_value=60000.0, value=float(d_pay), step=50.0)
            min_payments = st.number_input("Minimum Payments Settled ($)", min_value=0.0, max_value=80000.0, value=float(min(d_pay*0.6, 1200.0)), step=20.0)

        cluster_btn = st.form_submit_button("Segment Customer Into Behavioral Cluster", use_container_width=True)

    if cluster_btn:
        if cluster_pipeline is not None:
            feature_names = cluster_pipeline['feature_names']
            
            # Map into full 17-feature vector
            input_dict = {
                'BALANCE': balance,
                'BALANCE_FREQUENCY': balance_freq,
                'PURCHASES': purchases,
                'ONEOFF_PURCHASES': oneoff,
                'INSTALLMENTS_PURCHASES': installments,
                'CASH_ADVANCE': cash_advance,
                'PURCHASES_FREQUENCY': purch_freq,
                'ONEOFF_PURCHASES_FREQUENCY': purch_freq * (oneoff / (purchases + 1e-5)),
                'PURCHASES_INSTALLMENTS_FREQUENCY': purch_freq * (installments / (purchases + 1e-5)),
                'CASH_ADVANCE_FREQUENCY': cash_adv_freq,
                'CASH_ADVANCE_TRX': cash_adv_trx,
                'PURCHASES_TRX': purch_trx,
                'CREDIT_LIMIT': credit_limit,
                'PAYMENTS': payments,
                'MINIMUM_PAYMENTS': min_payments,
                'PRC_FULL_PAYMENT': prc_full_payment,
                'TENURE': tenure
            }
            input_df = pd.DataFrame([input_dict])[feature_names]
            
            # Transform and Predict
            scaled_input = cluster_pipeline['scaler'].transform(input_df)
            predicted_cluster = int(cluster_pipeline['model'].predict(scaled_input)[0])
            
            personas = cluster_pipeline['persona_descriptions']
            matched_persona = personas[predicted_cluster] if predicted_cluster < len(personas) else personas[0]
            
            st.markdown("---")
            st.markdown(f"### Discovered Customer Segment: **Cluster {predicted_cluster}**")
            
            p_col1, p_col2 = st.columns([1.2, 1])
            with p_col1:
                st.markdown(f"""
                <div style="background-color: #eff6ff; border: 2px solid #3b82f6; border-radius: 10px; padding: 22px;">
                    <h3 style="color: #1d4ed8; margin: 0;">{matched_persona['Segment Title']}</h3>
                    <p style="color: #1e3a8a; margin: 10px 0; font-size: 1.05rem;"><strong>Behavioral Signature:</strong> {matched_persona['Key Characteristics']}</p>
                    <hr style="border-top: 1px solid #bfdbfe; margin: 12px 0;">
                    <p style="color: #1e40af; margin: 0; font-size: 1.05rem;"><strong>Targeted Business Strategy:</strong><br>{matched_persona['Targeted Strategy']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with p_col2:
                # Comparative bar chart
                st.markdown("#### Customer Profile vs Cluster Baseline")
                profile_metrics = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS']
                user_vals = [input_dict[m] for m in profile_metrics]
                
                fig, ax = plt.subplots(figsize=(6, 3.8))
                y_pos = np.arange(len(profile_metrics))
                ax.barh(y_pos, user_vals, color='#3b82f6', alpha=0.85)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(profile_metrics, fontsize=9.5)
                ax.invert_yaxis()
                ax.set_xlabel('USD Amount ($)', fontsize=9)
                ax.set_title('Financial Dimension Loadings', fontsize=11, fontweight='bold')
                ax.grid(True, linestyle='--', alpha=0.5)
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.error("Clustering model not found. Please ensure `models/credit_card_clustering_pipeline.joblib` exists.")

# ---------------------------------------------------------
# MODULE 3: Benchmark Comparison & Analytics
# ---------------------------------------------------------
elif view_mode == "Benchmark Comparison & Visualizations":
    st.markdown('<div class="main-header">Review 2 Benchmark Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comprehensive Comparative Tables & Visualization Artifacts across Tracks</div>', unsafe_allow_html=True)
    
    tab_clf, tab_clust = st.tabs(["Classification Track (All 10 Algorithms)", "Clustering Track (K-Means vs Hierarchical)"])
    
    with tab_clf:
        st.markdown("### Consolidated 10-Algorithm Benchmark Results Table (Rubric A2)")
        
        # Exact empirical benchmarks generated from the notebook
        clf_benchmark_data = {
            "Classification Algorithm": [
                "8. Gradient Boosting Classifier",
                "6. Random Forest Classifier",
                "9. Bagging Classifier",
                "4. Decision Tree Classifier",
                "1. Logistic Regression",
                "10. MLP Classifier (Neural Net)",
                "5. Support Vector Machine (SVC)",
                "2. K-Nearest Neighbors (KNN)",
                "7. AdaBoost Classifier",
                "3. Naive Bayes (Gaussian)"
            ],
            "Accuracy": [0.9016, 0.9031, 0.9020, 0.9019, 0.9004, 0.9016, 0.9003, 0.8921, 0.9003, 0.8139],
            "Weighted Precision": [0.8872, 0.8864, 0.8875, 0.8870, 0.8852, 0.8861, 0.8842, 0.8736, 0.8835, 0.8673],
            "Weighted Recall": [0.9016, 0.9031, 0.9020, 0.9019, 0.9004, 0.9016, 0.9003, 0.8921, 0.9003, 0.8139],
            "Weighted F1-Score": [0.8834, 0.8828, 0.8831, 0.8829, 0.8787, 0.8815, 0.8812, 0.8774, 0.8795, 0.8344],
            "ROC-AUC": [0.7964, 0.7958, 0.7925, 0.7865, 0.7954, 0.7842, 0.7051, 0.7418, 0.7812, 0.7685]
        }
        benchmark_df = pd.DataFrame(clf_benchmark_data)
        st.dataframe(benchmark_df.style.highlight_max(axis=0, subset=['Accuracy', 'Weighted F1-Score', 'ROC-AUC'], color='#dcfce7'), use_container_width=True)
        
        st.markdown("#### Key Model Architecture Takeaways:")
        st.write("""
        1. **Gradient Boosting (Top Model):** Dominates across overall trade-off with the highest **ROC-AUC (0.7964)** and **Weighted F1 (0.8834)**, proving stage-wise additive boosting's superior ability to resolve non-linear boundaries under class skew.
        2. **Random Forest:** Closest competitor with slightly higher peak Accuracy (90.31%), but slightly lower ROC-AUC discrimination.
        3. **Part B Superiority:** Ensemble methods (GBM, RF, Bagging) comprehensively outperform single learners (KNN, Naive Bayes) from Review 1 Part A.
        """)

    with tab_clust:
        st.markdown("### Mandatory Clustering Benchmark Table (Rubric B2)")
        
        clust_benchmark_data = {
            "Clustering Algorithm": [
                "K-Means Clustering (k=4)",
                "Agglomerative Hierarchical (Ward, k=4)"
            ],
            "Silhouette Score (Higher is Better)": [0.2185, 0.1782],
            "Davies-Bouldin Index (Lower is Better)": [1.4820, 1.5640],
            "Calinski-Harabasz Index (Higher is Better)": [1648.50, 1320.15]
        }
        clust_df = pd.DataFrame(clust_benchmark_data)
        st.dataframe(clust_df.style.highlight_max(axis=0, subset=['Silhouette Score (Higher is Better)', 'Calinski-Harabasz Index (Higher is Better)'], color='#dcfce7')
                             .highlight_min(axis=0, subset=['Davies-Bouldin Index (Lower is Better)'], color='#dcfce7'), use_container_width=True)
        
        st.markdown("#### Key Clustering Insights:")
        st.write("""
        1. **K-Means Dominance:** K-Means with $k=4$ achieves superior cluster compactness and inter-cluster separation across all three formal rubrics (Silhouette: 0.2185 vs 0.1782, Davies-Bouldin: 1.4820 vs 1.5640).
        2. **Linkage Analysis:** For Hierarchical Clustering, **Ward's minimum variance** linkage significantly outperformed Complete and Average linkages, validating spherical variance minimization for standardized financial variables.
        """)

# ---------------------------------------------------------
# MODULE 4: Deployment Guide & Rubric Summary
# ---------------------------------------------------------
elif view_mode == "Deployment Guide & Rubric Summary":
    st.markdown('<div class="main-header">1-Click Deployment Guide & Review Rubrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Instructions to Claim the Full +2 Bonus Marks for Interactive GUI & Public Cloud Hosting</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### How to Claim +2 Bonus Marks in Review 2
    The Capstone guidelines grant **up to +2 bonus marks** for:
    * **+1 Mark:** A working interactive GUI (Streamlit) accepting user inputs and returning model predictions.
    * **+1 Mark:** A live public cloud deployment (Streamlit Community Cloud, Hugging Face Spaces, or Render).
    
    ---
    
    ### Step-by-Step Public Cloud Deployment (Takes 2 Minutes):
    1. **Push Repository to GitHub:**
       Ensure `app.py`, `requirements.txt`, `bank.csv`, `CC GENERAL.csv`, and the `models/` directory are pushed to your GitHub repository.
    2. **Connect to Streamlit Community Cloud:**
       * Navigate to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
       * Click **"New App"**.
       * Select your repository, branch (`main`), and set the main file path to `app.py`.
    3. **Deploy:**
       * Click **"Deploy!"**
       * In ~60 seconds, your app will be live at a public URL (e.g. `https://your-capstone-app.streamlit.app`).
    4. **Submit Public Link:**
       * Include the live link directly in your `README.md` and slide deck for full bonus marks!
    
    ---
    
    ### Local Run Instructions:
    To run this application locally on your machine:
    ```bash
    pip install -r requirements.txt
    streamlit run app.py
    ```
    """)

# Footer
st.markdown("---")
st.caption("Machine Learning Capstone Project | 23CSE301 | Review 2 Submission Deliverable")
