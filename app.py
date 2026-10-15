import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from datetime import datetime

st.set_page_config(
    page_title="E-Commerce Conversion & Intent Hub",
    page_icon="🛍️",
    layout="wide"
)

# --- CLEAN PROFESSIONAL LIGHT THEME STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #F8FAFC;
    }
    .hero-container {
        background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        opacity: 0.9;
    }
    .card {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .buy-box {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 1.2rem;
        border-radius: 8px;
        font-size: 1.2rem;
        font-weight: 700;
        text-align: center;
        border: 2px solid #34D399;
    }
    .nobuy-box {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 1.2rem;
        border-radius: 8px;
        font-size: 1.2rem;
        font-weight: 700;
        text-align: center;
        border: 2px solid #F87171;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize SQLite Database
def init_db():
    conn = sqlite3.connect('predictions_log.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            model_used TEXT,
            product_pages INTEGER,
            product_duration REAL,
            page_values REAL,
            bounce_rate REAL,
            exit_rate REAL,
            prediction_result TEXT,
            probability REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@st.cache_resource
def load_all_artifacts():
    try:
        # Models are our own from-scratch classes defined in scratch_ml.py (NumPy only)
        dt = joblib.load('decision_tree_model.pkl')
        knn = joblib.load('knn_model.pkl')
        lr = joblib.load('logistic_regression.pkl')
        svm = joblib.load('svm_model.pkl')
        preprocessor = joblib.load('preprocessor.pkl')
        metrics = joblib.load('model_metrics.pkl')
        feature_names = joblib.load('feature_names.pkl')
        ds_info = joblib.load('dataset_info.pkl')
        models = {
            "Decision Tree": dt,
            "Support Vector Machine (SVM)": svm,
            "K-Nearest Neighbors (KNN)": knn,
            "Logistic Regression": lr
        }
        # Mark the model with the highest test accuracy
        best = max(metrics, key=lambda m: metrics[m]['Accuracy'])
        models = {(f"{k} (Best Accuracy)" if k == best else k): v for k, v in models.items()}
        return models, preprocessor, metrics, feature_names, ds_info
    except Exception as e:
        st.error(f"⚠️ Model artifacts missing: {e}. Pehle 'python train_models.py' run karein!")
        return None, None, None, None, None

models_dict, preprocessor, metrics, feature_names, ds_info = load_all_artifacts()

# --- SIDEBAR ---
st.sidebar.title("⚙️ Dashboard Controls")
page = st.sidebar.radio("Navigation", ["🔮 Live Conversion Predictor", "📊 SOTA Model Analytics", "🗄️ Database Audit Logs"])

st.sidebar.markdown("---")
if models_dict:
    chosen_model_name = st.sidebar.selectbox("🧠 Select Active ML Model", list(models_dict.keys()))
    active_model = models_dict[chosen_model_name]

# --- HERO BANNER ---
st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🛍️ E-Commerce Conversion & Intent Hub</div>
    </div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# PAGE 1: LIVE PREDICTION
# -------------------------------------------------------------
if page == "🔮 Live Conversion Predictor":
    st.markdown("### 🎯 Real-Time Customer Session Simulation")
    st.markdown("Adjust visitor session telemetry parameters to check whether the customer will make a purchase or abandon the cart")

    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown('<div class="card"><h4>🖥️ Browsing Activity</h4>', unsafe_allow_html=True)
        admin_pages = st.slider("Administrative Pages Visited", 0, 27, 2)
        info_pages = st.slider("Informational Pages Visited", 0, 24, 0)
        product_related = st.slider("Product-Related Pages Visited", 1, 150, 25)
        product_duration = st.number_input("Time Spent on Products (Seconds)", min_value=0.0, max_value=8000.0, value=750.0, step=50.0)
        page_values = st.number_input("Page Value Score", min_value=0.0, max_value=360.0, value=18.5, step=1.0)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="card"><h4>⏱️ Session & Temporal Attributes</h4>', unsafe_allow_html=True)
        bounce_rates = st.slider("Bounce Rate (Avg)", 0.0, 0.2, 0.01, step=0.005)
        exit_rates = st.slider("Exit Rate (Avg)", 0.0, 0.2, 0.03, step=0.005)
        special_day = st.slider("Holiday / Special Day Proximity", 0.0, 1.0, 0.0, step=0.2)
        month = st.selectbox("Month of Visit", ["May", "Nov", "Mar", "Dec", "Oct", "Sep", "Aug", "Jul", "June", "Feb"])
        visitor_type = st.selectbox("Visitor Profile", ["Returning_Visitor", "New_Visitor", "Other"])
        weekend = st.selectbox("Is Weekend Session?", ["No (Weekday)", "Yes (Weekend)"])
        st.markdown('</div>', unsafe_allow_html=True)

    weekend_val = 1 if weekend.startswith("Yes") else 0

    # Raw session record - the from-scratch Preprocessor applies log / scaling / one-hot encoding
    input_dict = {
        'Administrative': [admin_pages],
        'Informational': [info_pages],
        'ProductRelated': [product_related],
        'ProductRelated_Duration': [product_duration],
        'BounceRates': [bounce_rates],
        'ExitRates': [exit_rates],
        'PageValues': [page_values],
        'SpecialDay': [special_day],
        'Month': [month],
        'VisitorType': [visitor_type],
        'Weekend': [weekend_val]
    }
    input_df = pd.DataFrame(input_dict)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Run Inference & Log to Database", type="primary", use_container_width=True):
        if active_model is not None and preprocessor is not None:
            model_input = preprocessor.transform(input_df)
            pred = active_model.predict(model_input)[0]          # uses the tuned decision threshold
            probs = active_model.predict_proba(model_input)[0]
            
            result_str = "Will Purchase" if pred == 1 else "Will Abandon"
            conf_prob = float(probs[1] * 100)

            try:
                conn = sqlite3.connect('predictions_log.db')
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO predictions (timestamp, model_used, product_pages, product_duration, page_values, bounce_rate, exit_rate, prediction_result, probability)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    chosen_model_name,
                    int(product_related),
                    float(product_duration),
                    float(page_values),
                    float(bounce_rates),
                    float(exit_rates),
                    result_str,
                    conf_prob
                ))
                conn.commit()
                conn.close()
            except Exception as db_err:
                pass

            st.markdown("---")
            st.markdown("### 📊 Inference Outcome Analysis")
            
            c1, c2 = st.columns(2, gap="large")
            with c1:
                if pred == 1:
                    st.markdown('<div class="buy-box">✅ HIGH CONVERSION INTENT<br><span style="font-size:0.95rem; font-weight:400;">Customer is likely to complete checkout.</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="nobuy-box">❌ CART ABANDONMENT RISK<br><span style="font-size:0.95rem; font-weight:400;">Customer is likely to exit without purchasing.</span></div>', unsafe_allow_html=True)
            with c2:
                mc1, mc2 = st.columns(2)
                with mc1:
                    st.metric("Purchase Probability", f"{conf_prob:.2f}%")
                with mc2:
                    st.metric("Decision Threshold", f"{active_model.threshold_*100:.1f}%")

# -------------------------------------------------------------
# PAGE 2: MODEL ANALYTICS
# -------------------------------------------------------------
elif page == "📊 SOTA Model Analytics":
    st.markdown("### 📈 Comprehensive Model Performance Benchmarking")
    n_test = ds_info['n_test'] if ds_info else "-"
    n_clean = ds_info['rows_clean'] if ds_info else "-"
    st.markdown(f"Comparative evaluation across **4 machine learning models implemented from scratch** "
                f"Models were trained on an 80% split of {n_clean:,} cleaned sessions and evaluated on the held-out {n_test:,} test sessions.")
    
    if metrics:
        df_metrics = pd.DataFrame(metrics).T[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].astype(float)
        st.dataframe(df_metrics.style.highlight_max(axis=0, color='#D1FAE5'), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        df_metrics.plot(kind='bar', ax=ax, colormap='viridis')
        ax.set_ylabel("Percentage (%)")
        ax.set_ylim(30, 100)
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig)

        st.markdown("### 📉 ROC Curves (Test Set)")
        fig_roc, ax_roc = plt.subplots(figsize=(6, 4.5))
        for m_name, m_data in metrics.items():
            ax_roc.plot(m_data['ROC_Curve']['fpr'], m_data['ROC_Curve']['tpr'],
                        label=f"{m_name} (AUC={m_data['ROC-AUC']:.1f}%)")
        ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Random guess")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend(fontsize=7)
        ax_roc.grid(linestyle='--', alpha=0.4)
        st.pyplot(fig_roc)



        st.markdown("### 🔍 Test Set Confusion Matrices")
        cols = st.columns(4)
        for idx, (m_name, m_data) in enumerate(metrics.items()):
            with cols[idx]:
                fig_cm, ax_cm = plt.subplots(figsize=(3, 2.5))
                sns.heatmap(np.array(m_data['Confusion_Matrix']), annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm)
                ax_cm.set_title(m_name, fontsize=8)
                st.pyplot(fig_cm)

# -------------------------------------------------------------
# PAGE 3: DATABASE LOGS
# -------------------------------------------------------------
elif page == "🗄️ Database Audit Logs":
    st.markdown("### 🗄️ SQLite MLOps Transaction Audit Log")
    st.markdown("Real-time persistent storage logging all simulation requests and outputs.")

    try:
        conn = sqlite3.connect('predictions_log.db')
        df_logs = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
        conn.close()

        if not df_logs.empty:
            st.success(f"Total Logged Inferences: {len(df_logs)} records secured in SQLite.")
            st.dataframe(df_logs, use_container_width=True)
            
            if st.button("🗑️ Purge Audit Logs", type="secondary"):
                conn = sqlite3.connect('predictions_log.db')
                cursor = conn.cursor()
                cursor.execute("DELETE FROM predictions")
                conn.commit()
                conn.close()
                st.success("Audit logs purged successfully!")
                st.rerun()
        else:
            st.info("No audit logs found yet. Pehle 'Live Conversion Predictor' se prediction run karein.")
    except Exception as e:
        st.error(f"Database error: {e}")