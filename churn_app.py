import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

# ==========================================
# PAGE CONFIG & MODERN MINIMALIST DARK THEME
# ==========================================
st.set_page_config(page_title="Telecom AI Strategy", layout="wide", page_icon="📡")

# Ultra-Clean Modern Dark CSS
st.markdown("""
<style>
    /* 1. Global Background - Deep Graphite */
    [data-testid="stAppViewContainer"], 
    [data-testid="stHeader"], 
    [data-testid="stSidebar"], 
    .main, .stApp, html, body {
        background-color: #0E1117 !important;
        color: #C9D1D9 !important;
        font-family: 'Inter', sans-serif;
    }

    /* 2. Professional Metric Cards */
    [data-testid="stMetric"] {
        background-color: #161B22 !important;
        border: 1px solid #30363D !important;
        border-radius: 12px !important;
        padding: 20px !important;
        box-shadow: 0 4px 0 rgba(0,0,0,0.1) !important;
    }
    
    /* Metric Value - Emerald Glow */
    [data-testid="stMetricValue"] > div {
        color: #2EA043 !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
    }
    [data-testid="stMetricLabel"] > div {
        color: #8B949E !important;
        text-transform: uppercase !important;
        font-size: 0.75em !important;
        font-weight: 600 !important;
        letter-spacing: 0.1em !important;
    }

    /* 3. Headers & Decorations */
    h1, h2, h3 {
        color: #F0F6FC !important;
        border-bottom: 2px solid #238636;
        padding-bottom: 10px;
        margin-bottom: 25px !important;
    }

    /* 4. The Action Card (Unified Strategy) */
    .strategy-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-left: 6px solid #2EA043;
        padding: 30px;
        border-radius: 8px;
        margin-top: 20px;
    }
    
    /* Sidebar Cleanup */
    [data-testid="stSidebar"] {
        background-color: #010409 !important;
        border-right: 1px solid #30363D !important;
    }
    
    /* Button & Input Styling */
    .stButton>button {
        background-color: #238636 !important;
        color: white !important;
        border-radius: 6px !important;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. DATA CORE (Cached)
# ==========================================
@st.cache_data
def load_and_clean_data():
    try:
        # Handling the file path dynamically
        df = pd.read_csv('telecom_churn.csv')
        df = df.fillna(method='ffill')
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            df[col] = df[col].clip(q1 - 1.5*iqr, q3 + 1.5*iqr)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}. Check if telecom_churn.csv is in the path.")
        return None

# ==========================================
# 2. AI ENGINES (Cached)
# ==========================================
@st.cache_data
def train_predictive_model(df):
    data = df.copy()
    le = LabelEncoder()
    for col in data.select_dtypes(include=['object']):
        if 'date' not in col: data[col] = le.fit_transform(data[col].astype(str))
    
    drop_cols = ['churn', 'customer_id', 'date_of_registration']
    X = data.drop(columns=[c for c in drop_cols if c in data.columns], errors='ignore')
    y = data['churn']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler, X.columns

@st.cache_resource
def train_smart_clustering(df):
    cluster_features = ['age', 'estimated_salary', 'calls_made', 'data_used']
    X_c = df[cluster_features].copy()
    scaler_c = StandardScaler()
    X_scaled = scaler_c.fit_transform(X_c)
    
    kmeans = KMeans(n_clusters=4, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    df_temp = X_c.copy()
    df_temp['cluster_id'] = cluster_labels
    cluster_stats = df_temp.groupby('cluster_id').median()
    global_medians = X_c.median()
    
    names = {}
    for i, row in cluster_stats.iterrows():
        if row['data_used'] > global_medians['data_used'] * 1.5: label = "Heavy Data User"
        elif row['estimated_salary'] > global_medians['estimated_salary'] * 1.5: label = "Premium High-Value"
        elif row['calls_made'] > global_medians['calls_made'] * 1.5: label = "High Frequency Caller"
        elif row['estimated_salary'] < global_medians['estimated_salary'] * 0.7: label = "Budget Conscious"
        else: label = "Standard Household"
        names[i] = label
            
    return kmeans, scaler_c, cluster_features, names, cluster_labels

# ==========================================
# 3. INITIALIZATION
# ==========================================
df_raw = load_and_clean_data()
if df_raw is not None:
    model_pred, scaler_pred, feature_list = train_predictive_model(df_raw)
    kmeans, scaler_cluster, c_features, seg_names, cluster_ids = train_smart_clustering(df_raw)
    df_raw['cluster_id'] = cluster_ids
    df_raw['segment'] = df_raw['cluster_id'].map(seg_names)

# ==========================================
# MAIN DASHBOARD
# ==========================================
st.title("� Telecom AI Retention Engine")
st.markdown("### Strategic Analysis & Recommendation Desk")

# Sidebar - Lookup
st.sidebar.markdown("## 🔎 Customer Lookup")
search_type = st.sidebar.radio("Analysis Mode", ["Scroll List", "Quick ID Search"])

if search_type == "Scroll List":
    selected_id = st.sidebar.selectbox("Select Target ID", df_raw['customer_id'].unique()[:500])
else:
    id_input = st.sidebar.text_input("Manual ID Entry")
    try: selected_id = int(id_input) if id_input else None
    except: selected_id = None

# ACTION CENTER
if selected_id in df_raw['customer_id'].values:
    user_data = df_raw[df_raw['customer_id'] == selected_id].iloc[0]
    
    # ML Prediction
    temp_df = df_raw.copy()
    le = LabelEncoder()
    for col in temp_df.select_dtypes(include=['object']):
        if 'date' not in col: temp_df[col] = le.fit_transform(temp_df[col].astype(str))
    
    drop_ml = ['churn', 'customer_id', 'date_of_registration', 'cluster_id', 'segment']
    X_user = temp_df[temp_df['customer_id'] == selected_id].drop(columns=[c for c in drop_ml if c in temp_df.columns], errors='ignore')
    user_scaled = scaler_pred.transform(X_user)
    
    probs = model_pred.predict_proba(user_scaled)[0]
    prob = probs[1] if len(probs) > 1 else (1.0 if model_pred.classes_[0] == 1 else 0.0)
    is_churn = model_pred.predict(user_scaled)[0]
    
    # Dashboard Grid
    st.subheader("� Performance Indicators")
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Churn Risk", f"{prob:.1%}")
    m2.metric("Behavioral Segment", user_data['segment'])
    m3.metric("Retention Status", "🔴 AT RISK" if is_churn == 1 else "🟢 STABLE")
    
    # AI STRATEGY ENGINE
    segment = user_data['segment']
    usage = user_data['data_used']
    salary = user_data['estimated_salary']
    
    strategy = ""
    rationale = ""
    cost = 0
    
    # LOGIC (REFINED)
    if is_churn == 0:
        if "Premium" in segment:
            strategy = "VIP Appreciation Reward"
            rationale = "Customer is high-value and stable. **Strategy**: Proactively offer a 12-month loyalty bundle with premium perk access."
            cost = 50
        elif usage > 5000:
            strategy = "Infinite Data Upgrade"
            rationale = "Heavy usage detected. **Strategy**: Move them to an unthrottled plan trial to increase long-term dependency."
            cost = 15
        else:
            strategy = "Personalized Care Sync"
            rationale = "Healthy signal. **Strategy**: Maintain engagement with quarterly benefit summaries via AI messaging."
            cost = 2
    else:
        if salary > 85000 and usage < 1500:
            strategy = "Proactive Plan Downsell"
            rationale = "Value-seeking behavior at risk. **Strategy**: Suggest a more economical plan to save the relationship without 100% loss."
            cost = 20
        elif "Heavy Data" in segment:
            strategy = "Data Defense: 50% Loyalty Credit"
            rationale = "At-risk heavy user. **Strategy**: Compete directly on price with a deep 6-month discount on data usage."
            cost = 120
        elif "Budget" in segment:
            strategy = "Direct Incentive: $30 Bill Credit"
            rationale = "Price sensitive. **Strategy**: A direct financial bridge to prevent immediate churn decision."
            cost = 30
        else:
            strategy = "Direct Expert Intervention"
            rationale = "Complex churn indicators. **Strategy**: Escalation to a Retention Specialist for a personalized win-back call."
            cost = 45

    # ROI Math
    roi = ( (1200 if "Premium" in segment else 600) * 0.45 ) - cost
    
    # UNIFIED STRATEGY CARD
    st.markdown(f"""
    <div class="strategy-card">
        <h2 style="color: #2EA043 !important; font-size: 1.5em; border-bottom: none; margin-bottom: 10px !important;">🎯 TARGET STRATEGY: {strategy}</h2>
        <p style="font-size: 1.1em; line-height: 1.6; color: #C9D1D9 !important;">{rationale}</p>
        <div style="display: flex; gap: 40px; margin-top: 25px;">
            <div>
                <span style="color: #8B949E; font-size: 0.8em; font-weight: bold;">EST. COST</span><br>
                <span style="color: #F0F6FC; font-size: 1.5em; font-weight: 900;">${cost}</span>
            </div>
            <div>
                <span style="color: #8B949E; font-size: 0.8em; font-weight: bold;">ANNUAL ROI IMPACT</span><br>
                <span style="color: #2EA043; font-size: 1.5em; font-weight: 900;">${roi:,.2f}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("� Raw Customer Telemetry"):
        st.dataframe(user_data.to_frame().T)

else:
    if selected_id: st.warning("Customer ID not found.")
    else: st.info("👋 Select or Search for a Customer ID in the sidebar to begin.")
