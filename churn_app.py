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
# PAGE CONFIG & ABSOLUTE DARK THEME
# ==========================================
st.set_page_config(page_title="Telecom AI Strategy", layout="wide", page_icon="📡")

# --- NUCLEAR DARK THEME ENFORCEMENT ---
st.markdown("""
<style>
    /* 1. Global Background - Deep Graphite */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="stAppViewBlockContainer"] {
        background-color: #0E1117 !important;
        background: #0E1117 !important;
    }

    /* 2. Text Visibility - Force High Contrast White */
    h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown, .stSelectbox label, .stTextInput label {
        color: #F0F6FC !important;
    }

    /* 3. Metric Cards - Dark & Defined */
    [data-testid="stMetric"] {
        background-color: #161B22 !important;
        border: 1px solid #30363D !important;
        border-radius: 12px !important;
        padding: 20px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5) !important;
    }
    [data-testid="stMetricValue"] > div { color: #39D353 !important; font-weight: 700 !important; }
    [data-testid="stMetricLabel"] > div { color: #8B949E !important; text-transform: uppercase !important; }

    /* 4. Unified Strategy Card - Emerald Bordered */
    .strategy-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-left: 6px solid #2EA043;
        padding: 30px;
        border-radius: 10px;
        margin: 25px 0;
    }
    
    /* Input Styling */
    div[data-baseweb="select"], input {
        background-color: #21262D !important;
        color: white !important;
        border: 1px solid #30363D !important;
    }
</style>
""", unsafe_allow_html=True)

# SIDEBAR STATUS
st.sidebar.success("✅ **SYSTEM v5.5: NUCLEAR DARK LOADED**")
st.sidebar.info("💡 **PRO-TIP**: If the background is still white on the website, you must **Commit and Push** these changes to GitHub.")

# ==========================================
# 1. DATA CORE (Cached)
# ==========================================
@st.cache_data
def load_and_clean_data():
    try:
        df = pd.read_csv('telecom_churn.csv')
        df = df.fillna(method='ffill')
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            df[col] = df[col].clip(q1 - 1.5*iqr, q3 + 1.5*iqr)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}. Check if telecom_churn.csv is present.")
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
    
    names = {}
    for i, row in cluster_stats.iterrows():
        if row['data_used'] > df_temp['data_used'].median() * 1.5: label = "Heavy Data User"
        elif row['estimated_salary'] > df_temp['estimated_salary'].median() * 1.5: label = "Premium High-Value"
        elif row['calls_made'] > df_temp['calls_made'].median() * 1.2: label = "High Frequency Caller"
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
st.title("📡 Telecom AI Retention Strategy")
st.write("---")

# Customer Lookup
st.sidebar.markdown("## � Target Analysis")
search_type = st.sidebar.radio("Method", ["Select ID", "Manual Entry"])

if search_type == "Select ID":
    selected_id = st.sidebar.selectbox("Choose ID", df_raw['customer_id'].unique()[:500])
else:
    id_input = st.sidebar.text_input("Enter ID Code")
    try: selected_id = int(id_input) if id_input else None
    except: selected_id = None

# PROCESS RESULTS
if selected_id in df_raw['customer_id'].values:
    user_data = df_raw[df_raw['customer_id'] == selected_id].iloc[0]
    
    # Inference
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
    
    # Metrics
    st.subheader("📈 Real-time System Analytics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Risk", f"{prob:.1%}")
    c2.metric("Behavioral Cohort", user_data['segment'])
    c3.metric("Current Status", "🔴 AT RISK" if is_churn == 1 else "🟢 STABLE")
    
    # Logic
    segment = user_data['segment']
    usage = user_data['data_used']
    salary = user_data['estimated_salary']
    
    strategy = "Priority Support"
    rationale = "General churn risk detected."
    cost = 40
    
    if is_churn == 0:
        if "Premium" in segment:
            strategy = "VIP Loyalty Rewards"
            rationale = "Elite customer. **Strategy**: Offer exclusive event access and zero-cost equipment upgrades."
            cost = 60
        else:
            strategy = "Automated Relationship Sync"
            rationale = "Stable patterns. **Strategy**: Maintain engagement via personalized AI value summaries."
            cost = 2
    else:
        if "Heavy Data" in segment:
            strategy = "Infinite Data Loyalty Grant"
            rationale = "High usage at risk. **Strategy**: Add 50GB bonus data per month for 12 months."
            cost = 100
        elif salary > 80000 and usage < 1000:
            strategy = "Value Recovery Downsell"
            rationale = "Customer is overpaying. **Strategy**: Proactively migrate to a cheaper plan to save the relationship."
            cost = 25
        else:
            strategy = "Retention Specialist Direct"
            rationale = "Behavioral instability. **Strategy**: Personalized human outreach with a flexi-credit offer."
            cost = 45

    roi = ( (1200 if "Premium" in segment else 600) * 0.50 ) - cost
    
    # Strategy Card
    st.markdown(f"""
    <div class="strategy-card">
        <h2 style="color: #39D353 !important; margin-top: 0; border-bottom: none;">� ACTION: {strategy}</h2>
        <p style="font-size: 1.1em; line-height: 1.6; color: #F0F6FC !important;">{rationale}</p>
        <hr style="border: 0; border-top: 1px solid #30363D; margin: 20px 0;">
        <div style="display: flex; gap: 40px;">
            <div>
                <span style="color: #8B949E; font-size: 0.8em; font-weight: 700;">IMPLEMENTATION COST</span><br>
                <span style="color: #FFFFFF; font-size: 1.5em; font-weight: 900;">${cost}</span>
            </div>
            <div>
                <span style="color: #8B949E; font-size: 0.8em; font-weight: 700;">ESTIMATED ANNUAL ROI</span><br>
                <span style="color: #39D353; font-size: 1.5em; font-weight: 900;">${roi:,.2f}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📂 Raw Telemetry Data"):
        st.dataframe(user_data.to_frame().T)

else:
    if selected_id: st.warning("ID Code Not Found.")
    else: st.info("👋 System Ready. Select or type a Customer ID in the sidebar to start analysis.")
