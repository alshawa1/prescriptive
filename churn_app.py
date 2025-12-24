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
# PAGE CONFIG & NUCLEAR DARK THEME
# ==========================================
st.set_page_config(page_title="Telecom Strategic AI", layout="wide", page_icon="🎯")

# Aggressive CSS to force Dark Mode regardless of system settings
st.markdown("""
<style>
    /* Force background on ALL potential overlays and containers */
    [data-testid="stAppViewContainer"], 
    [data-testid="stHeader"], 
    [data-testid="stSidebar"], 
    .main, 
    .stApp,
    html, body {
        background-color: #0B0E14 !important;
        color: #F8FAFC !important;
    }

    /* Target the specific sidebar container */
    section[data-testid="stSidebar"] > div {
        background-color: #161B22 !important;
    }

    /* Force all text in the app and sidebar to be bright white/off-white */
    p, span, label, h1, h2, h3, h4, .stMarkdown, .stSelectbox label, .stTextInput label {
        color: #F8FAFC !important;
    }

    /* Metric Card - Dark Industrial Style */
    [data-testid="stMetric"] {
        background-color: #1C2128 !important;
        border: 2px solid #30363D !important;
        border-radius: 12px !important;
        padding: 20px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5) !important;
    }
    [data-testid="stMetricValue"] > div {
        color: #58A6FF !important;
        font-weight: 800 !important;
        text-shadow: 0 0 10px rgba(88, 166, 255, 0.3) !important;
    }
    [data-testid="stMetricLabel"] > div {
        color: #8B949E !important;
        text-transform: uppercase !important;
        font-size: 0.8em !important;
        letter-spacing: 1px !important;
    }

    /* Specific Strategy Box (Ultra Dark Variant) */
    .unified-strategy-box {
        background-color: #0D1117;
        border: 2px solid #238636;
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0 0 20px rgba(35, 134, 54, 0.2);
        margin-top: 20px;
    }
    
    /* Input field styling for dark mode */
    .stSelectbox div[data-baseweb="select"], .stTextInput input {
        background-color: #21262D !important;
        color: white !important;
        border: 1px solid #30363D !important;
    }
</style>
""", unsafe_allow_html=True)

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
            # Standard IQR clipping
            iqr = q3 - q1
            df[col] = df[col].clip(q1 - 1.5*iqr, q3 + 1.5*iqr)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
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
    cluster_features = ['age', 'estimated_salary', 'calls_made', 'sms_sent', 'data_used']
    cluster_features = [f for f in cluster_features if f in df.columns]
    
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
        if row['data_used'] > global_medians['data_used'] * 1.5: label = "Heavy Data Consumer"
        elif row['estimated_salary'] > global_medians['estimated_salary'] * 1.5: label = "Premium High-Value"
        elif row['calls_made'] > global_medians['calls_made'] * 1.5: label = "Frequent Caller"
        elif row['estimated_salary'] < global_medians['estimated_salary'] * 0.7: label = "Budget-Conscious"
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
# MAIN INTERFACE
# ==========================================
st.title("🎯 Telecom Decision & Strategy Hub")
st.write("### AI-Powered Retention Engine")
st.markdown("---")

# Layout: Selection Sidebar
st.sidebar.header("🔍 Customer Lookup")
search_type = st.sidebar.radio("Method", ["Selection List", "Manual ID Entry"])

if search_type == "Selection List":
    selected_id = st.sidebar.selectbox("Choose Customer", df_raw['customer_id'].unique()[:500])
else:
    id_input = st.sidebar.text_input("Enter ID (e.g. 1, 15)")
    try:
        selected_id = int(id_input) if id_input else None
    except:
        selected_id = None

# PROCESS RESULTS
if selected_id in df_raw['customer_id'].values:
    user_data = df_raw[df_raw['customer_id'] == selected_id].iloc[0]
    
    # --- ML Inference ---
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
    
    # --- Dashboard Metrics ---
    st.subheader("� Real-time Analytics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Churn Risk", f"{prob:.1%}")
    m2.metric("Behavioral Segment", user_data['segment'])
    m3.metric("Retention Status", "🔴 AT RISK" if is_churn == 1 else "🟢 STABLE")
    
    # --- Strategy Recommendation Engine ---
    segment = user_data['segment']
    usage = user_data['data_used']
    salary = user_data['estimated_salary']
    
    strategy_title = ""
    ai_justification = ""
    est_cost = 0
    
    if is_churn == 0:
        if "Premium" in segment:
            strategy_title = "Action: VIP Excellence Bundle"
            ai_justification = "Customer is high-value and loyal. **Strategic Goal**: Reward stability with an invite-only service tier to prevent competitive poaching."
            est_cost = 60
        elif usage > 5000:
            strategy_title = "Action: Unlimited Data Speed Trial"
            ai_justification = "Heavy user found. **Strategic Goal**: Boost perceived value by removing speed caps for 3 months."
            est_cost = 10
        else:
            strategy_title = "Action: Periodic Care Notification"
            ai_justification = "Balanced usage patterns. **Strategic Goal**: Low-cost relational touchpoints via automated AI messaging."
            est_cost = 1
    else:
        # CHURN RISK MITIGATION
        if salary > 85000 and usage < 1200:
            strategy_title = "Action: Cost-Save Optimization Call"
            ai_justification = "Customer is 'Value Seeking'. They pay for more than they use. **Strategic Goal**: Proactively downsell to a cheaper plan to save the relationship."
            est_cost = 20
        elif "Heavy Data" in segment:
            strategy_title = "Action: 50% Data Loyalty Credit"
            ai_justification = "Data consumption is the primary bond. **Strategic Goal**: Massive price decrease for 4 months to neutralize competitor data offers."
            est_cost = 120
        elif "Budget" in segment:
            strategy_title = "Action: Direct Bill Subsidy ($25)"
            ai_justification = "Price sensitivity is the driver. **Strategic Goal**: Immediate financial relief to bridge the churn decision period."
            est_cost = 25
        else:
            strategy_title = "Action: Priority Win-Back Call"
            ai_justification = "Complex churn indicators. **Strategic Goal**: Direct human intervention from retention desk with flexi-credit."
            est_cost = 45

    # Unified Strategic Result Card
    roi = ( (1200 if "Premium" in segment else 600) * 0.50 ) - est_cost
    
    st.markdown(f"""
    <div class="unified-strategy-box">
        <h2 style="color: #39D353 !important; margin-top: 0;">🏆 BEST STRATEGY: {strategy_title}</h2>
        <p style="font-size: 1.15em; line-height: 1.6; color: #C9D1D9 !important;">{ai_justification}</p>
        <hr style="border-top: 1px solid #30363D; margin: 20px 0;">
        <div style="display: flex; gap: 50px;">
            <div>
                <span style="color: #8B949E; font-size: 0.9em; font-weight: 600;">ESTIMATED COST</span><br>
                <span style="color: #F8FAFC; font-size: 1.5em; font-weight: 800;">${est_cost}</span>
            </div>
            <div>
                <span style="color: #8B949E; font-size: 0.9em; font-weight: 600;">PROJECTED IMPACT (ANNUAL ROI)</span><br>
                <span style="color: #39D353; font-size: 1.5em; font-weight: 800;">${roi:,.2f}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("� Deep Dive: Customer Profile Data"):
        st.dataframe(user_data.to_frame().T)

else:
    if selected_id:
        st.warning("Customer ID not found.")
    else:
        st.info("👋 System Ready. Select or type a Customer ID to generate the optimized AI strategy.")
