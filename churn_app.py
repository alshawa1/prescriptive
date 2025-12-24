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
# PAGE CONFIG & PREMIUM CYBER-SPACE THEME
# ==========================================
st.set_page_config(page_title="AI Strategy Command", layout="wide", page_icon="⚖️")

# Nuclear CSS: Deep Space Violet & Gold Glassmorphism
st.markdown("""
<style>
    /* 1. Global Background - Deep Cosmic Indigo */
    [data-testid="stAppViewContainer"], 
    [data-testid="stHeader"], 
    [data-testid="stSidebar"], 
    .main, .stApp, html, body {
        background: radial-gradient(circle at top right, #16213E, #1A1A2E) !important;
        color: #E2E2E2 !important;
        font-family: 'Outfit', sans-serif;
    }

    /* 2. Glassmorphism for Metric Cards */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 20px !important;
        padding: 25px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
        transition: transform 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(255, 215, 0, 0.3) !important;
    }

    /* 3. Golden Highlights for Values */
    [data-testid="stMetricValue"] > div {
        color: #FFD700 !important;
        font-weight: 800 !important;
        text-shadow: 0 0 15px rgba(255, 215, 0, 0.4) !important;
        letter-spacing: -1px;
    }
    [data-testid="stMetricLabel"] > div {
        color: #B8B8B8 !important;
        text-transform: uppercase !important;
        font-size: 0.8em !important;
        letter-spacing: 2px !important;
    }

    /* 4. Sidebar - Modern Violet */
    section[data-testid="stSidebar"] > div {
        background-color: #0F0F1B !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    [data-testid="stSidebar"] * {
        color: #C1C1C1 !important;
    }

    /* 5. Headers & Titles */
    h1, h2, h3 {
        background: linear-gradient(90deg, #FFD700, #E94560);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
    }

    /* 6. The Command Card (Unified Strategy) */
    .command-card {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(15px);
        border: 2px solid rgba(255, 215, 0, 0.2);
        padding: 35px;
        border-radius: 24px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
        margin-top: 30px;
        position: relative;
        overflow: hidden;
    }
    .command-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 5px;
        background: linear-gradient(90deg, #FFD700, #E94560);
    }
    
    /* Input Styling */
    div[data-baseweb="select"], input {
        background-color: #16213E !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        color: white !important;
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
    
    drop_cols = ['churn', 'customer_id']
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
        if row['data_used'] > global_medians['data_used'] * 1.5: label = "Galactic User (Heavy Data)"
        elif row['estimated_salary'] > global_medians['estimated_salary'] * 1.5: label = "Nebula Gold (Premium)"
        elif row['calls_made'] > global_medians['calls_made'] * 1.5: label = "Solar Voice (High Caller)"
        else: label = "Standard Satellite"
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
# MAIN COMMAND CENTER
# ==========================================
st.title("🛰️ AI STRATEGY COMMAND")
st.write("---")

# Sidebar - High End
st.sidebar.markdown("### 🎛️ CUSTOMER SEARCH")
search_type = st.sidebar.radio("MODE", ["SELECT LIST", "CODE SCAN (ID)"])

if search_type == "SELECT LIST":
    selected_id = st.sidebar.selectbox("TARGET ID", df_raw['customer_id'].unique()[:500])
else:
    id_input = st.sidebar.text_input("ENTRY CODE")
    try: selected_id = int(id_input) if id_input else None
    except: selected_id = None

# ACTION CENTER
if selected_id in df_raw['customer_id'].values:
    user_data = df_raw[df_raw['customer_id'] == selected_id].iloc[0]
    
    # ML Inference
    temp_df = df_raw.copy()
    le = LabelEncoder()
    for col in temp_df.select_dtypes(include=['object']):
        if 'date' not in col: temp_df[col] = le.fit_transform(temp_df[col].astype(str))
    
    drop_ml = ['churn', 'customer_id', 'cluster_id', 'segment']
    X_user = temp_df[temp_df['customer_id'] == selected_id].drop(columns=[c for c in drop_ml if c in temp_df.columns], errors='ignore')
    user_scaled = scaler_pred.transform(X_user)
    
    probs = model_pred.predict_proba(user_scaled)[0]
    prob = probs[1] if len(probs) > 1 else (1.0 if model_pred.classes_[0] == 1 else 0.0)
    is_churn = model_pred.predict(user_scaled)[0]
    
    # Dashboard Grid
    st.subheader("📡 SYSTEM TELEMETRY")
    m1, m2, m3 = st.columns(3)
    m1.metric("CHURN PROBABILITY", f"{prob:.1%}")
    m2.metric("COHORT SEGMENT", user_data['segment'])
    m3.metric("MISSION STATUS", "🔴 RETENTION RISK" if is_churn == 1 else "🟢 STABLE ORBIT")
    
    # AI STRATEGY ENGINE
    segment = user_data['segment']
    usage = user_data['data_used']
    salary = user_data['estimated_salary']
    
    strategy = ""
    rationale = ""
    cost = 0
    
    if is_churn == 0:
        if "Nebula" in segment:
            strategy = "EXCLUSIVE: TITANIUM LOYALTY PASS"
            rationale = "High-value stable orbit. **Strategy**: Offer a personalized lifestyle concierge trial to permanently lock in this elite nebula cohort."
            cost = 75
        elif usage > 4000:
            strategy = "UPGRADE: INFINITE BANDWIDTH XP"
            rationale = "Heavy usage detected. **Strategy**: Proactively remove all data throttling to increase satisfaction and lifetime value."
            cost = 15
        else:
            strategy = "TACTIC: SMART CARE SYNC"
            rationale = "Healthy signal. **Strategy**: automated AI-generated relationship messages to maintain brand dominance."
            cost = 2
    else:
        if salary > 90000 and usage < 1500:
            strategy = "RESCUE: PLAN OPTIMIZER PRO"
            rationale = "Price/Value misalignment. Customer feels they are overpaying. **Strategy**: Propose a plan downgrade to lower churn probability while retaining base margin."
            cost = 25
        elif "Galactic" in segment:
            strategy = "SHIELD: 90 DAYS DATA CREDIT"
            rationale = "Heavy usage churn detected. **Strategy**: Massive data-specific credit to counteract cheaper competitive infrastructure."
            cost = 150
        elif "Budget" in segment:
            strategy = "BRIDGE: $40 INSTANT RELIEF"
            rationale = "Financial sensitivity. **Strategy**: One-time direct bill credit to solve the immediate economic churn threshold."
            cost = 40
        else:
            strategy = "DIRECT: PRIORITY COMMAND CALL"
            rationale = "Complex churn indicators. **Strategy**: Personalized call from the VIP Escalation team to solve qualitative issues."
            cost = 50

    roi = ( (1500 if "Nebula" in segment else 750) * 0.45 ) - cost
    
    # THE COMMAND BOX
    st.markdown(f"""
    <div class="command-card">
        <h2 style="color: #FFD700 !important; margin-top: 0; letter-spacing: 2px;">⚡ STRATEGIC COMMAND: {strategy}</h2>
        <p style="font-size: 1.15em; line-height: 1.7; color: #D1D1D1 !important;">{rationale}</p>
        <hr style="border: 0; border-top: 1px solid rgba(255, 215, 0, 0.15); margin: 25px 0;">
        <div style="display: flex; gap: 60px;">
            <div>
                <span style="color: #8B8B8B; font-size: 0.85em; font-weight: 700; letter-spacing: 1.5px;">EST. COST</span><br>
                <span style="color: #FFFFFF; font-size: 1.7em; font-weight: 900;">${cost}</span>
            </div>
            <div>
                <span style="color: #8B8B8B; font-size: 0.85em; font-weight: 700; letter-spacing: 1.5px;">PROJECTED IMPACT (ANNUAL ROI)</span><br>
                <span style="color: #FF007A; font-size: 1.7em; font-weight: 900;">${roi:,.2f}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📂 VIEW RAW TELEMETRY DATA"):
        st.write(user_data.to_frame().T)

else:
    if selected_id: st.warning("ID NOT FOUND")
    else: st.info("�️ SYSTEM ONLINE. SPECIFY TARGET CUSTOMER ID IN SIDEBAR.")
