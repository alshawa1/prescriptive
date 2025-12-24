import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ==========================================
# 0. THEME ARCHITECTURE (V7.1 - OBSIDIAN FORCE)
# ==========================================
st.set_page_config(page_title="AI STRATEGY COMMAND", layout="wide", page_icon="🚔")

# --- ULTIMATE CSS BLOWOUT ---
st.markdown("""
<style>
    /* 1. UNIVERSAL RESET - Force Obsidian Background on EVERYTHING */
    * {
        background-color: transparent !important;
    }
    
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stSidebar"], .main, .block-container {
        background-color: #050505 !important;
        background: #050505 !important;
        color: #E0E0E0 !important;
    }

    /* 2. TEXT FORCE - Ensure everything is readable */
    h1, h2, h3, h4, h5, h6, p, span, label, li, table, td, th {
        color: #E0E0E0 !important;
    }

    /* 3. AMBER ACCENTS - Brand Identity */
    h1, h2, h3 {
        background: linear-gradient(90deg, #F59E0B 0%, #D97706 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
        text-transform: uppercase;
    }

    /* 4. METRIC CARDS - Nuclear Deep Grey */
    [data-testid="stMetric"], .stMetric, div[data-testid="metric-container"] {
        background-color: #111111 !important;
        background: #111111 !important;
        border: 1px solid rgba(245, 158, 11, 0.4) !important;
        border-radius: 12px !important;
        padding: 25px !important;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.9) !important;
    }
    
    [data-testid="stMetricValue"] div, [data-testid="stMetricValue"] {
        color: #F59E0B !important;
        font-weight: 900 !important;
    }
    
    [data-testid="stMetricLabel"] div, [data-testid="stMetricLabel"] p {
        color: #888888 !important;
        text-transform: uppercase;
        font-size: 0.8rem;
    }

    /* 5. COMPONENTS - Tables, Selectors, Inputs */
    .stDataFrame, div[data-baseweb="select"], input, .stTextArea textarea {
        background-color: #111111 !important;
        color: white !important;
        border: 1px solid #333 !important;
    }

    /* 6. STRATEGY COMMAND HQ - Flagship Card */
    .strategy-hq {
        background: #0A0A0A;
        border: 1px solid #222;
        border-top: 6px solid #F59E0B;
        padding: 40px;
        border-radius: 10px;
        margin: 30px 0;
        box-shadow: 0 20px 60px rgba(0,0,0,0.8);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. CORE INTELLIGENCE (Verified)
# ==========================================
@st.cache_data
def get_verified_data():
    try:
        df = pd.read_csv('telecom_churn.csv')
        df = df.fillna(df.median(numeric_only=True))
        return df
    except:
        st.error("MISSING DATA: telecom_churn.csv not detected.")
        return None

@st.cache_resource
def train_ai_systems(df):
    data = df.copy()
    le = LabelEncoder()
    for col in data.select_dtypes(include=['object']):
        if 'date' not in col: data[col] = le.fit_transform(data[col].astype(str))
    
    X = data.drop(columns=['churn', 'customer_id'], errors='ignore')
    y = data['churn']
    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42).fit(X_s, y)
    
    c_feats = ['age', 'estimated_salary', 'data_used', 'calls_made']
    X_c = df[c_feats].copy()
    scaler_c = StandardScaler().fit(X_c)
    X_cs = scaler_c.transform(X_c)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_cs)
    
    centers = scaler_c.inverse_transform(kmeans.cluster_centers_)
    df_centers = pd.DataFrame(centers, columns=c_feats)
    medians = X_c.median()
    
    name_map = {}
    for i, row in df_centers.iterrows():
        if row['data_used'] > medians['data_used'] * 1.5: name = "Data Pioneer"
        elif row['estimated_salary'] > medians['estimated_salary'] * 1.5: name = "Elite Premium"
        elif row['calls_made'] > medians['calls_made'] * 1.3: name = "Talkative Connector"
        else: name = "Standard Operational"
        name_map[i] = name
        
    return model, scaler, kmeans, scaler_c, c_feats, name_map, kmeans.labels_

# ==========================================
# 2. INITIALIZATION
# ==========================================
df = get_verified_data()
if df is not None:
    ai_model, ai_scaler, ai_kmeans, ai_scaler_c, c_feats, ai_name_map, ai_labels = train_ai_systems(df)
    df['Cluster_ID'] = ai_labels
    df['Segment'] = df['Cluster_ID'].map(ai_name_map)

# ==========================================
# 3. INTERFACE
# ==========================================
st.title("🛰️ AI STRATEGIC COMMAND: OBSIDIAN v7.1")
st.write("---")

with st.sidebar:
    st.header("🎛️ TARGET SELECTION")
    search_type = st.radio("SEARCH BY", ["SCROLL LIST", "MANUAL ID"], horizontal=True)
    if search_type == "SCROLL LIST":
        target_id = st.selectbox("IDENTIFY ID", df['customer_id'].unique()[:500])
    else:
        id_str = st.text_input("ENTRY CODE")
        try: target_id = int(id_str) if id_str else None
        except: target_id = None

    st.write("---")
    st.success("✅ **SYSTEM ONLINE: v7.1**")
    st.warning("⚠️ **IMPORTANT**: You must PUSH these changes to GitHub for the link to turn black.")

if target_id in df['customer_id'].values:
    user = df[df['customer_id'] == target_id].iloc[0]
    
    # Analytics
    st.subheader("📡 TELEMETRY STREAM")
    col1, col2, col3, col4 = st.columns(4)
    
    X_target = df[df['customer_id'] == target_id].drop(columns=['churn', 'customer_id', 'Cluster_ID', 'Segment'], errors='ignore')
    for col in X_target.select_dtypes(include=['object']): X_target[col] = 0
    
    risk = ai_model.predict_proba(ai_scaler.transform(X_target))[0][1]
    
    col1.metric("CHURN PROBABILITY", f"{risk:.1%}")
    col2.metric("CURRENT COHORT", user['Segment'])
    col3.metric("ALERT SYSTEM", "CRITICAL" if risk > 0.5 else "NOMINAL")
    col4.metric("SALARY BRACKET", f"${user['estimated_salary']:,.0f}")

    # Retention HQ
    st.write("---")
    st.subheader("🎯 COMMAND STRATEGY")
    
    strategy, rationale, cost = "Standard Outreach", "User in stable orbit. Continue maintenance.", 5
    
    if risk < 0.3:
        if "Elite" in user['Segment']:
            strategy, rationale, cost = "VIP Concierge Upgrade", "Elite status identified. Pro-actively securing high-value asset.", 95
        else:
            strategy, rationale, cost = "AI Engagement Sync", "Maintenance mode. Automated touchpoints active.", 2
    else:
        if user['estimated_salary'] > 85000 and user['data_used'] < 1200:
            strategy, rationale, cost = "Value Optimization Call", "Account is overpaying. Immediate plan downsell recommended to save account.", 35
        elif "Data" in user['Segment']:
            strategy, rationale, cost = "Unlimited Bandwidth Grant", "Data reliance found. Neutralizing competitor offers with unlimited high-speed data.", 145
        else:
            strategy, rationale, cost = "Retention Specialist HQ", "Complex risk profile. Direct human intervention with flex-credit authority.", 65

    roi = (1500 if "Elite" in user['Segment'] else 700) * (1 - risk) - cost

    st.markdown(f"""
    <div class="strategy-hq">
        <h2 style="color: #F59E0B !important; margin: 0; letter-spacing: 2px;">🚀 COMMAND ACTION: {strategy}</h2>
        <p style="font-size: 1.25rem; color: #AAA !important; margin: 20px 0;">{rationale}</p>
        <hr style="border: 0; border-top: 1px solid #333; margin: 25px 0;">
        <div style="display: flex; gap: 60px;">
            <div>
                <span style="color: #666; font-size: 0.8rem; font-weight: 800; text-transform: uppercase;">EXECUTION COST</span><br>
                <span style="color: #FFF; font-size: 1.8rem; font-weight: 900;">${cost}</span>
            </div>
            <div>
                <span style="color: #666; font-size: 0.8rem; font-weight: 800; text-transform: uppercase;">PROJECTED ANNUAL ROI</span><br>
                <span style="color: #F59E0B; font-size: 1.8rem; font-weight: 900;">${roi:,.2f}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("👤 RAW TELEMETRY"):
        st.dataframe(user.to_frame().T)

else:
    st.info("🔭 **SYSTEM SCANNING...** Identify a Target ID in the sidebar to generate strategy.")
    
    # Global Plot
    pca = PCA(n_components=2)
    sample = df.sample(1000) if len(df) > 1000 else df
    pca_comp = pca.fit_transform(ai_scaler_c.transform(sample[c_feats]))
    df_viz = pd.DataFrame(pca_comp, columns=['PC1', 'PC2'])
    df_viz['Segment'] = sample['Segment'].values
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=df_viz, x='PC1', y='PC2', hue='Segment', alpha=0.7, palette='YlOrBr', ax=ax)
    ax.tick_params(colors='white')
    fig.patch.set_facecolor('#050505')
    ax.set_facecolor('#050505')
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    st.pyplot(fig)
