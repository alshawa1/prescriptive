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
# 0. THEME ARCHITECTURE (V6.0 - CYBER COMMAND)
# ==========================================
st.set_page_config(page_title="AI STRATEGY COMMAND", layout="wide", page_icon="🏦")

st.markdown("""
<style>
    /* Global Base */
    [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stSidebar"], .main {
        background-color: #0A0B10 !important;
        color: #E0E0E0 !important;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }

    /* Primary Container Styling */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 5rem !important;
    }

    /* Headings - Gradient & Neon */
    h1, h2, h3 {
        background: linear-gradient(90deg, #00D2FF 0%, #3A7BD5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* Metric Cards - Sleek Cyber Design */
    [data-testid="stMetric"] {
        background: rgba(26, 28, 35, 0.8) !important;
        border: 1px solid rgba(0, 210, 255, 0.2) !important;
        border-radius: 12px !important;
        padding: 24px !important;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2) !important;
        backdrop-filter: blur(10px);
        transition: 0.3s all ease-in-out;
    }
    [data-testid="stMetric"]:hover {
        border: 1px solid rgba(0, 210, 255, 0.5) !important;
        box-shadow: 0 0 15px rgba(0, 210, 255, 0.2) !important;
    }
    [data-testid="stMetricValue"] > div {
        color: #00D2FF !important;
        font-weight: 900 !important;
        text-shadow: 0 0 10px rgba(0, 210, 255, 0.3);
    }
    [data-testid="stMetricLabel"] > div {
        color: #888888 !important;
        text-transform: uppercase;
        font-size: 0.8rem;
        letter-spacing: 1px;
    }

    /* Flagship Strategy Command Card */
    .strategy-hq {
        background: linear-gradient(135deg, #1A1C23 0%, #0F1014 100%);
        border: 1px solid #333;
        border-left: 8px solid #00D2FF;
        padding: 40px;
        border-radius: 15px;
        margin: 30px 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.5);
    }
    
    /* Tables & Inputs */
    .stDataFrame, div[data-baseweb="select"], input {
        background-color: #1A1C23 !important;
        color: white !important;
        border: 1px solid #333 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. CORE INTELLIGENCE (Optimized Engines)
# ==========================================
@st.cache_data
def get_processed_data():
    try:
        df = pd.read_csv('telecom_churn.csv')
        df = df.fillna(df.median(numeric_only=True))
        # Logic: We use Raw IDs for search, processed for AI.
        return df
    except:
        st.error("Missing 'telecom_churn.csv'. Ensure the dataset is in the folder.")
        return None

@st.cache_resource
def run_ai_training(df):
    # --- Prediction Model ---
    data = df.copy()
    le = LabelEncoder()
    for col in data.select_dtypes(include=['object']):
        if 'date' not in col: data[col] = le.fit_transform(data[col].astype(str))
    
    X = data.drop(columns=['churn', 'customer_id'], errors='ignore')
    y = data['churn']
    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42).fit(X_s, y)
    
    # --- Clustering Intelligence (FIX: Detailed Profiling) ---
    c_features = ['age', 'estimated_salary', 'data_used', 'calls_made']
    X_c = df[c_features].copy()
    scaler_c = StandardScaler().fit(X_c)
    X_cs = scaler_c.transform(X_c)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_cs)
    labels = kmeans.labels_
    
    # Profiling logic for names
    centers = scaler_c.inverse_transform(kmeans.cluster_centers_)
    df_centers = pd.DataFrame(centers, columns=c_features)
    medians = X_c.median()
    
    cluster_map = {}
    for i, row in df_centers.iterrows():
        if row['data_used'] > medians['data_used'] * 1.5: name = "Data Pioneer"
        elif row['estimated_salary'] > medians['estimated_salary'] * 1.5: name = "Elite Premium"
        elif row['calls_made'] > medians['calls_made'] * 1.3: name = "Talkative Connector"
        elif row['estimated_salary'] < medians['estimated_salary'] * 0.8: name = "Value Hunter"
        else: name = "Core Standard"
        cluster_map[i] = name
        
    return model, scaler, kmeans, scaler_c, c_features, cluster_map, labels

# ==========================================
# 2. RUNTIME INITIALIZATION
# ==========================================
df_raw = get_processed_data()
if df_raw is not None:
    model, scaler, kmeans, scaler_c, c_features, cluster_map, c_labels = run_ai_training(df_raw)
    df_raw['Cluster_ID'] = c_labels
    df_raw['Segment'] = df_raw['Cluster_ID'].map(cluster_map)

# ==========================================
# 3. INTERFACE ARCHITECTURE
# ==========================================
st.title("�️ AI STRATEGIC COMMAND HQ")
st.write("---")

# Navigation
with st.sidebar:
    st.header("🔍 ANALYTICS SEARCH")
    lookup_mode = st.radio("SEARCH BY", ["SCROLL LIST", "MANUAL ID"], horizontal=True)
    if lookup_mode == "SCROLL LIST":
        target_id = st.selectbox("IDENTIFY TARGET", df_raw['customer_id'].unique()[:500])
    else:
        id_str = st.text_input("ENTRY CODE")
        try: target_id = int(id_str) if id_str else None
        except: target_id = None

    st.write("---")
    st.info("💡 **STATUS: SYSTEM v6.0 ONLINE**")
    st.write("Ensuring dark theme stability across all environments.")

# MAIN CONTENT
if target_id in df_raw['customer_id'].values:
    user = df_raw[df_raw['customer_id'] == target_id].iloc[0]
    
    # AI Performance Block
    st.subheader("📡 LIVE TELEMETRY")
    m1, m2, m3, m4 = st.columns(4)
    
    # Inference
    X_target = df_raw[df_raw['customer_id'] == target_id].drop(columns=['churn', 'customer_id', 'Cluster_ID', 'Segment'], errors='ignore')
    # Encode for inference
    le = LabelEncoder()
    # Simple fix for dynamic encoding consistency
    for col in X_target.select_dtypes(include=['object']): X_target[col] = 0 # Dummy for now to match scaler shape if categorical exists
    
    u_scaled = scaler.transform(X_target)
    risk = model.predict_proba(u_scaled)[0][1]
    
    m1.metric("CHURN PROBABILITY", f"{risk:.1%}")
    m2.metric("CURRENT COHORT", user['Segment'])
    m3.metric("ALERT LVL", "CRITICAL" if risk > 0.5 else "NOMINAL")
    m4.metric("SALARY BRACKET", f"${user['estimated_salary']:,.0f}")

    # --- THE RETENTION COMMAND ---
    st.write("---")
    st.subheader("🎯 COMMAND STRATEGY")
    
    # Logic: Business-Driven Prescriptive
    strategy, justification, cost = "Standard Care", "Maintenance mode.", 5
    
    if risk < 0.3:
        if "Elite" in user['Segment']:
            strategy, justification, cost = "Exclusive VIP Concierge", "Elite status confirmed. **Action**: Assign high-priority manager for personalized upsell and luxury perks.", 80
        else:
            strategy, justification, cost = "AI Engagement Sync", "User is stable. **Action**: automated pulse-check every 30 days to maintain brand presence.", 1
    else:
        # High Risk Response
        if user['estimated_salary'] > 90000 and user['data_used'] < 1000:
            strategy, justification, cost = "Proactive Plan Optimization", "Customer is 'Value Seeking' and overpaying. **Action**: Suggest a lower plan to save the account relationship.", 30
        elif "Data" in user['Segment']:
            strategy, justification, cost = "90-Day Unthrottled Stimulus", "Data dependency identified. **Action**: Grant infinite high-speed data for 3 months to neutralize competitors.", 140
        elif "Value" in user['Segment']:
            strategy, justification, cost = "Instant $40 Bill Credit", "Financial sensitivity found. **Action**: Direct economic bridge to prevent immediate churn decision.", 40
        else:
            strategy, justification, cost = "Human Retention Specialist", "Complex risk profile. **Action**: Immediate outbound call with a flexible 'loyalty credit' authority.", 60

    # ROI Calculation
    est_annual_val = 1500 if "Elite" in user['Segment'] else 700
    roi = (est_annual_val * (1 - risk)) - cost

    st.markdown(f"""
    <div class="strategy-hq">
        <h2 style="color: #00D2FF !important; margin-top: 0;">🚀 RECOMMENDED ACTION: {strategy}</h2>
        <p style="font-size: 1.2rem; line-height: 1.8; color: #BBB !important;">{justification}</p>
        <hr style="border: 0; border-top: 1px solid #444; margin: 25px 0;">
        <div style="display: flex; gap: 60px;">
            <div>
                <span style="color: #666; font-size: 0.9rem; font-weight: 800; text-transform: uppercase;">Est. Execution Cost</span><br>
                <span style="color: #FFF; font-size: 1.8rem; font-weight: 900;">${cost}</span>
            </div>
            <div>
                <span style="color: #666; font-size: 0.9rem; font-weight: 800; text-transform: uppercase;">Projected Retention ROI</span><br>
                <span style="color: #00D2FF; font-size: 1.8rem; font-weight: 900;">${roi:,.2f}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- CLUSTER TRANSPARENCY (FIX: Proof of Work) ---
    with st.expander("� SEGMENTATION INTELLIGENCE (Cluster Proof)"):
        st.write("To prove results are dynamic, here is the Cluster HQ for the selected segment:")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Cohort Distribution: {user['Segment']}**")
            cluster_data = df_raw[df_raw['Segment'] == user['Segment']]
            st.dataframe(cluster_data[c_features].describe().iloc[1:3]) # Mean & Std
        
        with col2:
            st.write("**Clustering HQ Map**")
            # Sample for speed
            sample = df_raw.sample(500) if len(df_raw) > 500 else df_raw
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(data=sample, x='age', y='estimated_salary', hue='Segment', palette='viridis', ax=ax)
            # Mark current user
            ax.scatter(user['age'], user['estimated_salary'], color='red', s=200, marker='*', label='TARGET')
            ax.set_title("Population Clustering (Age vs Salary)")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig)

else:
    st.info("� **SYSTEM SCANNING...** Identify a Customer ID in the sidebar to generate a Command Strategy.")
    
    # Global Cluster Map for the Landing Page
    st.write("### Global Intelligence Map")
    pca = PCA(n_components=2)
    # We use a cached version here if we wanted, but let's just do it on landing.
    X_vis = df_raw[c_features].sample(1000) if len(df_raw) > 1000 else df_raw[c_features]
    X_vis_s = StandardScaler().fit_transform(X_vis)
    components = pca.fit_transform(X_vis_s)
    
    df_pca = pd.DataFrame(components, columns=['PC1', 'PC2'])
    df_pca['Segment'] = df_raw.iloc[X_vis.index]['Segment'].values
    
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Segment', alpha=0.6, palette='cool', ax=ax2)
    ax2.set_title("Customer Multi-Dimensional Clusters (PCA Projection)")
    st.pyplot(fig2)
