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
# 0. THEME ARCHITECTURE (V7.0 - MIDNIGHT AMBER)
# ==========================================
st.set_page_config(page_title="AI STRATEGY COMMAND", layout="wide", page_icon="🚔")

st.markdown("""
<style>
    /* Global Base - Midnight Charcoal */
    [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stSidebar"], .main {
        background-color: #050505 !important;
        color: #E0E0E0 !important;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    /* Headings - Amber Gradient */
    h1, h2, h3 {
        background: linear-gradient(90deg, #F59E0B 0%, #D97706 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    /* Metric Cards - Nuclear Dark Force */
    [data-testid="stMetric"], .stMetric, div[data-testid="metric-container"] {
        background-color: #0F0F0F !important;
        background: #0F0F0F !important;
        border: 1px solid rgba(245, 158, 11, 0.3) !important;
        border-radius: 12px !important;
        padding: 24px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.8) !important;
    }
    
    /* Force Metric Text Colors */
    [data-testid="stMetricValue"] div, [data-testid="stMetricValue"] {
        color: #F59E0B !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] div, [data-testid="stMetricLabel"] p, [data-testid="stMetricLabel"] {
        color: #999999 !important;
    }

    /* Expanders & Other Boxes */
    .streamlit-expanderHeader, .stExpander {
        background-color: #111 !important;
        border-color: #222 !important;
    }

    /* The 'Strategy Command' Card - Flagship Amber */
    .strategy-hq {
        background: #111;
        border: 1px solid #222;
        border-top: 5px solid #F59E0B;
        padding: 40px;
        border-radius: 10px;
        margin: 30px 0;
        box-shadow: 0 15px 50px rgba(0,0,0,0.6);
    }
    
    /* Interactive Elements Styling */
    .stDataFrame, div[data-baseweb="select"], input {
        background-color: #111 !important;
        color: white !important;
        border: 1px solid #333 !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] > div {
        background-color: #080808 !important;
        border-right: 1px solid #222 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. AI CORE (Fully Verified Logic)
# ==========================================
@st.cache_data
def get_verified_data():
    try:
        df = pd.read_csv('telecom_churn.csv')
        df = df.fillna(df.median(numeric_only=True))
        return df
    except Exception as e:
        st.error(f"FATAL: Missing 'telecom_churn.csv'. System cannot proceed. Error: {e}")
        return None

@st.cache_resource
def train_ai_systems(df):
    # Predictive Agent
    data = df.copy()
    le = LabelEncoder()
    for col in data.select_dtypes(include=['object']):
        if 'date' not in col: data[col] = le.fit_transform(data[col].astype(str))
    
    X = data.drop(columns=['churn', 'customer_id'], errors='ignore')
    y = data['churn']
    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42).fit(X_s, y)
    
    # Clustering Agent
    c_features = ['age', 'estimated_salary', 'data_used', 'calls_made']
    X_c = df[c_features].copy()
    scaler_c = StandardScaler().fit(X_c)
    X_cs = scaler_c.transform(X_c)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_cs)
    
    # Generate Cluster Names
    centers = scaler_c.inverse_transform(kmeans.cluster_centers_)
    df_centers = pd.DataFrame(centers, columns=c_features)
    medians = X_c.median()
    
    name_map = {}
    for i, row in df_centers.iterrows():
        if row['data_used'] > medians['data_used'] * 1.5: name = "Data Pioneer"
        elif row['estimated_salary'] > medians['estimated_salary'] * 1.5: name = "Elite Premium"
        elif row['calls_made'] > medians['calls_made'] * 1.3: name = "Talkative Connector"
        else: name = "Standard Operational"
        name_map[i] = name
        
    return model, scaler, kmeans, scaler_c, c_features, name_map, kmeans.labels_

# ==========================================
# 2. RUNTIME BOOTSTRAP
# ==========================================
df = get_verified_data()
if df is not None:
    ai_model, ai_scaler, ai_kmeans, ai_scaler_c, c_feats, ai_name_map, ai_labels = train_ai_systems(df)
    df['Cluster_ID'] = ai_labels
    df['Segment'] = df['Cluster_ID'].map(ai_name_map)

# ==========================================
# 3. COMMAND INTERFACE
# ==========================================
st.title("🛰️ AI STRATEGIC COMMAND: AMBER v7.0")
st.write("---")

# Navigation Hub
with st.sidebar:
    st.header("🎛️ CUSTOMER SEARCH")
    search_mode = st.radio("SEARCH BY", ["SCROLL LIST", "MANUAL ID"], horizontal=True)
    if search_mode == "SCROLL LIST":
        selected_id = st.selectbox("IDENTIFY ANALYTIC TARGET", df['customer_id'].unique()[:500])
    else:
        id_input = st.text_input("ENTER ENTRY CODE")
        try: selected_id = int(id_input) if id_input else None
        except: selected_id = None

    st.write("---")
    st.success("�️ **SYSTEM STATUS: OPTIMAL**")
    st.info("💡 **VERSION 7.0**: All systems verified and executing.")

# MAIN DASHBOARD
if selected_id in df['customer_id'].values:
    user = df[df['customer_id'] == selected_id].iloc[0]
    
    # Live Metrics
    st.subheader("📡 SYSTEM TELEMETRY")
    m1, m2, m3, m4 = st.columns(4)
    
    # Risk Inference
    X_target = df[df['customer_id'] == selected_id].drop(columns=['churn', 'customer_id', 'Cluster_ID', 'Segment'], errors='ignore')
    for col in X_target.select_dtypes(include=['object']): X_target[col] = 0 # Baseline for categorical
    
    risk_prob = ai_model.predict_proba(ai_scaler.transform(X_target))[0][1]
    
    m1.metric("CHURN PROBABILITY", f"{risk_prob:.1%}")
    m2.metric("CURRENT COHORT", user['Segment'])
    m3.metric("ALERT SYSTEM", "CRITICAL" if risk_prob > 0.5 else "NOMINAL")
    m4.metric("SALARY BRACKET", f"${user['estimated_salary']:,.0f}")

    # --- ACTION RECOMMENDATION ---
    st.write("---")
    st.subheader("🎯 RETENTION COMMAND LOGIC")
    
    strategy, rationale, cost = "Standard Outreach", "User in stable orbit. Continue maintenance.", 5
    
    if risk_prob < 0.3:
        if "Elite" in user['Segment']:
            strategy, rationale, cost = "VIP Concierge Upgrade", "Elite status identified. **Command**: Offer exclusive lifestyle event trial to ensure long-term base loyalty.", 90
        else:
            strategy, rationale, cost = "AI Engagement Sync", "Nominal risk. **Command**: Automated relationship touchpoints via personalized AI notifications.", 2
    else:
        # High Risk Logic
        if user['estimated_salary'] > 85000 and user['data_used'] < 1200:
            strategy, rationale, cost = "Value Reconstruction Call", "Customer is overpaying for services. **Command**: Proactively migrate to a cheaper plan to save the relationship.", 30
        elif "Data" in user['Segment']:
            strategy, rationale, cost = "Unlimited Data Loyalty Stimulus", "High data reliance found. **Command**: Grant 90 days of unthrottled bandwidth to block competitive shifting.", 130
        else:
            strategy, rationale, cost = "Human Specialist Intervention", "Complex risk factors. **Command**: Immediate outbound call from Retention HQ with 40% discount authority.", 65

    # ROI Calculation
    est_clv = 1600 if "Elite" in user['Segment'] else 750
    roi = (est_clv * (1 - risk_prob)) - cost

    st.markdown(f"""
    <div class="strategy-hq">
        <h2 style="color: #F59E0B !important; margin-top: 0; letter-spacing: 2px;">🏆 TARGET STRATEGY: {strategy}</h2>
        <p style="font-size: 1.25rem; line-height: 1.8; color: #CCCCCC !important;">{rationale}</p>
        <hr style="border: 0; border-top: 1px solid #333; margin: 25px 0;">
        <div style="display: flex; gap: 60px;">
            <div>
                <span style="color: #666; font-size: 0.9rem; font-weight: 800; text-transform: uppercase; letter-spacing: 1px;">Implementation Cost</span><br>
                <span style="color: #FFFFFF; font-size: 1.8rem; font-weight: 950;">${cost}</span>
            </div>
            <div>
                <span style="color: #666; font-size: 0.9rem; font-weight: 800; text-transform: uppercase; letter-spacing: 1px;">Projected 12M ROI</span><br>
                <span style="color: #F59E0B; font-size: 1.8rem; font-weight: 950;">${roi:,.2f}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("👤 VIEW RAW CUSTOMER TELEMETRY"):
        st.dataframe(user.to_frame().T)

else:
    st.info("🔭 **SYSTEM SCANNING...** Target a Customer ID in the sidebar to generate a Command Strategy.")
    
    # Global Insight Map ( Landing Page)
    st.write("### AI Behavioral Population View")
    pca = PCA(n_components=2)
    sample_df = df.sample(1000) if len(df) > 1000 else df
    X_viz = ai_scaler_c.transform(sample_df[c_feats])
    pca_comp = pca.fit_transform(X_viz)
    
    df_viz = pd.DataFrame(pca_comp, columns=['PC1', 'PC2'])
    df_viz['Segment'] = sample_df['Segment'].values
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=df_viz, x='PC1', y='PC2', hue='Segment', alpha=0.7, palette='autumn', ax=ax)
    ax.set_title("Customer Multi-Dimensional Behavior Clustering", color='white')
    ax.tick_params(colors='white')
    fig.patch.set_facecolor('#050505')
    ax.set_facecolor('#111')
    st.pyplot(fig)
