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
# PAGE CONFIG & PREMIUM LIGHT THEME
# ==========================================
st.set_page_config(page_title="Telecom Strategic AI", layout="wide", page_icon="🎯")

# Custom CSS for high contrast and coordination
st.markdown("""
<style>
    /* High Contrast Text */
    .main .block-container h1, .main .block-container h2, .main .block-container h3, .main .block-container p, .main .block-container span {
        color: #0F172A !important;
    }
    
    /* Sidebar Text visibility */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
        color: #1E293B !important;
    }

    /* Metric Card Improvements */
    [data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        padding: 20px !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }
    [data-testid="stMetricValue"] > div {
        color: #2563EB !important;
    }
    [data-testid="stMetricLabel"] > div {
        color: #64748B !important;
    }

    /* Strategy Box Styling */
    .stSuccess {
        background-color: #F0FDF4 !important;
        border: 1px solid #BBF7D0 !important;
        color: #166534 !important;
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
# MAIN INTERFACE (RECOVERY & RECOMMENDATION)
# ==========================================
st.title("🎯 Telecom Retention & Strategy Engine")
st.markdown("---")

# Layout: Selection Sidebar
st.sidebar.header("🔍 Find Customer")
search_type = st.sidebar.radio("Lookup Method", ["Scroll List", "Manual ID Code"])

if search_type == "Scroll List":
    selected_id = st.sidebar.selectbox("Select Customer ID", df_raw['customer_id'].unique()[:500])
else:
    id_input = st.sidebar.text_input("Type ID (e.g. 10, 42)")
    try:
        selected_id = int(id_input) if id_input else None
    except:
        st.sidebar.error("Please enter a numeric ID")
        selected_id = None

# PROCESS RESULTS
if selected_id in df_raw['customer_id'].values:
    user_data = df_raw[df_raw['customer_id'] == selected_id].iloc[0]
    
    # --- Prediction Processing ---
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
    
    # --- UI Grid ---
    st.subheader("📊 Performance Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Churn Risk", f"{prob:.1%}")
    m2.metric("Customer Segment", user_data['segment'])
    m3.metric("Status", "🔴 AT RISK" if is_churn == 1 else "🟢 STABLE")
    
    st.write("### 💡 Recommended Strategy")
    
    segment = user_data['segment']
    usage = user_data['data_used']
    salary = user_data['estimated_salary']
    
    strategy_title = ""
    ai_justification = ""
    est_cost = 0
    
    # RECOMMENDATION ENGINE (ENHANCED LOGIC)
    if is_churn == 0:
        if "Premium" in segment:
            strategy_title = "Gold Membership Anniversary Offer"
            ai_justification = "Our analysis shows this customer is highly stable and profitable. **Best Strategy**: Lock in their loyalty by acknowledging their 'High-Value' Status with a premium perk."
            est_cost = 45
        elif usage > 5000:
            strategy_title = "Loyalty Speed Multiplier"
            ai_justification = "Heavy data usage detected. **Best Strategy**: Provide a free 'Premium Speed' boost for 30 days to reinforce value perception."
            est_cost = 5
        else:
            strategy_title = "Smart Engagement Check-in"
            ai_justification = "Healthy parameters. **Best Strategy**: Automated personalized content to maintain brand visibility."
            est_cost = 1
    else:
        # CHURN PREVENTION
        if salary > 80000 and usage < 1000:
            strategy_title = "Proactive Cost-Saver Optimization"
            ai_justification = "Customer exhibits 'Overpaying' behavior. **Best Strategy**: Offer a cheaper plan before they churn for a competitor. Saving the client with lower ARPU is better than 100% loss."
            est_cost = 25
        elif "Heavy Data" in segment:
            strategy_title = "Data Loyalty Shield (50% Off)"
            ai_justification = "Usage is high but risk is imminent. **Best Strategy**: Aggressive 6-month discount on data to nullify competitor pricing threats."
            est_cost = 120
        elif "Budget" in segment:
            strategy_title = "Financial Incentive: $30 Bill Credit"
            ai_justification = "Salary is in lower quartile. **Best Strategy**: Price sensitivity is the driver. A direct bill credit is the most effective retention tool."
            est_cost = 30
        else:
            strategy_title = "Strategic Retention Callback"
            ai_justification = "Behavioral driver unclear. **Best Strategy**: Direct reach-out from manager with an open-ended credit offer to diagnose and solve dissatisfaction."
            est_cost = 40

    # --- Unified Strategy Card ---
    st.markdown(f"""
    <div style="background-color: #F0FDF4; border: 1px solid #BBF7D0; padding: 25px; border-radius: 15px;">
        <h3 style="color: #166534; margin-top: 0;">🚀 RECOMMENDED STRATEGY: {strategy_title}</h3>
        <p style="color: #1E293B; font-size: 1.1em; line-height: 1.6;">{ai_justification}</p>
        <hr style="border: 0; border-top: 1px solid #BBF7D0; margin: 20px 0;">
        <div style="display: flex; gap: 40px;">
            <div>
                <span style="color: #64748B; font-weight: 600; text-transform: uppercase; font-size: 0.85em;">Estimated Cost</span><br>
                <span style="color: #0F172A; font-size: 1.4em; font-weight: 700;">${est_cost}</span>
            </div>
            <div>
                <span style="color: #64748B; font-weight: 600; text-transform: uppercase; font-size: 0.85em;">Projected ROI</span><br>
                <span style="color: #166534; font-size: 1.4em; font-weight: 700;">${roi:.2f}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("👤 Raw Customer Profile Data"):
        st.dataframe(user_data.to_frame().T)

else:
    if selected_id:
        st.warning("⚠️ Customer ID not found in current records.")
    else:
        st.info("👋 Welcome! Use the sidebar to select a customer and generate an AI retention strategy.")
