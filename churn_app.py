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
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Telecom AI Dashboard", layout="wide", page_icon="📡")

# ==========================================
# 1. DATA LOADING & CLEANING (Cached)
# ==========================================
@st.cache_data
def load_and_clean_data():
    try:
        df = pd.read_csv('telecom_churn.csv')
        # Cleaning
        df = df.fillna(method='ffill')
        
        # Outlier Treatment (Capping)
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            df[col] = df[col].clip(q1 - 1.5*iqr, q3 + 1.5*iqr)
            
        # Feature Engineering: Total Usage Intensity
        if all(c in df.columns for c in ['calls_made', 'sms_sent', 'data_used']):
             df['usage_intensity'] = (df['calls_made'] * 0.5) + (df['sms_sent'] * 0.1) + (df['data_used'] * 0.01)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# ==========================================
# 2. MODELING (Cached)
# ==========================================
@st.cache_data
def train_predictive_model(df):
    data = df.copy()
    le = LabelEncoder()
    # Encode categorical
    for col in data.select_dtypes(include=['object']):
        if 'date' not in col: data[col] = le.fit_transform(data[col].astype(str))
    
    # Features
    drop_cols = ['churn', 'customer_id', 'date_of_registration', 'usage_intensity']
    X = data.drop(columns=[c for c in drop_cols if c in data.columns], errors='ignore')
    y = data['churn']
    
    # Train
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler, X.columns

@st.cache_resource
def train_smart_clustering(df):
    # Features for behavioral segmentation
    cluster_features = ['age', 'estimated_salary', 'calls_made', 'sms_sent', 'data_used']
    cluster_features = [f for f in cluster_features if f in df.columns]
    
    X_c = df[cluster_features].copy()
    scaler_c = StandardScaler()
    X_scaled = scaler_c.fit_transform(X_c)
    
    # K-Means
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['cluster_id'] = kmeans.fit_predict(X_scaled)
    
    # Dynamic Cluster Naming based on medians
    cluster_stats = df.groupby('cluster_id')[cluster_features].median()
    global_medians = df[cluster_features].median()
    
    names = {}
    for i, row in cluster_stats.iterrows():
        if row['data_used'] > global_medians['data_used'] * 1.5:
            names[i] = "Heavy Data Consumer"
        elif row['estimated_salary'] > global_medians['estimated_salary'] * 1.5:
            names[i] = "Premium High-Value"
        elif row['calls_made'] > global_medians['calls_made'] * 1.5:
            names[i] = "Frequent Caller"
        elif row['estimated_salary'] < global_medians['estimated_salary'] * 0.7:
             names[i] = "Budget-Conscious"
        else:
            names[i] = "Standard Household"
            
    return kmeans, scaler_c, cluster_features, names

# ==========================================
# MAIN INITIALIZATION
# ==========================================
df_raw = load_and_clean_data()
if df_raw is not None:
    model_pred, scaler_pred, feature_list = train_predictive_model(df_raw)
    kmeans, scaler_cluster, c_features, seg_names = train_smart_clustering(df_raw)

# ==========================================
# APP UI
# ==========================================
st.title("� Telecom Intelligence & Retention Platform")

nav = st.sidebar.radio("Go To", ["Data Exploration", "Churn Prediction & AI Strategy"])

if nav == "Data Exploration":
    st.header("1. Descriptive Analytics")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Churn Overview")
        st.write(df_raw['churn'].value_counts(normalize=True).map(lambda x: f"{x:.1%}"))
        fig, ax = plt.subplots(figsize=(6,3))
        sns.countplot(x='churn', data=df_raw, palette='viridis')
        st.pyplot(fig)
    with c2:
        st.subheader("Top Features Correlated with Churn")
        num_df = df_raw.select_dtypes(include=[np.number])
        corr = num_df.corr()['churn'].sort_values(ascending=False).drop('churn').head(5)
        st.bar_chart(corr)

    st.subheader("Customer Segments (K-Means)")
    df_raw['segment'] = df_raw['cluster_id'].map(seg_names)
    fig2, ax2 = plt.subplots(figsize=(10,4))
    sns.countplot(x='segment', data=df_raw, palette='magma')
    plt.xticks(rotation=15)
    st.pyplot(fig2)

elif nav == "Churn Prediction & AI Strategy":
    st.header("2. Predictive & Prescriptive Analysis")
    
    cust_list = df_raw['customer_id'].unique()[:100]
    selected_id = st.selectbox("Select Customer ID to Analyze", cust_list)
    
    user_data = df_raw[df_raw['customer_id'] == selected_id].iloc[0]
    
    # --- PREDICTION ---
    # Prep for pred model
    temp_df = df_raw.copy()
    le = LabelEncoder()
    for col in temp_df.select_dtypes(include=['object']):
        if 'date' not in col: temp_df[col] = le.fit_transform(temp_df[col].astype(str))
    
    X_user = temp_df[temp_df['customer_id'] == selected_id].drop(columns=['churn', 'customer_id', 'date_of_registration', 'usage_intensity', 'cluster_id', 'segment'], errors='ignore')
    user_scaled = scaler_pred.transform(X_user)
    
    prob = model_pred.predict_proba(user_scaled)[0][1]
    is_churn = model_pred.predict(user_scaled)[0]
    
    st.write("---")
    res1, res2, res3 = st.columns(3)
    res1.metric("Predicted Churn Risk", f"{prob:.1%}")
    res2.metric("Customer Segment", user_data['segment'] if 'segment' in user_data else "Standard")
    res3.metric("Customer Status", "🔴 RISK" if is_churn == 1 else "🟢 STABLE")
    
    # --- RECOMMENDATION LOGIC ---
    st.subheader("💡 Strategic Recommendation")
    
    segment = user_data['segment'] if 'segment' in user_data else "Standard"
    usage = user_data['data_used']
    salary = user_data['estimated_salary']
    
    rec_title = ""
    reasoning = ""
    est_cost = 0
    
    if is_churn == 0:
        # Retention/Loyalty
        if "High-Value" in segment:
            rec_title = "Action: VIP Appreciation Offer"
            reasoning = "Customer is high-value and stable. Lock them in with an exclusive 12-month loyalty bonus to ensure long-term retention."
            est_cost = 50
        elif usage > df_raw['data_used'].median() * 1.5:
            rec_title = "Action: Priority Speed Upgrade"
            reasoning = "Happy heavy user. Offering a free speed priority trial improves satisfaction and prepares them for an upsell."
            est_cost = 5
        else:
            rec_title = "Action: Standard Engagement"
            reasoning = "Healthy customer relationship. Use personalized content to maintain brand top-of-mind."
            est_cost = 1
    else:
        # CHURN PREVENTION (High Risk)
        # 1. Check for "Overpaying" logic (High Salary + Low Usage)
        if salary > df_raw['estimated_salary'].median() * 1.2 and usage < df_raw['data_used'].median() * 0.5:
            rec_title = "Action: Plan Optimization (Downsell to Save)"
            reasoning = "Customer has high potential but low usage. They likely feel they are overpaying. Proactively suggest a 'Smart Saver' plan to prevent churn."
            est_cost = 20
        # 2. Check for "Usage Stress" logic (Heavy Data User)
        elif "Heavy Data" in segment:
            rec_title = "Action: 40% Data Discount (6 Months)"
            reasoning = "High data usage is their primary value. They are likely leaving for a cheaper data competitor. Neutralize the price threat."
            est_cost = 120
        # 3. Budget Conscious
        elif "Budget" in segment:
             rec_title = "Action: Cashback / Bill Credit"
             reasoning = "Price is the main driver. A one-time bill credit of $25 can bridge the gap for another quarter."
             est_cost = 25
        else:
            rec_title = "Action: Personalized Win-Back Call"
            reasoning = "Risk detected without clear usage driver. A human touch/direct feedback call with a flexible 1-month-free credit is recommended."
            est_cost = 40

    st.info(f"### {rec_title}")
    with st.expander("🔍 Strategic Justification", expanded=True):
        st.write(f"**Segment Behavior**: {segment}")
        st.write(f"**AI Reasoning**: {reasoning}")
        st.markdown(f"**Estimated Cost**: `${est_cost}`")
    
    # Financial Projection
    st.subheader("💰 Financial Impact")
    base_val = 600 # Assume $50/mo
    if "High-Value" in segment: base_val = 1200
    
    saved_val = base_val * 0.45 # Assumption of 45% effectiveness
    net_roi = saved_val - est_cost
    
    st.metric("Expected Net ROI (Annual)", f"${net_roi:.2f}")
