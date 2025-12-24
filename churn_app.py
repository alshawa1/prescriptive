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
st.set_page_config(page_title="Telecom Churn Dashboard", layout="wide")

# ==========================================
# 1. LOAD & PREPROCESS (Cached)
# ==========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('telecom_churn.csv')
        # Basic Preprocessing
        df = df.fillna(method='ffill')
        # Outlier Capping
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df[col] = np.where(df[col] < lower, lower, df[col])
            df[col] = np.where(df[col] > upper, upper, df[col])
        return df
    except FileNotFoundError:
        return None

# ==========================================
# 2. MODELS & LOGIC (Cached/Resource)
# ==========================================
@st.cache_data
def prepare_model_data(df):
    data = df.copy()
    le = LabelEncoder()
    # Feature Engineering
    if 'calls_made' in data.columns and 'sms_sent' in data.columns:
        data['total_interactions'] = data['calls_made'] + data['sms_sent']
    
    cat_cols = data.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if 'date' not in col:
            data[col] = le.fit_transform(data[col].astype(str))
    
    # Drop non-features
    X = data.drop(['churn', 'customer_id', 'date_of_registration'], axis=1, errors='ignore')
    y = data['churn']
    
    # Train Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X.columns

@st.cache_resource
def train_clustering_model(df):
    # Select features for clustering
    features = ['estimated_salary']
    if 'data_used' in df.columns: features.append('data_used')
    if 'calls_made' in df.columns: features.append('calls_made')
    
    X_cluster = df[features].copy()
    scaler_c = StandardScaler()
    X_cluster_scaled = scaler_c.fit_transform(X_cluster)
    
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X_cluster_scaled)
    
    df_temp = X_cluster.copy()
    df_temp['Cluster'] = clusters
    cluster_means = df_temp.groupby('Cluster').mean()
    
    cluster_labels = {}
    for c_id, row in cluster_means.iterrows():
        label = "Standard User"
        if row.get('data_used', 0) > df['data_used'].mean() * 1.5:
            label = "Heavy Data User"
        elif row.get('estimated_salary', 0) > df['estimated_salary'].mean() * 1.5:
            label = "High Net-Worth Individual"
        elif row.get('data_used', 0) < df['data_used'].mean() * 0.5:
            label = "Low Engagement / Budget"
        cluster_labels[c_id] = label
        
    return kmeans, cluster_labels, features, scaler_c

# ==========================================
# MAIN APP EXECUTION
# ==========================================
df = load_data()

if df is None:
    st.error("File 'telecom_churn.csv' not found!")
    st.stop()

# Initialize models
model, scaler, feature_names = prepare_model_data(df)
kmeans_model, cluster_names, cluster_features, scaler_cluster = train_clustering_model(df)

st.title("📊 Telecom Churn Analytics & Recommendation System")

sidebar_opt = st.sidebar.selectbox("Navigation", ["Descriptive Analytics", "Predict & Recommend"])

if sidebar_opt == "Descriptive Analytics":
    st.header("Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(x='churn', data=df, ax=ax)
        st.pyplot(fig)
    with col2:
        st.subheader("Key Statistics")
        st.dataframe(df.describe())
    st.subheader("Variable Distributions")
    num_col = st.selectbox("Select Column to Visualize", df.select_dtypes(include=[np.number]).columns)
    fig2, ax2 = plt.subplots(figsize=(8,3))
    sns.histplot(df[num_col], kde=True, ax=ax2)
    st.pyplot(fig2)

elif sidebar_opt == "Predict & Recommend":
    st.header("🔮 Prediction & Recommendations")
    
    # 1. Selection
    st.subheader("Select Customer for Analysis")
    cust_id = st.selectbox("Customer ID", df['customer_id'].head(100).unique())
    customer_row = df[df['customer_id'] == cust_id].iloc[0]
    st.write("### Customer Details")
    st.write(customer_row)
    
    # 2. Prediction Logic
    # We need to process the data exactly like the training set
    data_encoded = df.copy()
    if 'calls_made' in data_encoded.columns and 'sms_sent' in data_encoded.columns:
        data_encoded['total_interactions'] = data_encoded['calls_made'] + data_encoded['sms_sent']
    
    le = LabelEncoder()
    for col in data_encoded.select_dtypes(include=['object']):
        if 'date' not in col: data_encoded[col] = le.fit_transform(data_encoded[col].astype(str))
    
    X_full = data_encoded.drop(['churn', 'customer_id', 'date_of_registration'], axis=1, errors='ignore')
    user_features = X_full.loc[df['customer_id'] == cust_id]
    user_scaled = scaler.transform(user_features)
    
    pred_prob = model.predict_proba(user_scaled)[0][1]
    pred_class = model.predict(user_scaled)[0]
    
    st.write("---")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.metric("Churn Probability", f"{pred_prob:.2%}")
    with col_p2:
        status = "🔴 High Risk (Churn)" if pred_class == 1 else "🟢 Low Risk (Retained)"
        st.metric("Prediction Status", status)
        
    # 3. Smart Recommendations
    st.write("---")
    st.subheader("💡 Smart AI Recommendations")
    
    # Identify Cluster
    user_cluster_data = customer_row[cluster_features].to_frame().T
    user_c_scaled = scaler_cluster.transform(user_cluster_data)
    cluster_id = kmeans_model.predict(user_c_scaled)[0]
    segment_name = cluster_names.get(cluster_id, "Standard Segment")
    
    st.info(f"Customer Segment Identified: **{segment_name}**")
    
    rec_title = ""
    rec_desc = ""
    est_cost = 0
    
    if pred_class == 0:
        if "Heavy Data" in segment_name:
            rec_title = "Upsell: Unlimited 5G Add-on"
            rec_desc = "User loves data and is loyal. Upsell premium speed to increase ARPU."
            est_cost = 10
        elif "High Net-Worth" in segment_name:
            rec_title = "Relationship: Exclusive Event Invite"
            rec_desc = "Maintain loyalty with exclusive perks for high-value clients."
            est_cost = 150
        else:
            rec_title = "Maintain: 'Thank You' Message"
            rec_desc = "Standard engagement message to maintain brand presence."
            est_cost = 1
    else:
        if "Heavy Data" in segment_name:
            rec_title = "Retention: 50% Off Data Plan for 6 Months"
            rec_desc = "Critical Data User at risk. Data discounts are the best hook for them."
            est_cost = 120
        elif "High Net-Worth" in segment_name:
            rec_title = "Retention: VIP Concierge & Free Device Upgrade"
            rec_desc = "High Value Customer. A full device subsidy is justified to prevent loss."
            est_cost = 500
        elif "Low Engagement" in segment_name:
            rec_title = "Win-Back: 'We Miss You' Free Recharge"
            rec_desc = "Price sensitive, low usage user. A small gift can restart engagement."
            est_cost = 20
        else:
            rec_title = "Retention: 1 Month Free Service"
            rec_desc = "Generic retention offer to show customer appreciation."
            est_cost = 30
            
    st.markdown(f"**Recommended Action**: ### {rec_title}")
    with st.expander("ℹ️ Why this strategy?", expanded=True):
        st.write(f"**Customer Segment**: {segment_name}")
        st.write(f"**Reasoning**: {rec_desc}")
    
    # 4. ROI
    st.subheader("Financial Impact")
    customer_val = 600
    if "High Net-Worth" in segment_name: customer_val = 1200
    if "Low Engagement" in segment_name: customer_val = 300
    
    gain = customer_val * 0.40 # 40% retention boost assumption
    roi = gain - est_cost
    st.metric("Expected Net ROI", f"${roi:.2f}")
