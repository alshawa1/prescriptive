import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Set page config
st.set_page_config(page_title="Telecom Churn Dashboard", layout="wide")

# ==========================================
# 1. LOAD & PREPROCESS
# ==========================================
@st.cache_data
def load_data():
    # Attempt to load the dataset
    try:
        df = pd.read_csv('telecom_churn.csv')
        
        # Basic Preprocessing
        df = df.fillna(method='ffill')
        
        # Outlier Capping (Simplified for App)
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

df = load_data()

if df is None:
    st.error("File 'telecom_churn.csv' not found in the directory!")
    st.stop()

# Feature Engineering
if 'calls_made' in df.columns and 'sms_sent' in df.columns:
    df['total_interactions'] = df['calls_made'] + df['sms_sent']

# Encoding for Model
@st.cache_data
def prepare_model_data(df):
    data = df.copy()
    le = LabelEncoder()
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

model, scaler, feature_names = prepare_model_data(df)

# ==========================================
# 2. DASHBOARD UI
# ==========================================
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
    
    # User Selection
    st.subheader("Select Customer for Analysis")
    cust_id = st.selectbox("Customer ID", df['customer_id'].head(100).unique())
    
    customer_row = df[df['customer_id'] == cust_id].iloc[0]
    st.write("### Customer Details")
    st.write(customer_row)
    
    # Predict
    # Prepare single row
    # We need to replicate the exact encoding/feature steps for this single row.
    # For simplicity in this demo app, we grab the *Encoded* data from our training set corresponding to this index
    # (In a real app, we would transform the raw inputs of this specific customer)
    
    # Find the row in the processed X
    # Re-running encoding logic just for index matching (simplified)
    data_encoded = df.copy()
    le = LabelEncoder()
    for col in data_encoded.select_dtypes(include=['object']):
        if 'date' not in col: data_encoded[col] = le.fit_transform(data_encoded[col].astype(str))
    
    X_full = data_encoded.drop(['churn', 'customer_id', 'date_of_registration'], axis=1, errors='ignore')
    customer_features = X_full.loc[df['customer_id'] == cust_id]
    
    # Scale
    customer_features_scaled = scaler.transform(customer_features)
    
    # Prediction
    pred_prob = model.predict_proba(customer_features_scaled)[0][1]
    pred_class = model.predict(customer_features_scaled)[0]
    
    st.write("---")
    col_pred1, col_pred2 = st.columns(2)
    with col_pred1:
        st.metric("Churn Probability", f"{pred_prob:.2%}")
    with col_pred2:
        status = "🔴 High Risk (Churn)" if pred_class == 1 else "🟢 Low Risk (Retained)"
        st.metric("Prediction Status", status)
        
from sklearn.cluster import KMeans

# ==========================================
# CLUSTERING MODEL (RecSys)
# ==========================================
@st.cache_resource
def train_clustering_model(df, scaler):
    # Select features for clustering (Usage behavior)
    # We use original numerical columns for clustering interpretation
    features = ['estimated_salary']
    if 'data_used' in df.columns: features.append('data_used')
    if 'calls_made' in df.columns: features.append('calls_made')
    
    X_cluster = df[features].copy()
    X_cluster_scaled = scaler.fit_transform(X_cluster)
    
    # 4 Clusters: e.g., Low Value, High Value, Heavy Users, etc.
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X_cluster_scaled)
    
    # Analyze Cluster Characteristics to assign labels dynamically
    df_temp = X_cluster.copy()
    df_temp['Cluster'] = clusters
    cluster_means = df_temp.groupby('Cluster').mean()
    
    # Simple logic to name clusters based on Data Usage & Salary
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
        
    return kmeans, cluster_labels, features

kmeans_model, cluster_names, cluster_features = train_clustering_model(df, StandardScaler())

# ... [Inside Predict & Recommend Section] ...

    # RECOMMENDATION SYSTEM
    st.write("---")
    st.subheader("💡 Smart AI Recommendations")
    
    # 1. Identify Cluster
    # Get user features for clustering
    user_cluster_data = customer_row[cluster_features].to_frame().T
    # Scale (using a fresh scaler fit on full data for consistency in this simplified snippet)
    # Note: In prod, use the same scaler object from training.
    scaler_cluster = StandardScaler().fit(df[cluster_features]) 
    user_scaled = scaler_cluster.transform(user_cluster_data)
    
    cluster_id = kmeans_model.predict(user_scaled)[0]
    segment_name = cluster_names.get(cluster_id, "Standard Segment")
    
    st.info(f"Customer Segment Identified: **{segment_name}**")
    
    # 2. Generate Recommendation based on Segment & Churn Risk
    rec_title = ""
    rec_desc = ""
    est_cost = 0
    
    if pred_class == 0:
        # Low Risk
        if "Heavy Data" in segment_name:
            rec_title = "Upsell: Unlimited 5G Add-on"
            rec_desc = "User loves data and is loyal. Upsell premium speed."
            est_cost = 10
        elif "High Net-Worth" in segment_name:
            rec_title = "Relationship: Exclusive Event Invite"
            rec_desc = "Maintain loyalty with exclusive perks."
            est_cost = 150
        else:
            rec_title = "Maintain: 'Thank You' Message"
            rec_desc = "Standard engagement to keep satisfaction high."
            est_cost = 1
    else:
        # High Risk (Churn)
        if "Heavy Data" in segment_name:
            rec_title = "Retention: 50% Off Data Plan for 6 Months"
            rec_desc = "Critical Data User at risk. Aggressive discount needed."
            est_cost = 120
        elif "High Net-Worth" in segment_name:
            rec_title = "Retention: VIP Concierge & Free Device Upgrade"
            rec_desc = "High Value Customer. Device subsidy is justified to retain."
            est_cost = 500
        elif "Low Engagement" in segment_name:
            rec_title = "Win-Back: 'We Miss You' Free Recharge"
            rec_desc = "Low value, price sensitive. Small freebie to re-engage."
            est_cost = 20
        else:
            rec_title = "Retention: 1 Month Free Service"
            rec_desc = "Standard churn prevention offer."
            est_cost = 30 # Average monthly ARPU
            
    st.markdown(f"**Recommended Action**: ### {rec_title}")
    
    with st.expander("ℹ️ Why this strategy?", expanded=True):
        st.write(f"**Customer Segment**: {segment_name}")
        st.write(f"**Reasoning**: {rec_desc}")
        st.info("This strategy is designed to maximize retention for this specific profile while optimizing cost.")
    
    # 3. ROI Calculator
    st.write(f"**Estimated Implementation Cost**: ${est_cost}")
    
    # ROI Logic
    customer_val = 600 # Assume $50/mo
    if "High Net-Worth" in segment_name: customer_val = 1200 # Higher value
    if "Low Engagement" in segment_name: customer_val = 300 # Lower value
    
    retention_prob_gain = 0.40 # Strategy effectiveness
    saved_revenue = customer_val * retention_prob_gain
    
    roi = saved_revenue - est_cost
    
    st.metric("Projected Net ROI", f"${roi:.2f}", delta_color="normal")

