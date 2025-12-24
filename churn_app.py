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

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

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
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Create a temp DF to calculate medians
    df_temp = X_c.copy()
    df_temp['cluster_id'] = cluster_labels
    cluster_stats = df_temp.groupby('cluster_id').median()
    global_medians = X_c.median()
    
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
            
    return kmeans, scaler_c, cluster_features, names, cluster_labels

# ==========================================
# MAIN INITIALIZATION
# ==========================================
df_raw = load_and_clean_data()
if df_raw is not None:
    model_pred, scaler_pred, feature_list = train_predictive_model(df_raw)
    kmeans, scaler_cluster, c_features, seg_names, cluster_ids = train_smart_clustering(df_raw)
    df_raw['cluster_id'] = cluster_ids
    df_raw['segment'] = df_raw['cluster_id'].map(seg_names)

# ==========================================
# APP UI
# ==========================================
st.title("📡 Telecom Intelligence & Retention Platform")

nav = st.sidebar.radio("Go To", ["Data Exploration", "Churn Prediction & AI Strategy"])

if nav == "Data Exploration":
    st.header("1. Descriptive Analytics")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Churn Overview")
        st.write(df_raw['churn'].value_counts(normalize=True).map(lambda x: f"{x:.1%}"))
        fig, ax = plt.subplots(figsize=(8,5), dpi=100)
        sns.countplot(x='churn', data=df_raw, palette='viridis', ax=ax)
        ax.set_title("Distribution of Churn (Target Variable)", fontsize=12)
        ax.set_xlabel("Churned? (0=No, 1=Yes)", fontsize=10)
        ax.set_ylabel("Number of Customers", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
    with c2:
        st.subheader("Top Drivers of Churn")
        num_df = df_raw.select_dtypes(include=[np.number])
        corr = num_df.corr()['churn'].sort_values(ascending=False).drop('churn').head(7)
        st.info("Features with strongest correlation to leaving")
        st.bar_chart(corr)

    st.subheader("Customer Behavioral Segments")
    fig2, ax2 = plt.subplots(figsize=(10,5), dpi=100)
    sns.countplot(y='segment', data=df_raw, palette='magma', ax=ax2)
    ax2.set_title("Distribution of AI-Generated Customer Segments", fontsize=12)
    ax2.set_xlabel("Count", fontsize=10)
    ax2.set_ylabel("Customer Segment", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig2)

elif nav == "Churn Prediction & AI Strategy":
    st.header("2. Predictive & Prescriptive Analysis")
    
    st.subheader("Enter Customer Code / Select ID")
    search_type = st.radio("Search By:", ["Select List", "Manual Entry"])
    
    if search_type == "Select List":
        selected_id = st.selectbox("Select Customer ID", df_raw['customer_id'].unique()[:500])
    else:
        selected_id_input = st.text_input("Type Customer ID Code (e.g. 1, 10, 50)")
        try:
            selected_id = int(selected_id_input) if selected_id_input else None
        except ValueError:
            st.error("Please enter a valid numeric ID")
            selected_id = None

    if selected_id in df_raw['customer_id'].values:
        user_data = df_raw[df_raw['customer_id'] == selected_id].iloc[0]
        
        st.write("### 👤 Customer Profile")
        st.dataframe(user_data.to_frame().T)
        
        # --- PREDICTION ---
        temp_df = df_raw.copy()
        le = LabelEncoder()
        for col in temp_df.select_dtypes(include=['object']):
            if 'date' not in col: temp_df[col] = le.fit_transform(temp_df[col].astype(str))
        
        # Consistent drop set
        drop_for_ml = ['churn', 'customer_id', 'date_of_registration', 'usage_intensity', 'cluster_id', 'segment']
        X_user = temp_df[temp_df['customer_id'] == selected_id].drop(columns=[c for c in drop_for_ml if c in temp_df.columns], errors='ignore')
        user_scaled = scaler_pred.transform(X_user)
        
        # Robust Prediction
        probs = model_pred.predict_proba(user_scaled)[0]
        prob = probs[1] if len(probs) > 1 else (1.0 if model_pred.classes_[0] == 1 else 0.0)
        is_churn = model_pred.predict(user_scaled)[0]
        
        st.write("---")
        res1, res2, res3 = st.columns(3)
        res1.metric("Predicted Churn Risk", f"{prob:.1%}")
        res2.metric("Customer Segment", user_data['segment'], help="Segment identified by K-Means based on behavior")
        res3.metric("Customer Status", "🔴 RISK" if is_churn == 1 else "🟢 STABLE")
        
        # --- RECOMMENDATION LOGIC ---
        st.subheader("💡 Strategic Recommendation (Prescriptive)")
        
        segment = user_data['segment']
        usage = user_data['data_used']
        salary = user_data['estimated_salary']
        
        rec_title = ""
        reasoning = ""
        est_cost = 0
        
        if is_churn == 0:
            if "High-Value" in segment:
                rec_title = "Action: VIP Appreciation Offer"
                reasoning = "Customer is high-value and stable. Lock them in with an exclusive 12-month loyalty bonus."
                est_cost = 50
            elif usage > df_raw['data_used'].median() * 1.5:
                rec_title = "Action: Priority Speed Upgrade"
                reasoning = "Satisfied heavy user. Offering a speed trial improves retention and sets up upsell."
                est_cost = 5
            else:
                rec_title = "Action: Standard Engagement"
                reasoning = "Healthy relationship. Maintain brand presence with personalized weekly tips."
                est_cost = 1
        else:
            if salary > df_raw['estimated_salary'].median() * 1.2 and usage < df_raw['data_used'].median() * 0.5:
                rec_title = "Action: Plan Optimization (Downgrade Save)"
                reasoning = "Customer is high potential but low usage. Likely overpaying. Suggesting a cheaper plan saves the client."
                est_cost = 20
            elif "Heavy Data" in segment:
                rec_title = "Action: 40% Data Discount (6 Months)"
                reasoning = "They need data but price is a threat. Neutralize competitors with aggressive discounting."
                est_cost = 120
            elif "Budget" in segment:
                 rec_title = "Action: $25 Instant Bill Credit"
                 reasoning = "Price sensitive user. A bridge credit buys time and stops immediate churn."
                 est_cost = 25
            else:
                rec_title = "Action: Win-Back Feedback Call"
                reasoning = "Risk without clear behavior driver. Human reach-out with flexible credit is best."
                est_cost = 40

        st.success(f"### {rec_title}")
        with st.expander("🔍 Strategic Justification & Prescriptive Details", expanded=True):
            st.write(f"**Customer Segment**: {segment}")
            st.write(f"**AI Decision Reasoning**: {reasoning}")
            st.write(f"**Monthly Usage Context**: {usage:.1f} MB | Salary Context: ${salary:,.0f}")
        
        # Financial Impact
        st.subheader("💰 Annual ROI Projection")
        base_val = 600
        if "High-Value" in segment: base_val = 1200
        saved_val = base_val * 0.45 
        net_roi = saved_val - est_cost
        st.metric("Estimated Net ROI", f"${net_roi:.2f}")
    
    elif selected_id is not None:
        st.warning("Customer ID not found in the dataset range.")
