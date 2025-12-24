import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="AI Strategic Command",
    layout="wide",
    page_icon="🛰️"
)

st.markdown("""
<style>
:root {
    --bg-main: #0B0E14;
    --bg-card: #121826;
    --bg-sidebar: #0F1320;
    --border-soft: #1F2937;
    --text-main: #E5E7EB;
    --text-muted: #9CA3AF;
    --accent: #F59E0B;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-main) !important;
    color: var(--text-main);
    font-family: Inter, system-ui, sans-serif;
}

section[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border-soft);
}

h1, h2, h3 {
    color: var(--accent) !important;
    font-weight: 800;
    letter-spacing: 1px;
}

p, label, span {
    color: var(--text-main);
}

[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border-soft);
    border-radius: 14px;
    padding: 22px;
}

[data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-weight: 800;
    font-size: 1.8rem;
}

[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
}

.strategy-box {
    background: linear-gradient(160deg, #121826, #0F172A);
    border: 1px solid var(--border-soft);
    border-left: 6px solid var(--accent);
    border-radius: 14px;
    padding: 36px;
    margin-top: 30px;
}

.strategy-box h2 {
    margin-top: 0;
}

input, textarea, select, div[data-baseweb="select"] {
    background-color: #0F172A !important;
    color: white !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 8px;
}

.stDataFrame {
    background-color: var(--bg-card);
}

.stExpander {
    background-color: var(--bg-card);
    border: 1px solid var(--border-soft);
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("telecom_churn.csv")
    df = df.fillna(df.median(numeric_only=True))
    return df

@st.cache_resource
def build_ai(df):
    data = df.copy()
    le = LabelEncoder()
    for c in data.select_dtypes(include="object"):
        data[c] = le.fit_transform(data[c].astype(str))

    X = data.drop(columns=["churn", "customer_id"], errors="ignore")
    y = data["churn"]

    scaler = StandardScaler().fit(X)
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        random_state=42
    ).fit(scaler.transform(X), y)

    cluster_cols = ["age", "estimated_salary", "data_used", "calls_made"]
    Xc = df[cluster_cols]
    scaler_c = StandardScaler().fit(Xc)
    kmeans = KMeans(n_clusters=4, n_init=10, random_state=42).fit(scaler_c.transform(Xc))

    centers = scaler_c.inverse_transform(kmeans.cluster_centers_)
    med = Xc.median()

    names = {}
    for i, r in enumerate(centers):
        if r[2] > med["data_used"] * 1.5:
            names[i] = "Data Power User"
        elif r[1] > med["estimated_salary"] * 1.5:
            names[i] = "Elite Premium"
        elif r[3] > med["calls_made"] * 1.3:
            names[i] = "Voice Intensive"
        else:
            names[i] = "Balanced Core"

    return model, scaler, kmeans, scaler_c, cluster_cols, names

df = load_data()
model, scaler, km, scaler_c, ccols, cmap = build_ai(df)
df["Cluster"] = km.labels_
df["Segment"] = df["Cluster"].map(cmap)

st.title("🛰️ AI Strategic Command Center")
st.write("---")

with st.sidebar:
    st.header("🎯 Customer Targeting")
    mode = st.radio("Lookup Mode", ["Select", "Manual"])
    if mode == "Select":
        cid = st.selectbox("Customer ID", df["customer_id"].unique()[:500])
    else:
        cid = st.text_input("Enter ID")
        cid = int(cid) if cid.isdigit() else None

    st.success("SYSTEM ONLINE")
    st.caption("Version 7.1 · Visual Optimized")

if cid in df["customer_id"].values:
    user = df[df["customer_id"] == cid].iloc[0]

    X_u = df[df["customer_id"] == cid].drop(
        columns=["churn", "customer_id", "Cluster", "Segment"],
        errors="ignore"
    )
    for c in X_u.select_dtypes(include="object"):
        X_u[c] = 0

    risk = model.predict_proba(scaler.transform(X_u))[0][1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Churn Risk", f"{risk:.1%}")
    c2.metric("Segment", user["Segment"])
    c3.metric("Status", "⚠️ HIGH RISK" if risk > 0.5 else "✅ STABLE")
    c4.metric("Salary", f"${user['estimated_salary']:,.0f}")

    if risk < 0.3:
        strategy = "AI Engagement Flow"
        cost = 5
        reason = "Low risk detected. Maintain loyalty with automation."
    elif "Elite" in user["Segment"]:
        strategy = "VIP Retention Protocol"
        cost = 90
        reason = "High value elite customer at risk."
    elif "Data" in user["Segment"]:
        strategy = "Unlimited Data Boost"
        cost = 120
        reason = "Heavy data dependency identified."
    else:
        strategy = "Human Retention Call"
        cost = 60
        reason = "Complex churn indicators detected."

    clv = 1600 if "Elite" in user["Segment"] else 800
    roi = (clv * (1 - risk)) - cost

    st.markdown(f"""
    <div class="strategy-box">
        <h2>🎯 Strategy: {strategy}</h2>
        <p style="font-size:1.15rem;line-height:1.7">{reason}</p>
        <hr style="border:0;border-top:1px solid #1F2937;margin:25px 0">
        <b>Cost:</b> ${cost}<br>
        <b>Projected ROI:</b> ${roi:,.2f}
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Raw Customer Data"):
        st.dataframe(user.to_frame().T)

else:
    st.info("Select a customer to activate strategic mode.")

    pca = PCA(n_components=2)
    samp = df.sample(800) if len(df) > 800 else df
    Xp = pca.fit_transform(scaler_c.transform(samp[ccols]))
    viz = pd.DataFrame(Xp, columns=["PC1", "PC2"])
    viz["Segment"] = samp["Segment"].values

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=viz, x="PC1", y="PC2", hue="Segment", alpha=0.75, ax=ax)
    ax.set_facecolor("#0B0E14")
    fig.patch.set_facecolor("#0B0E14")
    ax.tick_params(colors="white")
    ax.set_title("Behavioral Cluster Map", color="#F59E0B")
    st.pyplot(fig)
