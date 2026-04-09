import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>

/* ===== GALAXY BACKGROUND ===== */
.stApp {
    background-image:url("purple background.jpg")
    color: white;
    overflow: hidden;
}

/* Moving Stars */
.stApp::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 200%;
    height: 200%;
    background-image: radial-gradient(white 1px, transparent 1px);
    background-size: 50px 50px;
    animation: starsMove 100s linear infinite;
    opacity: 0.15;
}

/* Star Animation */
@keyframes starsMove {
    from {transform: translate(0, 0);}
    to {transform: translate(-500px, -500px);}
}

/* Sidebar Galaxy */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0f2027,#203a43,#2c5364);
}

/* Glass Metric Cards */
.metric-card {
    background: rgba(255,255,255,0.08);
    padding:20px;
    border-radius:20px;
    backdrop-filter: blur(12px);
    box-shadow: 0 0 30px rgba(0,255,255,0.4);
    text-align:center;
    transition: 0.4s;
}
.metric-card:hover {
    transform: scale(1.08);
    box-shadow: 0 0 50px rgba(255,0,255,0.7);
}

/* Neon Galaxy Button */
.stButton > button {
    background: linear-gradient(90deg,#00ffff,#ff00ff,#00ffff);
    background-size: 200% auto;
    color:white;
    border:none;
    border-radius:12px;
    padding:10px 25px;
    font-size:18px;
    font-weight:bold;
    box-shadow:0 0 25px #00ffff;
    animation: glowMove 3s linear infinite;
    transition: 0.3s;
}

@keyframes glowMove {
    0% {background-position: 0% center;}
    100% {background-position: 200% center;}
}

.stButton > button:hover {
    transform: scale(1.1);
    box-shadow:0 0 60px #ff00ff;
}

</style>
""", unsafe_allow_html=True)

st.title("🌌Customer Segmentation Dashboard")

# ---------------- SIDEBAR ----------------
st.sidebar.header("🛠 Manual Prediction")

age = st.sidebar.slider("Age", 18, 70, 35)
income = st.sidebar.slider("Income", 10000, 100000, 50000)
spending = st.sidebar.slider("Total Spending", 100, 5000, 1000)
web = st.sidebar.slider("Web Purchases", 0, 20, 10)
store = st.sidebar.slider("Store Purchases", 0, 20, 10)
visits = st.sidebar.slider("Web Visits / Month", 0, 30, 10)
recency = st.sidebar.slider("Recency (Days)", 0, 100, 30)

# ---------------- SAMPLE DATA ----------------
@st.cache_data
def load_data():
    np.random.seed(42)
    data = pd.DataFrame({
        "Age": np.random.randint(18, 70, 200),
        "Income": np.random.randint(10000, 100000, 200),
        "Total_Spending": np.random.randint(100, 5000, 200),
        "NumWebPurchases": np.random.randint(0, 20, 200),
        "NumStorePurchases": np.random.randint(0, 20, 200),
        "WebVisitsMonth": np.random.randint(0, 30, 200),
        "Recency": np.random.randint(0, 100, 200)
    })
    return data

df = load_data()

# ---------------- CLUSTERING ----------------
scaler = StandardScaler()
scaled = scaler.fit_transform(df)

kmeans = KMeans(n_clusters=4, random_state=42)
df["Cluster"] = kmeans.fit_predict(scaled)

sil_score = silhouette_score(scaled, df["Cluster"])

# ---------------- METRICS ----------------
col1, col2, col3, col4 = st.columns(4)

col1.markdown(f"<div class='metric-card'><h3>👥 Total Customers</h3><h2>{len(df)}</h2></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='metric-card'><h3>💰 Avg Income</h3><h2>{int(df['Income'].mean())}</h2></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='metric-card'><h3>🛍 Avg Spending</h3><h2>{int(df['Total_Spending'].mean())}</h2></div>", unsafe_allow_html=True)
col4.markdown(f"<div class='metric-card'><h3>📊 Silhouette Score</h3><h2>{round(sil_score,2)}</h2></div>", unsafe_allow_html=True)

st.write("")

# ---------------- 2D PCA VISUALIZATION ----------------
st.subheader("✨ 2D Cluster Visualization")

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled)

fig = go.Figure()
colors = ["#00ffff", "#ff00ff", "#39ff14", "#ff073a"]

for i in range(4):
    fig.add_trace(go.Scatter(
        x=pca_data[df["Cluster"]==i,0],
        y=pca_data[df["Cluster"]==i,1],
        mode="markers",
        marker=dict(size=10, color=colors[i]),
        name=f"Cluster {i}"
    ))

fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font_color="white"
)

st.plotly_chart(fig, use_container_width="stretch")

# ---------------- 3D VISUALIZATION ----------------
st.subheader("🌠 3D Customer Segmentation")

pca3 = PCA(n_components=3)
pca3_data = pca3.fit_transform(scaled)

fig3d = px.scatter_3d(
    x=pca3_data[:,0],
    y=pca3_data[:,1],
    z=pca3_data[:,2],
    color=df["Cluster"].astype(str),
)
fig3d.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white")

st.plotly_chart(fig3d, use_container_width="stretch")

# ---------------- ELBOW METHOD ----------------
st.subheader("📉 Elbow Method")

inertia = []
for k in range(1,10):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(scaled)
    inertia.append(km.inertia_)

fig_elbow = px.line(x=range(1,10), y=inertia)
fig_elbow.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white")

st.plotly_chart(fig_elbow, use_container_width="stretch")

# ---------------- DONUT CHART ----------------
st.subheader("🍩 Cluster Distribution")

cluster_counts = df["Cluster"].value_counts().reset_index()
cluster_counts.columns = ["Cluster","Count"]

fig_donut = px.pie(cluster_counts, names="Cluster", values="Count", hole=0.5)
fig_donut.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white")

st.plotly_chart(fig_donut, use_container_width="stretch")

# ---------------- PREDICTION ----------------
cluster = None

if st.sidebar.button("🚀 Predict Segment"):

    progress = st.sidebar.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

    user = scaler.transform([[age,income,spending,web,store,visits,recency]])
    cluster = kmeans.predict(user)[0]

    st.sidebar.success(f"✨ Predicted Cluster: {cluster}")

    st.subheader("💡 Expect Customer Expectation & Business Suggestion")

    if cluster == 0:
        st.info("🎯 Budget Customers → Provide Discounts & Combo Offers")
    elif cluster == 1:
        st.info("💎 Premium Customers → Provide Loyalty Rewards & VIP Access")
    elif cluster == 2:
        st.info("📈 High Potential → Retarget with Email & App Notifications")
    else:
        st.info("🛒 Regular Shoppers → Cross-sell & Personalized Recommendations")

    st.markdown("### 📊 Customer Profile Summary")
    st.markdown(f"""
    - 👤 Age: {age}  
    - 💰 Income: {income}  
    - 🛍 Spending: {spending}  
    - 🌐 Web Purchases: {web}  
    - 🏬 Store Purchases: {store}  
    - 📅 Recency: {recency} days  
    """)

    with st.spinner("Analyzing Customer Behavior..."):
        time.sleep(2)

st.balloons()
st.metric("Total Clusters Created", 4)
st.success("✨ Customer Segmentation Completed!")

# ---------------- DOWNLOAD REPORT ----------------
if cluster is not None:
    report = f"""
Customer Segmentation Report

Predicted Cluster: {cluster}

Customer Details:
Age: {age}
Income: {income}
Spending: {spending}
Web Purchases: {web}
Store Purchases: {store}
Recency: {recency}

Business Strategy Applied Based on Cluster.
"""
    st.download_button("📄 Download AI Summary Report",
                       report,
                       file_name="customer_report.txt")

# ---------------- BULK CSV ----------------
st.subheader("📂 Bulk Prediction Using CSV")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    bulk = pd.read_csv(uploaded)
    bulk = bulk.reindex(columns=scaler.feature_names_in_, fill_value=0)
    bulk_scaled = scaler.transform(bulk)
    bulk["Cluster"] = kmeans.predict(bulk_scaled)
    st.dataframe(bulk.head())
    st.download_button("⬇ Download Result", bulk.to_csv(index=False), "result.csv")

# ---------------- BUSINESS RECOMMENDATION ----------------
st.subheader("🤖 Company Suggestion")

st.markdown("""
- 🎯 Target specific customer groups  
- 💰 Increase revenue using premium offers  
- 📢 Improve digital marketing strategy  
- 📊 Make smart business decisions  
- 🧠 Predict churn customers early  
- 🔁 Automate campaign personalization  
""")