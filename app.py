"""
app.py
------
Streamlit application for Customer Segmentation.
Upload a CSV, get cluster assignments, visualize segments, download results.
"""

import io
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA

# ── Segment metadata ──────────────────────────────────────────────────────────
SEGMENT_DESCRIPTIONS = {
    "VIP Customers": {
        "description": "Highest lifetime value; premium spenders with frequent, recent purchases.",
        "strategy": "Exclusive loyalty rewards, early-access offers, dedicated account manager.",
        "icon": "👑",
    },
    "Loyal Customers": {
        "description": "Regular buyers with strong engagement and consistent spending.",
        "strategy": "Loyalty programs, personalised recommendations, birthday discounts.",
        "icon": "⭐",
    },
    "Potential Customers": {
        "description": "Moderate spenders showing growth signals; not yet fully committed.",
        "strategy": "Targeted upsell campaigns, product education, limited-time offers.",
        "icon": "🚀",
    },
    "New Customers": {
        "description": "Recently acquired; low tenure and purchase history.",
        "strategy": "Welcome series, onboarding guides, first-purchase incentives.",
        "icon": "🌱",
    },
    "At-Risk Customers": {
        "description": "Low recent activity; at risk of churning.",
        "strategy": "Win-back campaigns, personalised outreach, reactivation discounts.",
        "icon": "⚠️",
    },
}

# ── Prediction helpers ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model ...")
def load_model():
    bundle = joblib.load("models/kmeans_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return bundle["model"], scaler, bundle["label_map"], bundle["feature_cols"]


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Total_Spending"] = df.get("Lifetime_Value", df.get("Total_Spending", 0))
    membership = df["Membership_Years"] if "Membership_Years" in df.columns else pd.Series(np.ones(len(df)))
    df["Purchase_Frequency"] = df.get("Total_Purchases", pd.Series(np.zeros(len(df)))) / membership.clip(lower=0.1)
    days = df["Days_Since_Last_Purchase"] if "Days_Since_Last_Purchase" in df.columns else pd.Series([30] * len(df))
    df["RFM_Recency"] = 1 / (days + 1)
    df["RFM_Frequency"] = df.get("Total_Purchases", pd.Series(np.zeros(len(df))))
    df["RFM_Monetary"] = df.get("Lifetime_Value", pd.Series(np.zeros(len(df))))
    age = df["Age"] if "Age" in df.columns else pd.Series([35] * len(df))
    df["Customer_Age_Group"] = pd.cut(age, bins=[0, 30, 50, 200], labels=[0, 1, 2]).astype(float)
    return df


def _preprocess_input(df: pd.DataFrame, feature_cols: list, scaler) -> np.ndarray:
    df = df.copy()
    if "Gender" in df.columns:
        df["Gender_Encoded"] = (df["Gender"] == "Male").astype(int)
    df = _engineer_features(df)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df[feature_cols] = df[feature_cols].fillna(0)
    return scaler.transform(df[feature_cols].values)


def predict(raw_df: pd.DataFrame) -> pd.DataFrame:
    model, scaler, label_map, feature_cols = load_model()
    X_scaled = _preprocess_input(raw_df, feature_cols, scaler)
    clusters = model.predict(X_scaled)
    df = raw_df.copy()
    df["Cluster"] = clusters
    df["Segment"] = df["Cluster"].map(label_map)
    df["Segment_Description"] = df["Segment"].map(
        lambda s: SEGMENT_DESCRIPTIONS.get(s, {}).get("description", "Unknown")
    )
    df["Marketing_Strategy"] = df["Segment"].map(
        lambda s: SEGMENT_DESCRIPTIONS.get(s, {}).get("strategy", "Unknown")
    )
    df["Icon"] = df["Segment"].map(
        lambda s: SEGMENT_DESCRIPTIONS.get(s, {}).get("icon", "❓")
    )
    return df


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
h1 { color: #1f2937; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
try:
    model, scaler, label_map, feature_cols = load_model()
except Exception as e:
    st.error(f"No trained model found. Run `python main.py` first.\n\n{e}")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Controls")
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "📤 Upload Customer CSV",
        type=["csv"],
        help="Upload a CSV with customer features.",
    )
    st.markdown("---")
    st.markdown("### 📖 Segment Guide")
    for seg, info in SEGMENT_DESCRIPTIONS.items():
        st.markdown(f"**{info['icon']} {seg}**")
        st.caption(info["description"])
        st.markdown("")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎯 Customer Segmentation Dashboard")
st.markdown(
    "Upload your customer data to automatically assign ML-powered segments "
    "and uncover actionable marketing insights."
)
st.markdown("---")

# ── Require file upload ───────────────────────────────────────────────────────
if uploaded_file is None:
    st.info("👈 Upload a CSV file in the sidebar to get started.")
    st.stop()

raw_df = pd.read_csv(uploaded_file)
result_df = predict(raw_df)

# ── KPIs ──────────────────────────────────────────────────────────────────────
n_customers = len(result_df)
n_segments = result_df["Segment"].nunique()
top_segment = result_df["Segment"].value_counts().idxmax()
avg_ltv = result_df["Lifetime_Value"].mean() if "Lifetime_Value" in result_df.columns else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("👥 Total Customers", f"{n_customers:,}")
col2.metric("🏷️ Segments Found", n_segments)
col3.metric("🏆 Top Segment", top_segment)
col4.metric("💰 Avg Lifetime Value", f"${avg_ltv:,.0f}")

st.markdown("---")

# ── Segment Distribution + Pie ────────────────────────────────────────────────
chart1, chart2 = st.columns(2)

seg_counts = result_df["Segment"].value_counts().reset_index()
seg_counts.columns = ["Segment", "Count"]

with chart1:
    fig_bar = px.bar(
        seg_counts, x="Segment", y="Count",
        color="Segment", title="Customer Segment Distribution",
        color_discrete_sequence=px.colors.qualitative.Set2,
        text="Count",
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig_bar, use_container_width=True)

with chart2:
    fig_pie = px.pie(
        seg_counts, values="Count", names="Segment",
        title="Segment Share",
        color_discrete_sequence=px.colors.qualitative.Set2,
        hole=0.4,
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    fig_pie.update_layout(height=380)
    st.plotly_chart(fig_pie, use_container_width=True)

# ── PCA Cluster Visualization ─────────────────────────────────────────────────
st.subheader("🔵 PCA Cluster Visualization")

try:
    X_scaled = _preprocess_input(raw_df, feature_cols, scaler)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    ev = pca.explained_variance_ratio_

    pca_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "Segment": result_df["Segment"].values,
    })

    fig_pca = px.scatter(
        pca_df, x="PC1", y="PC2", color="Segment",
        title=f"PCA Plot — PC1 ({ev[0]:.1%}) × PC2 ({ev[1]:.1%})",
        color_discrete_sequence=px.colors.qualitative.Set2,
        opacity=0.6,
    )
    fig_pca.update_traces(marker=dict(size=5))
    fig_pca.update_layout(height=450)
    st.plotly_chart(fig_pca, use_container_width=True)
except Exception as e:
    st.warning(f"PCA visualization skipped: {e}")

# ── Cluster Statistics ────────────────────────────────────────────────────────
st.subheader("📊 Cluster Statistics")

summary_cols = [
    "Lifetime_Value", "Total_Purchases", "Days_Since_Last_Purchase",
    "Login_Frequency", "Average_Order_Value", "Membership_Years",
]
available = [c for c in summary_cols if c in result_df.columns]
if available:
    cluster_stats = result_df.groupby("Segment")[available].mean().round(2)
    st.dataframe(
        cluster_stats.style.background_gradient(cmap="YlOrRd", axis=0),
        use_container_width=True,
    )

# ── Segment Descriptions ──────────────────────────────────────────────────────
st.subheader("🏷️ Segment Descriptions & Marketing Strategies")

unique_segs = result_df["Segment"].unique()
cols = st.columns(min(len(unique_segs), 3))
for i, seg in enumerate(unique_segs):
    info = SEGMENT_DESCRIPTIONS.get(seg, {})
    count = (result_df["Segment"] == seg).sum()
    pct = count / n_customers * 100
    with cols[i % len(cols)]:
        st.markdown(f"""
**{info.get('icon', '❓')} {seg}**
- 👥 Customers: **{count:,}** ({pct:.1f}%)
- 📝 {info.get('description', '')}
- 🎯 *{info.get('strategy', '')}*
""")

# ── Customer Table ────────────────────────────────────────────────────────────
st.subheader("📋 Customer Assignment Table")

display_cols = ["Segment", "Cluster", "Icon"] + [
    c for c in ["Age", "Lifetime_Value", "Total_Purchases", "Days_Since_Last_Purchase"]
    if c in result_df.columns
]
st.dataframe(result_df[display_cols].head(200), use_container_width=True)

# ── Download ──────────────────────────────────────────────────────────────────
st.subheader("⬇️ Download Results")
csv_buffer = io.StringIO()
result_df.to_csv(csv_buffer, index=False)
st.download_button(
    label="📥 Download Segmented CSV",
    data=csv_buffer.getvalue(),
    file_name="customer_segments.csv",
    mime="text/csv",
)

st.markdown("---")
st.caption("Built with Streamlit · Scikit-learn · MLflow | Customer Segmentation ML Project")
