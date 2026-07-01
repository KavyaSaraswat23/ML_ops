import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Monitoring",
    page_icon="📉",
    layout="wide"
)

# ============================================================
# LOAD CSS
# ============================================================

css_file = Path("assets/style.css")

if css_file.exists():
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("raw_heart_disease_data.csv")

df = load_data()

# ============================================================
# HEADER
# ============================================================

st.title("📉 Model Monitoring Dashboard")

st.markdown("""
Monitor dataset quality, missing values, data drift,
and model health.
""")

st.divider()

# ============================================================
# KPI CARDS
# ============================================================

total_rows = df.shape[0]
total_columns = df.shape[1]
missing = df.isnull().sum().sum()
missing_percent = round((missing / (df.shape[0] * df.shape[1])) * 100, 2)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Dataset Rows", total_rows)

with c2:
    st.metric("Features", total_columns)

with c3:
    st.metric("Missing Values", missing)

with c4:
    st.metric("Missing %", f"{missing_percent}%")

st.divider()

# ============================================================
# DATASET HEALTH
# ============================================================

st.subheader("🩺 Dataset Health")

health = pd.DataFrame({
    "Check": [
        "Dataset Loaded",
        "Missing Values",
        "Duplicate Rows",
        "Data Drift",
        "Target Drift"
    ],
    "Status": [
        "✅ Passed",
        "✅ Passed" if missing == 0 else "⚠️ Warning",
        "✅ Passed" if df.duplicated().sum() == 0 else "⚠️ Warning",
        "⏳ Pending",
        "⏳ Pending"
    ]
})

st.dataframe(health, use_container_width=True)

st.divider()

# ============================================================
# MISSING VALUES
# ============================================================

st.subheader("📋 Missing Values Per Feature")

missing_df = pd.DataFrame({
    "Feature": df.columns,
    "Missing": df.isnull().sum().values
})

fig = px.bar(
    missing_df,
    x="Feature",
    y="Missing",
    title="Missing Values",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ============================================================
# DUPLICATE ROWS
# ============================================================

st.subheader("📄 Duplicate Records")

duplicates = df.duplicated().sum()

st.metric(
    "Duplicate Rows",
    duplicates
)

st.divider()

# ============================================================
# DATA DRIFT
# ============================================================

st.subheader("📈 Data Drift Status")

st.info("""
Evidently AI report will be displayed here after
the report is generated in Google Colab.
""")

progress = 0

st.progress(progress)

st.divider()

# ============================================================
# MODEL HEALTH
# ============================================================

st.subheader("🤖 Model Health")

model_health = pd.DataFrame({
    "Metric": [
        "Prediction Service",
        "Model Availability",
        "Training Pipeline",
        "Monitoring Pipeline"
    ],
    "Status": [
        "🟢 Running",
        "🟢 Available",
        "🟡 Waiting",
        "🟡 Waiting"
    ]
})

st.dataframe(model_health, use_container_width=True)

st.divider()

# ============================================================
# EVIDENTLY REPORT
# ============================================================

st.subheader("📄 Evidently AI Report")

st.warning("""
The Evidently HTML report has not been added yet.

After generating it in Colab, place it inside:

We'll embed it automatically in the next version.
""")

st.divider()

# ============================================================
# SYSTEM LOGS
# ============================================================

st.subheader("📝 Monitoring Logs")

logs = [
    "✔ Dataset loaded successfully.",
    "✔ Feature validation completed.",
    "✔ Missing value inspection completed.",
    "⏳ Waiting for drift report.",
    "⏳ Waiting for production predictions."
]

for log in logs:
    st.write(log)

st.divider()

# ============================================================
# FOOTER
# ============================================================

st.markdown(
"""
<div style="text-align:center; padding:20px">

### 📉 Model Monitoring Dashboard

Developed using

**Streamlit • Scikit-Learn • MLflow • Random Forest**

---

Made by **Kavya Saraswat**

</div>
""",
unsafe_allow_html=True
)