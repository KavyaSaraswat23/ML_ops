import streamlit as st
from pathlib import Path

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# LOAD CUSTOM CSS
# -----------------------------
css_file = Path("assets/style.css")

if css_file.exists():
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/2966/2966480.png",
    width=80
)

st.sidebar.title("AI Healthcare")

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Heart Disease Prediction System**

    Machine Learning Pipeline

    • Random Forest

    • MLflow

    • Evidently AI

    • Streamlit
    """
)

st.sidebar.markdown("---")

# -----------------------------
# HERO SECTION
# -----------------------------
col1, col2 = st.columns([3,2])

with col1:

    st.markdown("# ❤️ AI Heart Disease Prediction")

    st.markdown(
        """
Predict cardiovascular disease using a **Machine Learning pipeline**
built with **Scikit-Learn**, **MLflow**, **Evidently AI**, and **Streamlit**.

This application demonstrates a complete end-to-end MLOps workflow:

- Data preprocessing
- Feature engineering
- Random Forest classification
- MLflow model versioning
- Evidently monitoring
- Interactive prediction dashboard
"""
    )

# with col2:

st.divider()

# -----------------------------
# KPI CARDS
# -----------------------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Patients",
        "303"
    )

with c2:
    st.metric(
        "Features",
        "15"
    )

with c3:
    st.metric(
        "Best Recall",
        "92%"
    )

with c4:
    st.metric(
        "Algorithm",
        "Random Forest"
    )

st.divider()

# -----------------------------
# WORKFLOW
# -----------------------------
st.subheader("🔄 Prediction Workflow")

workflow = """
Patient Data

⬇

Outlier Handling

⬇

Feature Engineering

⬇

Encoding + Scaling

⬇

Correlation Filtering

⬇

Random Forest

⬇

Heart Disease Prediction
"""

st.code(workflow)

# -----------------------------
# FEATURES
# -----------------------------
st.subheader("🚀 Features")

col1, col2, col3 = st.columns(3)

with col1:

    st.info(
        """
### 📊 Dashboard

Visualize

- Patient statistics

- Disease distribution

- Feature importance

- Correlation heatmap
"""
    )

with col2:

    st.success(
        """
### 🩺 Prediction

Predict heart disease

using trained

Random Forest model

with probability score.
"""
    )

with col3:

    st.warning(
        """
### 🤖 MLOps

✔ MLflow

✔ Evidently

✔ Monitoring

✔ Model Versioning
"""
    )

st.divider()

# -----------------------------
# TECHNOLOGY STACK
# -----------------------------
st.subheader("🛠 Technology Stack")

tech1, tech2, tech3, tech4 = st.columns(4)

tech1.markdown("### 🐍 Python")
tech2.markdown("### 🌲 Scikit-Learn")
tech3.markdown("### 📈 MLflow")
tech4.markdown("### ⚡ Streamlit")


# -----------------------------
# FOOTER
# -----------------------------
st.markdown(
    """
---
<center>

### ❤️ AI Heart Disease Prediction System

Developed by **Kavya Saraswat**

Machine Learning • MLOps • Streamlit

</center>
""",
    unsafe_allow_html=True,
)