import streamlit as st
from pathlib import Path

# ----------------------------------------------------
# Page Config
# ----------------------------------------------------

st.set_page_config(
    page_title="About",
    page_icon="ℹ️",
    layout="wide"
)

# ----------------------------------------------------
# Load CSS
# ----------------------------------------------------

css = Path("assets/style.css")

if css.exists():
    with open(css) as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

# ----------------------------------------------------
# Header
# ----------------------------------------------------

st.title("ℹ️ About This Project")

st.markdown(
"""
This application is an **End-to-End Machine Learning Operations (MLOps)**
project for predicting the presence of heart disease using clinical
patient information.
"""
)

st.divider()

# ----------------------------------------------------
# Project Overview
# ----------------------------------------------------

col1, col2 = st.columns([2,1])

with col1:

    st.subheader("📌 Project Overview")

    st.markdown("""
### Objectives

- Predict Heart Disease

- Build an End-to-End ML Pipeline

- Track Experiments using MLflow

- Monitor Data Drift using Evidently AI

- Deploy using Streamlit

---

### Dataset

The dataset contains clinical information collected from
patients including

- Age

- Sex

- Chest Pain Type

- Cholesterol

- Blood Pressure

- ECG

- Heart Rate

- ST Depression

and other medical measurements.

The target variable is:

**Presence of Heart Disease**
""")

with col2:

    st.info("""
### Project Statistics

Algorithm

Random Forest

---

Features

15

---

Patients

303

---

Language

Python

---

Framework

Streamlit
""")

st.divider()

# ----------------------------------------------------
# Tech Stack
# ----------------------------------------------------

st.header("🛠 Technology Stack")

c1,c2,c3,c4 = st.columns(4)

with c1:
    st.success("""
### Python

Pandas

NumPy

Scikit-Learn
""")

with c2:
    st.success("""
### MLOps

MLflow

Model Registry

Versioning
""")

with c3:
    st.success("""
### Monitoring

Evidently AI

Data Drift

Reports
""")

with c4:
    st.success("""
### Deployment

Streamlit

Plotly

GitHub
""")

st.divider()

# ----------------------------------------------------
# Workflow
# ----------------------------------------------------

st.header("🔄 Machine Learning Workflow")

workflow = """
Raw Dataset

↓

EDA

↓

Feature Engineering

↓

Preprocessing Pipeline

↓

Random Forest Training

↓

Hyperparameter Tuning

↓

MLflow Logging

↓

Model Evaluation

↓

Deployment

↓

Prediction
"""

st.code(workflow)

st.divider()

# ----------------------------------------------------
# Model Pipeline
# ----------------------------------------------------

st.header("🧠 Model Pipeline")

pipeline = """
Outlier Replacer

↓

Feature Generator

↓

StandardScaler

↓

One Hot Encoder

↓

Correlation Filter

↓

Random Forest Classifier
"""

st.code(pipeline)

st.divider()

# ----------------------------------------------------
# Future Improvements
# ----------------------------------------------------

st.header("🚀 Future Improvements")

st.checkbox("Deploy on Streamlit Cloud", value=True)

st.checkbox("Docker Container")

st.checkbox("CI/CD Pipeline")

st.checkbox("Automated Retraining")

st.checkbox("Model Monitoring Alerts")

st.checkbox("REST API")

st.checkbox("Cloud Deployment (AWS/GCP)")

st.divider()

# ----------------------------------------------------
# Developer
# ----------------------------------------------------

st.header("👨‍💻 Developer")

st.markdown("""
**Name**

Kavya Saraswat

---

Machine Learning Engineer

MLOps Enthusiast

Python Developer
""")

st.divider()

st.markdown(
"""
<div style="text-align:center; padding:20px">

### ℹ️ About This Project

Developed using

**Streamlit • Scikit-Learn • MLflow • Random Forest**

---

Made by **Kavya Saraswat**

</div>
""",
unsafe_allow_html=True
)