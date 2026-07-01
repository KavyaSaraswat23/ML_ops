import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
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
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "raw_heart_disease_data.csv"
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


df = load_data()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 Dashboard")

st.sidebar.markdown("---")

st.sidebar.success(
    "Dataset Loaded Successfully"
)

st.sidebar.info(f"""
Rows : **{df.shape[0]}**

Columns : **{df.shape[1]}**
""")

# ============================================================
# HEADER
# ============================================================

st.title("📊 Heart Disease Dashboard")

st.write(
    "Overview of the Heart Disease dataset used for training the Machine Learning model."
)

st.divider()

# ============================================================
# KPI CARDS
# ============================================================

total_patients = len(df)

average_age = round(df["age"].mean(), 1)

male_count = (df["sex"] == 1).sum()

female_count = (df["sex"] == 0).sum()

heart_patients = (df["presence_of_heart_disease"] == 1).sum()

healthy_patients = (df["presence_of_heart_disease"] == 0).sum()

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "👨‍⚕️ Total Patients",
        total_patients
    )

with c2:

    st.metric(
        "🩺 Average Age",
        average_age
    )

with c3:

    st.metric(
        "❤️ Heart Disease",
        heart_patients
    )

with c4:

    st.metric(
        "💚 Healthy",
        healthy_patients
    )

st.divider()

# ============================================================
# SECOND ROW
# ============================================================

c1, c2 = st.columns(2)

with c1:

    st.subheader("Dataset Information")

    info_df = pd.DataFrame({

        "Attribute":[
            "Rows",
            "Columns",
            "Male Patients",
            "Female Patients"
        ],

        "Value":[
            total_patients,
            df.shape[1],
            male_count,
            female_count
        ]

    })

    st.dataframe(
        info_df,
        use_container_width=True
    )

with c2:

    st.subheader("Target Distribution")

    target_df = pd.DataFrame({

        "Category":[
            "Healthy",
            "Heart Disease"
        ],

        "Patients":[
            healthy_patients,
            heart_patients
        ]

    })

    st.dataframe(
        target_df,
        use_container_width=True
    )

st.divider()

# ============================================================
# DATA PREVIEW
# ============================================================

st.subheader("📄 Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)

st.divider()

# ============================================================
# SUMMARY
# ============================================================

st.subheader("📈 Statistical Summary")

st.dataframe(
    df.describe(),
    use_container_width=True
)

st.divider()

# ============================================================
# MISSING VALUES
# ============================================================

st.subheader("🚨 Missing Values")

missing = df.isnull().sum()

missing_df = pd.DataFrame({

    "Column":missing.index,
    "Missing Values":missing.values

})

st.dataframe(
    missing_df,
    use_container_width=True
)

st.divider()

# ============================================================
# FOOTER
# ============================================================

st.markdown(
"""
---
### ❤️ AI Heart Disease Prediction Dashboard

Developed using

- Streamlit
- Scikit-Learn
- MLflow
- Evidently AI

"""
)
st.subheader("❤️ Heart Disease Distribution")

target_counts = df["presence_of_heart_disease"].value_counts()

labels = ["Heart Disease", "Healthy"]

fig = go.Figure(
    data=[
        go.Pie(
            labels=labels,
            values=[
                target_counts[1],
                target_counts[0]
            ],
            hole=0.6
        )
    ]
)

fig.update_layout(
    height=420,
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("👨 Gender Distribution")

gender = df["sex"].replace(
    {
        1: "Male",
        0: "Female"
    }
)

gender_count = gender.value_counts()

fig = px.pie(
    names=gender_count.index,
    values=gender_count.values,
    hole=0.5,
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("📈 Age Distribution")

fig = px.histogram(
    df,
    x="age",
    nbins=20,
    template="plotly_white"
)

fig.update_layout(
    xaxis_title="Age",
    yaxis_title="Patients"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("🫀 Chest Pain Types")

cp = df["chest_pain_type"].value_counts()

fig = px.bar(
    x=cp.index.astype(str),
    y=cp.values,
    labels={
        "x":"Chest Pain Type",
        "y":"Patients"
    },
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("🩸 Cholesterol Distribution")

fig = px.box(
    df,
    y="serum_cholesterol",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("❤️ Maximum Heart Rate Achieved")

fig = px.histogram(
    df,
    x="maximum_heart_rate_achieved",
    nbins=25,
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("🔥 Correlation Heatmap")

corr = df.corr(numeric_only=True)

fig = px.imshow(
    corr,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    aspect="auto"
)

fig.update_layout(
    height=700
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("⭐ Feature Importance")

st.info(
    """
This chart will display the Random Forest feature importance
after we connect the trained MLflow model.
"""
)

st.subheader("📄 Dataset")

st.dataframe(
    df,
    use_container_width=True,
    height=450
)

st.subheader("📊 Statistical Summary")

st.dataframe(
    df.describe(),
    use_container_width=True
)

# ==========================================================
# FOOTER
# ==========================================================

st.markdown(
"""
<div style="text-align:center; padding:20px">

### 📊 Heart Disease Dashboard

Developed using

**Streamlit • Scikit-Learn • MLflow • Random Forest**

---

Made by **Kavya Saraswat**

</div>
""",
unsafe_allow_html=True
)
