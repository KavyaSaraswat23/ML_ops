import streamlit as st
import pandas as pd
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Model Performance",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# LOAD CSS
# ============================================================

css_file = Path("assets/style.css")

if css_file.exists():
    with open(css_file) as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

# ============================================================
# HEADER
# ============================================================

st.title("📈 Model Performance Dashboard")

st.write(
"""
This page presents the evaluation results of the trained
Random Forest Heart Disease Prediction model.
"""
)

st.divider()

# ============================================================
# MODEL INFORMATION
# ============================================================

st.header("🤖 Model Information")

col1, col2 = st.columns(2)

with col1:

    st.info(
        """
### Model

- **Algorithm:** Random Forest Classifier
- **Dataset:** UCI Heart Disease Dataset
- **Target Variable:** Presence of Heart Disease
- **Evaluation Metric:** Recall Score
"""
    )

with col2:

    st.info(
        """
### Training Details

- **Cross Validation:** Stratified K-Fold (10 Folds)
- **Hyperparameter Tuning:** GridSearchCV
- **Class Weight:** Balanced
- **Random State:** 22
"""
    )

st.divider()

# ============================================================
# MODEL PERFORMANCE METRICS
# ============================================================

st.header("📈 Evaluation Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:

    st.metric(
        "Accuracy",
        "83.33%"
    )

with col2:

    st.metric(
        "Precision",
        "64.29%"
    )

with col3:

    st.metric(
        "Recall",
        "90.00%"
    )

with col4:

    st.metric(
        "F1 Score",
        "75.00%"
    )

with col5:

    st.metric(
        "ROC-AUC",
        "88.85%"
    )

st.divider()

# ============================================================
# MODEL PIPELINE
# ============================================================

st.header("⚙️ Machine Learning Pipeline")

pipeline_df = pd.DataFrame({

    "Step": [

        "Outlier Replacement",

        "Feature Engineering",

        "Feature Scaling",

        "Categorical Encoding",

        "Correlation Filtering",

        "Random Forest Classification"

    ],

    "Description": [

        "Replace extreme values using IQR method",

        "Generate new informative features",

        "StandardScaler applied to numerical features",

        "One-Hot Encoding + Ordinal Encoding",

        "Remove highly correlated features",

        "Predict Heart Disease"

    ]

})

st.dataframe(
    pipeline_df,
    use_container_width=True,
    hide_index=True
)

st.divider()

# ============================================================
# MODEL SUMMARY
# ============================================================

st.header("📝 Model Summary")

st.success(
"""
The Random Forest model was trained using a complete machine
learning pipeline consisting of preprocessing,
feature engineering,
encoding,
correlation filtering,
and hyperparameter optimization.

The model achieved a **Recall of 90%**, making it highly suitable
for detecting patients with heart disease while minimizing false negatives.
"""
)

st.divider()

# ============================================================
# CONFUSION MATRIX
# ============================================================

import matplotlib.pyplot as plt
import numpy as np

st.header("📊 Confusion Matrix")

cm = np.array([
    [21, 5],
    [1, 9]
])

fig, ax = plt.subplots(figsize=(6,5))

im = ax.imshow(cm)

ax.set_xticks([0,1])
ax.set_yticks([0,1])

ax.set_xticklabels(["Healthy","Heart Disease"])
ax.set_yticklabels(["Healthy","Heart Disease"])

ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold"
        )

st.pyplot(fig)

st.divider()

# ============================================================
# CONFUSION MATRIX EXPLANATION
# ============================================================

col1, col2 = st.columns(2)

with col1:

    st.info(f"""
### Confusion Matrix Summary

✔ True Negatives : **21**

✔ False Positives : **5**

✔ False Negatives : **1**

✔ True Positives : **9**
""")

with col2:

    st.success("""
### Interpretation

The model correctly identifies most patients
with heart disease.

Only **1 patient** having heart disease
was incorrectly classified as healthy,
which explains the high Recall score.
""")

st.divider()

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

st.header("📋 Classification Report")

report_df = pd.DataFrame({

    "Class":[
        "Healthy (0)",
        "Heart Disease (1)",
        "Macro Average",
        "Weighted Average"
    ],

    "Precision":[
        0.95,
        0.64,
        0.80,
        0.87
    ],

    "Recall":[
        0.81,
        0.90,
        0.85,
        0.83
    ],

    "F1 Score":[
        0.88,
        0.75,
        0.81,
        0.84
    ],

    "Support":[
        26,
        10,
        36,
        36
    ]

})

st.dataframe(
    report_df,
    use_container_width=True,
    hide_index=True
)

st.divider()

# ============================================================
# PERFORMANCE ANALYSIS
# ============================================================

st.header("📈 Performance Analysis")

st.markdown("""
### Key Observations

- The model achieved an overall **Accuracy of 83.33%**.

- The **Recall of 90%** indicates excellent detection of patients with heart disease.

- The model sacrifices some Precision (64.29%) in order to reduce false negatives.

- Only **one patient with heart disease** was missed during testing.

- This behaviour is desirable in healthcare applications where missing a diseased patient is more critical than producing additional false alarms.
""")

st.divider()

# ============================================================
# MODEL STRENGTHS
# ============================================================

st.header("⭐ Model Strengths")

strengths = pd.DataFrame({

    "Strength":[
        "High Recall",
        "Balanced Random Forest",
        "Hyperparameter Tuned",
        "Cross Validated",
        "Complete ML Pipeline"
    ],

    "Description":[
        "Detects most heart disease cases.",
        "Handles class imbalance effectively.",
        "Optimized using GridSearchCV.",
        "Validated using Stratified K-Fold.",
        "Includes preprocessing and feature engineering."
    ]

})

st.dataframe(
    strengths,
    use_container_width=True,
    hide_index=True
)

st.divider()

# ============================================================
# IMPORTS
# ============================================================

import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# METRIC VISUALIZATION
# ============================================================

st.header("📊 Performance Visualization")

metrics_df = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],

    "Score": [
        83.33,
        64.29,
        90.00,
        75.00,
        88.85
    ]

})

fig = px.bar(
    metrics_df,
    x="Metric",
    y="Score",
    text="Score",
    title="Model Performance Metrics"
)

fig.update_layout(
    yaxis_title="Percentage",
    xaxis_title="Evaluation Metric",
    yaxis=dict(range=[0,100]),
    height=450
)

fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# ============================================================
# PRECISION VS RECALL
# ============================================================

st.header("🎯 Precision vs Recall")

comparison = pd.DataFrame({

    "Metric":[
        "Precision",
        "Recall"
    ],

    "Value":[
        64.29,
        90.00
    ]

})

fig2 = px.pie(
    comparison,
    values="Value",
    names="Metric",
    hole=0.55,
    title="Precision vs Recall"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

st.divider()

# ============================================================
# CONFUSION MATRIX BREAKDOWN
# ============================================================

st.header("📈 Prediction Distribution")

prediction_df = pd.DataFrame({

    "Category":[
        "True Negative",
        "False Positive",
        "False Negative",
        "True Positive"
    ],

    "Count":[
        21,
        5,
        1,
        9
    ]

})

fig3 = px.bar(
    prediction_df,
    x="Category",
    y="Count",
    text="Count",
    title="Prediction Distribution"
)

fig3.update_layout(height=450)

st.plotly_chart(
    fig3,
    use_container_width=True
)

st.divider()

# ============================================================
# MODEL HEALTH
# ============================================================

st.header("💚 Model Health")

recall = 90
accuracy = 83.33
roc = 88.85

col1, col2, col3 = st.columns(3)

with col1:

    st.progress(recall/100)

    st.caption(f"Recall : {recall}%")

with col2:

    st.progress(accuracy/100)

    st.caption(f"Accuracy : {accuracy}%")

with col3:

    st.progress(roc/100)

    st.caption(f"ROC-AUC : {roc}%")

st.divider()

# ============================================================
# FINAL CONCLUSION
# ============================================================

st.header("🏁 Final Conclusion")

st.success("""
### Overall Performance

The trained Random Forest model demonstrates strong predictive
performance for heart disease classification.

Highlights:

• Accuracy: **83.33%**

• Recall: **90.00%**

• ROC-AUC: **88.85%**

• Only **1 false negative** on the testing dataset.

The model prioritizes identifying patients with heart disease,
making it well suited for healthcare screening applications
where minimizing missed diagnoses is essential.
""")

st.divider()

# ============================================================
# TECHNOLOGY STACK
# ============================================================

st.header("🛠 Technology Stack")

stack = pd.DataFrame({

    "Component":[
        "Programming Language",
        "Machine Learning",
        "Pipeline",
        "Model Tracking",
        "Deployment",
        "Visualization"
    ],

    "Technology":[
        "Python",
        "Scikit-Learn",
        "Pipeline API",
        "MLflow",
        "Streamlit",
        "Plotly"
    ]

})

st.dataframe(
    stack,
    use_container_width=True,
    hide_index=True
)

st.divider()

# ==========================================================
# FOOTER
# ==========================================================


st.markdown(
"""
<div style="text-align:center; padding:20px">

### 📈 Model Performance Dashboard

Developed using

**Streamlit • Scikit-Learn • MLflow • Random Forest**

---

Made by **Kavya Saraswat**

</div>
""",
unsafe_allow_html=True
)