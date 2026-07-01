import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from pathlib import Path
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

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="MLflow Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 MLflow Experiment Dashboard")

st.write(
"""
View all experiments,
runs,
metrics,
parameters
and registered models.
"""
)

st.divider()

# ==========================================================
# CONNECT TO LOCAL TRACKING
# ==========================================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = MlflowClient()

# ==========================================================
# EXPERIMENTS
# ==========================================================

experiments = client.search_experiments()

if len(experiments) == 0:

    st.warning("No experiments found.")

    st.stop()

experiment_names = [e.name for e in experiments]

selected = st.selectbox(

    "Select Experiment",

    experiment_names

)

experiment = next(
    e for e in experiments
    if e.name == selected
)

st.success(f"Experiment ID : {experiment.experiment_id}")

st.divider()

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id]
)

if len(runs) == 0:

    st.warning("No runs found.")

    st.stop()

run_names = []

for run in runs:

    run_names.append(run.info.run_id)

selected_run = st.selectbox(

    "Select Run",

    run_names

)

run = next(

    r for r in runs

    if r.info.run_id == selected_run

)

st.header("Run Information")

col1,col2,col3 = st.columns(3)

with col1:

    st.metric(

        "Run ID",

        run.info.run_id[:10]

    )

with col2:

    st.metric(

        "Status",

        run.info.status

    )

with col3:

    st.metric(

        "Lifecycle",

        run.info.lifecycle_stage

    )

st.divider()

st.header("Logged Parameters")

params = pd.DataFrame(

    run.data.params.items(),

    columns=["Parameter","Value"]

)

st.dataframe(

    params,

    use_container_width=True

)

st.header("Run Tags")

tags = pd.DataFrame(

    run.data.tags.items(),

    columns=["Tag","Value"]

)

st.dataframe(

    tags,

    use_container_width=True

)

st.header("Registered Models")

models = client.search_registered_models()

if len(models)==0:

    st.info("No Registered Models")

else:

    for model in models:

        st.subheader(model.name)

        versions = client.search_model_versions(

            f"name='{model.name}'"

        )

        for version in versions:

            st.write(

                f"Version : {version.version}"

            )

            st.write(

                f"Current Stage : {version.current_stage}"

            )

            st.write(

                f"Run ID : {version.run_id}"

            )

            st.divider()
# ==========================================================
# FOOTER
# ==========================================================

st.markdown(
"""
<div style="text-align:center; padding:20px">

### 📈 MLflow Experiment Dashboard

Developed using

**Streamlit • Scikit-Learn • MLflow • Random Forest**

---

Made by **Kavya Saraswat**

</div>
""",
unsafe_allow_html=True
)