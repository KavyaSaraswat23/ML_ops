import streamlit as st
import pandas as pd
from pathlib import Path

from utils.model_loader import load_model

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ==========================================================
# LOAD CSS
# ==========================================================

css_file = Path("assets/style.css")

if css_file.exists():
    with open(css_file) as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

# ==========================================================
# LOAD MODEL
# ==========================================================

try:

    model = load_model()

    MODEL_READY = True

except Exception as e:

    MODEL_READY = False

    st.error(f"Unable to load model.\n\n{e}")

# ==========================================================
# CATEGORY MAPPINGS
# ==========================================================

SEX_MAPPING = {
    "Male": 1,
    "Female": 0
}

YES_NO_MAPPING = {
    "No": 0,
    "Yes": 1
}

CHEST_PAIN_MAPPING = {

    "Typical Angina":
        "cp_typical_angina",

    "Atypical Angina":
        "cp_atypical_angina",

    "Non-anginal Pain":
        "cp_non_anginal_pain",

    "Asymptomatic":
        "cp_asymptomatic"
}

RESTING_ECG_MAPPING = {

    "Normal":
        "ecg_normal",

    "ST-T Wave Abnormality":
        "ecg_st_t_wave_abnormality",

    "Left Ventricular Hypertrophy":
        "ecg_left_ventricular_hypertrophy"
}

THAL_MAPPING = {

    "Normal":
        "thal_normal",

    "Fixed Defect":
        "thal_fixed_defect",

    "Reversible Defect":
        "thal_reversable_defect"
}

SLOPE_MAPPING = {

    "Upsloping":
        "slope_upsloping",

    "Flat":
        "slope_flat",

    "Downsloping":
        "slope_downsloping"
}

# ==========================================================
# PAGE TITLE
# ==========================================================

st.title("❤️ Heart Disease Prediction")

st.markdown(
"""
Predict the **risk of heart disease**
using the trained Random Forest model.

The model automatically performs:

- Outlier Handling
- Feature Engineering
- Feature Scaling
- One-Hot Encoding
- Correlation Filtering
- Random Forest Classification
"""
)

st.divider()

# ==========================================================
# MODEL STATUS
# ==========================================================

if MODEL_READY:

    st.success("✅ Trained Model Loaded Successfully")

else:

    st.error(
        """
Model could not be loaded.

Please place

models/final_rf_model.pkl

inside the models folder.
"""
    )

st.divider()

# ==========================================================
# PATIENT INFORMATION HEADER
# ==========================================================

st.header("🩺 Patient Information")

st.caption(
    "Fill all patient details before clicking Predict."
)

col1, col2 = st.columns(2)

# ==========================================================
# LEFT COLUMN
# ==========================================================

with col1:

    age = st.number_input(
        "Age (Years)",
        min_value=18,
        max_value=100,
        value=55,
        step=1
    )

    sex = st.selectbox(
        "Sex",
        options=list(SEX_MAPPING.keys())
    )

    chest_pain = st.selectbox(
        "Chest Pain Type",
        options=list(CHEST_PAIN_MAPPING.keys())
    )

    resting_blood_pressure = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=70,
        max_value=250,
        value=120,
        step=1
    )

    serum_cholesterol = st.number_input(
        "Serum Cholesterol (mg/dL)",
        min_value=100,
        max_value=600,
        value=220,
        step=1
    )

    fasting_blood_sugar = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL",
        options=list(YES_NO_MAPPING.keys())
    )

    resting_ecg = st.selectbox(
        "Resting ECG Result",
        options=list(RESTING_ECG_MAPPING.keys())
    )

# ==========================================================
# RIGHT COLUMN
# ==========================================================

with col2:

    maximum_heart_rate_achieved = st.number_input(
        "Maximum Heart Rate Achieved",
        min_value=60,
        max_value=250,
        value=150,
        step=1
    )

    exercise_induced_angina = st.selectbox(
        "Exercise Induced Angina",
        options=list(YES_NO_MAPPING.keys())
    )

    st_depression_induced_by_exercise = st.number_input(
        "ST Depression",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        options=list(SLOPE_MAPPING.keys())
    )

    number_major_vessels = st.selectbox(
        "Number of Major Vessels",
        options=[0.0, 1.0, 2.0, 3.0]
    )

    thal = st.selectbox(
        "Thal",
        options=list(THAL_MAPPING.keys())
    )

# ==========================================================
# CONVERT UI VALUES TO MODEL VALUES
# ==========================================================

sex = SEX_MAPPING[sex]

fasting_blood_sugar = YES_NO_MAPPING[fasting_blood_sugar]

exercise_induced_angina = YES_NO_MAPPING[exercise_induced_angina]

chest_pain_type = CHEST_PAIN_MAPPING[chest_pain]

resting_ecg_results = RESTING_ECG_MAPPING[resting_ecg]

thal = THAL_MAPPING[thal]

slope_peak_exercise_st_segment = SLOPE_MAPPING[slope]

st.divider()

st.subheader("📋 Patient Summary")

# ==========================================================
# CREATE INPUT DATAFRAME
# ==========================================================

patient_df = pd.DataFrame({

    "age": [int(age)],

    "sex": [int(sex)],

    "chest_pain_type": [chest_pain_type],

    "resting_blood_pressure": [int(resting_blood_pressure)],

    "serum_cholesterol": [int(serum_cholesterol)],

    "fasting_blood_sugar": [int(fasting_blood_sugar)],

    "resting_ecg_results": [resting_ecg_results],

    "maximum_heart_rate_achieved": [
        int(maximum_heart_rate_achieved)
    ],

    "exercise_induced_angina": [
        int(exercise_induced_angina)
    ],

    "st_depression_induced_by_exercise": [
        float(st_depression_induced_by_exercise)
    ],

    "slope_peak_exercise_st_segment": [
        slope_peak_exercise_st_segment
    ],

    "number_major_vessels": [
        float(number_major_vessels)
    ],

    "thal": [
        thal
    ]

})

# ==========================================================
# VERIFY INPUT TYPES
# ==========================================================

expected_dtypes = {

    "age": "int64",

    "sex": "int64",

    "chest_pain_type": "object",

    "resting_blood_pressure": "int64",

    "serum_cholesterol": "int64",

    "fasting_blood_sugar": "int64",

    "resting_ecg_results": "object",

    "maximum_heart_rate_achieved": "int64",

    "exercise_induced_angina": "int64",

    "st_depression_induced_by_exercise": "float64",

    "slope_peak_exercise_st_segment": "object",

    "number_major_vessels": "float64",

    "thal": "object"

}

# ==========================================================
# DISPLAY SUMMARY
# ==========================================================

left, right = st.columns([3, 2])

with left:

    st.dataframe(
        patient_df,
        use_container_width=True,
        hide_index=True
    )

with right:

    st.markdown("### Data Types")

    dtype_df = pd.DataFrame({

        "Column": patient_df.columns,

        "Current": patient_df.dtypes.astype(str).values,

        "Expected": [
            expected_dtypes[col]
            for col in patient_df.columns
        ]

    })

    st.dataframe(
        dtype_df,
        use_container_width=True,
        hide_index=True
    )

# ==========================================================
# VALIDATION
# ==========================================================

validation_errors = []

if age < 18:
    validation_errors.append(
        "Age should be at least 18."
    )

if resting_blood_pressure <= 0:
    validation_errors.append(
        "Blood pressure must be positive."
    )

if serum_cholesterol <= 0:
    validation_errors.append(
        "Serum cholesterol must be positive."
    )

if maximum_heart_rate_achieved <= 0:
    validation_errors.append(
        "Maximum heart rate must be positive."
    )

if st_depression_induced_by_exercise < 0:
    validation_errors.append(
        "ST depression cannot be negative."
    )

# ==========================================================
# SHOW VALIDATION RESULT
# ==========================================================

if validation_errors:

    st.warning("Please correct the following:")

    for error in validation_errors:
        st.write(f"• {error}")

else:

    st.success("✅ Input validation passed.")

st.divider()

# ==========================================================
# PREDICT BUTTON
# ==========================================================

predict_btn = st.button(

    "❤️ Predict Heart Disease",

    use_container_width=True,

    disabled=(
        not MODEL_READY
        or len(validation_errors) > 0
    )

)

# ==========================================================
# PREDICTION
# ==========================================================

if predict_btn:

    if not MODEL_READY:

        st.error("❌ Model is not loaded.")

    else:

        try:

            # --------------------------------------------------
            # RUN MODEL
            # --------------------------------------------------

            prediction = model.predict(patient_df)[0]

            probability = model.predict_proba(patient_df)[0]

            healthy_probability = probability[0] * 100

            disease_probability = probability[1] * 100

            st.divider()

            st.header("🩺 Prediction Result")

            # --------------------------------------------------
            # RESULT CARD
            # --------------------------------------------------

            if prediction == 1:

                st.error(
                    "⚠️ High Risk of Heart Disease"
                )

            else:

                st.success(
                    "✅ Low Risk of Heart Disease"
                )

            # st.divider()

            # --------------------------------------------------
            # METRICS
            # --------------------------------------------------

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "❤️ Heart Disease Probability",
                    f"{disease_probability:.2f}%"
                )

                st.progress(
                    disease_probability / 100
                )

            with col2:

                st.metric(
                    "💚 Healthy Probability",
                    f"{healthy_probability:.2f}%"
                )

                st.progress(
                    healthy_probability / 100
                )

            st.divider()

            # --------------------------------------------------
            # RISK LEVEL
            # --------------------------------------------------

            st.subheader("📊 Risk Assessment")

            if disease_probability >= 80:

                st.error(
                    "🔴 Risk Level : VERY HIGH"
                )

            elif disease_probability >= 60:

                st.warning(
                    "🟠 Risk Level : HIGH"
                )

            elif disease_probability >= 40:

                st.info(
                    "🟡 Risk Level : MODERATE"
                )

            else:

                st.success(
                    "🟢 Risk Level : LOW"
                )

            st.divider()

            # --------------------------------------------------
            # RECOMMENDATION
            # --------------------------------------------------

            st.subheader("💡 Recommendation")

            if prediction == 1:

                st.warning(
                    """
The model predicts a **high probability of heart disease**.

Recommended Next Steps:

• Visit a Cardiologist

• ECG Test

• Echocardiography

• Blood Test

• Exercise Stress Test

• Lifestyle Modification

⚠️ This prediction is only an AI estimation and **NOT** a medical diagnosis.
"""
                )

            else:

                st.success(
                    """
The model predicts a **low probability of heart disease**.

Recommended:

• Maintain Healthy Lifestyle

• Regular Exercise

• Balanced Diet

• Routine Medical Check-up

• Monitor Blood Pressure

• Maintain Healthy Weight
"""
                )

            st.divider()

            # --------------------------------------------------
            # MODEL CONFIDENCE
            # --------------------------------------------------

            st.subheader("🤖 Model Confidence")

            confidence = max(probability) * 100

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            st.progress(
                confidence / 100
            )

            st.divider()

            # --------------------------------------------------
            # FINAL SUMMARY
            # --------------------------------------------------

            st.subheader("📋 Final Summary")

            summary = patient_df.copy()

            summary["Prediction"] = (
                "Heart Disease"
                if prediction == 1
                else "Healthy"
            )

            summary["Heart Disease Probability"] = (
                f"{disease_probability:.2f}%"
            )

            summary["Healthy Probability"] = (
                f"{healthy_probability:.2f}%"
            )

            summary["Confidence"] = (
                f"{confidence:.2f}%"
            )

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True
            )

        except Exception as e:

            st.error("❌ Prediction Failed")

            st.exception(e)

# ==========================================================
# PREDICTION HISTORY
# ==========================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if predict_btn and MODEL_READY:

    history = patient_df.copy()

    history["Prediction"] = (
        "Heart Disease"
        if prediction == 1
        else "Healthy"
    )

    history["Heart Disease Probability"] = round(
        disease_probability,
        2
    )

    history["Healthy Probability"] = round(
        healthy_probability,
        2
    )

    history["Confidence"] = round(
        confidence,
        2
    )

    st.session_state.prediction_history.append(history)

# ==========================================================
# SHOW HISTORY
# ==========================================================

if len(st.session_state.prediction_history) > 0:

    st.divider()

    st.header("📈 Prediction History")

    history_df = pd.concat(
        st.session_state.prediction_history,
        ignore_index=True
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    # ------------------------------------------------------

    csv = history_df.to_csv(index=False)

    st.download_button(

        label="⬇ Download History (CSV)",

        data=csv,

        file_name="prediction_history.csv",

        mime="text/csv",

        use_container_width=True

    )

    # ------------------------------------------------------

    json_data = history_df.to_json(
        orient="records",
        indent=4
    )

    st.download_button(

        label="⬇ Download History (JSON)",

        data=json_data,

        file_name="prediction_history.json",

        mime="application/json",

        use_container_width=True

    )

# ==========================================================
# RESET HISTORY
# ==========================================================

st.divider()

col1, col2 = st.columns(2)

with col1:

    if st.button(
        "🗑 Clear Prediction History",
        use_container_width=True
    ):

        st.session_state.prediction_history = []

        st.success("Prediction history cleared.")

        st.rerun()

with col2:

    st.button(
        "🔄 New Prediction",
        use_container_width=True
    )

# ==========================================================
# ABOUT MODEL
# ==========================================================

with st.expander("ℹ About the Prediction Model"):

    st.markdown(
        """
### Model Information

**Algorithm**

- Random Forest Classifier

**Pipeline**

- Outlier Replacement
- Feature Engineering
- Standard Scaling
- One-Hot Encoding
- Correlation Filtering

**Evaluation Metric**

- Recall Score

**Purpose**

The model predicts the probability of heart disease using
clinical information provided by the patient.

---

⚠ This application is intended for educational purposes only.

It should **not** be used as a substitute for professional
medical diagnosis.
"""
    )

# ==========================================================
# FOOTER
# ==========================================================

st.markdown(
"""
<div style="text-align:center; padding:20px">

### ❤️ Heart Disease Prediction System

Developed using

**Streamlit • Scikit-Learn • MLflow • Random Forest**

---

Made by **Kavya Saraswat**

</div>
""",
unsafe_allow_html=True
)