import joblib
import streamlit as st
import __main__

from utils.custom_transformers import (
    OutlierReplacer,
    FeatureGenerator,
    CorrelationFilter
)

__main__.OutlierReplacer = OutlierReplacer
__main__.FeatureGenerator = FeatureGenerator
__main__.CorrelationFilter = CorrelationFilter

@st.cache_resource
def load_model():

    model = joblib.load(
        "models/final_rf_model.pkl"
    )

    return model