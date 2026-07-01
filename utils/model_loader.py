import joblib
import streamlit as st
import __main__
from pathlib import Path

from utils.custom_transformers import (
    OutlierReplacer,
    FeatureGenerator,
    CorrelationFilter
)

# Register custom transformers so joblib can deserialize the pipeline
__main__.OutlierReplacer = OutlierReplacer
__main__.FeatureGenerator = FeatureGenerator
__main__.CorrelationFilter = CorrelationFilter

# Absolute path to project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Absolute path to model
MODEL_PATH = BASE_DIR / "models" / "final_rf_model.pkl"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at:\n{MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)
