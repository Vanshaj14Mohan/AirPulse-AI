# ==========================================================
# 🌍 AirPulse AI
# AQI Classification Page
# ==========================================================

# ===========================
# Import Libraries
# ===========================

import streamlit as st
import pandas as pd
import numpy as np

from utils import (
    load_original_dataset,
    load_classifier,
    load_country_encoder,
    load_city_encoder,
    load_scaler
)


import streamlit as st

def show_classification():
    # ======================================================
    # Load Data, Model & Preprocessing Objects
    # ======================================================

    original_df = load_original_dataset()

    classifier = load_classifier()

    country_encoder = load_country_encoder()

    city_encoder = load_city_encoder()

    scaler = load_scaler()


    # ======================================================
    # Page Header
    # ======================================================

    st.title("🏷 AQI Classification")

    st.markdown("""
    ### Classify Air Quality using Machine Learning

    Enter the environmental and weather conditions below to
    classify the air quality into a pollution severity category
    using the trained XGBoost classification model.
    """)

    st.markdown("---")

    st.title("🏷 AQI Classification")