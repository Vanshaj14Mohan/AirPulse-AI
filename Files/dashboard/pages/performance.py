# ==========================================================
# 🌍 AirPulse AI
# Model Performance Page
# ==========================================================

# ===========================
# Import Libraries
# ===========================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# from utils import (
#     load_original_dataset,
#     load_regressor,
#     load_classifier
# )

from utils import (
    load_original_dataset,
    load_processed_dataset,
    load_regressor,
    load_classifier
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ==========================================================
# MODEL PERFORMANCE PAGE
# ==========================================================

def show_performance():
    # ======================================================
    # Page Header
    # ======================================================

    st.title("📈 Model Performance")

    st.markdown("""
    ### Evaluate AirPulse AI Machine Learning Models

    Analyze the performance of the trained regression and
    classification models using standard machine learning
    evaluation metrics and visualizations.
    """)

    st.markdown("---")


    # ======================================================
    # Load Data & Models
    # ======================================================

    original_df = load_original_dataset()

    regressor = load_regressor()

    classifier = load_classifier()


    # ======================================================
    # Model Overview
    # ======================================================

    st.subheader("🤖 Model Overview")

    col1, col2 = st.columns(2)


    # ------------------------------------------------------
    # Regression Model
    # ------------------------------------------------------

    with col1:

        st.markdown("### 📈 AQI Regression")

        st.info("""
        **Model:** XGBoost Regressor

        **Purpose:** Predict the numerical Air Quality Index (AQI).

        **Task:** Regression
        """)


    # ------------------------------------------------------
    # Classification Model
    # ------------------------------------------------------

    with col2:

        st.markdown("### 🏷 AQI Classification")

        st.info("""
        **Model:** XGBoost Classifier

        **Purpose:** Classify air quality into pollution severity categories.

        **Task:** Classification
        """)


    st.markdown("---")


    # ======================================================
    # Dataset Information
    # ======================================================

    st.subheader("📊 Evaluation Dataset")

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "🌍 Countries",
            original_df["country"].nunique()
        )


    with col2:

        st.metric(
            "🏙 Cities",
            original_df["city"].nunique()
        )


    with col3:

        st.metric(
            "📄 Records",
            f"{len(original_df):,}"
        )


    with col4:

        st.metric(
            "🌫 Average AQI",
            round(original_df["aqi"].mean(), 2)
        )


    st.markdown("---")


    # ======================================================
    # Evaluation Section
    # ======================================================

    st.subheader("📋 Model Evaluation")

    st.write("""
    The following sections will present detailed performance
    metrics for both the AQI regression and classification models.
    """)


    st.info(
        "🚀 Phase 6.5.1 - Model Performance Setup Completed"
    )
