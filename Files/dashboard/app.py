# ==========================================================
# 🌍 AirPulse AI
# Intelligent Global Air Quality Monitoring &
# Prediction System
# ==========================================================

# ===========================
# Import Libraries
# ===========================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ===========================
# Page Configuration
# ===========================

st.set_page_config(
    page_title="AirPulse AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# Import Utility Functions
# ===========================

from utils import (
    load_original_dataset,
    load_processed_dataset,
    load_regressor,
    load_classifier
)

# ===========================
# Load Dataset
# ===========================

original_df = load_original_dataset()
processed_df = load_processed_dataset()

# ===========================
# Load Models
# ===========================

regressor = load_regressor()
classifier = load_classifier()

# ===========================
# Sidebar
# ===========================

st.sidebar.title("🌍 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📊 Dashboard",
        "🤖 AQI Prediction",
        "🏷 AQI Classification",
        "📈 Model Performance",
        "ℹ About"
    ]
)

# ==========================================================
# HOME PAGE
# ==========================================================

if page == "🏠 Home":
    st.title("🌍 AirPulse AI")
    st.markdown("""
    ### Intelligent Global Air Quality Monitoring & Prediction System

    AirPulse AI is an intelligent machine learning-based application
    developed to analyze global air quality data, predict Air Quality
    Index (AQI), classify pollution severity, and provide interactive
    visualizations for environmental monitoring and decision making.
    """)

    st.markdown("---")

    # =====================================================
    # Dataset Overview
    # =====================================================

    st.subheader("📊 Dataset Overview")

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

    # =====================================================
    # Project Features
    # =====================================================

    st.subheader("🚀 Features")

    left, right = st.columns(2)

    with left:
        st.success("AQI Prediction")
        st.success("AQI Classification")
        st.success("Interactive Dashboard")

    with right:
        st.success("Machine Learning Models")
        st.success("Data Analytics")
        st.success("Pollution Insights")

    st.markdown("---")

    # =====================================================
    # Technologies Used
    # =====================================================

    st.subheader("🛠 Technologies Used")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.info("🐍 Python")
        st.info("📊 Pandas")
        st.info("📈 Plotly")

    with c2:
        st.info("🤖 Scikit-Learn")
        st.info("⚡ XGBoost")
        st.info("🌐 Streamlit")

    with c3:
        st.info("📉 Matplotlib")
        st.info("🎨 Seaborn")
        st.info("💾 Joblib")

    st.markdown("---")

    # =====================================================
    # Workflow
    # =====================================================

    st.subheader("🔄 AirPulse AI Workflow")
    st.markdown("""

    **1️⃣ Data Collection**

    Global Air Quality Dataset

    **2️⃣ Exploratory Data Analysis**

    Understanding pollution patterns

    **3️⃣ Data Preprocessing**

    Cleaning, Encoding & Scaling

    **4️⃣ Machine Learning**

    Regression & Classification Models

    **5️⃣ Advanced Modeling**

    XGBoost Implementation

    **6️⃣ Dashboard Development**

    Interactive Air Quality Monitoring

    """)

    st.markdown("---")

    # =====================================================
    # Dataset Preview
    # =====================================================

    with st.expander("📋 Dataset Preview (Click to View Top 10 Records)"):
        st.write("### Top 10 Records")
        st.dataframe(
            original_df.head(10),
            use_container_width=True,
            hide_index=True
            )

    # =====================================================
    # Dataset Information
    # =====================================================

    with st.expander("📌 Dataset Information"):
        st.write("### Shape")
        st.write(original_df.shape)
        st.write("### Columns")
        st.write(original_df.columns.tolist())
        st.write("### Data Types")
        st.dataframe(original_df.dtypes.astype(str))
        st.write("### Statistical Summary")
        st.dataframe(original_df.describe())

# ==========================================================
# Remaining Pages
# ==========================================================

elif page == "📊 Dashboard":
    st.title("📊 Dashboard")
    st.info("Coming in Phase 6.2")

elif page == "🤖 AQI Prediction":
    st.title("🤖 AQI Prediction")
    st.info("Coming in Phase 6.3")

elif page == "🏷 AQI Classification":
    st.title("🏷 AQI Classification")
    st.info("Coming in Phase 6.4")

elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    st.info("Coming in Phase 6.5")

elif page == "ℹ About":
    st.title("ℹ About")
    st.info("Coming in Phase 6.6")
