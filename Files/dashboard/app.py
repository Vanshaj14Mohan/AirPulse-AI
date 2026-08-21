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
    load_dashboard_dataset,
    load_regressor,
    load_classifier
)

from pages.prediction import show_prediction

# ===========================
# Load Dataset
# ===========================

original_df = load_original_dataset()
processed_df = load_processed_dataset()
dashboard_df = load_dashboard_dataset()

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
# Remaining Pages of the project
# ==========================================================

# ==========================================================
# DASHBOARD PAGE
# ==========================================================
# ==========================================================
# DASHBOARD PAGE
# ==========================================================

elif page == "📊 Dashboard":

    # ======================================================
    # Dashboard Header
    # ======================================================

    st.title("📊 AirPulse AI Dashboard")
    st.markdown("""
    ### Interactive Global Air Quality Analytics

    Explore worldwide air quality trends using interactive filters,
    visualizations, and environmental insights.
    """)

    st.markdown("---")

    # ======================================================
    # Dashboard Filters
    # ======================================================

    st.subheader("🎛 Dashboard Filters")
    col1, col2, col3 = st.columns(3)

    # ----------------------------
    # Country Filter
    # ----------------------------

    with col1:
        countries = sorted(dashboard_df["country"].unique())
        selected_country = st.selectbox(
            "🌍 Select Country",
            ["All"] + list(countries)
        )

    # ----------------------------
    # City Filter
    # ----------------------------

    with col2:
        if selected_country == "All":
            cities = sorted(
                dashboard_df["city"].unique()
            )
        else:
            cities = sorted(
                dashboard_df[
                    dashboard_df["country"] == selected_country
                ]["city"].unique()
            )

        selected_city = st.selectbox(
            "🏙 Select City",
            ["All"] + list(cities)
        )

    # ----------------------------
    # Month Filter
    # ----------------------------

    with col3:
        if "month" in dashboard_df.columns:
            months = sorted(dashboard_df["month"].unique())
            selected_month = st.selectbox(
                "📅 Select Month",
                ["All"] + list(months)
            )
        else:
            selected_month = "All"
            st.warning("Month column not available.")
    st.markdown("---")

    # ======================================================
    # Apply Filters
    # ======================================================

    filtered_df = dashboard_df.copy()

    if selected_country != "All":
        filtered_df = filtered_df[
            filtered_df["country"] == selected_country
        ]
    if selected_city != "All":
        filtered_df = filtered_df[
            filtered_df["city"] == selected_city
        ]
    if selected_month != "All":
        filtered_df = filtered_df[
            filtered_df["month"] == selected_month
        ]

    # ======================================================
    # No Data Check
    # ======================================================

    if filtered_df.empty:
        st.warning("No records found for the selected filters.")
    else:

        # ======================================================
        # KPI Cards
        # ======================================================

        st.subheader("📈 Dashboard Overview")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "🌍 Countries",
                filtered_df["country"].nunique()
            )

        with col2:
            st.metric(
                "🏙 Cities",
                filtered_df["city"].nunique()
            )

        with col3:
            st.metric(
                "📄 Total Records",
                f"{len(filtered_df):,}"
            )

        with col4:
            st.metric(
                "🌫 Average AQI",
                round(filtered_df["aqi"].mean(), 2)
            )

        st.markdown("---")

        # ======================================================
        # AQI Distribution
        # ======================================================

        st.subheader("📊 AQI Distribution")
        fig = px.histogram(
            filtered_df,
            x="aqi",
            nbins=40,
            title="Distribution of Air Quality Index",
            color_discrete_sequence=["royalblue"]
        )

        fig.update_layout(
            xaxis_title="AQI",
            yaxis_title="Frequency",
            template="plotly_white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown("---")

        # ======================================================
        # Top 10 Most Polluted Cities
        # ======================================================

        st.subheader("🏭 Top 10 Most Polluted Cities")
        city_df = (
            filtered_df
            .groupby("city")["aqi"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        fig = px.bar(
            city_df,
            x="city",
            y="aqi",
            color="aqi",
            color_continuous_scale="Reds",
            title="Top 10 Cities by Average AQI"
        )

        fig.update_layout(
            xaxis_title="City",
            yaxis_title="Average AQI",
            template="plotly_white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown("---")

        # ======================================================
        # Country-wise AQI
        # ======================================================

        st.subheader("🌍 Country-wise Average AQI")
        country_df = (
            filtered_df
            .groupby("country")["aqi"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )

        fig = px.bar(
            country_df,
            x="country",
            y="aqi",
            color="aqi",
            color_continuous_scale="Viridis",
            title="Average AQI by Country"
        )

        fig.update_layout(
            xaxis_title="Country",
            yaxis_title="Average AQI",
            template="plotly_white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.info("🚀 Phase 6.2 - Part 1 Completed")

# elif page == "🤖 AQI Prediction":
#     st.title("🤖 AQI Prediction")
#     st.info("Coming in Phase 6.3")

elif page == "🤖 AQI Prediction":
    show_prediction()

elif page == "🏷 AQI Classification":
    st.title("🏷 AQI Classification")
    st.info("Coming in Phase 6.4")

elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    st.info("Coming in Phase 6.5")

elif page == "ℹ About":
    st.title("ℹ About")
    st.info("Coming in Phase 6.6")
