import streamlit as st

def show_about():
    # ======================================================
    # Page Title
    # ======================================================
    st.title("ℹ About AirPulse AI")

    st.write(
        """
        AirPulse AI is a machine learning-powered air quality analysis
        application designed to analyze air pollution data, predict the
        Air Quality Index (AQI), and classify air quality into different
        pollution severity categories.
        """
    )

    st.divider()

    # ======================================================
    # Project Overview
    # ======================================================

    st.header("🌍 Project Overview")

    st.write(
        """
        Air pollution is a major environmental concern that can affect
        human health and quality of life. AirPulse AI uses machine learning
        techniques to analyze air quality data and provide meaningful
        insights about pollution levels.

        The application combines data analysis, visualization, regression,
        and classification models into one interactive dashboard.
        """
    )

    st.divider()

    # ======================================================
    # Key Features
    # ======================================================

    st.header("✨ Key Features")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Data Analysis")

        st.write(
            """
            - Explore air quality data
            - Visualize pollution trends
            - Analyze important air quality metrics
            """
        )

        st.subheader("🎯 AQI Prediction")

        st.write(
            """
            - Predict numerical AQI values
            - Use trained machine learning models
            - Analyze prediction results
            """
        )

    with col2:
        st.subheader("🏷 AQI Classification")

        st.write(
            """
            - Classify air quality into categories
            - Good
            - Moderate
            - Unhealthy
            - Hazardous
            """
        )

        st.subheader("📈 Model Performance")

        st.write(
            """
            - Regression evaluation metrics
            - Classification evaluation metrics
            - Confusion matrix
            - Classification report
            """
        )

    st.divider()

    # ======================================================
    # Machine Learning Models
    # ======================================================

    st.header("🤖 Machine Learning Models")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Regression Model")

        st.write(
            """
            **XGBoost Regressor**

            Used to predict the numerical Air Quality Index (AQI).
            """
        )

    with col2:
        st.subheader("Classification Model")

        st.write(
            """
            **XGBoost Classifier**

            Used to classify air quality into different pollution
            severity categories.
            """
        )

    st.divider()

    # ======================================================
    # Technologies Used
    # ======================================================

    st.header("🛠 Technologies Used")

    st.write(
        """
        - Python
        - Streamlit
        - Pandas
        - NumPy
        - Scikit-learn
        - XGBoost
        - Plotly
        """
    )

    st.divider()

    # ======================================================
    # Project Highlights
    # ======================================================

    st.header("🚀 Project Highlights")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Regression Model",
            "XGBoost Regressor"
        )

    with col2:
        st.metric(
            "Classification Accuracy",
            "99.58%"
        )

    with col3:
        st.metric(
            "AQI Categories",
            "4"
        )

    st.divider()

    # ======================================================
    # Project Goal
    # ======================================================

    st.header("🎯 Project Goal")

    st.write(
        """
        The goal of AirPulse AI is to demonstrate how machine learning
        and data visualization can be used together to analyze air quality
        data and generate useful predictions and classifications.

        The project provides an interactive platform where users can
        explore air quality information and understand the performance
        of machine learning models.
        """
    )

    # ======================================================
    # Footer
    # ======================================================

    st.divider()

    st.caption(
        "AirPulse AI | Machine Learning Powered Air Quality Analysis"
    )