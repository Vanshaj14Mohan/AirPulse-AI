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
    processed_df = load_processed_dataset()
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

    # st.info(
    #     "🚀 Phase 6.5.1 - Model Performance Setup Completed"
    # )
    # ======================================================
    # REGRESSION MODEL PERFORMANCE
    # ======================================================

    st.subheader("📊 Regression Model Performance")

    st.markdown("""
    The regression model predicts the numerical Air Quality Index (AQI).
    Performance is evaluated using MAE, RMSE and R² score.
    """)

    # ======================================================
    # Prepare Features and Target
    # ======================================================

    X = processed_df.drop("aqi", axis=1)
    y = processed_df["aqi"]

    # ======================================================
    # Train-Test Split
    # ======================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


    # ======================================================
    # Make Predictions
    # ======================================================

    y_pred = regressor.predict(X_test)

    # ======================================================
    # Calculate Evaluation Metrics
    # ======================================================

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    # ======================================================
    # Display Metrics
    # ======================================================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "MAE",
            f"{mae:.4f}"
        )

    with col2:
        st.metric(
            "RMSE",
            f"{rmse:.4f}"
        )

    with col3:
        st.metric(
            "R² Score",
            f"{r2:.4f}"
        )

    # ======================================================
    # CLASSIFICATION MODEL PERFORMANCE
    # ======================================================

    st.markdown("---")
    
    st.subheader("🏷 Classification Model Performance")

    st.markdown("""
    The classification model predicts the pollution severity
    category of the air quality using four AQI categories:
    Good, Moderate, Unhealthy and Hazardous.
    """)

    # ======================================================
    # Create AQI Categories
    # ======================================================

    def categorize_aqi(aqi):
        if aqi <= 50:
            return 0
        elif aqi <= 100:
            return 1
        elif aqi <= 150:
            return 2
        else:
            return 3

    # ======================================================
    # Prepare Features and Target
    # ======================================================

    X_class = processed_df.drop("aqi", axis=1)
    y_class = processed_df["aqi"].apply(categorize_aqi)

    # ======================================================
    # Train-Test Split
    # ======================================================

    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
        X_class,
        y_class,
        test_size=0.2,
        random_state=42
    )

    # ======================================================
    # Make Predictions
    # ======================================================

    y_pred_class = classifier.predict(X_test_class)

    # ======================================================
    # Calculate Classification Metrics
    # ======================================================

    accuracy = accuracy_score(
        y_test_class,
        y_pred_class
    )

    precision = precision_score(
        y_test_class,
        y_pred_class,
        average="weighted"
    )

    recall = recall_score(
        y_test_class,
        y_pred_class,
        average="weighted"
    )

    f1 = f1_score(
        y_test_class,
        y_pred_class,
        average="weighted"
    )

    # ======================================================
    # Display Classification Metrics
    # ======================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Accuracy",
            f"{accuracy:.4f}"
        )

    with col2:
        st.metric(
            "Precision",
            f"{precision:.4f}"
        )

    with col3:
        st.metric(
            "Recall",
            f"{recall:.4f}"
        )

    with col4:
        st.metric(
            "F1 Score",
            f"{f1:.4f}"
        )

    # ======================================================
    # CONFUSION MATRIX
    # ======================================================

    st.markdown("---")

    st.subheader("🔍 Confusion Matrix")

    st.markdown("""
    The confusion matrix shows how accurately the classification
    model predicts each AQI category and where misclassifications occur.
    """)


    # ======================================================
    # Calculate Confusion Matrix
    # ======================================================

    cm = confusion_matrix(
        y_test_class,
        y_pred_class
    )


    # ======================================================
    # AQI Category Labels
    # ======================================================

    category_labels = [
        "Good",
        "Moderate",
        "Unhealthy",
        "Hazardous"
    ]


    # ======================================================
    # Create Confusion Matrix DataFrame
    # ======================================================

    cm_df = pd.DataFrame(
        cm,
        index=category_labels,
        columns=category_labels
    )

    # ======================================================
    # Create Heatmap
    # ======================================================

    fig = px.imshow(
        cm_df,
        text_auto=True,
        labels=dict(
            x="Predicted Category",
            y="Actual Category",
            color="Number of Samples"
        ),
        x=category_labels,
        y=category_labels,
        title="AQI Classification Confusion Matrix"
    )


    fig.update_layout(
        xaxis_title="Predicted Category",
        yaxis_title="Actual Category"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )
