import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="AirPulse AI",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 AirPulse AI")

st.subheader(
    "Intelligent Global Air Quality Monitoring & Prediction System"
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Dashboard",
        "AQI Prediction",
        "AQI Classification",
        "Model Performance",
        "About"
    ]
)

df = pd.read_csv("../data/processed_air_quality.csv")

if page == "Home":
    st.header("Welcome to AirPulse AI")
    st.write(
        """
        AirPulse AI is an intelligent air quality monitoring
        and prediction system developed using Machine Learning.

        This application provides:

        - AQI Prediction
        - AQI Classification
        - Interactive Visualizations
        - Air Quality Insights
        """
    )
    
st.subheader("Dataset Preview")
st.dataframe(df.head())

