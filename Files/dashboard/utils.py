# ==========================================================
# 🌍 AirPulse AI
# Utility Functions
# Common helper functions used across all Streamlit pages
# ==========================================================

# ===========================
# Import Libraries
# ===========================

import streamlit as st
import pandas as pd
import joblib

# ==========================================================
# DATA LOADING FUNCTIONS
# ==========================================================

# ===========================
# Load Original Dataset
# ===========================

@st.cache_data
def load_original_dataset():
    df = pd.read_csv("../data/globalAirQuality.csv")
    return df


# ===========================
# Load Processed Dataset
# ===========================

@st.cache_data
def load_processed_dataset():
    df = pd.read_csv("../data/processed_air_quality.csv")
    return df


# ===========================
# Load Dashboard Dataset
# ===========================

@st.cache_data
def load_dashboard_dataset():
    df = pd.read_csv("../data/dashboard_data.csv")
    return df

# ==========================================================
# MODEL LOADING FUNCTIONS
# ==========================================================

# ===========================
# Load Regression Model
# ===========================

@st.cache_resource
def load_regressor():
    regressor = joblib.load(
        "../models/xgboost_regressor.pkl"
    )

    return regressor


# ===========================
# Load Classification Model
# ===========================

@st.cache_resource
def load_classifier():
    classifier = joblib.load(
        "../models/xgboost_classifier.pkl"
    )

    return classifier

# ==========================================================
# AQI HELPER FUNCTIONS
# ==========================================================

# ===========================
# AQI Category
# ===========================

def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy"
    else:
        return "Hazardous"


# ===========================
# AQI Color
# ===========================

def get_aqi_color(aqi):
    if aqi <= 50:
        return "green"
    elif aqi <= 100:
        return "orange"
    elif aqi <= 150:
        return "red"
    else:
        return "darkred"


# ==========================================================
# DATASET INFORMATION
# ==========================================================

def dataset_summary(df):
    return {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Countries": df["country"].nunique(),
        "Cities": df["city"].nunique(),
        "Average AQI": round(df["aqi"].mean(), 2)
    }


# ==========================================================
# FEATURE LIST
# ==========================================================

def get_feature_columns():
    return [
        "country",
        "city",
        "latitude",
        "longitude",
        "pm25",
        "pm10",
        "no2",
        "so2",
        "o3",
        "co",
        "temperature",
        "humidity",
        "wind_speed",
        "hour",
        "day",
        "month"
    ]

# ==========================================================
# PREPROCESSING OBJECT LOADING FUNCTIONS
# ==========================================================

# ===========================
# Load Country Encoder
# ===========================

@st.cache_resource
def load_country_encoder():
    encoder = joblib.load(
        "../models/country_encoder.pkl"
    )

    return encoder

# ===========================
# Load City Encoder
# ===========================

@st.cache_resource
def load_city_encoder():
    encoder = joblib.load(
        "../models/city_encoder.pkl"
    )

    return encoder


# ===========================
# Load Scaler
# ===========================

@st.cache_resource
def load_scaler():
    scaler = joblib.load(
        "../models/scaler.pkl"
    )

    return scaler