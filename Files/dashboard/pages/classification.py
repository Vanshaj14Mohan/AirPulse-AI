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

    # ======================================================
    # LOCATION INFORMATION
    # ======================================================

    st.subheader("📍 Location Information")

    col1, col2 = st.columns(2)


    # ======================================================
    # Country Selection
    # ======================================================

    with col1:

        countries = sorted(
            original_df["country"].dropna().unique()
        )

        selected_country = st.selectbox(
            "🌍 Select Country",
            countries
        )

    # ======================================================
    # City Selection
    # ======================================================

    with col2:

        cities = sorted(
            original_df[
                original_df["country"] == selected_country
            ]["city"].dropna().unique()
        )

        selected_city = st.selectbox(
            "🏙 Select City",
            cities
        )

    # ======================================================
    # GET LOCATION COORDINATES
    # ======================================================

    location_data = original_df[
        (original_df["country"] == selected_country) &
        (original_df["city"] == selected_city)
    ]

    latitude = location_data["latitude"].iloc[0]

    longitude = location_data["longitude"].iloc[0]


    st.info(
        f"📍 Coordinates: Latitude = {latitude}, "
        f"Longitude = {longitude}"
    )

    st.markdown("---")

    # ======================================================
    # POLLUTION PARAMETERS
    # ======================================================

    st.subheader("🌫 Pollution Parameters")

    col1, col2, col3 = st.columns(3)


    # PM2.5
    with col1:

        pm25 = st.number_input(
            "PM2.5",
            min_value=0.0,
            value=30.0,
            step=0.1
        )


    # PM10
    with col2:

        pm10 = st.number_input(
            "PM10",
            min_value=0.0,
            value=60.0,
            step=0.1
        )


    # NO2
    with col3:

        no2 = st.number_input(
            "NO₂",
            min_value=0.0,
            value=30.0,
            step=0.1
        )

    # ======================================================
    # Remaining Pollution Parameters
    # ======================================================

    col1, col2, col3 = st.columns(3)


    # SO2
    with col1:

        so2 = st.number_input(
            "SO₂",
            min_value=0.0,
            value=5.0,
            step=0.1
        )


    # O3
    with col2:

        o3 = st.number_input(
            "O₃",
            min_value=0.0,
            value=30.0,
            step=0.1
        )


    # CO
    with col3:

        co = st.number_input(
            "CO",
            min_value=0.0,
            value=1.0,
            step=0.01
        )

    st.markdown("---")

    # ======================================================
    # WEATHER PARAMETERS
    # ======================================================

    st.subheader("🌤 Weather Parameters")

    col1, col2, col3 = st.columns(3)


    # Temperature
    with col1:

        temperature = st.number_input(
            "🌡 Temperature (°C)",
            value=20.0,
            step=0.1
        )


    # Humidity
    with col2:

        humidity = st.number_input(
            "💧 Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            step=0.1
        )


    # Wind Speed
    with col3:

        wind_speed = st.number_input(
            "💨 Wind Speed",
            min_value=0.0,
            value=5.0,
            step=0.1
        )


    st.markdown("---")

    # ======================================================
    # TIME INFORMATION
    # ======================================================

    st.subheader("🕒 Time Information")

    col1, col2, col3 = st.columns(3)


    # Hour
    with col1:

        hour = st.slider(
            "🕐 Hour",
            min_value=0,
            max_value=23,
            value=12
        )


    # Day
    with col2:

        day = st.slider(
            "📅 Day",
            min_value=1,
            max_value=31,
            value=15
        )


    # Month
    with col3:

        month = st.slider(
            "🗓 Month",
            min_value=1,
            max_value=12,
            value=6
        )


    st.markdown("---")

    # ======================================================
    # CLASSIFICATION BUTTON
    # ======================================================

    classify_button = st.button(
        "🏷 Classify AQI",
        type="primary",
        use_container_width=True
    )


    # ======================================================
    # AQI CLASSIFICATION
    # ======================================================

    if classify_button:

        try:

            # ==================================================
            # Encode Country and City
            # ==================================================

            country_encoded = country_encoder.transform(
                [selected_country]
            )[0]

            city_encoded = city_encoder.transform(
                [selected_city]
            )[0]


            # ==================================================
            # Create Input DataFrame
            # ==================================================

            input_data = pd.DataFrame({

                "country": [country_encoded],

                "city": [city_encoded],

                "latitude": [latitude],

                "longitude": [longitude],

                "pm25": [pm25],

                "pm10": [pm10],

                "no2": [no2],

                "so2": [so2],

                "o3": [o3],

                "co": [co],

                "temperature": [temperature],

                "humidity": [humidity],

                "wind_speed": [wind_speed],

                "hour": [hour],

                "day": [day],

                "month": [month]

            })


            # ==================================================
            # Apply Same Scaling Used During Training
            # ==================================================

            input_scaled = scaler.transform(
                input_data
            )


            # ==================================================
            # Make Classification Prediction
            # ==================================================

            prediction = classifier.predict(
                input_scaled
            )[0]


            # ==================================================
            # Display Classification Result
            # ==================================================

            st.markdown("---")

            st.subheader("🎯 Classification Result")


            st.metric(
                "🏷 Predicted AQI Category",
                str(prediction)
            )


            # ==================================================
            # Category Message
            # ==================================================

            if str(prediction) == "Good":

                st.success(
                    "🟢 Air quality is considered good."
                )

            elif str(prediction) == "Moderate":

                st.info(
                    "🟡 Air quality is considered moderate."
                )

            elif str(prediction) == "Unhealthy":

                st.warning(
                    "🟠 Air quality is unhealthy."
                )

            elif str(prediction) == "Hazardous":

                st.error(
                    "⛔ Air quality is hazardous."
                )

            else:

                st.info(
                    f"Predicted classification: {prediction}"
                )


        except Exception as e:

            st.error(
                f"❌ Classification failed: {e}"
            )
