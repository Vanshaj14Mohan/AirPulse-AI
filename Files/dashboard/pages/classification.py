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

    st.title("🏷 AQI Classification")