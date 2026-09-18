import streamlit as st
import plotly.express as px

from utils import load_dashboard_dataset

def show_dashboard():

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
    # Load Dashboard Dataset
    # ======================================================

    dashboard_df = load_dashboard_dataset()

    # ======================================================
    # Dashboard Filters
    # ======================================================

    st.subheader("🎛 Dashboard Filters")
    col1, col2, col3 = st.columns(3)

    # ----------------------------
    # Country Filter
    # ----------------------------

    with col1:

        countries = sorted(
            dashboard_df["country"].unique()
        )

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
            months = sorted(
                dashboard_df["month"].unique()
            )

            selected_month = st.selectbox(
                "📅 Select Month",
                ["All"] + list(months)
            )

        else:
            selected_month = "All"
            st.warning(
                "Month column not available."
            )

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

        st.warning(
            "No records found for the selected filters."
        )

        return

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