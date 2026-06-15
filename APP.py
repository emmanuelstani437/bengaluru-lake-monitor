import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# ==========================
# MODEL LOADING
# ==========================
lr_model = joblib.load("logistic_model.pkl")
lr_scaler = joblib.load("scaler.pkl")
kmeans_model = joblib.load("kmeans_model.pkl")
kmeans_scaler = joblib.load("kmeans_scaler.pkl")
lake_df = pd.read_csv("lake_app_dataset.csv")

# ==========================
# PAGE CONFIGURATION
# ==========================
st.set_page_config(
    page_title="Bengaluru Lake Health Monitoring",
    page_icon="🌊",
    layout="wide"
)

# Hero Image
try:
    image = Image.open("images/bellandur_lake.jpg")
    st.image(image, use_container_width=True)
except FileNotFoundError:
    st.warning("Hero image not found. Please ensure 'images/bellandur_lake.jpg' exists.")

# Main Title
st.title("🌊 Bengaluru Lake Health Monitoring System")

st.markdown("""
### AI-Powered Water Quality Assessment and Pollution Source Identification

Monitor Bengaluru's lakes, evaluate water quality,
identify pollution severity, and analyze dominant
pollution sources using Machine Learning.
""")

st.info(
    "This system combines Logistic Regression and K-Means Clustering "
    "to assess lake health and identify likely pollution contributors."
)

st.divider()

# ==========================
# NAVIGATION
# ==========================
if "page" not in st.session_state:
    st.session_state["page"] = ""

col1, col2 = st.columns(2)

with col1:
    st.subheader("🧪 Analyze New Water Sample")
    st.write("""
    Enter water quality parameters from a new sample
    and predict pollution severity.
    """)
    if st.button("Analyze New Sample"):
        st.session_state["page"] = "sample"
        st.rerun()

with col2:
    st.subheader("🌊 Check Your Beloved Lake")
    st.write("""
    Select a lake from Bengaluru and view its
    pollution profile.
    """)
    if st.button("Check Lake Health"):
        st.session_state["page"] = "lake"
        st.rerun()

st.divider()

# ==========================
# SAMPLE PAGE
# ==========================
if st.session_state["page"] == "sample":

    if st.button("⬅ Back to Home", key="back_sample"):
        st.session_state["page"] = ""
        st.rerun()

    st.header("🧪 Water Sample Analysis")

    try:
        analysis_image = Image.open("images/analysis.jpg")
        st.image(analysis_image, use_container_width=True)
    except FileNotFoundError:
        pass

    st.subheader("Enter Water Quality Parameters")

    col_input1, col_input2 = st.columns(2)

    with col_input1:
        conductivity = st.number_input(
            "Conductivity (µmho/cm)",
            min_value=142.0, max_value=3090.0, value=142.0,
            help="Measures water's ability to conduct electricity. Higher values may indicate dissolved salts and pollutants."
        )
        nitrate = st.number_input(
            "Nitrate (mg/L)",
            min_value=0.3, max_value=31.0, value=0.3,
            help="Represents nitrogen contamination from sewage, fertilizers and agricultural runoff."
        )
        ammonical_n = st.number_input(
            "Ammonical Nitrogen (mg/L)",
            min_value=0.2, max_value=200.0, value=0.2,
            help="Strong indicator of untreated sewage and organic waste contamination."
        )
        total_coliform = st.number_input(
            "Total Coliform (MPN/100ml)",
            min_value=14.0, max_value=49000000.0, value=14.0, format="%.0f",
            help="Indicates contamination from human and animal waste."
        )
        turbidity = st.number_input(
            "Turbidity (NTU)",
            min_value=1.0, max_value=212.0, value=1.0,
            help="Measures water clarity. High values indicate suspended particles and pollution."
        )
        cod = st.number_input(
            "Chemical Oxygen Demand - COD (mg/L)",
            min_value=10.0, max_value=450.0, value=10.0,
            help="Amount of oxygen required to chemically oxidize organic water contaminants."
        )

    with col_input2:
        chlorides = st.number_input(
            "Chlorides (mg/L)",
            min_value=12.0, max_value=570.0, value=12.0,
            help="Can indicate industrial discharge, sewage contamination and salinity."
        )
        tds = st.number_input(
            "Total Dissolved Solids - TDS (mg/L)",
            min_value=10.0, max_value=2156.0, value=10.0,
            help="Represents dissolved minerals, salts, metals and pollutants present in water."
        )
        total_hardness = st.number_input(
            "Total Hardness (mg/L)",
            min_value=36.0, max_value=996.0, value=36.0,
            help="Measures dissolved calcium and magnesium salts."
        )
        phosphate = st.number_input(
            "Phosphate (mg/L)",
            min_value=0.1, max_value=8.0, value=0.1,
            help="High phosphate levels promote algal blooms and eutrophication in lakes."
        )
        bod = st.number_input(
            "Biological Oxygen Demand - BOD (mg/L)",
            min_value=1.0, max_value=100.0, value=1.0,
            help="Amount of oxygen consumed by bacteria while decomposing organic matter."
        )

    st.write("---")

    if st.button("Analyze Sample", type="primary"):

        lr_sample = pd.DataFrame([[
            conductivity, nitrate, ammonical_n, total_coliform,
            turbidity, chlorides, tds, total_hardness, phosphate
        ]], columns=[
            "Conductivity", "Nitrate", "Ammonical_N", "Total_Coliform",
            "Turbidity", "Chlorides", "TDS", "Total_Hardness", "Phosphate"
        ])

        lr_scaled = lr_scaler.transform(lr_sample)
        pollution_prediction = lr_model.predict(lr_scaled)[0]

        kmeans_sample = pd.DataFrame([[
            cod, conductivity, tds, chlorides,
            nitrate, phosphate, bod, ammonical_n
        ]], columns=[
            "COD", "Conductivity", "TDS", "Chlorides",
            "Nitrate", "Phosphate", "BOD", "Ammonical_N"
        ])

        kmeans_scaled = kmeans_scaler.transform(kmeans_sample)
        cluster_prediction = kmeans_model.predict(kmeans_scaled)[0]

        st.subheader("Predicted Pollution Severity")
        st.success(f"Prediction: {pollution_prediction}")

        cluster_map = {
            0: "Chemical / Industrial Influence",
            1: "Mixed Pollution",
            2: "Relatively Healthy",
            3: "Extreme Pollution Hotspot"
        }

        pollution_source = cluster_map.get(cluster_prediction, "Unknown")

        st.subheader("Likely Dominant Pollution Source")
        st.info(pollution_source)

        if cluster_prediction == 0:
            sewage, industrial, nutrient = 25, 60, 15
        elif cluster_prediction == 1:
            sewage, industrial, nutrient = 45, 40, 15
        elif cluster_prediction == 2:
            sewage, industrial, nutrient = 20, 15, 65
        else:
            sewage, industrial, nutrient = 70, 20, 10

        st.subheader("Pollution Source Breakdown")

        st.progress(sewage / 100)
        st.write(f"🟤 Sewage Pollution: {sewage}%")

        st.progress(industrial / 100)
        st.write(f"🏭 Industrial Pollution: {industrial}%")

        st.progress(nutrient / 100)
        st.write(f"🌾 Nutrient Runoff: {nutrient}%")

# ==========================
# LAKE PAGE
# ==========================
elif st.session_state["page"] == "lake":

    if st.button("⬅ Back to Home", key="back_lake"):
        st.session_state["page"] = ""
        st.rerun()

    st.header("🌊 Lake Health Analysis")

    lake_names = sorted(lake_df["Lake_Name"].dropna().unique())
    selected_lake = st.selectbox("Select a Lake", lake_names)
    lake_data = lake_df[lake_df["Lake_Name"] == selected_lake]

    st.subheader(selected_lake)
    st.write(f"Number of samples available: {len(lake_data)}")

    avg_cod = pd.to_numeric(lake_data["COD"], errors="coerce").mean()
    avg_bod = pd.to_numeric(lake_data["BOD"], errors="coerce").mean()
    avg_nitrate = pd.to_numeric(lake_data["Nitrate"], errors="coerce").mean()
    avg_ammonical = pd.to_numeric(lake_data["Ammonical_N"], errors="coerce").mean()
    avg_conductivity = pd.to_numeric(lake_data["Conductivity"], errors="coerce").mean()
    avg_tds = pd.to_numeric(lake_data["TDS"], errors="coerce").mean()
    avg_chlorides = pd.to_numeric(lake_data["Chlorides"], errors="coerce").mean()
    avg_phosphate = pd.to_numeric(lake_data["Phosphate"], errors="coerce").mean()

    kmeans_input = pd.DataFrame([[
        avg_cod, avg_conductivity, avg_tds, avg_chlorides,
        avg_nitrate, avg_phosphate, avg_bod, avg_ammonical
    ]], columns=[
        "COD", "Conductivity", "TDS", "Chlorides",
        "Nitrate", "Phosphate", "BOD", "Ammonical_N"
    ])

    scaled = kmeans_scaler.transform(kmeans_input)
    cluster = kmeans_model.predict(scaled)[0]

    # FIX: All 4 clusters now correctly handled
    if cluster == 0:
        st.warning("🏭 Chemical / Industrial Influenced Lake")
    elif cluster == 1:
        st.error("🟤 Sewage Dominated Lake")
    elif cluster == 2:
        st.success("🟢 Healthy Lake")
    else:
        st.error("🚨 Extreme Pollution Hotspot")

    # ==========================
    # POLLUTION CONTRIBUTION
    # ==========================
    # FIX: This section is now correctly indented inside the 'lake' elif block

    sewage_score = avg_bod * 0.45 + avg_ammonical * 0.35 + avg_nitrate * 0.20
    industrial_score = avg_cod * 0.45 + avg_tds * 0.25 + avg_conductivity * 0.30
    other_score = avg_phosphate * 10
    total_score = sewage_score + industrial_score + other_score

    if total_score > 0:
        sewage_pct = round(sewage_score / total_score * 100, 1)
        industrial_pct = round(industrial_score / total_score * 100, 1)
        other_pct = round(other_score / total_score * 100, 1)
    else:
        sewage_pct = industrial_pct = other_pct = 0.0

    st.subheader("Pollution Source Breakdown")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🟤 Sewage Contribution", f"{sewage_pct}%")
        st.progress(sewage_pct / 100)

    with col2:
        st.metric("🏭 Industrial Contribution", f"{industrial_pct}%")
        st.progress(industrial_pct / 100)

    with col3:
        st.metric("🌾 Other Sources", f"{other_pct}%")
        st.progress(other_pct / 100)

    st.divider()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Average COD", f"{avg_cod:.2f}")
    m2.metric("Average BOD", f"{avg_bod:.2f}")
    m3.metric("Average Nitrate", f"{avg_nitrate:.2f}")
    m4.metric("Average Ammonical Nitrogen", f"{avg_ammonical:.2f}")