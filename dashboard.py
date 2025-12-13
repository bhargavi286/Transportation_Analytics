# dashboard.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------------------------
# Load Model & Data
# ---------------------------
MODEL_PATH = "D:/models/fuel_efficiency_model.pkl"
DATA_PATH = "data/cleaned/master_analytics.csv"

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

st.set_page_config(page_title="Transportation Analytics Dashboard", layout="wide")
st.title("🚛 Transportation Analytics & Fuel Efficiency Prediction")

# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.header("Predict Fuel Efficiency")

vehicle_id = st.sidebar.selectbox("Vehicle ID", df["Vehicle_ID"].unique())
route = st.sidebar.selectbox("Route", df["Route"].unique())
distance = st.sidebar.number_input("Distance (km)", min_value=0.0, value=10.0)
vehicle_type = st.sidebar.selectbox("Vehicle Type", df["Vehicle_Type"].unique())
capacity = st.sidebar.number_input("Vehicle Capacity", min_value=0, value=50)

if st.sidebar.button("Predict Fuel Efficiency"):
    # Prepare input for model
    input_df = pd.DataFrame({
        "Distance_km": [distance],
        "Capacity": [capacity],
        "Vehicle_ID": [vehicle_id],
        "Vehicle_Type": [vehicle_type],
        "Route": [route]
    })
    
    # One-hot encoding (must match training)
    cat_cols = ["Vehicle_Type", "Route"]
    input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)
    
    # Align columns with training data
    model_features = model.feature_names_in_  # sklearn 1.2+
    for col in model_features:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[model_features]
    
    prediction = model.predict(input_encoded)[0]
    st.success(f"Predicted Fuel Efficiency: **{prediction:.2f} km/L**")

# ---------------------------
# Data Exploration
# ---------------------------
st.header("📊 Dataset Overview")
st.dataframe(df.head())

st.subheader("Fuel Efficiency Distribution")
fig, ax = plt.subplots(figsize=(8,4))
sns.histplot(df["Fuel_Efficiency_km_per_L"], kde=True, ax=ax)
st.pyplot(fig)

st.subheader("Fuel Efficiency by Vehicle Type")
fig2, ax2 = plt.subplots(figsize=(8,4))
sns.boxplot(x="Vehicle_Type", y="Fuel_Efficiency_km_per_L", data=df, ax=ax2)
st.pyplot(fig2)

st.subheader("Fuel Efficiency vs Distance")
fig3, ax3 = plt.subplots(figsize=(8,4))
sns.scatterplot(x="Distance_km", y="Fuel_Efficiency_km_per_L", hue="Vehicle_Type", data=df, ax=ax3)
st.pyplot(fig3)
