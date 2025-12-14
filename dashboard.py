import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os


st.set_page_config(
    page_title="Transportation Analytics Dashboard",
    layout="wide"
)

st.title(" Transportation Analytics Dashboard")


DATA_PATH = r"C:\Users\admin\Desktop\Transportation_Analytics\data\cleaned\master_analytics.csv"
MODEL_PATH = r"D:\models\fuel_efficiency_model.pkl"


if not os.path.exists(DATA_PATH):
    st.error(" master_analytics.csv not found")
    st.stop()

df = pd.read_csv(DATA_PATH)
st.success(" Dataset loaded successfully")


if not os.path.exists(MODEL_PATH):
    st.error(" Model file not found. Run model_training.py")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
    st.success(" Model loaded successfully")
except Exception:
    st.error(" Model file corrupted. Re-run model_training.py")
    st.stop()


st.subheader("Dataset Overview")
st.dataframe(df.head())


st.subheader(" Fuel Efficiency per Vehicle")

fig1, ax1 = plt.subplots(figsize=(8, 4))
sns.barplot(
    x="Vehicle_ID",
    y="Fuel_Efficiency_km_per_L",
    data=df,
    ax=ax1
)
ax1.set_title("Fuel Efficiency per Vehicle")
st.pyplot(fig1)


st.subheader("Delivery Delay Distribution")

fig2, ax2 = plt.subplots(figsize=(8, 4))
sns.histplot(df["Delay_Minutes"], bins=10, kde=True, ax=ax2)
ax2.set_title("Delivery Delay Distribution")
st.pyplot(fig2)

st.subheader(" Correlation Heatmap")

numeric_df = df.select_dtypes(include=["int64", "float64"])
fig3, ax3 = plt.subplots(figsize=(8, 5))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax3)
ax3.set_title("Correlation Heatmap")
st.pyplot(fig3)


st.subheader(" Predict Fuel Efficiency")

distance = st.number_input("Distance (km)", min_value=1.0)
fuel_used = st.number_input("Fuel Used (litres)", min_value=0.1)
delay = st.number_input("Delay (minutes)", min_value=0.0)

if st.button("Predict"):
    try:
        input_data = [[distance, fuel_used, delay, 0, 0, 0, 0, 0, 0]]
        prediction = model.predict(input_data)
        st.success(f" Predicted Fuel Efficiency: {prediction[0]:.2f} km/L")
    except Exception as e:
        st.error("Prediction failed. Check model features.")
