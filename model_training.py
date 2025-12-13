import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

print(" Fuel Efficiency Model Training Started...")


DATA_PATH = "data/cleaned/master_analytics.csv"
SAVE_DIR = "D:/models"  
os.makedirs(SAVE_DIR, exist_ok=True)
MODEL_PATH = os.path.join(SAVE_DIR, "fuel_efficiency_model.pkl")


df = pd.read_csv(DATA_PATH)


feature_cols = ["Distance_km", "Capacity", "Fuel_Consumed_L"]  
categorical_cols = ["Vehicle_Type", "Route"]

X_num = df[feature_cols]


encoder = OneHotEncoder(drop='first', dtype=int)
X_cat = encoder.fit_transform(df[categorical_cols]).toarray()


import numpy as np
X = np.hstack([X_num.values, X_cat])
y = df["Fuel_Efficiency_km_per_L"].values

print(f"Features used: {X.shape[1]}, Rows: {X.shape[0]}")


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)


y_pred = model.predict(X)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(" Model Trained Successfully!")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")


joblib.dump(model, MODEL_PATH)
print(f" Model saved as {MODEL_PATH}")
