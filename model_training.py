import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print("Fuel Efficiency Model Training Started...")


DATA_PATH = os.path.join("data", "cleaned", "master_analytics.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "fuel_efficiency_model.pkl")


if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("master_analytics.csv not found")

df = pd.read_csv(DATA_PATH)


FEATURES = [
    "Distance_km",
    "Fuel_Consumed_L",
    "Delay_Minutes"
]

TARGET = "Fuel_Efficiency_km_per_L"


for col in FEATURES + [TARGET]:
    if col not in df.columns:
        raise ValueError(f"Missing column in dataset: {col}")

X = df[FEATURES]
y = df[TARGET]

print(f"Features used: {len(FEATURES)}, Rows: {len(df)}")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Trained Successfully!")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")


os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"Model saved at: {MODEL_PATH}")
