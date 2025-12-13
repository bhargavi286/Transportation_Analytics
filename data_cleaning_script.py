import pandas as pd
import os


raw_path = r"C:\Users\admin\Desktop\Transportation_Analytics\data\raw"
cleaned_path = r"C:\Users\admin\Desktop\Transportation_Analytics\data\cleaned"
os.makedirs(cleaned_path, exist_ok=True)


vehicles = pd.read_csv(os.path.join(raw_path, "vehicles.csv"))
vehicles.drop_duplicates(inplace=True)
vehicles.to_csv(os.path.join(cleaned_path, "vehicles_clean.csv"), index=False)


drivers = pd.read_csv(os.path.join(raw_path, "drivers.csv"))
drivers.drop_duplicates(inplace=True)
drivers.to_csv(os.path.join(cleaned_path, "drivers_clean.csv"), index=False)


gps_routes = pd.read_csv(os.path.join(raw_path, "gps_routes.csv"))
gps_routes.drop_duplicates(inplace=True)
gps_routes.to_csv(os.path.join(cleaned_path, "gps_routes_clean.csv"), index=False)


fuel_logs = pd.read_csv(os.path.join(raw_path, "fuel_logs.csv"))
fuel_logs.drop_duplicates(inplace=True)
fuel_logs.to_csv(os.path.join(cleaned_path, "fuel_logs_clean.csv"), index=False)


delivery_logs = pd.read_csv(os.path.join(raw_path, "delivery_logs.csv"))
delivery_logs.drop_duplicates(inplace=True)
delivery_logs.to_csv(os.path.join(cleaned_path, "delivery_logs_clean.csv"), index=False)


maintenance = pd.read_csv(os.path.join(raw_path, "maintenance.csv"))
maintenance.drop_duplicates(inplace=True)
maintenance.to_csv(os.path.join(cleaned_path, "maintenance_clean.csv"), index=False)

print(" All CSV files cleaned and saved to cleaned folder successfully!")
