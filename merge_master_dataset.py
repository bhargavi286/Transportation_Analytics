import pandas as pd
import os


cleaned_path = r"C:\Users\admin\Desktop\Transportation_Analytics\data\cleaned"
master_file = os.path.join(cleaned_path, "master_analytics.csv")

vehicles = pd.read_csv(os.path.join(cleaned_path, "vehicles_clean.csv"))
drivers = pd.read_csv(os.path.join(cleaned_path, "drivers_clean.csv"))
gps_routes = pd.read_csv(os.path.join(cleaned_path, "gps_routes_clean.csv"))
fuel_logs = pd.read_csv(os.path.join(cleaned_path, "fuel_logs_clean.csv"))
delivery_logs = pd.read_csv(os.path.join(cleaned_path, "delivery_logs_clean.csv"))
maintenance = pd.read_csv(os.path.join(cleaned_path, "maintenance_clean.csv"))


master = gps_routes.merge(vehicles, on="Vehicle_ID", how="left")
master = master.merge(fuel_logs, on="Trip_ID", how="left")
master = master.merge(delivery_logs, on="Trip_ID", how="left")
master = master.merge(maintenance, on="Vehicle_ID", how="left")


master["Driver_ID"] = ["D001","D002","D003","D004","D005"]


master.to_csv(master_file, index=False)
print(" Master dataset created successfully!")
