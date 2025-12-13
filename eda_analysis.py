import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


cleaned_path = r"C:\Users\admin\Desktop\Transportation_Analytics\data\cleaned"
master_file = os.path.join(cleaned_path, "master_analytics.csv")


df = pd.read_csv(master_file)


plt.figure(figsize=(8,5))
sns.barplot(x="Vehicle_ID", y="Fuel_Efficiency_km_per_L", data=df)
plt.title("Fuel Efficiency per Vehicle")
plt.savefig(os.path.join(cleaned_path,"fuel_efficiency_per_vehicle.png"))
plt.close()


plt.figure(figsize=(8,5))
sns.histplot(df["Delay_Minutes"], bins=10, kde=True)
plt.title("Delivery Delay Distribution")
plt.savefig(os.path.join(cleaned_path,"delay_distribution.png"))
plt.close()


numeric_df = df.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig(os.path.join(cleaned_path,"correlation_heatmap.png"))
plt.close()

print(" All EDA charts created successfully and saved inside /data/cleaned/")
