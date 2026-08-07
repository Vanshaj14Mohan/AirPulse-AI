import pandas as pd

df = pd.read_csv("E:\\AirPulse AI\\Files\\data\\globalAirQuality.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month
df.drop('timestamp', axis=1, inplace=True)

def categorize_aqi(aqi):
    if aqi <= 50: return 0
    elif aqi <= 100: return 1
    elif aqi <= 150: return 2
    else: return 3
df['aqi_category'] = df['aqi'].apply(categorize_aqi)

df.to_csv("E:\\AirPulse AI\\Files\\data\\dashboard_data.csv", index=False)
print("dashboard_data.csv created successfully.")
