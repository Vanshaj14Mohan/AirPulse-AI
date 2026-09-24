import pandas as pd

country_map = {
    'US': 'United States', 'GB': 'United Kingdom', 'FR': 'France', 'DE': 'Germany', 
    'ES': 'Spain', 'IT': 'Italy', 'CA': 'Canada', 'MX': 'Mexico', 'BR': 'Brazil', 
    'AR': 'Argentina', 'ZA': 'South Africa', 'EG': 'Egypt', 'KE': 'Kenya', 
    'NG': 'Nigeria', 'AE': 'United Arab Emirates', 'SA': 'Saudi Arabia', 
    'QA': 'Qatar', 'IN': 'India', 'JP': 'Japan', 'KR': 'South Korea', 
    'CN': 'China', 'HK': 'Hong Kong', 'SG': 'Singapore', 'TH': 'Thailand', 
    'MY': 'Malaysia', 'ID': 'Indonesia', 'AU': 'Australia', 'NZ': 'New Zealand', 
    'RU': 'Russia', 'TR': 'Turkey', 'IR': 'Iran', 'PK': 'Pakistan', 
    'PH': 'Philippines', 'VN': 'Vietnam', 'PL': 'Poland', 'SE': 'Sweden', 
    'FI': 'Finland', 'CH': 'Switzerland'
}

# Update dashboard data only, to avoid breaking ML LabelEncoders
df = pd.read_csv("E:\\AirPulse AI\\Files\\data\\dashboard_data.csv")
df['country'] = df['country'].map(country_map).fillna(df['country'])
df.to_csv("E:\\AirPulse AI\\Files\\data\\dashboard_data.csv", index=False)
print("Updated dashboard_data.csv")
