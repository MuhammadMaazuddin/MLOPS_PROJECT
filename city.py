import pandas as pd
import numpy as np

# --------------------------
# ORIGINAL CITY DATA CREATION
# --------------------------
rows = 1000
time = pd.date_range("2024-01-01", periods=rows, freq="H")

# Temperature pattern
temp = 25 + 5 * np.sin(np.linspace(0, 15 * np.pi, rows)) + np.random.normal(0, 1, rows)

# Pollution increases with temp + random spikes
pm25 = 40 + (temp - 25) * 2 + np.random.normal(0, 5, rows)
pm10 = pm25 + np.random.normal(3, 2, rows)
no2 = np.random.normal(30, 10, rows)
so2 = np.random.normal(12, 5, rows)

# Calculate AQI simplistically
aqi = (pm25 * 0.6) + (pm10 * 0.3) + (no2 * 0.1)

risk_level = pd.cut(
    aqi,
    [-10, 100, 200, 1000],
    labels=["low", "medium", "high"]
)

city_df = pd.DataFrame({
    "timestamp": time,
    "temperature": temp.round(2),
    "pm25": pm25.round(2),
    "pm10": pm10.round(2),
    "no2": no2.round(2),
    "so2": so2.round(2),
    "aqi": aqi.round(2),
    "risk_level": risk_level
})

# ----------------------------------------
# UNIFIED SCHEMA (for all FL nodes)
# ----------------------------------------
UNIFIED_COLUMNS = [
    "timestamp",
    "heart_rate",
    "steps",
    "body_temp",
    "temperature",
    "humidity",
    "pm25",
    "pm10",
    "respiratory_cases",
    "viral_infections",
    "co2_level",
    "noise_level",
    "risk_level"
]

# Create unified DF with all columns = 0
unified_df = pd.DataFrame(0, index=city_df.index, columns=UNIFIED_COLUMNS)

# Fill in the columns that exist in CITY dataset
unified_df["timestamp"] = city_df["timestamp"]
unified_df["temperature"] = city_df["temperature"]
unified_df["pm25"] = city_df["pm25"]
unified_df["pm10"] = city_df["pm10"]
unified_df["risk_level"] = city_df["risk_level"]

# Optional: drop timestamp during model training later  
# (keep it here for record keeping)
    
# Save unified CSV
unified_df.to_csv("data/city_B.csv", index=False)

print("Unified city dataset created as city_unified.csv!")
