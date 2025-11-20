import pandas as pd
import numpy as np

# ----------------------------
# ORIGINAL WEATHER DATA
# ----------------------------
rows = 1000
time = pd.date_range("2024-01-01", periods=rows, freq="H")

temp = 25 + 7 * np.sin(np.linspace(0, 20 * np.pi, rows)) + np.random.normal(0, 1, rows)
humidity = 80 - (temp - 25) * 2 + np.random.normal(0, 3, rows)
wind_speed = np.random.uniform(1, 15, rows)
rainfall = np.random.exponential(0.3, rows)

risk_score = (temp - 27) / 10 + (humidity - 60) / 100
risk_level = pd.cut(risk_score, [-10, 0.1, 0.5, 10], labels=["low", "medium", "high"])

weather_df = pd.DataFrame({
    "timestamp": time,
    "temperature": temp.round(2),
    "humidity": humidity.round(2),
    "wind_speed": wind_speed.round(2),
    "rainfall_mm": rainfall.round(2),
    "risk_level": risk_level
})


# --------------------------------------
# UNIFIED SCHEMA (for all FL datasets)
# --------------------------------------
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

# Create unified frame with all features = 0
unified_df = pd.DataFrame(0, index=weather_df.index, columns=UNIFIED_COLUMNS)

# Fill available weather columns
unified_df["timestamp"] = weather_df["timestamp"]
unified_df["temperature"] = weather_df["temperature"]
unified_df["humidity"] = weather_df["humidity"]
unified_df["risk_level"] = weather_df["risk_level"]

# Save unified dataset
unified_df.to_csv("data/weather_B.csv", index=False)

print("Unified weather dataset created as weather_unified.csv!")
