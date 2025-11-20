import pandas as pd
import numpy as np

# ------------------------------
# ORIGINAL WEARABLE DATA
# ------------------------------
rows = 1000
time = pd.date_range("2024-01-01", periods=rows, freq="min")

heart_rate = np.random.normal(75, 10, rows)
body_temp = np.random.normal(37, 0.4, rows)
steps = np.random.randint(0, 30, rows)

calories = steps * np.random.uniform(0.03, 0.06)  # not needed for unified schema

risk_score = (
    (heart_rate - 60) / 80 +
    (body_temp - 36.5) * 2
)

risk_level = pd.cut(
    risk_score,
    bins=[-10, 0.2, 0.6, 10],
    labels=["low", "medium", "high"]
)

wearable_df = pd.DataFrame({
    "timestamp": time,
    "heart_rate": heart_rate.astype(int),
    "body_temp": body_temp.round(2),
    "steps": steps,
    "calories_burned": calories.round(2),
    "risk_level": risk_level
})


# ----------------------------------------
# UNIFIED SCHEMA COLUMNS
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

# Create unified empty DF with all columns = 0
unified_df = pd.DataFrame(0, index=wearable_df.index, columns=UNIFIED_COLUMNS)

# Fill supported columns from wearable data
unified_df["timestamp"] = wearable_df["timestamp"]
unified_df["heart_rate"] = wearable_df["heart_rate"]
unified_df["steps"] = wearable_df["steps"]
unified_df["body_temp"] = wearable_df["body_temp"]
unified_df["risk_level"] = wearable_df["risk_level"]

# Save unified dataset
unified_df.to_csv("data/wearable_B.csv", index=False)

print("Unified wearable dataset created as wearable_unified.csv!")
