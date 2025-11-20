import pandas as pd
import numpy as np

# --------------------------
# ORIGINAL HOSPITAL DATA
# --------------------------
rows = 1000
time = pd.date_range("2024-01-01", periods=rows, freq="H")

respiratory_cases = np.random.poisson(5, rows)
cardiac_cases = np.random.poisson(3, rows)
viral_infections = np.random.poisson(8, rows)

total_patients = respiratory_cases + cardiac_cases + viral_infections

risk_score = total_patients / 25
risk_level = pd.cut(risk_score, [-10, 0.3, 0.7, 10], labels=["low", "medium", "high"])

hospital_df = pd.DataFrame({
    "timestamp": time,
    "respiratory_cases": respiratory_cases,
    "cardiac_cases": cardiac_cases,
    "viral_infections": viral_infections,
    "total_patients": total_patients,
    "risk_level": risk_level
})

# ----------------------------------------
# UNIFIED SCHEMA COLUMNS (same for all FL nodes)
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
unified_df = pd.DataFrame(0, index=hospital_df.index, columns=UNIFIED_COLUMNS)

# Fill available columns from hospital dataset
unified_df["timestamp"] = hospital_df["timestamp"]
unified_df["respiratory_cases"] = hospital_df["respiratory_cases"]
unified_df["viral_infections"] = hospital_df["viral_infections"]
unified_df["risk_level"] = hospital_df["risk_level"]

# Save unified dataset
unified_df.to_csv("data/hospital_B.csv", index=False)

print("Unified hospital dataset created as hospital_unified.csv!")
