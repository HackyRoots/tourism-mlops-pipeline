# data_register.py
# Data Registration: validates the dataset stored in the repository and prints a summary.
# The CSV lives inside the GitHub repo (tourism_project/data), so no external dataset store is needed.
import sys
import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"

# Columns described in the data dictionary
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier", "Occupation", "Gender",
    "NumberOfPersonVisiting", "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips",
    "Passport", "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
    "PitchSatisfactionScore", "ProductPitched", "NumberOfFollowups", "DurationOfPitch",
]

df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded from {DATA_PATH}")

# ---- Validation checks ----
problems = []

missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
if missing_cols:
    problems.append(f"Missing expected columns: {missing_cols}")

if "ProdTaken" in df.columns and not set(df["ProdTaken"].dropna().unique()).issubset({0, 1}):
    problems.append("Target column ProdTaken must contain only 0 and 1")

if "CustomerID" in df.columns and df["CustomerID"].duplicated().any():
    problems.append("CustomerID contains duplicate values")

if problems:
    print("\nVALIDATION FAILED")
    for p in problems:
        print(" -", p)
    sys.exit(1)

print("\nVALIDATION PASSED: all expected columns are present, target is binary, CustomerID is unique")

# ---- Summary ----
extra_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]
print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Extra (unlisted) columns: {extra_cols}")
print("\nColumn types:")
print(df.dtypes.to_string())
print("\nMissing values per column:")
print(df.isna().sum().to_string())
print("\nTarget distribution (ProdTaken):")
print(df["ProdTaken"].value_counts().to_string())
print(f"Purchase rate: {df['ProdTaken'].mean():.2%}")
print("\nFirst 5 rows:")
print(df.head().to_string())
print("\nDataset registered successfully.")
