# prep.py
# Data Preparation: loads the dataset from the repo, cleans it, splits it and saves the splits locally.
# The workflow hands the four split files to the next job as an artifact.
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
TARGET = "ProdTaken"

df = pd.read_csv(DATA_PATH)
print("Raw shape:", df.shape)

# 1. Remove repeated customers (same CustomerID), then drop the columns that carry no predictive
#    information (row index and customer identifier). Rows that merely look alike once the ID is
#    removed are different customers, so they are kept.
df = df.drop_duplicates(subset="CustomerID")
df = df.drop(columns=["Unnamed: 0", "CustomerID"], errors="ignore")

# 2. Clean text columns: trim spaces and fix the inconsistent Gender label ("Fe Male" -> "Female")
text_cols = df.select_dtypes(include="object").columns
for col in text_cols:
    df[col] = df[col].str.strip()
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# 3. Remove rows without a target label
df = df.dropna(subset=[TARGET])
print("Clean shape:", df.shape)
print("Missing values left:", int(df.isna().sum().sum()), "(handled by the imputer inside the model pipeline)")

# 4. Split into features / target and then into train and test sets (stratified keeps the class ratio)
X = df.drop(columns=[TARGET])
y = df[TARGET]
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Save the splits locally (project root) so the workflow can pass them to the training job
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print(f"Train: {Xtrain.shape}  purchase rate {ytrain.mean():.2%}")
print(f"Test : {Xtest.shape}  purchase rate {ytest.mean():.2%}")
print("Saved: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")
