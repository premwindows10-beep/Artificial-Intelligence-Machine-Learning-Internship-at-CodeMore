# ==============================
# 1. Import Libraries
# ==============================
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ==============================
# 2. Load Dataset
# ==============================
df = pd.read_csv("Smartphone_Usage_And_Addiction_Analysis_7500_Rows (1).csv")

print("Columns:\n", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())

# ==============================
# 3. Drop Irrelevant Columns
# ==============================
for col in ['transaction_id', 'user_id']:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# ==============================
# 4. Handle Missing Values
# ==============================
# Numeric columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

# Categorical columns
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# ==============================
# 5. Encode Categorical Variables
# ==============================
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# ==============================
# 6. Feature Engineering
# ==============================
df['total_usage'] = (
    df['social_media_hours'] +
    df['gaming_hours'] +
    df['work_study_hours']
)

# ==============================
# 7. Feature Scaling (IMPORTANT FIX)
# ==============================
scaler = StandardScaler()

# Exclude target column from scaling
target_col = 'addicted_label'

feature_cols = df.drop(columns=[target_col]).columns
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# ==============================
# 8. Final Checks
# ==============================
print("\nProcessed Data:\n", df.head())

print("\nShape of dataset:", df.shape)

print("\nMissing values:\n", df.isnull().sum())

print("\nTask Completed Successfully ✅")
