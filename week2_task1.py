# ==============================
# 1. Import Libraries
# ==============================
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# 2. Load Dataset
# ==============================
df = pd.read_csv("Smartphone_Usage_And_Addiction_Analysis_7500_Rows (1).csv")

print("First 5 rows:\n", df.head())

# ==============================
# 3. Basic Information
# ==============================
print("\nShape:", df.shape)
print("\nInfo:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# ==============================
# 4. Data Cleaning
# ==============================
# Drop irrelevant columns
for col in ['transaction_id', 'user_id']:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# Handle missing values
# Numeric
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

# Categorical
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing values after cleaning:\n", df.isnull().sum())

# ==============================
# 5. Basic Analysis
# ==============================
print("\nGender Distribution:\n", df['gender'].value_counts())
print("\nStress Level Distribution:\n", df['stress_level'].value_counts())
print("\nAddiction Label Distribution:\n", df['addicted_label'].value_counts())

# ==============================
# 6. Visualization (EDA)
# ==============================

# Histogram - Screen Time
plt.figure()
plt.hist(df['daily_screen_time_hours'])
plt.title("Daily Screen Time Distribution")
plt.xlabel("Hours")
plt.ylabel("Frequency")
plt.show()

# Bar chart - Addiction
plt.figure()
df['addicted_label'].value_counts().plot(kind='bar')
plt.title("Addiction Count")
plt.xlabel("Addicted Label")
plt.ylabel("Count")
plt.show()

# Scatter Plot - Screen Time vs Sleep
plt.figure()
plt.scatter(df['daily_screen_time_hours'], df['sleep_hours'])
plt.xlabel("Screen Time")
plt.ylabel("Sleep Hours")
plt.title("Screen Time vs Sleep")
plt.show()

# ==============================
# 7. Correlation Analysis
# ==============================
corr = df.corr(numeric_only=True)

print("\nCorrelation Matrix:\n", corr)

# ==============================
# 8. Insights (Print Simple Observations)
# ==============================
print("\nBasic Insights:")
print("- Most users have moderate screen time.")
print("- Higher screen time may reduce sleep hours.")
print("- Addiction levels vary based on usage patterns.")

print("\nTask Completed Successfully ✅")