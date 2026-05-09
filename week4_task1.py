# =========================================
# 1. Import Required Libraries
# =========================================
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

# =========================================
# 2. Load Dataset
# =========================================
df = pd.read_csv("Smartphone_Usage_And_Addiction_Analysis_7500_Rows (1).csv")

print("Dataset Loaded Successfully ✅")
print("\nFirst 5 Rows:\n")
print(df.head())

# =========================================
# 3. Data Cleaning
# =========================================

# Remove unnecessary columns
for col in ['transaction_id', 'user_id']:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# Handle missing values

# Numeric columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

# Categorical columns
cat_cols = df.select_dtypes(include=['object']).columns

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After Cleaning:\n")
print(df.isnull().sum())

# =========================================
# 4. Encode Categorical Variables
# =========================================
le = LabelEncoder()

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# =========================================
# 5. Feature Engineering
# =========================================
df['total_usage'] = (
    df['social_media_hours'] +
    df['gaming_hours'] +
    df['work_study_hours']
)

# =========================================
# 6. Remove Target Column
# (Unsupervised Learning does not use labels)
# =========================================
if 'addicted_label' in df.columns:
    X = df.drop('addicted_label', axis=1)
else:
    X = df.copy()

# =========================================
# 7. Feature Scaling
# =========================================
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# =========================================
# 8. Elbow Method
# =========================================
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure(figsize=(7,5))
plt.plot(range(1, 11), wcss, marker='o')

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")

plt.show()

# =========================================
# 9. Apply K-Means Clustering
# =========================================
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to dataset
df['Cluster'] = clusters

print("\nCluster Counts:\n")
print(df['Cluster'].value_counts())

# =========================================
# 10. Visualize Clusters
# =========================================
plt.figure(figsize=(7,5))

plt.scatter(
    df['daily_screen_time_hours'],
    df['sleep_hours'],
    c=df['Cluster']
)

plt.xlabel("Daily Screen Time Hours")
plt.ylabel("Sleep Hours")

plt.title("K-Means Clustering")

plt.show()

# =========================================
# 11. Final Output
# =========================================
print("\nFirst 5 Rows with Cluster Labels:\n")
print(df.head())

print("\nTask Completed Successfully ✅")