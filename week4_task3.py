# =========================================
# 1. Import Required Libraries
# =========================================
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# =========================================
# 2. Load Dataset
# =========================================
df = pd.read_csv("Smartphone_Usage_And_Addiction_Analysis_7500_Rows (1).csv")

print("Dataset Loaded Successfully ✅")

# =========================================
# 3. Data Cleaning
# =========================================

# Remove unnecessary columns
for col in ['transaction_id', 'user_id']:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# Handle Missing Values

# Numeric columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

# Categorical columns
cat_cols = df.select_dtypes(include=['object']).columns

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

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
# 6. Prepare Features
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

print("\nData Scaling Completed ✅")

# =========================================
# 8. PCA (Principal Component Analysis)
# =========================================
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("\nPCA Shape:", X_pca.shape)

# =========================================
# 9. PCA Visualization
# =========================================
plt.figure(figsize=(7,5))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1]
)

plt.title("PCA Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.show()

# =========================================
# 10. t-SNE
# =========================================
tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30
)

X_tsne = tsne.fit_transform(X_scaled)

print("\nt-SNE Shape:", X_tsne.shape)

# =========================================
# 11. t-SNE Visualization
# =========================================
plt.figure(figsize=(7,5))

plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1]
)

plt.title("t-SNE Visualization")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")

plt.show()

# =========================================
# 12. Explained Variance (PCA)
# =========================================
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

# =========================================
# 13. Final Output
# =========================================
print("\nTask Completed Successfully ✅")