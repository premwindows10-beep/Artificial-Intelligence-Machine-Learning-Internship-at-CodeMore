# ==============================
# 1. Import Libraries
# ==============================
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==============================
# 2. Load Dataset
# ==============================
df = pd.read_csv("Smartphone_Usage_And_Addiction_Analysis_7500_Rows (1).csv")

print("Dataset Loaded")
print("Shape:", df.shape)

# ==============================
# 3. Data Cleaning
# ==============================
# Drop unnecessary columns
df.drop(['transaction_id', 'user_id'], axis=1, inplace=True)

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# ==============================
# 4. Encode Categorical Variables
# ==============================
le = LabelEncoder()

for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# ==============================
# 5. Feature Engineering
# ==============================
df['total_usage'] = (
    df['social_media_hours'] +
    df['gaming_hours'] +
    df['work_study_hours']
)

# ==============================
# 6. Split Features & Target
# ==============================
X = df.drop('addicted_label', axis=1)
y = df['addicted_label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 7. Feature Scaling
# ==============================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# 8. Train Model
# ==============================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ==============================
# 9. Prediction & Evaluation
# ==============================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# ==============================
# 10. Final Output
# ==============================
print("\nTask Completed Successfully ✅")