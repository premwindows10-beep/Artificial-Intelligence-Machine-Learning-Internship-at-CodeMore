# ==============================
# 1. Import Libraries
# ==============================
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# 2. Load Dataset
# ==============================
df = pd.read_csv("Smartphone_Usage_And_Addiction_Analysis_7500_Rows (1).csv")

print("Dataset Loaded Successfully")
print("Shape:", df.shape)

# ==============================
# 3. Data Cleaning
# ==============================
# Drop unnecessary columns
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

# Encode categorical variables
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# ==============================
# 4. Feature Engineering
# ==============================
df['total_usage'] = (
    df['social_media_hours'] +
    df['gaming_hours'] +
    df['work_study_hours']
)

# ==============================
# 5. Split Features & Target
# ==============================
X = df.drop('addicted_label', axis=1)
y = df['addicted_label']

# ==============================
# 6. Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 7. Feature Scaling (IMPORTANT)
# ==============================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# 8. Model 1: Logistic Regression
# ==============================
model1 = LogisticRegression(max_iter=1000)
model1.fit(X_train, y_train)

y_pred1 = model1.predict(X_test)

print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, y_pred1))
print(classification_report(y_test, y_pred1))

# ==============================
# 9. Model 2: Decision Tree
# ==============================
model2 = DecisionTreeClassifier(random_state=42)
model2.fit(X_train, y_train)

y_pred2 = model2.predict(X_test)

print("\n--- Decision Tree ---")
print("Accuracy:", accuracy_score(y_test, y_pred2))
print(classification_report(y_test, y_pred2))

# ==============================
# 10. Model Tuning (Decision Tree)
# ==============================
model_tuned = DecisionTreeClassifier(max_depth=5, random_state=42)
model_tuned.fit(X_train, y_train)

y_pred_tuned = model_tuned.predict(X_test)

print("\n--- Tuned Decision Tree ---")
print("Accuracy:", accuracy_score(y_test, y_pred_tuned))

# ==============================
# 11. Overfitting Check
# ==============================
print("\n--- Overfitting Check ---")
print("Train Accuracy:", model_tuned.score(X_train, y_train))
print("Test Accuracy:", model_tuned.score(X_test, y_test))

# ==============================
# 12. Final Output
# ==============================
print("\nTask Completed Successfully ✅")