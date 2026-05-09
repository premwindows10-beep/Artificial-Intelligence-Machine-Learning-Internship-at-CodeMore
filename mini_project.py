import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

# =========================================
# Page Configuration
# =========================================
st.set_page_config(
    page_title="Mini AI Project",
    page_icon="🤖",
    layout="wide"
)

# =========================================
# Custom CSS for 3D UI
# =========================================
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(to right, #0f172a, #1e1b4b);
        color: white;
    }

    .title {
        text-align: center;
        font-size: 50px;
        font-weight: bold;
        color: cyan;
        margin-top: 20px;
        text-shadow: 3px 3px 15px rgba(0,255,255,0.6);
    }

    .subtitle {
        text-align: center;
        font-size: 22px;
        color: #d1d5db;
        margin-bottom: 40px;
    }

    .card {
        background: rgba(255,255,255,0.08);
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
        transition: 0.3s;
        margin-bottom: 25px;
    }

    .card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 15px 35px rgba(0,255,255,0.4);
    }

    .card-title {
        font-size: 28px;
        font-weight: bold;
        color: cyan;
        margin-bottom: 10px;
    }

    .footer {
        text-align: center;
        font-size: 20px;
        margin-top: 40px;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================
# Header
# =========================================
st.markdown('<div class="title">🤖 Mini AI Project</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">End-to-End Machine Learning Workflow using Smartphone Usage Dataset</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Developed by Prem Kumar Mahato</div>',
    unsafe_allow_html=True
)

# =========================================
# Load Dataset
# =========================================
try:
    df = pd.read_csv("Smartphone_Usage_And_Addiction_Analysis_7500_Rows (1).csv")

    st.success("Dataset Loaded Successfully ✅")

    # =========================================
    # Data Cleaning
    # =========================================
    for col in ['transaction_id', 'user_id']:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # Fill Missing Values
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns

    for col in num_cols:
        df[col] = df[col].fillna(df[col].mean())

    cat_cols = df.select_dtypes(include=['object']).columns

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Encode categorical values
    le = LabelEncoder()

    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # Feature Engineering
    df['total_usage'] = (
        df['social_media_hours'] +
        df['gaming_hours'] +
        df['work_study_hours']
    )

    # =========================================
    # Dataset Preview
    # =========================================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📂 Dataset Preview</div>', unsafe_allow_html=True)

    st.dataframe(df.head())

    st.markdown('</div>', unsafe_allow_html=True)

    # =========================================
    # Machine Learning Model
    # =========================================
    X = df.drop('addicted_label', axis=1)
    y = df['addicted_label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🤖 Machine Learning Model</div>', unsafe_allow_html=True)

    st.write("Model Used: Logistic Regression")
    st.write(f"Model Accuracy: {accuracy:.2f}")

    st.markdown('</div>', unsafe_allow_html=True)

    # =========================================
    # K-Means Clustering
    # =========================================
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

    clusters = kmeans.fit_predict(X_scaled)

    df['Cluster'] = clusters

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🧠 K-Means Clustering</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(8,5))

    ax.scatter(
        df['daily_screen_time_hours'],
        df['sleep_hours'],
        c=df['Cluster']
    )

    ax.set_xlabel("Daily Screen Time")
    ax.set_ylabel("Sleep Hours")
    ax.set_title("Cluster Visualization")

    st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)

    # =========================================
    # Insights
    # =========================================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📊 Project Insights</div>', unsafe_allow_html=True)

    st.write("• Users with higher screen time generally have lower sleep hours.")
    st.write("• Clustering helped identify different user behavior groups.")
    st.write("• Logistic Regression achieved good prediction accuracy.")

    st.markdown('</div>', unsafe_allow_html=True)

except FileNotFoundError:
    st.error("Dataset file not found. Please keep CSV file in the same folder as app.py")

# =========================================
# Footer
# =========================================
st.markdown(
    '<div class="footer">🚀 AI & Machine Learning Internship Project</div>',
    unsafe_allow_html=True
)