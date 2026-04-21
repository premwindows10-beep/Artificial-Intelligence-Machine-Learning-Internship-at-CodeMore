# Artificial-Intelligence-Machine-Learning-Internship-at-CodeMore
🎯 Objective

To perform data preprocessing and feature engineering to make the dataset suitable for machine learning models.

🛠️ Steps Performed
1.Data Loading
Loaded dataset using Pandas
Explored structure and column details
2.Data Cleaning
Removed irrelevant columns (transaction_id, user_id)
Handled missing values:
Numerical → filled with mean
Categorical → filled with mode
3.Encoding Categorical Variables
Converted text data (e.g., gender, stress level) into numerical form using Label Encoding
4.Feature Engineering
Created a new feature:
total_usage = social_media + gaming + work/study hours
5.Feature Scaling
Applied StandardScaler to normalize feature values
Excluded target column (addicted_label) from scaling
6.Final Validation
Checked dataset shape and structure
Ensured no missing values remain
