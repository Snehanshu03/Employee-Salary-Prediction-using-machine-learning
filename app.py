import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv('Salary Data.csv').dropna()

# Sidebar: Collect user inputs
st.sidebar.header("Enter Employee Details")

def get_unique_sorted(col):
    return sorted(df[col].dropna().unique())

age = st.sidebar.slider("Age", int(df['Age'].min()), int(df['Age'].max()), 30)
gender = st.sidebar.selectbox("Gender", get_unique_sorted('Gender'))
education = st.sidebar.selectbox("Education Level", get_unique_sorted('Education Level'))
job_title = st.sidebar.selectbox("Job Title", get_unique_sorted('Job Title'))
experience = st.sidebar.slider("Years of Experience", 0, int(df['Years of Experience'].max()), 3)

# Model selector (placeholder, for prediction routing)
model_option = st.sidebar.selectbox("Select Model", options=["Random Forest", "XGBoost", "Linear Regression"])

# Encode categorical data
label_encoders = {}
for col in ['Gender', 'Education Level', 'Job Title']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# User input formatting
def encode_input(row, fitted_encoders):
    row_encoded = row.copy()
    for col, le in fitted_encoders.items():
        row_encoded[col] = le.transform([row[col]])[0]
    return row_encoded

user_input = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "Education Level": education,
    "Job Title": job_title,
    "Years of Experience": experience
}])
user_input_encoded = encode_input(user_input, label_encoders)

# Prepare feature matrix
X = df.drop('Salary', axis=1)
y = df['Salary']

# Split test set and load trained model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = joblib.load("best_salary_predictor.pkl")
r2 = round(r2_score(y_test, model.predict(X_test)), 3)

# Prediction
salary_pred = model.predict(user_input_encoded)[0]

# Streamlit output
st.title("Employee Salary Predictor")
st.write(f"#### Prediction Based on Selected Model ({model_option})")
st.success(f"**Predicted Salary:** ${salary_pred:,.2f}")

# Show R2 Score
st.subheader("Model R2 Accuracy Score")
st.metric("R2 Score (Pretrained Model)", r2)

# Feature importance
st.subheader("Feature Importance")
if model_option in ["Random Forest", "XGBoost"]:
    import matplotlib.pyplot as plt
    import numpy as np
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        features = X.columns
        idx = np.argsort(importance)[::-1]
        plt.figure(figsize=(6, 3))
        plt.barh(features[idx], importance[idx])
        plt.xlabel("Importance")
        plt.tight_layout()
        st.pyplot(plt)
    else:
        st.write("Selected model does not support feature importances.")
else:
    st.write("Feature importance not available for Linear Regression.")

# Data preview section
with st.expander("See Sample of Training Data"):
    st.dataframe(df.head(10))
