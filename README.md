# Employee Salary Prediction using Machine learning

A user-friendly Streamlit web application that predicts employee salaries based on demographic and professional attributes using a pre-trained machine learning model.

Problem Statement

Accurate prediction of employee salaries is a critical challenge faced by organizations and HR departments worldwide. Traditional compensation methods often rely on broad industry benchmarks or subjective judgment, which may overlook the complex interplay of demographic and professional factors.

This project addresses these limitations by leveraging machine learning to build a predictive system that accounts for key attributes such as:

Age

Gender

Education Level

Job Title

Years of Experience

By introducing data-driven objectivity, the system aims to:

Provide unbiased, fact-based salary predictions

Minimize reliance on manual assumptions and stereotypes

Streamline compensation planning, helping businesses make fair decisions and professionals understand their market value

 Predict salary by entering:
- Age
- Gender
- Education Level
- Job Title
- Years of Experience

---

## Demo

![App Preview]<img width="1915" height="1028" alt="sample-ui" src="https://github.com/user-attachments/assets/62d1396f-c51e-4ce9-b506-a9ec68404a85" />
  
![App Preview]<img width="1913" height="1024" alt="sample-ui_2" src="https://github.com/user-attachments/assets/d18b722f-f2b2-40e7-8201-87d85ac0d28b" />
  


---

## Project Overview

This project uses machine learning models such as **Random Forest Regressor** and **XGBoost** to train on employee salary data. Model training is handled by `Model training.py` to select and save the best performer.

The web app (`app.py`) lets users:
- Enter employee details
- Predict salary using the saved model `best_salary_predictor.pkl`
- View model R² accuracy
- See feature importance charts

---


---

## How to Run Locally

### 1. Clone the Repository


### 2. (Optional) Create and Activate a Virtual Environment


### 3. Install Dependencies


### 3. Install Dependencies
python -m venv venv
source venv/bin/activate # On Linux/Mac
venv\Scripts\activate.bat # On Windows


### 3. Install Dependencies

pip install -r requirements.txt


### 4. Launch the Streamlit App

streamlit run app.py


The app will open in your browser at `https://employee-salary-prediction-using-machine-learning-usdvhiqebant.streamlit.app/`.

---

## 🧪 Model Information

- **RandomForestRegressor** (best model, by default)
- Trained/tested using `train_test_split`
- R² Score used for evaluation

> The training script (`Model-training.py`) performs:
> - Data cleaning and encoding
> - Model training and evaluation
> - Model selection and saving (with `joblib`)

---

## ✅ Sample Prediction Features

| Feature               | Type         | Example            |
|-----------------------|--------------|--------------------|
| Age                   | Numeric      | `29`               |
| Gender                | Categorical  | `Male`             |
| Education Level       | Categorical  | `Master's Degree`  |
| Job Title             | Categorical  | `Data Analyst`     |
| Years of Experience   | Numeric      | `4`                |

---

## Feature Importance

For tree-based models, feature importances are displayed in a bar chart, available in the UI for Random Forest and XGBoost options.

![alt text](/images/Feature%20Importance.png)

---





