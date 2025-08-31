import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# dataset
df = pd.read_csv("Salary Data.csv") 

# Drop missing or empty rows
df.dropna(inplace=True)

# Encode categorical variables
label_encoders = {}
for col in ['Gender', 'Education Level', 'Job Title']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Defining features and target
X = df.drop('Salary', axis=1)
y = df['Salary']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

models = {
    'RandomForest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42, objective='reg:squarederror'),
    'LinearRegression': LinearRegression()
}

results = {}
best_model_name = None
best_r2 = float('-inf')
best_model = None

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    results[name] = r2
    if r2 > best_r2:
        best_r2 = r2
        best_model_name = name
        best_model = model

#  Display Model Accuracy

print("\nModel Performance (R2 Score on Test Set):")
print("-" * 38)
print("{:<17}{}".format("Model", "R2 Score"))
print("-" * 38)
for name, r2 in results.items():
    print("{:<17}{:.3f}".format(name, r2))
print("-" * 38)

print(f"\nBest model ({best_model_name}) ")



# Saving the best model
import joblib

filename = 'best_salary_predictor.pkl'
joblib.dump(best_model, filename)
print(f"Best model ({best_model_name}) saved as {filename}")
