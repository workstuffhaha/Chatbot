import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer

# Load data
data = pd.read_csv(r"C:\Users\Prachee\Desktop\Projects\Chatbot\Dataset\heart.csv")


# Identify categorical and numerical columns
categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

data = data.drop('FastingBS', axis=1)  # Drop as done previously
data = data.drop('ST_Slope_Flat', axis=1, errors='ignore')  # Drop as done previously

# Define transformations
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols)
])

# Split data
X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']

print(X.shape)
print(y.shape)

print("this is for X")
print(X.head())
print("this is for y")
print(y.head())
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Fit transformer
X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)
X_test = preprocessor.transform(X_test)

# Train Logistic Regression model
log_reg_model = LogisticRegression(max_iter=1000)
kf = KFold(n_splits=6, random_state=42, shuffle=True)
cv_results = cross_val_score(log_reg_model, X_train, y_train, cv=kf)
print(f"Logistic Regression Cross-validation Mean Accuracy: {np.round(cv_results.mean(), 4)}")

log_reg_model.fit(X_train, y_train)
print(f"Logistic Regression Train Score: {log_reg_model.score(X_train, y_train)}")
print(f"Logistic Regression Test Score: {log_reg_model.score(X_test, y_test)}")

# Save model and preprocessor
with open("model1.pkl", "wb") as model_file:
    pickle.dump(log_reg_model, model_file)

with open("preprocessor1.pkl", "wb") as preprocessor_file:
    pickle.dump(preprocessor, preprocessor_file)

# Define function for app.py
def predict_heart_disease(user_input):
    user_input = np.array(user_input).reshape(1, -1)
    transformed_input = preprocessor.transform(user_input)
    prediction = log_reg_model.predict(transformed_input)
    return "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"
