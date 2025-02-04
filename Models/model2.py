import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Read the dataset
df = pd.read_csv(r"C:\Users\Prachee\Desktop\Projects\Chatbot\Dataset\Student Depression Dataset.csv")

# Convert 'Dietary Habits' to numerical values
df['Dietary Habits'] = df['Dietary Habits'].replace({'Healthy': 1, 'Moderate': 2, 'Unhealthy': 3, 'Others': 4})

# Drop unnecessary columns
columns_to_drop = [
    'Profession', 'Academic Pressure', 'Work Pressure', 'Study Satisfaction', 
    'Job Satisfaction', 'Sleep Duration', 'Have you ever had suicidal thoughts ?', 
    'Financial Stress', 'Family History of Mental Illness', 'Depression'
]
df.drop(columns=columns_to_drop, inplace=True)

# Define features and target
X = df[['Age', 'CGPA', 'Dietary Habits', 'Work/Study Hours']]
y = df['Dietary Habits']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

# Feature scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train Logistic Regression model
lr = LogisticRegression(max_iter=200)
lr.fit(x_train_scaled, y_train)

# Save model and scaler
with open("student_depression_model.pkl", "wb") as model_file:
    pickle.dump(lr, model_file)

with open("student_depression_scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

# Accuracy report
accuracy = lr.score(x_test_scaled, y_test)
print("Accuracy on test data:", accuracy)

# Prediction function for UI
def predict_depression_risk(user_input):
    user_input = np.array(user_input).reshape(1, -1)
    transformed_input = scaler.transform(user_input)
    prediction = lr.predict(transformed_input)
    
    mapping = {1: "Healthy", 2: "Moderate", 3: "Unhealthy", 4: "Others"}
    return f"Dietary Habits Prediction: {mapping[prediction[0]]}"
