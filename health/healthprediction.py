import pandas as pd
from sklearn.linear_model import LogisticRegression

# Step 1: Create simple health data
data = {
    "Glucose": [80, 85, 90, 120, 140, 150, 160, 170],
    "BMI": [22, 24, 26, 30, 32, 35, 36, 38],
    "Age": [25, 30, 35, 40, 45, 50, 55, 60],
    "Outcome": [0, 0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[["Glucose", "BMI", "Age"]]
y = df["Outcome"]

model = LogisticRegression()
model.fit(X, y)

glucose = float(input("Enter Glucose Level: "))
bmi = float(input("Enter BMI: "))
age = float(input("Enter Age: "))

prediction = model.predict([[glucose, bmi, age]])

if prediction[0] == 1:
    print("⚠ High Risk of Diabetes")
else:
    print("✅ Low Risk of Diabetes")
