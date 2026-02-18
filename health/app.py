from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Create simple dataset
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

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    glucose = float(request.form["glucose"])
    bmi = float(request.form["bmi"])
    age = float(request.form["age"])

    prediction = model.predict([[glucose, bmi, age]])

    if prediction[0] == 1:
        result = "⚠ High Risk of Diabetes"
    else:
        result = "✅ Low Risk of Diabetes"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
