from flask import Flask, render_template, request
from joblib import load
import numpy as np
import os

app = Flask(__name__)

# Define model path
model_path = os.path.join("static", "salary_model.pkl")
model = load(model_path)

@app.route('/', methods=["GET", "POST"])
def salary_prediction():
    prediction = None
    if request.method == "POST":
        try:
            experience = float(request.form["experience"])
            prediction = model.predict(np.array([experience]).reshape(-1, 1))[0]
            prediction = round(prediction, 2)
        except ValueError:
            return "Por favor ingresa un número válido"
    return render_template('layout.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)