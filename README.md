# Your Salary, Predicted!
**Author:** Carlos Andrés Lincango.  
**Alias:** _Ustedcan_  

**EDX user's mame:** Carlos_Lincango.  
**Github user's name:** [Ustedcan.]  
**LinkedIn:** Carlos Andrés Lincango.

<a href="https://linkedin.com/in/carlos-andr%c3%a9s-lincango-2b5a60132/" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="carlos-andr%c3%a9s-lincango-2b5a60132/" height="30" width="40" /></a>


**Country:** Ecuador.  
**City:** Quito.  
**Date:** 2024-12-23

#### Video Demo: <URL https://youtu.be/7yWmAOPNjRc?si=aOxbeOQs6zmGWqDH>
## Description:
### Introduction:
This is a simple Machine Learning project that builds on the knowledge acquired in programming from CS50’s Introduction to Computer Science and applies it to my field of expertise, Data Science and Machine Learning. I hold a Bachelor of Business Administration and completed a Master’s Degree in Information Systems and Data Science, but I had never taken a programming course before.

Thanks to the knowledge acquired, I was able to improve the understanding of the logic of ML algorithms and their application and deployment to solve real-life problems.

The objective of this project was to develop a web application that uses a simple Machine Learning algorithm to predict a person's salary based on their years of experience, all framed within the programming fundamentals acquired in Harvard's CS50.

I am excited to present my final project for CS50’s Introduction to Computer Science. I extend my heartfelt gratitude to David Malan and the entire staff for their invaluable support and guidance.

## Project Structure:
```
PROYECTO FINAL/
│
├── __pycache__/
├── .venv/
├── images/
│
├── static/
│   ├── Salary_dataset_clean.csv
│   ├── Salary_dataset.csv
│   ├── salary_model.pkl
│   ├── style.css
│   ├── test_data.csv
│
├── templates/
│   └── layout.html
│
├── 1.1.1
├── app.py
├── preprocessing.py
├── requirements.txt
├── training.py

```
The project consists of a Flask application called `app.py,` which is essentially the web application where users can make salary predictions. It also includes preprocessing and training scripts for a simple linear regression model named `preprocessing.py` and `training.py`. A serialized pickle file called `salary_model.pkl` contains the trained model, which will be used for batch predictions. Additionally, there are a CSV file, `Salary_dataset.csv`, which is the dataset used to train the algorithm. It contains two variables: `Years of Experience`, which holds information about the number of years a person has worked, and `Salary`, which indicates the salary a person receives based on their years of experience.


## Workflow:

#### 1. Business understanding:
In this stage of the project, the problem to be solved was defined, the dataset to be used was decided, and the following stages that make up the project were defined.

The idea of the project was to develop a web application that allows its users to predict their salary based on their years of experience.

#### 2. Dependency Installation:
The libraries and frameworks to be used for developing the project were defined. Primarily, frameworks for Machine Learning such as Numpy, Pandas, Scikit Learn were used, along with the Flask framework for web application development, among others.

The specific details of each library can be found in the `requirements.txt` file.

```py
Flask==2.2.5               # Framework for the web application
Flask-WTF==1.1.1           # For handling forms in Flask
scikit-learn==1.3.1        # For machine learning algorithms
pandas==2.1.1              # For data manipulation and analysis
numpy==1.26.0              # Library for numerical operations
Jinja2==3.1.2              # Templating engine used by Flask
joblib==1.1.1              # For saving and loading machine learning models
```

#### 3. Preprocessing:
In the world of Data Science, there are several "steps" that must be followed before training an ML model. One of these steps is data preprocessing. In this step, an initial exploration of the data is performed, the type of data maintained by the dataset is examined, missing values are checked, and missing data is imputed.

The preprocessing applied to the dataset was as follows:  
- Missing values were checked.  

- The `Unnamed: 0` column, which contained information that did not contribute to the model, was removed. In fact, the dataset contained three variables, but the `Unnamed: 0` column functioned more as a unique identifier than as a predictor variable.  

- Once preprocessing was completed, a new .csv file called `Salary_dataset_clean.csv` was created, which was used to train the model.

<details>
    <summary>Click here to expand Python code</summary>

```py
import pandas as pd
import os

def main():
    """Loads, preprocesses, and saves a salary dataset."""

    # Define file paths using os.path for platform independence
    data_path = os.path.join("static", "Salary_dataset.csv") 
    output_path = os.path.join("static", "Salary_dataset_clean.csv")

    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        return  # Exit the function if the file isn't found

    if data.empty:
        print("Error: Dataset is empty")
        return

    print("Dataset loaded successfully.")

    # Display initial information
    print("\nFirst 10 rows:")
    print(data.head(10))

    print("\nData types:")
    print(data.dtypes)

    print("\nDataset shape:", data.shape)

    # Data preprocessing: Drop the 'Unnamed: 0' column
    try:
        dataset = data.drop(columns=['Unnamed: 0'])
        print("\nColumn 'Unnamed: 0' dropped successfully.")
    except KeyError:
        print("\nWarning: Column 'Unnamed: 0' not found. Skipping drop operation.")
        dataset = data.copy() 

    expected_columns = ['YearsExperience', 'Salary']
    if list(dataset.columns) == expected_columns: 
        print("Columns match expected format.")
    else:
        print("Warning: Columns do not match expected format.")
        print(f"Actual columns: {list(dataset.columns)}")

    print("\nDataset Columns:", dataset.columns)

    # Check for missing values
    nulos = dataset.isnull().sum()
    print("\nNumber of missing values:\n", nulos)

    # Save the cleaned dataset
    try:
        dataset.to_csv(output_path, index=False)
        print(f"\nDataset saved successfully to {output_path}")
    except Exception as e:
        print(f"Error saving dataset: {e}")


if __name__ == "__main__":
    main()
```
</details>

#### 4. Training:
Once preprocessing was completed, we trained the linear regression model with the cleaned data. The predictor variable was separated from the outcome variable, and the data was split into training and testing sets with an 80/20 ratio, 80% of the data was used to train the model, and 20% was used to test it. The evaluation metrics were then printed.

```py
print("\nModel evaluation:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
```
#####  **Model Evaluation Metrics:**

```py
Model evaluation:
Mean Squared Error: 49830096.86
R^2 Score: 0.90
Mean Absolute Error: 6286.45
```
To enable the application to make new predictions based on new information entered by users, it is necessary to store the trained model information in a pickle file that will contain the serialized model, i.e., a binary representation of the model. This is done through a .pkl file and the joblib library.

The output of this process is the trained model in .pkl format named `salary_model.pkl`

```py
# Save the model
    model_path = os.path.join("static", "salary_model.pkl")
    dump(model, model_path)
    print(f"\nModel saved successfully to {model_path}")
```

<details>
    <summary>Click here to expand Python code</summary>

```py
import numpy as np
import pandas as pd
import os
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def main():
    """Loads, trains, and evaluates a salary dataset."""
    
    # Define file paths using os.path for platform independence
    data_path = os.path.join("static", "Salary_dataset_clean.csv")
    
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        return  # Exit the function if the file isn't found
    
    if data.empty:
        print("Error: Dataset is empty")
        return
    
    print("Dataset loaded successfully.")
    
    # Split the data into features and target
    X = data['YearsExperience'].values.reshape(-1, 1)
    y = data['Salary']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print("\nModel evaluation:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    
    # Save the model
    model_path = os.path.join("static", "salary_model.pkl")
    dump(model, model_path)
    print(f"\nModel saved successfully to {model_path}")
    
    # Save the test data
    test_data = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    test_data_path = os.path.join("static", "test_data.csv")
    test_data.to_csv(test_data_path, index=False)
    print(f"\nTest data saved successfully to {test_data_path}")


if __name__ == "__main__":
    main()
```
</details>

#### 5. Layout and Visual Structure Design:
For the frontend of the application, a basic interface was designed that contains the name of the application, an input box that only accepts a numeric value representing the person's years of experience, ranging from 0 to 100 years.

A for loop in Jinja is applied to generate dynamic rendering based on the user's input and the prediction made by the model.

```HTML
{% if prediction is not none %}
        <div class="result">
            <h2>Result:</h2>
            <p>With {{ request.form['experience'] }} years of experience, the estimated salary is: ${{ prediction }} per year</p>
        </div>
        {% endif %}
```

<details>
    <summary>Click here to expand CSS</summary>

``` css
/*This code was adapted from Chat Gpt*/
/* body */
body {
    font-family: 'Arial', sans-serif;
    line-height: 1.6;
    color: #333;
    background-color: #f8f9fa;

    /* Flexbox */
    display: flex;                
    justify-content: center;      
    align-items: center;          
    height: 100vh;                
    margin: 0;                    
}

/* container */
.container {
    background-color: #fff;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    text-align: center;
}

/* Header */
h1 {
    color: #007bff;
    font-size: 24px;
    margin-bottom: 10px;
}

/* Form */
form {
    margin: 20px 0;
}

input[type="number"] {
    padding: 10px;
    margin: 10px 0;
    border: 1px solid #ccc;
    border-radius: 5px;
    width: 100%;
    max-width: 300px;
    font-size: 16px;
    box-sizing: border-box;
}

button {
    background-color: #007bff;
    color: #fff;
    padding: 10px 20px;
    border: none;
    border-radius: 5px;
    font-size: 16px;
    cursor: pointer;
    transition: background-color 0.3s ease;
}

button:hover {
    background-color: #0056b3;
}

/* result */
.result {
    margin-top: 20px;
    padding: 15px;
    background-color: #e9ecef;
    border-radius: 5px;
    color: #212529;
    text-align: left;
    font-size: 18px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

/* Responsive */
@media (max-width: 600px) {
    body {
        padding: 10px;
    }

    input[type="number"],
    button {
        width: 100%;
    }
}
```
</details>

The design approach was to provide a simple, user-centered interface that is also aesthetically pleasing. A responsive structure was used to ensure the project is functional and accessible on any device.

#### 6. Application Development:
For the development of the web application, the POST request type was considered on the '/' route becuase sensitive information (such as years of experience) is being sent from the client to the server. POST allows this data to be sent in the body of the request, which maintains the privacy of the information and prevents the parameters from being visible in the URL, as is the case with GET. This approach is more suitable when processing form data, especially when making predictions or handling information that should not be exposed in the URL for security reasons.

```py
 if request.method == "POST":
```
When a user submits the form, Flask receives a POST request with the entered work experience data. In the block `if request.method == "POST"`, Flask takes the value of `experience`, converts it to a number, and passes it to the Machine Learning model to make the prediction. The prediction result is rounded and passed to the `layout.html` template, which then displays it to the user. This allows the server to process the data and return the result without exposing sensitive data in the URL.


```py
 experience = float(request.form["experience"])
            prediction = model.predict(np.array([experience]).reshape(-1, 1))[0]
```

```py
return render_template('layout.html', prediction=prediction)
```



<details>
    <summary>Click here to expand Python code</summary>

```py
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
```
</details>

And this is it — the conclusion of the project. Below are images showcasing how the web application works and the results achieved from developing this project.

### Main Page
This is the main page of the application. Users can view a brief description of the system's purpose and a form to input their years of experience.

![Main page of the application](./images/Screenshot%202024-12-23%20214557.png)

### User Input
The form where the user enters their years of work experience. This numeric input field ensures a valid number is entered for the salary prediction process.

![User input form](./images/Screenshot%202024-12-23%20215526.png)

### Prediction
Once the user enters their experience, the model predicts the estimated salary and displays it in this section.

![Salary prediction result](./images/Screenshot%202024-12-23%20215741.png)




<!--Links de referencia-->
[Ustedcan.]: https://github.com/Ustedcan
[Carlos Andrés Lincango.]: https://www.linkedin.com/in/carlos-andr%C3%A9s-lincango-2b5a60132/