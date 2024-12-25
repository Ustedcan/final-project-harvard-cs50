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