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