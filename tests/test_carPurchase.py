import read_carPurchase as rcp 
import time
import pandas as pd
import numpy as np


# Test case to ensure the correct loading of data
def test_load_data():
    data = rcp.loadData()
    startTime = time.time()
    assert isinstance(data, pd.DataFrame), "Loaded data should be a DataFrame."

    expected_columns = ['Customer Name', 'Customer e-mail', 'Country', 'Gender', 
                        'Annual Salary', 'Credit Card Debt', 'Net Worth', 'Car Purchase Amount']

    for col in expected_columns:
        assert col in data.columns, f"Expected column {col} not found in loaded data."
    endTime = time.time()
    print(endTime - startTime)


# Test case to ensure the correct shape of data
def test_shapeData():
    x_train, x_test, y_train, y_test, scaled_input_df, scaled_output_df, scaler = rcp.splitData()

    
    startTime = time.time()
    
    assert scaled_input_df.shape[1] == 5, "Expected 5 features in the X data after preprocessing."
    assert scaled_output_df.shape[1] == 1, "Expected Y data to have a single column."
    endTime = time.time()
    print(endTime - startTime)


# Test case to ensure the correct columns for Input
def test_colInput():
    input_df, output_df = rcp.preprocessData()
    
    startTime = time.time()
    
    input_columns = ['Gender', 'Age', 'Annual Salary', 'Credit Card Debt', 'Net Worth']
    # Convert the NumPy array to a DataFrame
    X_df = pd.DataFrame(input_df, columns=input_columns)
    # Check if X_df is a DataFrame
    assert isinstance(X_df, pd.DataFrame)
    # Check that the columns have been dropped for X
    assert "Customer Name" not in X_df.columns
    assert "Customer e-mail" not in X_df.columns
    assert "Country" not in X_df.columns
    assert "Car Purchase Amount" not in X_df.columns
    
    endTime = time.time()
    print(endTime - startTime)
    
    
# Test case to ensure the correct columns for Output
def test_colOutput():
    input_df, output_df = rcp.preprocessData()
    
    startTime = time.time()
    # Convert the NumPy array to a DataFrame
    # Allows to use coloumns, numpy array doesn't have coloumns
    output_df = pd.DataFrame(output_df, columns=['Car Purchase Amount'])
    
    # Check if Y_df is a DataFrame and has the correct column name
    assert isinstance(output_df, pd.DataFrame)
    assert output_df.columns == 'Car Purchase Amount'
    
    endTime = time.time()
    print(endTime - startTime)
    

# Test case to ensure the correct range of data
def test_rangeData():
    input_df, output_df = rcp.preprocessData()
    x_train, x_test, y_train, y_test, scaled_input_df, scaled_output_df, scaler = rcp.splitData()
    excel_file = "Car_Purchasing_Data.xlsx"
    df = pd.read_excel(excel_file)
    
    startTime = time.time()
    range_input_result = len(input_df)
    range_output_result = len(output_df)
    expected_range = len(df)
    
    assert range_input_result and range_output_result == expected_range
    
    assert 0 <= np.min(scaled_input_df) <= 1, "X min should be scaled between 0 and 1."
    assert 0 <= np.min(scaled_output_df) <= 1, "Y min should be scaled between 0 and 1."
    assert 0 <= np.max(scaled_input_df) <= 1, "X max should be scaled between 0 and 1."
    assert 0 <= np.max(scaled_output_df) <= 1, "Y max should be scaled between 0 and 1."
    
    endTime = time.time()
    print(endTime - startTime)


# Test case to ensure the correct splitting of data
def test_splitData():

    x_train, x_test, y_train, y_test, scaled_input_df, scaled_output_df, scaler = rcp.splitData()
    excel_file = "Car_Purchasing_Data.xlsx"
    df = pd.read_excel(excel_file)
    
    startTime = time.time()
    
    size_result = len(np.concatenate((x_train, x_test)))
    expected_size = len(df)
    assert size_result == expected_size
    
    # Check proportions for train-test split
    assert x_train.shape[0] / scaled_input_df.shape[0] == 0.8
    assert x_test.shape[0] / scaled_input_df.shape[0] == 0.2
    assert y_train.shape[0] / scaled_output_df.shape[0] == 0.8
    assert y_test.shape[0] / scaled_output_df.shape[0] == 0.2
    
    endTime = time.time()
    
    print(endTime - startTime)
