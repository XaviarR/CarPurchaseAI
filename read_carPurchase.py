import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
import pickle
import numpy as np
from joblib import dump, load

sns.set_theme(style="darkgrid")

class RegressionModelFactory:
    @staticmethod
    def create_model(model_type):
        if model_type == 'linear':
            return LinearRegression()
        elif model_type == 'svm':
            return SVR()
        elif model_type == 'random_forest':
            return RandomForestRegressor()
        elif model_type == 'gradient_boosting':
            return GradientBoostingRegressor()
        elif model_type == 'xgb':
            return XGBRegressor()
        elif model_type == 'lasso':
            return Lasso()
        else:
            raise ValueError("Unsupported model type")
        
        
# Load from excel file
def loadData(): 
    # Import excel file
    excel_file = "Car_Purchasing_Data.xlsx"
    # Read excel file with pandas
    df = pd.read_excel(excel_file)
    
    return df

# Create and Reshape Data Frames
# Input_DF has multiple coloumns
# Output_DF is Car Purchase Amount
def preprocessData():  
    load = loadData()
    # Using pandas create datafram that uses columns of number values
    numeric_df = load.select_dtypes(include=['int64', 'float64'])


    input_df = numeric_df.copy()
    input_df = input_df.drop(["Car Purchase Amount"],axis=1)

    # input_df = input_df.values.reshape(-1,1)
    output_df = load["Car Purchase Amount"].values.reshape(-1,1)

    return input_df, output_df

# Transform and split data
# Transform scales the data in relationship of 0 to 1
# Test split seperated data for testing purposes with relationship of test_size being 20%
def splitData():
    input_df, output_df = preprocessData()
    
    # Scale input_df and output_df
    scaler = MinMaxScaler()
    scaled_input_df = scaler.fit_transform(input_df)
    scaled_output_df = scaler.fit_transform(output_df)

    # Create train/test with scaled dfs
    x = scaled_input_df
    y = scaled_output_df

    # using the train/test split function
    x_train, x_test, y_train, y_test = train_test_split(x ,y ,random_state=42 ,test_size=0.20 ,shuffle=True)
    
    return x_train, x_test, y_train, y_test, scaled_input_df, scaled_output_df, scaler

# Evaluate Fastest Model
# Regressiong Models differ depending on relationship of graph eg; random ploted graph would be terrible for linear model
def trainModels():
    x_train, x_test, y_train, y_test, scaled_input_df, scaled_output_df, scaler = splitData()
    
    model_types = [
        'linear', 'svm', 'random_forest', 'gradient_boosting', 'xgb', 'lasso'
    ]
    
    trained_models = {}
    
    for model_type in model_types:
        model = RegressionModelFactory.create_model(model_type)
        model.fit(x_train, y_train.ravel())
        trained_models[model_type] = model
        
    return trained_models
    


def evaluateModels():
    x_train, x_test, y_train, y_test, scaled_input_df, scaled_output_df, scaler = splitData()
    trained_models = trainModels()
    
    rmse_values = {}
    preds_reshaped = {}  # Create a dictionary to store reshaped predictions
    
    for name, model in trained_models.items():
        preds = model.predict(x_test)
        rmse_values[name] = mean_squared_error(y_test, preds, squared=False)
        
    #     # Reshape the preds array
    #     preds_reshaped[name] = preds.reshape(-1, 1)
        
    # y_test_reshaped = y_test.reshape(-1, 1)  # Reshape y_test
    
    # for name, preds in preds_reshaped.items():
    #     comparison_df = pd.DataFrame({'Actual': y_test_reshaped.flatten(), 'Predicted': preds.flatten()})
    #     print(f"Comparison DataFrame for {name}:")
    #     print(f"RMSE for {name}: {rmse_values[name]}")
    #     print(comparison_df)
    #     # print(rmse_values[name])
        
    return rmse_values

def plotPerformance():
    
    rmse_values = evaluateModels()
    
    plt.figure(figsize=(10,7))
    models = list(rmse_values.keys())
    rmse = list(rmse_values.values())
    bars = plt.bar(models, rmse, color=['blue', 'green', 'red', 'purple', 'orange', 'grey'])

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.00001, round(yval, 5), ha='center', va='bottom', fontsize=10)

    plt.xlabel('Models')
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.title('Model RMSE Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Find shape of split data
def train_test_shape():
    
    x_train, x_test, y_train, y_test, scaled_input_df, scaled_output_df, scaler = splitData()
    
    print("X_train:",x_train.shape)
    print("X_test: ", x_test.shape)
    print("Y_train: ", y_train.shape)
    print("Y_test: ", y_test.shape)
    print("\n")

    print("X_train: ", x_train[:5])
    print("X_test: ", x_test[:5])
    print("\n")
    print("Y_train: ", y_train[:5])
    print("Y_test: ", y_test[:5])
    print("\n")

# Plot Prediction of data using Linear Regression (Testing data set)
def linear_test():
    
    x_train, x_test, y_train, y_test, scaled_input_df, scaled_output_df, scaler = splitData()
    
    # Perform linear regression and plotting for each column in y_pred
    # Column with the same order of name as dataframe
    y_columns = ["Gender", "Age", "Annual Salary", "Credit Card Debt", "Net Worth"]
    for col_idx, col_name in enumerate(y_columns):  # Assuming y_columns contains your column names
        regr = LinearRegression()
        regr.fit(x_test, y_test[:, col_idx])
        r_squared = regr.score(x_test, y_test[:, col_idx])
        print(f"Test R-squared value for column {col_name}: {r_squared}")

        y_pred = regr.predict(x_test)
        
        plt.title("Test Data")
        plt.scatter(x_test, y_test[:, col_idx], color='b', label='Actual')
        plt.plot(x_test, y_pred, color='k', label='Predicted')
        plt.xlabel('Car Purchase Amount')
        plt.ylabel(col_name)  # Use the column name as the y-label
        plt.legend()
        plt.show()
    
# Plot prediction of data using Linear Regression (training data set)    
def linear_train():
    
    x_train, x_test, y_train, y_test, scaled_input_df, scaled_output_df, scaler = splitData()
    
    # Perform linear regression and plotting for each column in y_pred
    # Column with the same order of name as dataframe
    y_columns = ["Gender", "Age", "Annual Salary", "Credit Card Debt", "Net Worth"]
    for col_idx, col_name in enumerate(y_columns):  # Assuming y_columns contains your column names
        regr = LinearRegression()
        regr.fit(x_train, y_train[:, col_idx])
        r_squared = regr.score(x_train, y_train[:, col_idx])
        print(f"Train R-squared value for column {col_name}: {r_squared}")

        y_pred = regr.predict(x_train)
        
        plt.title("Train data")
        plt.scatter(x_train, y_train[:, col_idx], color='b', label='Actual')
        plt.plot(x_train, y_pred, color='k', label='Predicted')
        plt.xlabel('Car Purchase Amount')
        plt.ylabel(col_name)  # Use the column name as the y-label
        plt.legend()
        plt.show()

# Display head/tail of dataframe
def head_tail():
    
    df = loadData()
    
    # Print first 5 rows
    print(df.head())
    # Print last 5 rows
    print(df.tail())

# Display shape of dataframe
def shape_data():
    
    df = loadData()
    
    # Get the shape of the DataFrame
    shape = df.shape
    # Print the number of rows and columns
    print("Number of rows:", shape[0])
    print("Number of columns:", shape[1])

# Display consice summary of dataframe
def summary_data():
    df = loadData()
    
    # Display concise summary
    print("\nHere is the consice summary of the dataset\n")
    df.info()

# Check for null values from dataframe
def null_check():
    df = loadData()
    # Check for null values and count them
    print("\nHere are null values\n")
    null_counts = df.isnull().sum()
    # Print the count of null values for each column
    print(null_counts)

# Describe statistics of dataframe
def describe_data():
    
    df = loadData()
    
    # Get overall statistics about the dataset
    print("\nHere is the summary of overall dataset\n")
    summary = df.describe()
    # Print the summary statistics
    print(summary)

# Plot scatter graph of input and output dataframe from preprocessData()
def scatter_graph():  
    input_df, output_df = preprocessData()
    numeric_df = pd.DataFrame(input_df, columns=input_df.columns)  # Convert input_df to DataFrame
    output_df = pd.DataFrame(output_df, columns=["Car Purchase Amount"])  # Convert output_df to DataFrame
    
    combined_df = numeric_df.join(output_df)
    
    # Specify the column you want to use as the x-axis
    x_column = 'Car Purchase Amount'
    # Iterate over other columns for comparison
    for column in combined_df.columns:
        if column != x_column:
            sns.scatterplot(x=x_column, y=column, data=combined_df,)
            plt.title(f"Scatter Plot: {x_column} vs {column}")
            plt.show()
           
# Plot and display pair graph of entire data 
def pairplot_graph():
    input_df, output_df = preprocessData()
    numeric_df = pd.DataFrame(input_df, columns=input_df.columns)  # Convert input_df to DataFrame
    output_df = pd.DataFrame(output_df, columns=["Car Purchase Amount"])  # Convert output_df to DataFrame
    
    combined_df = numeric_df.join(output_df)
    
    sns.pairplot(combined_df)
    plt.show()
    
# Train best model using the whole data set
def train_entireData():
    
    x_train, x_test, y_train, y_test, scaled_input_df, scaled_output_df, scaler = splitData()
    
    # Create instance
    xg = LinearRegression()
    
    # Train Model
    xg.fit(scaled_input_df, scaled_output_df)
    
    # Predict
    xg_preds = xg.predict(scaled_input_df)
    
    # Evaluate
    xg_rmse = mean_squared_error(scaled_output_df, xg_preds, squared=False)
    result = xg_rmse.reshape(-1,1)
    
    print(f"Prediction is: {scaler.inverse_transform(result)}")
    
    return xg

def saveModel():
    xg = train_entireData()
    # Save
    filename = "car_purchseFinal.joblib"
    dump(xg, open(filename, "wb"))
    
def loadModelrwawr():
    # x_train, x_test, y_train, y_test, scaled_input_df, scaled_output_df, scaler = splitData()
    scaler, sc1 = scalerFunc()
    
    filename = "car_purchseFinal.joblib"
    # Load saved model
    loaded_model = load(open(filename, "rb"))
    
    gender = int(input("Enter your gender (0 for male, 1 for female): "))

    age = int(input("Enter your age: "))

    annualSalary = float(input("Enter your annual salary: "))

    creditDebt = float(input("Enter your credit debt: "))

    networth = float(input("Enter your networth: "))
    
    
    # Using loaded model use this data
    input_data = scaler.transform([[gender, age, annualSalary, creditDebt, networth]])
    print(input_data)
     

    pred = loaded_model.predict(input_data)
    pred = pred.reshape(1,-1)
    
    
    # Display prediction as readable value not scaled 0 to 1 
    print("Predicted Car_Purchase_Amount based on input:",sc1.inverse_transform(pred))

# Different training model with different random state test split
def new_test_set():
    input_df, output_df = preprocessData()
    
    # Scale input_df and output_df
    scaler = MinMaxScaler()
    scaled_input_df = scaler.fit_transform(input_df)
    sc1 = MinMaxScaler()
    scaled_output_df = sc1.fit_transform(output_df)

    # Create train/test with scaled dfs
    x = scaled_input_df
    y = scaled_output_df
    x_train, x_test, y_train, y_test = train_test_split(x ,y ,random_state=58 ,test_size=0.20 ,shuffle=True)
    
    xg =  LinearRegression() 
    
    xg.fit(x_test, y_test)
    
    xg_preds = xg.predict(x_test)
    
    xg_rmse = mean_squared_error(y_test, xg_preds, squared=False)
    
    print(f"Linear RMSE: {xg_rmse}")
    
    final_preds = xg_rmse.reshape(1,-1)
    
    print("Predicted Annual Salary based on input:",sc1.inverse_transform(final_preds))
    
    
def scalerFunc():
    
    input_df, output_df = preprocessData()
    
    scaler = MinMaxScaler()
    scaled_input_df = scaler.fit_transform(input_df)
    sc1 = MinMaxScaler()
    scaled_output_df = sc1.fit_transform(output_df)
    return scaler, sc1
    
    

def loadModel():
    scaler, sc1 = scalerFunc()
    input_df, output_df = preprocessData()
    
    filename = "car_purchseFinal.joblib"
    # Load the trained model
    loaded_model = load(open(filename, "rb"))

    user_input = []
    for column in ['Gender', 'Age', 'Annual Salary', 'Credit Card Debt', 'Net Worth']:
        while True:
            value = input(f"Enter value for {column}: ")
            if value.strip() == "":
                print("Value cannot be empty. Please try again.")
            else:
                try:
                    value = float(value)
                    user_input.append(value)
                    break
                except ValueError:
                    print("Invalid input. Please enter a valid number.")

    # Convert user inputs to a NumPy array and preprocess
    user_input = np.array([user_input])
    scaled_user_input = scaler.transform(user_input)
    
    # Use the loaded model to make a prediction
    predicted_amount = loaded_model.predict(scaled_user_input)
    
    predicted_amount = predicted_amount.reshape(1,-1)
    
    # Display the prediction to the user
    print("Predicted Car Purchase Amount:", sc1.inverse_transform(predicted_amount))  # Access the single prediction value directly




    
# head_tail()
# shape_data()
# summary_data()
# describe_data()

# scatter_graph()
# pairplot_graph()

# new_test_set()
# train_entireData()
# trainModels()
# evaluateModels()
# plotPerformance()

# saveModel()
# loadModel()
