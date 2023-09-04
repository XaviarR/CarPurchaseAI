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

sns.set_theme(style="darkgrid")

# Import excel file
excel_file = "DiamondValues(10000).xlsx"
# Read excel file with pandas
df = pd.read_excel(excel_file)


input_df = df.select_dtypes(include=['int64', 'float64'])
input_df = input_df.drop(["Price"],axis=1)

output_df = df["Price"].values.reshape(-1,1)


scaler = MinMaxScaler()

scaled_input_df = scaler.fit_transform(input_df)
scaled_output_df = scaler.fit_transform(output_df)


# Create train/test with scaled dfs
x = scaled_input_df
y = scaled_output_df

# using the train/test split function
x_train, x_test, y_train, y_test = train_test_split(x ,y ,random_state=50 ,test_size=0.20 ,shuffle=True)

def test_trainData():
    # Intit Models
    regr = LinearRegression()
    svm = SVR() 
    rf = RandomForestRegressor() 
    gbr = GradientBoostingRegressor() 
    xg =  XGBRegressor() 
    las = Lasso()

    
    # Train Models
    regr.fit(x_test, y_test)
    svm.fit(x_test, y_test)
    rf.fit(x_test, y_test)
    gbr.fit(x_test, y_test)
    xg.fit(x_test, y_test)
    las.fit(x_test, y_test)

    
    # Predict from data
    regr_preds = regr.predict(x_test)
    svm_preds = svm.predict(x_test)
    rf_preds = rf.predict(x_test)
    gbr_preds = gbr.predict(x_test)
    xg_preds = xg.predict(x_test)
    las_preds = las.predict(x_test)

        
    # Evaluate performance
    regr_rmse = mean_squared_error(y_test, regr_preds, squared=False)
    svm_rmse = mean_squared_error(y_test, svm_preds, squared=False)
    rf_rmse = mean_squared_error(y_test, rf_preds, squared=False)
    gbr_rmse = mean_squared_error(y_test, gbr_preds, squared=False)
    xg_rmse = mean_squared_error(y_test, xg_preds, squared=False)
    las_rmse = mean_squared_error(y_test, las_preds, squared=False)

    
    # Display Results
    print(f"Linear Regression Model: {regr_rmse}")
    print(f"Support Vector Machine RMSE: {svm_rmse}")
    print(f"Random Forest RMSE: {rf_rmse}")
    print(f"Gradient Boosting Regressor RMSE: {gbr_rmse}")
    print(f"XGBRegressor RMSE: {xg_rmse}")
    print(f"Lasso RMSE: {las_rmse}")

    
    # choose the best model
    model_objects = [regr, svm, rf, gbr, xg, las]
    rmse_values = [regr_rmse, svm_rmse, rf_rmse, gbr_rmse, xg_rmse, las_rmse]

    best_model_index = rmse_values.index(min(rmse_values))
    best_model_object = model_objects[best_model_index]

    # print(f"The best model is {models[best_model_index]} with RMSE: {rmse_values[best_model_index]}")

    # visualize the models results
    # Create a bar chart
    models = ['Linear Regression', 'Support Vector Machine', 'Random Forest', 'Gradient Boosting Regressor', 'XGBRegressor', 'Lasso']
    plt.figure(figsize=(10,7))
    bars = plt.bar(models, rmse_values, color=['blue', 'green', 'red', 'purple', 'orange', 'grey'])

    # Add RMSE values on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.00001, round(yval, 6), ha='center', va='bottom', fontsize=10)

    plt.xlabel('Models')
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.title('Model RMSE Comparison')
    plt.xticks(rotation=45)  # Rotate model names for better visibility
    plt.tight_layout()

    # Display the chart
    plt.show()

def train_entireData():
    # Create instance
    xg = XGBRegressor()
    
    # Train Model
    xg.fit(scaled_input_df, scaled_output_df)
    
    # Predict
    xg_preds = xg.predict(scaled_input_df)
    
    # Evaluate
    xg_rmse = mean_squared_error(scaled_output_df, xg_preds, squared=False)
    
    print(f"Prediction is: {xg_rmse}")

    # # Save
    # filename = "final.sav"
    # pickle.dump(xg, open(filename, "wb"))

    # loaded_model = pickle.load(open(filename, "rb"))
    # result = loaded_model.score(scaled_input_df, scaled_output_df)
    # print(result)
    
# train_entireData()

# test_trainData()

# sns.pairplot(df)
# plt.show()