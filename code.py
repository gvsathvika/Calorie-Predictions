"""Calories Burnt Prediction"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

calories = pd.read_csv(r'C:\Users\HP\Downloads\Calories Burnt Prediction-20241210T173301Z-001\Calories Burnt Prediction\Calories Burnt Prediction\ml_project\calories.csv')
exercise_data = pd.read_csv(r'C:\Users\HP\Downloads\Calories Burnt Prediction-20241210T173301Z-001\Calories Burnt Prediction\Calories Burnt Prediction\ml_project\exercise.csv')
calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)

calories_data.replace({"Gender":{'male':0,'female':1}}, inplace=True)

X = calories_data.drop(columns=['User_ID','Calories'], axis=1)
Y = calories_data['Calories']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

model = XGBRegressor()
model.fit(X_train.values, Y_train)

input_data = (0,68,190,94,29,105,40.8)

# changing input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)

print('The number of calories burnt ', float(int(prediction[0])))
