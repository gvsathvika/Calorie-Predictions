"""Calories Burnt Prediction"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from flask import Flask, render_template, request
app = Flask(__name__, static_url_path='/static')

@app.route('/')
def student():
    return render_template('index.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        res = request.form
        l1 = []
    for key,value in res.items():
        l1.append((float(value)))

    calories = pd.read_csv(r'C:\Users\HP\Downloads\Calories Burnt Prediction-20241210T173301Z-001\Calories Burnt Prediction\Calories Burnt Prediction\ml_project\calories.csv')
    exercise_data = pd.read_csv(r'C:\Users\HP\Downloads\Calories Burnt Prediction-20241210T173301Z-001\Calories Burnt Prediction\Calories Burnt Prediction\ml_project\exercise.csv')
    calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)
    calories_data.replace({"Gender":{'male':0,'female':1}}, inplace=True)
    
    X = calories_data.drop(columns=['User_ID','Calories'], axis=1)
    Y = calories_data['Calories']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    model = XGBRegressor()
    model.fit(X_train, Y_train)
    
    
    input_data = list(l1)

    # changing input_data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = model.predict(input_data_reshaped)
    
    prediction = float(int(prediction[0]))
    
    print('The number of calories burnt ', float(int(prediction)))
    return render_template("result.html",predict = prediction, result = res)

if __name__ == '__main__':
    app.run()