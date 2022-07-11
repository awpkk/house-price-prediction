from django .shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

def home(request):
    return render(request, "home.html")

def predict(request):
    return render(request, "predict.html")

def result(request):
    data = pd.read_csv(r'C:\Users\dell\USA_housing.csv')
    data = data.drop(['Address'], axis=1)
    X = data.drop('Price', axis=1)
    y = data['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    model = LinearRegression()
    model.fit(X_train, y_train)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])

    pred = model.predict(np.array([val1,val2,val3,val4,val5]).reshape(1,-1))
    pred = round(pred[0])

    price = "The predicted price is $"+str(pred)

    return render(request, "predict.html",{"result2":price})