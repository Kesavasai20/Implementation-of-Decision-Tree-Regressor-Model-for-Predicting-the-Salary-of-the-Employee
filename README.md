# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. calculate Mean square error,data prediction and r2.

## Program:
```py
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: K KESAVA SAI
RegisterNumber: 212223230105
```
```PY
import pandas as pd
data = pd.read_csv(r'salary.csv')
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred) 
r2
dt.predict([[5,6]])
```

## Output:
## data.head():
![image](https://github.com/Kesavasai20/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/138849303/c0df42cc-1840-43af-9326-c5f653b70873)
## data.info():
![image](https://github.com/Kesavasai20/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/138849303/43f82072-f0fd-4ca5-8e71-be256bb5dad5)
## isnull and sum()
![image](https://github.com/Kesavasai20/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/138849303/d484497c-7201-4c8e-981b-983c34779eb7)
## position head()
![image](https://github.com/Kesavasai20/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/138849303/3a73b376-2677-4531-840c-a0029c4cfae2)
## Value of MSE
![image](https://github.com/Kesavasai20/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/138849303/2429dcda-d459-4048-b08a-27da60c2c847)
## value of r2
![image](https://github.com/Kesavasai20/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/138849303/176808fc-2e4d-40a8-a8ae-d56b0c5744de)
## data.prediction()
![image](https://github.com/Kesavasai20/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/138849303/035d1bfa-805d-417e-9227-713d4bf62406)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
