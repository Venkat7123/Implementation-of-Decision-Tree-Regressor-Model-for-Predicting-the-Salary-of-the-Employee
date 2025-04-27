# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the salary dataset, encode categorical data, and handle missing values.
2. Split the dataset into features (X) and target (y), then into training and testing sets.
3. Train a DecisionTreeRegressor model on the training data.
4. Predict salaries on test data and evaluate using Mean Squared Error (MSE) and R² score.

## Program:

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Venkatachalam S
RegisterNumber:  212224220121

```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

df = pd.read_csv('/content/Salary.csv')

df.head()

df.info()

df.isnull().sum()

le = LabelEncoder()

df['Position'] = le.fit_transform(df['Position'])
df.head()

x = df[['Position','Level']]
y = df[['Salary']]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)

dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
y_pred

mse = metrics.mean_squared_error(y_test,y_pred)
mse

r2 = metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```
## Output:
![image](https://github.com/user-attachments/assets/1622cb78-9d1f-49d0-aa88-ece029caee10)
![image](https://github.com/user-attachments/assets/06254ae4-9d21-40ac-9c22-d48a638effc1)
![image](https://github.com/user-attachments/assets/4f5a2ccb-0270-4ccf-a87e-c56e11783fd6)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
