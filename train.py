import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
data=pd.read_csv("https://raw.githubusercontent.com/JatinSadhwani02/Employee-Salary-Predictio-in-Machine-Learning/master/Salary.csv")

X=data.iloc[:,0].values.reshape(-1,1)
y=data.iloc[:,1].values.reshape(-1,1)
lr=LinearRegression()
lr.fit(X,y)

with open("moidel.pkl","wb") as f:
    pickle.dump(lr,f)