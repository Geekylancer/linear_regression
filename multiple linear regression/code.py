# Import Libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 


# Import dataset
dataset = pd.read_csv('50_startups.csv')


# Applying One hot encoding
dummy = pd.get_dummies(dataset['State'],drop_first = True)
df = pd.concat([dataset,dummy],axis = 1)
df1 = df.drop('State',axis = 1)


X = df1.drop('Profit',axis = 1)
y = df1['Profit']


# Splitting dataset into training set and testing set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 10)


# Fitting regression model
from sklearn.linear_model import LinearRegression
regressor  = LinearRegression()
regressor.fit(X_train,y_train)


# Predicting test dataset
regressor.predict(X_test)


# Obtaining accuracy of the model
regressor.score(X_test,y_test)

