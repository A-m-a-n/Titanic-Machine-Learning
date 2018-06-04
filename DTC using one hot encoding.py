# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 23:32:36 2018

@author: Aman

Code for titanic problem using decision tree and one-hot-encoding

"""

#DecissionTree and Predict methods are very important in this example. This is the real starting/building of ML
#Here we will be playing with more columns. However DecisionTreeClassifier algorithm works only on numeric/continuous data/columns
#Henceforth we need to convert  catogerical columns to dummy columns
#This technique is called one-hot encoding

import pandas as pd
from sklearn import tree
import io
import pydot #if we need to use any external .exe files.... Here we are using dot.exe
import os

os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin'

titanic_train = pd.read_csv("F:/Data Science/Titanic/Data/titanic_train.csv")

#EDA
titanic_train.shape
titanic_train.info()

#Transformation of non numneric cloumns
#There is an exception with the pclass. Though it's coinncidentally is a number but it's a classification but not a number.
#titanic_train1 = titanic_train[['Pclass', 'Sex', 'Embarked', 'Fare']]

#Convert categoric to One hot encoding using get_dummies
titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'Sex', 'Embarked'])
titanic_train1.shape
titanic_train1.info()
titanic_train1.describe

#now the drop non numerical columns where we will not be applying logic. Something like we will not apply logic on names, passengerID ticket id etc...
X_train = titanic_train1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'],1) 
y_train = titanic_train['Survived']

#print(type(y_train)) It is a series
#print(type(X_train)) X_train is a dataframe

dt = tree.DecisionTreeClassifier()

#.fit builds the model. In this case the model building is using Decission Treee Algorithm
dt.fit(X_train,y_train)
type(dt)

#visualize the decission tree
dot_data = io.StringIO()  

tree.export_graphviz(dt, out_file = dot_data, feature_names = X_train.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
os.chdir( "F:\Data Science\Titanic\Decision Trees" )
graph.write_pdf("DT-AugmentedColumns.pdf")

#predict the outcome using decission tree
titanic_test = pd.read_csv("F:/Data Science/Titanic/Data/titanic_test.csv")
titanic_test.info() #Found that one row has Fare = null in test data. Instead of dropping this column, let's take the mean of it.
titanic_test.Fare[titanic_test['Fare'].isnull()] = titanic_test['Fare'].mean()

#Now apply same get_dummies and drop columns on test data as well like above we did for train data
titanic_test1 = pd.get_dummies(titanic_test, columns=['Pclass', 'Sex', 'Embarked'])
X_test = titanic_test1.drop(['PassengerId','Age','Cabin','Ticket', 'Name'], 1)
#Apply the model on Furture/test data
titanic_test['Survived'] = dt.predict(X_test)
os.chdir( "F:\Data Science\Titanic\Submission" )
titanic_test.to_csv("Submission_RefinedVer1.csv", columns=['PassengerId', 'Survived'], index=False)