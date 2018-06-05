# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 18:07:34 2018
Code for titanic problem using decision tree, one-hot-encoding, parameter tuning and main focus is on BAGGING
@author: Aman
"""
import pandas as pd
import os
from sklearn import preprocessing
from sklearn import tree
from sklearn import ensemble
from sklearn import model_selection

#changes working directory
os.chdir( "F:/Data Science/Titanic/Data/" )
titanic_train = pd.read_csv("titanic_train.csv")
titanic_train.shape
titanic_train.info()

titanic_test = pd.read_csv('titanic_test.csv')
titanic_test.shape
titanic_test.info()

titanic_test["Survived"] = None

#Let's excercise by concatinating both train and test data
#Concatenation is Bcoz to have same number of rows and columns so that our job will be easy
titanic = pd.concat([titanic_train, titanic_test])
titanic.shape
titanic.info()

#Extract and create title column from name
def extract_title(name):
     return name.split(',')[1].split('.')[0].strip()
#The map(aFunction, aSequence) function applies a passed-in function to each item in an iterable object 
#and returns a list containing all the function call results.
titanic['Title'] = titanic['Name'].map(extract_title)

#Imputation work for missing data with default values
mean_imputer = preprocessing.Imputer() #By defalut parameter is mean and let it use default one.
mean_imputer.fit(titanic_train[['Age','Fare']]) 
#Age is missing in both train and test data.
#Fare is NOT missing in train data but missing test data. Since we are playing on tatanic union data, we are applying mean imputer on Fare as well..
titanic[['Age','Fare']] = mean_imputer.transform(titanic[['Age','Fare']])

#creaate categorical age column from age
#It's always a good practice to create functions so that the same can be applied on test data as well
def convert_age(age):
    if(age >= 0 and age <= 10): 
        return 'Child'
    elif(age <= 25): 
        return 'Young'
    elif(age <= 50): 
        return 'Middle'
    else: 
        return 'Old'
#Convert numerical Age column to categorical Age1 column
titanic['Age1'] = titanic['Age'].map(convert_age)

#Create a new column FamilySize by combining SibSp and Parch and seee we get any additioanl pattern recognition than individual
titanic['FamilySize'] = titanic['SibSp'] +  titanic['Parch'] + 1
def convert_familysize(size):
    if(size == 1): 
        return 'Single'
    elif(size <=3): 
        return 'Small'
    elif(size <= 6): 
        return 'Medium'
    else: 
        return 'Large'
#Convert numerical FamilySize column to categorical FamilySize1 column
titanic['FamilySize1'] = titanic['FamilySize'].map(convert_familysize)

#Now we got 3 new columns, Title, Age1, FamilySize1
#convert categorical columns to one-hot encoded columns including  newly created 3 categorical columns
#There is no other choice to convert categorical columns to get_dummies in Python
titanic1 = pd.get_dummies(titanic, columns=['Sex','Pclass','Embarked', 'Age1', 'Title', 'FamilySize1'])
titanic1.shape
titanic1.info()

#Drop un-wanted columns for faster execution and create new set called titanic2
titanic2 = titanic1.drop(['PassengerId','Name','Age','Ticket','Cabin','Survived'], axis=1, inplace=False)
#See how may columns are there after 3 additional columns, one hot encoding and dropping
titanic2.shape 
titanic2.info()
#Splitting tain and test data
X_train = titanic2[0:titanic_train.shape[0]] #0 t0 891 records
X_train.shape
X_train.info()
y_train = titanic_train['Survived']

#oob scrore is computed as part of model construction process
dt_estimator = tree.DecisionTreeClassifier()
#This is what the real Bagging model is
#In-order to specify, which model to be used is what base_estimator is: In this case we are building using Decission Tree Classifier
bt_estimator = ensemble.BaggingClassifier(base_estimator= dt_estimator, max_features =8, oob_score=False, random_state=4521)
#n_estimators means how many no. of tree to be grown
#base_estimator__ (Double underscore__ acts as prefix)
bt_grid = {'n_estimators':[7], 'base_estimator__max_depth':[3,4,5]}

grid_bt_estimator = model_selection.GridSearchCV(bt_estimator, bt_grid, cv=10, n_jobs=1)
grid_bt_estimator.fit(X_train, y_train)
print(grid_bt_estimator.grid_scores_) #In SK Learn Verion 0.18

print(grid_bt_estimator.best_score_)
print(grid_bt_estimator.best_params_)
print(grid_bt_estimator.score(X_train, y_train))

X_test = titanic2[titanic_train.shape[0]:]
X_test.shape
X_test.info()
titanic_test['Survived'] = grid_bt_estimator.predict(X_test)

os.chdir( "F:\Data Science\Titanic\Submission" )
titanic_test.to_csv('submission_Bagging.csv', columns=['PassengerId','Survived'],index=False)
