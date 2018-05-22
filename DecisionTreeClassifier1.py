

"""
Code for titanic problem using plain decision tree
"""

import os
import pandas as pd
from sklearn import tree
import io
import pydot #if we need to use any external .exe files.... Here we are using dot.exe

os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin'

"""
os.chdir( "F:/Data Science/Titanic/Data/" )
os.getcwd()
"""
titanic_train = pd.read_csv("F:/Data Science/Titanic/Data/titanic_train.csv")

#EDA
titanic_train.shape
titanic_train.info()
X_titanic_train = titanic_train[['Pclass', 'Parch']]
y_titanic_train = titanic_train['Survived']

#build the decision tree model
dt = tree.DecisionTreeClassifier()
dt.fit(X_titanic_train, y_titanic_train)

#visualize the decission tree
dot_data = io.StringIO() 
tree.export_graphviz(dt, out_file = dot_data, feature_names = X_titanic_train.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
os.chdir( "F:\Data Science\Titanic\Decision Trees" )
graph.write_pdf("DS-DT.pdf")
#os.getcwd()

#Predict the outcome using decision tree
titanic_test = pd.read_csv("F:/Data Science/Titanic/Data/titanic_test.csv")
X_test = titanic_test[['Pclass', 'Parch']]
titanic_test['Survived'] = dt.predict(X_test)
os.chdir( "F:\Data Science\Titanic\Submission" )
titanic_test.to_csv("submission.csv", columns=['PassengerId','Survived'], index=False)