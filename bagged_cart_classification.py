"""@author: Tanzim"""
# Bagged Decision Trees for Classification
import pandas as pd
filename = "PUT THE .csv file"
colnames = ['Column names in quotes seperated by comma']
dataset = pd.read_csv(filename, names=colnames).values
# separate array into input and output components
X = dataset[:,0:8] # rows:columns
Y = dataset[:,8]
from sklearn.tree import DecisionTreeClassifier
cart = DecisionTreeClassifier()
from sklearn.ensemble import BaggingClassifier
model = BaggingClassifier(base_estimator=cart, n_estimators=100, random_state=0)
from sklearn.model_selection import KFold
kfold = KFold(n_splits=10, random_state=0)
from sklearn.model_selection import cross_val_score
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())