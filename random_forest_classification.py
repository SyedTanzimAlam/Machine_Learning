"""@author: Tanzim"""
# Random Forest Classification
import pandas as pd
filename = "PUT THE .csv file"
colnames = ['Column names in quotes seperated by comma']
dataset = pd.read_csv(filename, names=colnames).values
# separate array into input and output components
X = dataset[:,0:8] # rows:columns
Y = dataset[:,8]
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_features=3)
from sklearn.model_selection import KFold
kfold = KFold(n_splits=10, random_state=0)
from sklearn.model_selection import cross_val_score
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())