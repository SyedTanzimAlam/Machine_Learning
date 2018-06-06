"""@author: Tanzim"""
# Save Model Using joblib
import pandas as pd
filename = "PUT THE .csv file"
colnames = ['Column names in quotes seperated by comma']
dataset = pd.read_csv(filename, names=colnames).values
# separate array into input and output components
X = dataset[:,0:8] # rows:columns
Y = dataset[:,8]
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
# Fit the model on training set
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, Y_train)
# save the model to disk
from sklearn.externals.joblib import dump
filename = 'finalized_model.sav'
dump(model, filename)
# some time later...
# load the model from disk
from sklearn.externals.joblib import load
load_model = load(filename)
result = load_model.score(X_test, Y_test)
print(result)