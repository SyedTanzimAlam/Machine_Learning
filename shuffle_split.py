"""@author: Tanzim"""
# Evaluate using Shuffle Split Cross Validation
import pandas as pd
filename = "PUT THE .csv file"
colnames = ['Column names in quotes seperated by comma']
dataset = pd.read_csv(filename, names=colnames).values
# separate array into input and output components
X = dataset[:,0:8] # rows:columns
Y = dataset[:,8]
# Evaluation
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
kfold = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))