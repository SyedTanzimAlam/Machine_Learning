"""@author: Tanzim"""
# Grid Search for Hyperparameter Tuning
import pandas as pd
filename = "PUT THE .csv file"
colnames = ['Column names in quotes seperated by comma']
dataset = pd.read_csv(filename, names=colnames).values
# separate array into input and output components
X = dataset[:,0:8] # rows:columns
Y = dataset[:,8]
from sklearn.linear_model import Ridge
model = Ridge()
import numpy as np
alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
from sklearn.model_selection import GridSearchCV
param_grid = dict(alpha=alphas)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X, Y)
print(grid.best_score_)
print(grid.best_estimator_.alpha)