"""@author: Tanzim"""
# Randomized for Algorithm Tuning
import pandas as pd
filename = "PUT THE .csv file"
colnames = ['Column names in quotes seperated by comma']
dataset = pd.read_csv(filename, names=colnames).values
# separate array into input and output components
X = dataset[:,0:8] # rows:columns
Y = dataset[:,8]
from sklearn.linear_model import Ridge
model = Ridge()
from scipy.stats import uniform
param_grid = {'alpha': uniform()}
from sklearn.model_selection import RandomizedSearchCV
rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100, random_state=0)
rsearch.fit(X, Y)
print(rsearch.best_score_)
print(rsearch.best_estimator_.alpha)