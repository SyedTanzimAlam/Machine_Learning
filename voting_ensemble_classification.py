"""@author: Tanzim"""
# Voting Ensemble for Classification
import pandas as pd
filename = "PUT THE .csv file"
colnames = ['Column names in quotes seperated by comma']
dataset = pd.read_csv(filename, names=colnames).values
# separate array into input and output components
X = dataset[:,0:8] # rows:columns
Y = dataset[:,8]
# create the sub models
estimators = []
from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression()
estimators.append(('logistic', model1))
from sklearn.tree import DecisionTreeClassifier
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
from sklearn.svm import SVC
model3 = SVC()
estimators.append(('svm', model3))
# create the ensemble model
from sklearn.ensemble import VotingClassifier
ensemble = VotingClassifier(estimators)
from sklearn.model_selection import KFold
kfold = KFold(n_splits=10, random_state=0)
from sklearn.model_selection import cross_val_score
results = cross_val_score(ensemble, X, Y, cv=kfold)
print(results.mean())







