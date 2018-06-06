"""@author: Tanzim"""
# Create a pipeline that extracts features from the data then creates a model
# load data
import pandas as pd
filename = "PUT THE .csv file"
colnames = ['Column names in quotes seperated by comma']
dataset = pd.read_csv(filename, names=colnames).values
# separate array into input and output components
X = dataset[:,0:8] # rows:columns
Y = dataset[:,8]
# create feature union
features = []
from sklearn.decomposition import PCA
features.append(('pca', PCA(n_components=3)))
from sklearn.feature_selection import SelectKBest
features.append(('select_best', SelectKBest(k=6)))
from sklearn.pipeline import FeatureUnion
feature_union = FeatureUnion(features)
# create pipeline
estimators = []
from sklearn.linear_model import LogisticRegression
estimators.append(('feature_union', feature_union))
estimators.append(('logistic', LogisticRegression()))
from sklearn.pipeline import Pipeline
model = Pipeline(estimators)
# evaluate pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kfold = KFold(n_splits=10, random_state=0)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())