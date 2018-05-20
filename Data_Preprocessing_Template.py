"""@author: Tanzim"""
### 1. DATA PREPROCESSING :

# Import libraries:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset and the tables:

"""Import dataset"""
dataset = pd.read_csv("fill in filename")

"""create independent variable table"""
x = dataset.iloc[ : , columns].values

"""create dependent variable table"""
y = dataset.iloc[ : , columns].values

# Dealing with Missing data:

"""Import library"""
from sklearn.preprocessing import Imputer

"""create object for class"""
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)

"""Fit object along the column with NaN as needed"""
imputer = imputer.fit(x[:,1:3])

"""Replace missing values with mean values as needed"""
x[:,1:3] = imputer.transform(x[:,1:3])

# Encoding Categorical Values (present in both independent and dependent variables):

#on Independent variable:

"""Import library"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

"""create object for class"""
labelencoder_x = LabelEncoder()

"""call method on the column"""
x[:,0] = labelencoder_x.fit_transform(x[:,0])

"""If categorical values are unordered, we need to use one hot encoder to indicate that unordered behaviour """

"""create object for class"""
onehotencoder = OneHotEncoder(categorical_feature=[0])

"""call method on the column"""
x = onehotencoder.fit_transform(x).toarray()

#on Dependent variable if needed (Only label encoding is needed):

"""create object for class"""
labelencoder_y = LabelEncoder()

"""call method"""
y = labelencoder_y.fit_transform(y)

# Split training_set and Test_set:

"""Import library"""
from sklearn.cross_validation import train_test_split

"""create objects"""
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Feature Scaling:

""" We can do Standardization or Normalization"""

"""Import library"""
from sklearn.preprocessing import StandardScaler

"""create object for class"""
sc_x = StandardScaler()
sc_y = StandardScaler()

""" For test set we only do transform and no need to do feature scaling for y_test at all
    For classification questions with categorical features,
    we need not do feature scaling for y_train and y_test"""

"""call method"""
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
y_train = sc_y.fit_transform(y_train)
