#compaaring linear with ridge regression for various values of alphaa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge,LinearRegression
import mglearn

boston=load_boston()

X,y=mglearn.datasets.load_extended_boston()

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=60)

ridge=Ridge(alpha=0.01).fit(X_train,y_train)
#lr= LinearRegression().fit(X_train,y_train)
print('training set score {:.2f}'.format(ridge.score(X_train,y_train)))
print("test score {:.2f}".format(ridge.score(X_test,y_test)))
