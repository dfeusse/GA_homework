#http://blog.data-reverie.com/2013/03/first-steps-with-random-forests-and.html
import pylab as pl
import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

iris = datasets.load_iris()
#print iris

#X = the data; y = the target
X, y = iris.data, iris.target

SEED = 42
n = 10
for i in range(n):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=.20, random_state=i*SEED)

rf = RandomForestClassifier(n_estimators=1000)
rf.fit(X_train, y_train)

# so here you predict off of x_test, the compare results to y_test to see how accurate it is
z = rf.predict(X_test)
print z
print y_test

print rf.predict_proba(X_test)
print rf.score(X_test, y_test)
