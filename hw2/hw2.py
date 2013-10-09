import pylab as pl
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import cross_validation

hw_file = 'carSalesSearch.csv'
read_file = pd.read_csv(hw_file)

y = read_file['Sales']
y = np.array(y)
print y.shape

X = read_file['SearchIndex']
X = np.array(X)
#print X.shape
X = X.reshape(24,1)
print X.shape

SEED = 42
n = 10
for i in range(n):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            X, y, test_size=.20, random_state=i*SEED)

print '$$$$$'
print X_train.shape
print y_train.shape

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
# LinearRegression(copy_X=True, fit_intercept=True, normalize=False)
print 'regression coefficient: '
print (regr.coef_)

#the mean square error
print 'the mean square error: '
print np.mean((regr.predict(X_test)-y_test)**2)

print 'variance score: '
print regr.score(X_test, y_test)
