import numpy as np
from sklearn import linear_model

y = np.arange(100)
# print y

#correct way to create np array for clf.fit
X = np.array([[2*i] for i in y])
print X.shape

#incorrect way to create np array for clf.fit
# X = np.array([2*i for i in y])                                                            
# print X.shape                                                                 

regr = linear_model.LinearRegression()
regr.fit(X,y)

print(regr.coef_)
print regr.score(X,y)
