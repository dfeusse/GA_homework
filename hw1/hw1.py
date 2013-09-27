from __future__ import division
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn import svm

iris = datasets.load_iris()
# view format of data
#print iris.data.shape, iris.target.shape

iris_X = iris.data
iris_Y = iris.target

iris_X_train, iris_X_test, iris_Y_train, iris_Y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.33, random_state=0)

# see shape of training and test data sets
#print iris_X_train.shape, iris_Y_train.shape
#print iris_X_test.shape, iris_Y_test.shape

clf = svm.SVC(kernel="linear", C=1)
scores = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=5)
print scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#print clf.score(iris_X_test, iris_Y_test)

print '-------'

# knn calculation
knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_Y_train)
prediction = knn.predict(iris_X_test)
print prediction

def accuracy(prediction, iris_Y_test):
    diff_count = 0
    for i, j in zip(prediction, iris_Y_test):
        if i != j:
            diff_count +=1
    return diff_count

print "Only " + str(accuracy(prediction, iris_Y_test)) + " data point was classified incorrectly"

print "In other words, there was " + str(int((1 - accuracy(prediction, iris_Y_test)/len(prediction))*100))+"%" + " accuracy"

knn.score(iris_X_test, iris_Y_test)
