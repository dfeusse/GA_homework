import pylab as pl
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import cross_validation

data_file = 'uclaAdmissionData.csv'
df = pd.read_csv(data_file)
# print df.head()
# print df[:30]
df.columns = ['admit', 'gre', 'gpa', 'prestige']
# print df[:10]
print df.describe()

#standard deviation of all columns
# print df.std()
#frequency table cutting presitge and whether or not someone was admitted
# print pd.crosstab(df['admit'], df['prestige'], rownames=['admit'])

#target is admit, either yes(1) or no(1)
#data is the columns gre, gpa, prestige

#y = THE TARGET
y = df['admit']
y = np.array(y)

X = df.loc[:,'gre':]
X = np.array(X)
print X.shape
# print X[:10]

SEED = 42
n = 10

for i in range(n):
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=.20, random_state=i*SEED)

print '$$$$$$ pone $$$$$$$'	
print X_train.shape
print y_train.shape

# log_regr = linear_model.LinearRegression()
logreg = linear_model.LogisticRegression()
print logreg.fit(X_train,y_train)
print '------- admission prediction -------'
print logreg.predict(X_test)

print '------- admission probability -------'
print logreg.predict_proba(X_test)

#create dataframe of actual result, predicted result, probabilites
actualAdmission = y_test
predictedAdmission = logreg.predict(X_test)
probabilityAdmission = logreg.predict_proba(X_test)

print type(probabilityAdmission)
cara = pd.DataFrame(probabilityAdmission)
print cara.head()

d = {'actual': actualAdmission, 'predicted': predictedAdmission}
df = pd.DataFrame(d)
print df.head()

ipone = df.join(cara)
print ipone.head()
ipone.to_csv('yes.csv')
