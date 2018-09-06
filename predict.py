# -*- coding: utf-8 -*-
"""
@author: han0305
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('creditcard.csv')

print(data.columns)
print(data.shape)
print(data.head(5))
data.hist(figsize=(20,20))
plt.show()

fraud = data[data["Class"]==1]
valid = data[data["Class"]==0]

outlier_fraction = len(fraud)/float(len(valid))
print("outlier fraction {} :".format(outlier_fraction))
print("fraud cases {} :".format(len(fraud)))
print("valid cases {} :".format(len(valid)))

corrmat  = data.corr()
fig = plt.figure(figsize = (12,9))
sns.heatmap(corrmat,vmax=.8,square=True)
plt.show()

columns = data.columns.tolist()
print(columns)

columns = [c for c in columns if c not in ["Class"]]
target = "Class" 
x=data[columns]
y=data[target]

from sklearn.metrics import classification_report,accuracy_score 
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

state = 1
classifiers  = {"Isolation Forest" : IsolationForest(max_samples=len(x),contamination = outlier_fraction,random_state = state), "Local Outlier Factor":LocalOutlierFactor(n_neighbors=20,contamination=outlier_fraction)}
n_outliers = len(fraud)
plt.figure(figsize=(9, 7))
n_outliers = len(fraud)


for i, (clf_name, clf) in enumerate(classifiers.items()):
    
    
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
    
 
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    n_errors = (y_pred != Y).sum()
    
    
    print('{}: {}'.format(clf_name, n_errors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))