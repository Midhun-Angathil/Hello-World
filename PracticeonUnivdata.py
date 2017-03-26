#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 15:19:48 2017

@author: midhun
"""

# Importing Libraries
import os
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn import tree

# Changing working directory
os.chdir("/home/midhun/My working directory For Machine Learning (Spyder)")

# Reading the file
url = "/home/midhun/My working directory For Machine Learning (Spyder)/UniversalBank.csv"
#names = ["ID", "Age", "Experience"	, "Income", "ZIP", "Code", "Family	", "CCAvg", "Education", "Mortgage	Personal", "Loan", "Securities", "Account", "CD", "Account", "Online	CreditCard"]
data = pd.read_csv(url)

# Shape
print(data.shape)

# Head
print(data.tail(50))

# Summary statistics
print(data.describe())

# Class Distribution
print(data.groupby('Personal Loan').size())

'''
#deleting attributes with NaN values
del data['Account.1']

print ()
print("Printing after deleting one attribute: \n")
print(data.head(20))
print()
'''

# Boxplots and whiskers
data.plot(kind='box', subplots = True, layout = (7,2), figsize = (10,20), sharex = False, sharey = False)
plt.show()

'''
# Dealing with outliers (works only on normally distributed data)
grouped = data.groupby('ID')

#statBefore = pd.DataFrame({'mean': grouped['Income'].mean(), 'median': grouped['Income'].median(), 'std' : grouped['Income'].std()})

statBefore = pd.DataFrame({'q1': grouped['Income'].quantile(.25), \
'median': grouped['Income'].median(), 'q3' : grouped['Income'].quantile(.75)})

def is_outlier(row):
    iq_range = statBefore.loc[row.ID]['q3'] - statBefore.loc[row.ID]['q1']
    median = statBefore.loc[row.ID]['median']
    if row.Income > (median + (1.5* iq_range)) or row.Income < (median - (1.5* iq_range)):
        return True
    else:
        return False
    
#apply the function to the original df:
data.loc[:, 'outlier'] = data.apply(is_outlier, axis = 1)
       
#filter to only non-outliers:
data_no_outliers = data[~(data.outlier)]
print(data.shape)
#print(data_no_outliers.shape)
print()
print("Printing dataset sample after removing outliers... \n")
print(data.head(20))
print("Summary Statistics: \n")
print(data.describe())
print()
print("With Outliers: \n")
print()
print(data.describe())

# Boxplots and whiskers after removing outliers
data.plot(kind='box', subplots = True, layout = (8,2), figsize = (10,20), sharex = False, sharey = False)
plt.show()
'''

# histogram
data.hist(figsize = (10,20))
plt.show()

#scatter plot matrix
scatter_matrix(data, figsize = (10,20))
plt.show()


# Split out valdation and train datasets
Y = data['Personal Loan']
del data['Personal Loan']
X = data.values 
validation_size = 0.20
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = validation_size, random_state = 1 )

# Apply decision Tree
MyTree = tree.DecisionTreeClassifier()
MyTree = MyTree.fit(X_train, Y_train)

skores = MyTree.score(X_train, Y_train)
print(skores)

# Applying k-fold cross-validation
folds = cross_val_score(tree.DecisionTreeClassifier(), X, Y, scoring = 'accuracy', cv=10)
print()
print(folds)

mean_folds = folds.mean()

print()
print(("My mean accuracy of 10-fold cross validation is: %f") % (mean_folds * 100))

# Predicting the values on test set
prediction = MyTree.predict(X_test)
print()
print("My predictions: \n")
print(prediction)
print()

# Applying confusion matrix and classification report
print(confusion_matrix(Y_test, prediction))
print()
print(classification_report(Y_test, prediction))
print() 