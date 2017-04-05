
# Load libraries
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
import pydotplus
from sklearn import tree   
from IPython.display import Image
from sklearn import svm

# change working directory
import os
os.chdir("/home/midhun/My working directory For Machine Learning (Spyder)")

# loading dataset
url = "/home/midhun/My working directory For Machine Learning (Spyder)/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'target']
dataset = pandas.read_csv(url, names = names)

# shape
print(dataset.shape)

# head
print(dataset.head(20))

# description
print(dataset.describe())

# Class distribution
print(dataset.groupby('target').size())

# box and whisker plots
dataset.plot(kind='box', subplots = True, layout = (2,2), sharex = False, sharey = False)
plt.show()

# histograms
dataset.hist()
plt.show()

# scatter plot matrix
scatter_matrix(dataset)
plt.show()

#Split-out validation dataset
array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = validation_size, random_state = seed)

# Applying logistic regression
model = LogisticRegression()
model = model.fit(X_train, Y_train)

# checking the accuracy of training set
score = model.score(X_train, Y_train)

print(("The accuracy of training data set is: %f percent") % (score * 100))
print("\n")
print("Oops! Looks like it has overfitting problem :D")
print()
print("proceeding to model evaluation using 10-fold cross-validation...")
print()

# evaluate the model using 10-fold cross-validation
scores = cross_val_score(LogisticRegression(), X, Y, scoring='accuracy', cv=10)
print(scores)
print()
# Mean of the scores from 10-fold cross-validation
print(scores.mean())
print()
print("Negative! Got a lot of work to do!")
print()

# Predicting values
predicted = model.predict(X_validation)
print("This is the predicted values: \n")
print(predicted)

# Confusion_matrix
print(confusion_matrix(Y_validation, predicted))
print(classification_report(Y_validation, predicted))

# Applying Neural Networks
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes = (1,), \
                    random_state = None)
clf = clf.fit(X_train, Y_train)

res = clf.score(X_train, Y_train)

print(("Accuracy from MLP is %f") % (res))

results = cross_val_score(MLPClassifier(), X, Y, scoring = 'accuracy', cv=10)
print()
print(results)
print()
mean_of_result = results.mean()
print(("The average is: %f") % (mean_of_result))
print("\n")

# Predicting values
predict = clf.predict(X_validation)
print("The predicted values for MLPClassification is: \n")
print(predict)
print("\n")

# Confusion matrix
print(confusion_matrix(Y_validation, predict))
print()
print(classification_report(Y_validation, predict))

#prediction using Decision Trees
Dtree = tree.DecisionTreeClassifier()
Dtree = Dtree.fit(X_train, Y_train)

''' we have Python module pydotplus installed, we can generate a PDF file (or 
any other supported file type) directly in Python:'''

'''
dot_data = tree.export_graphviz(Dtree, out_file = None, feature_names=dataset.columns,  
                         class_names=dataset.target,  
                         filled=True, rounded=True,  
                         special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("iris.pdf")
print(Image(graph.create_png()))
'''

skore = Dtree.score(X_train, Y_train)
print(skore)

# Applying 10-fold cross validation
kfold = cross_val_score(tree.DecisionTreeClassifier(), X, Y, scoring='accuracy', cv=10)
print("DTREE accuracies:\n")
print(kfold)
print()
Dtree_mean = kfold.mean()
print()
print(("kfold mean is: %f") % (Dtree_mean * 100))

# Putting up predictions from Decision Trees
predikt = Dtree.predict(X_validation)
print("The predicted values for DecisionTreeClassifier is: \n")
print(predikt)
print("\n")

# Confusion matrix for predicted & actual values of Decision Tree
print(confusion_matrix(Y_validation, predikt))
print()
print(classification_report(Y_validation, predikt))

# Applying SVM
svm_classify = svm.SVC()
svm_classify = svm_classify.fit(X_train, Y_train)
print()
print("Accuracy using svm without k-fold cross validation: \n")
sckores = svm_classify.score(X_train, Y_train)
print(sckores)

# Applying k-fold cross validation
cross_vals = cross_val_score(svm.SVC(), X, Y, scoring = 'accuracy', cv=10)
print()
print("printing k-fold cross validation results of svm \n")
print(cross_vals) 
print()
print("taking the mean of the results from k-fold \n")
print(cross_vals.mean() * 100)

# Predictions from SVM
predhict = svm_classify.predict(X_validation)
print()
print("The predicted values for SVM classifier is: \n")
print(predhict)
print()

# Confusion matrix and classification report
print(confusion_matrix(Y_validation, predhict))
print()
print(classification_report(Y_validation, predhict))
print()


