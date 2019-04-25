# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 14:59:51 2018

@author: kohul
"""
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 
from sklearn.ensemble import AdaBoostClassifier #For Classification
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
##########################################################################################################

###########################################################################################################
dataset1 = 'C:/Users/kohul/OneDrive/Documents/ML/titanic.csv'

data1 = pd.read_csv(dataset1, sep=',')

#####Creating Dummies###########
def createDummy(data):
    data = pd.get_dummies(data, drop_first=True)
    return data

###############################
def Check(data):
    for col in data.columns:
        if len(data[col].unique()) == len(data[col]):
            data = data.drop(col, axis=1)
            print(col)
    return data


# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred): 
      
    print("Confusion Matrix: ", 
        confusion_matrix(y_test, y_pred)) 
      
    print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
      
    print("Report : ", 
    classification_report(y_test, y_pred))

#def remove_category_col(data):
#    for col in data.columns:
#        if isinstance(data[col][1], str):
#            data = data.drop(col, axis=1)
#            print(col)
#    return data

##############################Data Set 1#################################################
#Data Processing
Y=data1.iloc[:,1:2].values
Y = (np.where(Y>np.median(Y),1,0))
data1 = data1.drop(['Survived','Name','Ticket','Cabin'],axis=1)
data1 = Check(data1)
X = createDummy(data1)
print(X.columns)

###############################SVM######################################################
X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(X, Y, test_size=0.3, random_state=10)
#
print("/////////////////////////Linear Kernel////////////////////////////")
model = svm.SVC(decision_function_shape='ovr', kernel='linear')
model.fit(X_train_1, Y_train_1)
y_pred = model.predict(X_test_1)
cal_accuracy(Y_test_1, y_pred)

print("/////////////////////////RBF Kernel////////////////////////////")
model = svm.SVC(decision_function_shape='ovr', kernel='rbf')
model.fit(X_train_1, Y_train_1)
y_pred = model.predict(X_test_1)
cal_accuracy(Y_test_1, y_pred)

print("/////////////////////////Sigmoid Kernel////////////////////////////")
model = svm.SVC(decision_function_shape='ovr', kernel='sigmoid')
model.fit(X_train_1, Y_train_1)
y_pred = model.predict(X_test_1)
cal_accuracy(Y_test_1, y_pred)

##################################Decision Tree#############################################

def train_using_gini(X_train, y_train, depth): 
  
    # Creating the classifier object 
    clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=depth) 
  
    # Performing training 
    clf_gini.fit(X_train, y_train) 
    return clf_gini 
# Function to make predictions 
def prediction(X_test, clf_object): 
  
    # Predicton on test with giniIndex 
    y_pred_tree = clf_object.predict(X_test)  
    return y_pred_tree


#####################################################################################

print("///////////////////////////////////////////Default Decition Tree///////////////////////////////////")

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100)
clf_gini.fit(X_train_1, Y_train_1) 
y_pred_tree = prediction(X_test_1, clf_gini)
cal_accuracy(Y_test_1, y_pred_tree)

#####################################################################################
    
accuracy_test  = []
accuracy_train = []
depth = []
for i in range(1,20):
    clf_gini = train_using_gini(X_train_1, Y_train_1, i)
    y_pred_tree = prediction(X_test_1, clf_gini)
    y_pred_tree_train = prediction(X_train_1, clf_gini)
    accuracy_test.append(accuracy_score(Y_test_1,y_pred_tree)*100)
    accuracy_train.append(accuracy_score(Y_train_1,y_pred_tree_train)*100)
    depth.append(i)
#    cal_accuracy(Y_test_1, y_pred_tree)
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.plot(depth,accuracy_test)
plt.plot(depth,accuracy_train)
#fig, ax1 = plt.subplots()
#ax1.plot(depth,accuracy_test)
#ax2 = ax1.twinx()
#ax2.plot(depth,accuracy_train)
##fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
#######################################################################################
print("///////////////////////////////////////////Default Boosting///////////////////////////////////")
dt = DecisionTreeClassifier(criterion = "gini", random_state = 100) 
clf = GradientBoostingClassifier(n_estimators=200,learning_rate=1)
clf.fit(X_train_1,Y_train_1)
y_pred_boost = clf.predict(X_test_1)
cal_accuracy(Y_test_1, y_pred_boost) 
#################################Boosting#################################################
accuracy_test  = []
accuracy_train = []
depth = []
for i in range(1,20):
    dt = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=i) 
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=i,learning_rate=0.05)
    clf.fit(X_train_1,Y_train_1)
    y_pred_boost = clf.predict(X_test_1)
    y_pred_boost_train = clf.predict(X_train_1)
    accuracy_test.append(accuracy_score(Y_test_1,y_pred_boost)*100)
    accuracy_train.append(accuracy_score(Y_train_1,y_pred_boost_train)*100)
    depth.append(i)
#    cal_accuracy(Y_test_1, y_pred_boost) 
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.plot(depth,accuracy_test)
plt.plot(depth,accuracy_train)
plt.show()

print("///////////////////////////////////////////Chosen Decition Tree///////////////////////////////////")
clf_gini = train_using_gini(X_train_1, Y_train_1,3)
y_pred_tree = prediction(X_test_1, clf_gini)
cal_accuracy(Y_test_1, y_pred_tree)
print("///////////////////////////////////////////Chosen Boosting///////////////////////////////////")
dt = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3) 
clf = GradientBoostingClassifier(n_estimators=200,max_depth = 4, learning_rate=0.05)
clf.fit(X_train_1,Y_train_1)
y_pred_boost = clf.predict(X_test_1)
cal_accuracy(Y_test_1, y_pred_boost) 
################################Data set2################################################
#data processing

##############################SVM######################################################

#################################Decision Tree#############################################

#################################Boosting#################################################
